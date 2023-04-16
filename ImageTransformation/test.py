#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Inference Phase for Deep Learning Models with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-09 UTC + 08:00, Chinese Standard Time(CST)

# ===============================================================
# Step 0. Auxiliary functions or scripts
# Step 1. Parse the JSON configure file
# Step 2. Create Dataset and DatasetLoader and Solver for Model
# Step 3. Inference Phase for Models or Networks
# ===============================================================
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
import os
import argparse
import datetime
import time
import random

# ----- standard library -----
import torch
import imageio
import numpy as np

# ----- custom library -----
import options.options_configure as option
from utils import utiliy
from solver import create_solver
from data_improcessing import create_dataset
from data_improcessing import create_dataloader


# --------------------------------------
# Step 0. Auxiliary functions or scripts
def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")


# set the random seed, 
# the result of deep learning can be complete recovery, 
# relieve affect to final result because of some random operations.
def set_seed(seed=42):
    if seed is None:
        seed = random.randint(1, 10000)
    print("========> Random Seed is {}".format(seed))

    # Even if you don't use them, you still have to import
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------
# Step 3. Inference Phase for Models or Networks
def main_inference_phase():
    # Firstly, where is your inference JSON configure about Model, Dataset, Training ... 
    path2json_config = r"./options/test/test_EDSR_x4.json"

    # Step 1. Parse the JSON configure file
    arg_parser = argparse.ArgumentParser(description="Train Deep Learning Modles")
    arg_parser.add_argument("-opt", type=str, required=False, default=path2json_config, help="Path to options JSON file.")
    option_config = option.parse_config(arg_parser.parse_args().opt)
    option_config = option.dict_to_nonedict(option_config)

    set_seed(seed=option_config["manual_seed"])

    # -----------------------------------------------------------------
    # Step 1. initial configure
    scale = option_config["scale"]
    degradation_process = option_config["degradation"]

    model_name = option_config["networks"]["which_model"].upper()
    if option_config["self_ensemble"]:
        model_name += "plus"
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # create test dataloader
    # 可以同时一系列的基准数据集进行测试, 也可以一个一个的进行测试
    benchmark_names = []
    test_loaders = []
    for _, dataset_config in sorted(option_config["datasets"].items()):
        test_set = create_dataset(dataset_config)
        test_loader = create_dataloader(test_set, dataset_config)
        test_loaders.append(test_loader)
        print("========> Test Dataset: {}, Number of images: {}".format(test_set.name(), len(test_set)))
        benchmark_names.append(test_set.name())
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Create solver and load model
    solver = create_solver(option_config)

    print("========> Start Test and Inference Phase")
    printbar()
    print("==============================================================")
    print("Method: {} || Scale: {} || Degradation: {}".format(model_name, scale, degradation_process))

    for benchmark, test_loader in zip(benchmark_names, test_loaders):
        print("========> Test Benchmark Dataset: {}".format(benchmark))

        reconstructed_list = []
        path_list = []
        total_psnr = []
        total_ssim = []
        total_time = []

        # 测试合成数据集还是真实数据集, 即是否需要 Ground Trues (GT)
        need_GT = False if test_loader.dataset.__class__.__name__.find("Paired") < 0 else True

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch, need_GT=need_GT)

            # ------------------------------------------
            # calculate forward time (inference time)
            # 该推断时间需要注意, 跟 CPU, 进程数量有关系, 在各种方法对比的时候仅供参考, 并不能很准确
            time_start = time.time()
            solver.test()
            time_end = time.time()
            total_time.append((time_end - time_start))
            # ------------------------------------------

            visuals = solver.get_current_visual(need_GT=need_GT)
            reconstructed_list.append(visuals["reconstructed_imgs"])

            # calculate metrics PSNR and SSIM using Python
            if need_GT: # PSNR and SSIM are reference metrics 有参指标
                psnr, ssim = utiliy.calculate_metrics(visuals["reconstructed_imgs"], visuals["ground_trues"], crop_border=scale)

                total_psnr.append(psnr)
                total_ssim.append(ssim)
                path_list.append(os.path.basename(batch["groundtrue_path"][0]).replace("GT", model_name))

                print("{}/{} {} || PSNR(dB)/SSIM: {:.2f}/{:.4f} || Timer: {:.4f} seconds.".format(
                    iter+1, len(test_loader), os.path.basename(batch["input_path"][0]),
                    psnr, ssim, (time_end - time_end) ))

            else: # 对真实图像进行测试, 即没有 GT
                path_list.append(os.path.basename(batch["input_path"][0]))
                print("{}/{} {} || Timer: {:.4f} seconds.".format(
                    iter+1, len(test_loader), os.path.basename(batch["input_path"][0]), (time_end - time_end) ))
        # ------------------Ending this benchmark dataset-----------------

        # --------------------------------
        # 对整个测试数据集进行评估
        if need_GT:
            print("========> {} Average PSNR(dB) || SSIM || Speed in Inference Phase for {}".format(model_name, benchmark))
            print("PSNR: {:.2f},     SSIM: {:.4f},    Speed: {:.4f}".format(
                sum(total_psnr)/len(total_psnr),
                sum(total_ssim)/len(total_ssim),
                sum(total_time)/len(total_time) ))
        else:
            print("========> {} Average Speed in Inference Phase  for {} is {:.4f} seconds.".format(
                model_name, benchmark, sum(total_time)/len(total_time) ))
        # ------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------
        # Saving the reconstructed images for further subjective evaluation or calculate PSNR/SSIM on MATLAB.
        reconstructed_path = r"./result_imgs/Reconstructed/"

        if need_GT:
            save_img_path = os.path.join(reconstructed_path+degradation_process, model_name, benchmark, "x{}".format(scale))
        else:
            save_img_path = os.path.join(reconstructed_path+benchmark, model_name, "x{}".format(scale))

        print("========> Saving Reconstructed Images of {} in {}\n".format(benchmark, save_img_path))
        os.makedirs(save_img_path, exist_ok=True)
        for img, name in zip(reconstructed_list, path_list):
            imageio.imwrite(os.path.join(save_img_path, name), img)
        # ----------------------------------------------------------------------------------------------------

    print("==============================================================")
    printbar()
    print("================= Finishing Inference ========================")


# -------------------------------
if __name__ == "__main__":
    main_inference_phase()
# --------------------------------------------
# python test.py | tee test_log_Set5.txt
# python test.py | tee test_log_historical.txt
# python test.py | tee test_log_Set14.txt
# python test.py | tee test_log_BSD100.txt
# python test.py | tee test_log_Urban100.txt
# python test.py | tee test_log_Manga109.txt
# --------------------------------------------