#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Train Phase for Deep Learning Models with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-09 UTC + 08:00, Chinese Standard Time(CST)

# ==================================================
# Step 0. Auxiliary functions or scripts
# Step 1. Parse the JSON configure file
# Step 2. Create Dataset and DatasetLoader
# Step 3. Create Deep Learning Models or Networks
# Step 4. Train Models or Networks
# Step 5. Train Phase for Models or Networks
# ==================================================
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
import datetime
import argparse
import random

# ----- standard library -----
import torch
import numpy as np
from tqdm import tqdm

# ----- custom library -----
import options.options_configure as option
from data_improcessing import create_dataset
from data_improcessing import create_dataloader
from solver import create_solver
from utils import utiliy


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


# ------------------------------------------
# Step 2. Create Dataset and DatasetLoader
def get_data(opt):
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print("========> Train Dataset: {}, Number of images: {}.".format(train_set.name(), len(train_set)))

            if train_loader is None:
                raise ValueError("[Error] The training data does not exist.")
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print("========> Val Dataset: {}, Number of images: {}.".format(val_set.name(), len(val_set)))

            if val_loader is None:
                raise ValueError("[Error] The training data does not exist.")
        else:
            raise NotImplementedError("[Error] Dataset phase {} in *.json is not Implemented.".format(phase))

    return train_loader, val_loader, train_set, val_set


# -------------------------------------------------
# Step 3. Create Deep Learning Models or Networks
# ---------------------------------
# Step 4. Train Models or Networks
# 函数风格的训练循环, 对脚本形式上作了简单的函数封装
# ------------------------------------------------
def train_model(opt, dataload_train, dataload_val, train_set, val_set):
    # 模型的创建以及其训练等一些操作的解决方案
    solver = create_solver(opt)

    scale = opt["scale"] # for image super-resolution
    model_name = opt["networks"]["which_model"].upper()

    print("================== Start Training =========================")
    printbar()
    print("============================================================================")

    solver_log = solver.get_current_log()
    num_epoch = int(opt["solver"]["num_epochs"])
    start_epoch = solver_log["epoch"] # 考虑断点续训情况

    print("Method: {} || Scale for SR: {} || Epoch Range: {}~{}".format(model_name, scale, start_epoch, num_epoch))

    for epoch in range(start_epoch, num_epoch + 1):
        print("\n========> Training Epoch: {}/{}, Learning Rate: {:f}".format(epoch, num_epoch, solver.get_current_learning_rate()))

        # Initialization
        solver_log["epoch"] = epoch

        # Training Model
        train_loss_list = []
        with tqdm(total=len(dataload_train), desc="Epoch: {}/{}".format(epoch, num_epoch), miniters=1) as t:
            for iter, batch in enumerate(dataload_train):
                solver.feed_data(batch)
                iter_loss = solver.train_step()
                batch_size = batch["input_img"].size(0)
                train_loss_list.append(iter_loss * batch_size)

                t.set_postfix_str("Batch Loss: {:.4f}".format(iter_loss))
                t.update()

        solver_log["records"]["train_loss"].append(sum(train_loss_list)/len(train_set))
        solver_log["records"]["learning_rate"].append(solver.get_current_learning_rate())

        print("\nEpoch: {}/{}, Average Train Loss is : {:.6f}".format(epoch, num_epoch, sum(train_loss_list)/len(train_set)))
        # -------------------------------------------------------------------------------------------------------------------

        print("\n========> Validating on {} epoch.".format(epoch))
        psnr_list = []
        ssim_list = []
        val_loss_list = []

        for iter, batch in enumerate(dataload_val):
            solver.feed_data(batch)
            iter_loss = solver.test()
            val_loss_list.append(iter_loss)

            # Calculate evaluation metrics
            visuals = solver.get_current_visual()
            psnr, ssim = utiliy.calculate_metrics(visuals["reconstructed_imgs"], visuals["ground_trues"], crop_border=scale)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if opt["save_image"]: # 是否需要保存主观对比结果
                solver.save_current_visual(epoch, iter)

        solver_log["records"]["val_loss"].append(sum(val_loss_list)/len(val_loss_list))
        solver_log["records"]["psnr"].append(sum(psnr_list)/len(psnr_list))
        solver_log["records"]["ssim"].append(sum(ssim_list)/len(ssim_list))
        # --------------------------------------------------------------------------------

        # record the best epoch (best denotes the PSNR)
        epoch_is_best = False
        if solver_log["best_pred"] < (sum(psnr_list)/len(psnr_list)):
            solver_log["best_pred"] = (sum(psnr_list)/len(psnr_list))
            epoch_is_best = True
            solver_log["best_epoch"] = epoch

        print("{} PSNR: {:.2f}, SSIM: {:.4f}, Loss: {:.6f}, Best PSNR: {:.2f} in Epoch: {}".format(
            val_set.name(), sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list), 
            sum(val_loss_list)/len(val_loss_list), solver_log["best_pred"], solver_log["best_epoch"]
        ))

        # 保存记录到硬盘
        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()

        # update learning rate accorading to the learning rate strategy schedule.
        solver.update_learning_rate()
    # -------------------------------------------epoch iterator--------------------------------------

    printbar()
    print("================== Finishing Training =========================")


# --------------------------------------------
# Step 5. Train Phase for Models or Networks
def main_train_phase():
    # Firstly, where is your train JSON configure about Model, Dataset, Training ... 
    path2json_config = r"./options/train/train_EDSR_x4.json"
    

    # Step 1. Parse the JSON configure file
    arg_parser = argparse.ArgumentParser(description="Train Deep Learning Modles")
    arg_parser.add_argument("-opt", type=str, required=False, default=path2json_config, help="Path to options JSON file.")
    option_config = option.parse_config(arg_parser.parse_args().opt)

    set_seed(seed=option_config["manual_seed"])

    train_loader, val_loader, train_set, val_set = get_data(option_config)

    train_model(option_config, train_loader, val_loader, train_set, val_set)


# -------------------------------
if __name__ == "__main__":
    main_train_phase()
# -------------------------------------------
# python train.py | tee EDSR_train_log.txt
# -------------------------------------------