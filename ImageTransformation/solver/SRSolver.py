#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The single image super-resolution (SISR) solver with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-11 UTC + 08:00, Chinese Standard Time(CST)
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
import os
import pathlib
from collections import OrderedDict

# ----- standard library -----
import torch
import torchvision
import pandas as pd
import imageio

# ----- custom library -----
from .base_solver import BaseSolver
from networks import create_model, init_weights
from utils import utiliy


class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.train_opt = opt["solver"]
        self.InputImgs = self.Tensor()
        self.GroundTrues = self.Tensor()
        self.ReconstructedImgs = None

        # 以 字典 形式记录 loss value, metrics, and learning_rate
        self.records = {"train_loss": [],
                        "val_loss": [],
                        "psnr": [],
                        "ssim": [],
                        "learning_rate": []}

        # according to the configuration file for creating model using Deep Learning
        self.model = create_model(opt)

        # ------------Training phase settings for Model----------------------
        if self.is_train:
            self.model.train()

            # ------------Loss Function----------------------
            # set loss function L1 norm or L2 norm
            loss_type = self.train_opt["loss_type"]
            if loss_type == "L1":
                self.criterion_pixel = torch.nn.L1Loss()
            elif loss_type == "L2":
                self.criterion_pixel = torch.nn.MSELoss()
            else:
                raise NotImplementedError("Loss type {} is not Implemented.".format(loss_type))

            if self.use_gpu:
                self.criterion_pixel = self.criterion_pixel.cuda()
            # ---------------------------------------------------------

            # ------------Optimizer for Model ----------------------
            # set the optimizer for the Model when training phase
            # 对参数权重是否需要进行衰减, 以达到正则化效果和约束
            weight_decay = self.train_opt["weight_deacy"] if self.train_opt["weight_decay"] else 0

            # 选择优化器
            optimizer_type = self.train_opt["optim_type"].upper()
            if optimizer_type == "ADAM": 
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                 lr=self.train_opt["learning_rate"],
                                                 weight_decay=weight_decay)
            else:
                raise NotImplementedError("Optimizer type {} is not Implemented.".format(optimizer_type))
            # ---------------------------------------------------------------------------------------------

            # ------------Learning rate scheduler for training Model ----------------------
            if self.train_opt["learning_rate_scheme"].lower() == "multisteplr":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                    self.train_opt["learning_rate_steps"],
                                                                    self.train_opt["learning_rate_gamma"])
            elif self.train_opt["learning_rate_scheme"].lower() == "cosineanealinglr":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR()
            else:
                raise NotImplementedError("Only MultiStepLR scheme is supported.")
            # -----------------------------------------------------------------------
        # ------------Ending Training phase settings for Model-------------------------

        self.load() # custom function to load or initialize the Model firstly.
        self.print_network() # custom function to print the model information firstly.

        print("========> Solver Initialized : {} || GPU: {}".format(self.__class__.__name__, self.use_gpu))

        if self.is_train:
            print("Optimizer is : {} ".format(self.optimizer))
            print("Learning_rate scheduler milestones : {}, lambda: {}".format(self.scheduler.milestones, self.scheduler.gamma))
    # ------------Ending Initializing __init__ constructor for class-------------------------


    def load(self):
        """load or initialize network.
        """
        # 断点续训, 或者加载已经训练好的模式进行测试和推断 inference phase
        if (self.is_train and self.opt["solver"]["pretrain"]) or not self.is_train:
            model_path = self.opt["solver"]["pretrained_path"]
            if model_path is None:
                raise ValueError("[Error] The pretrained_path does not declarate in *.json.")

            print("========> Loading model from {}".format(model_path))
            if self.is_train: # 断点续训
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint["state_dict"])

                if self.opt["solver"]["pretrain"] == "resume":
                    self.current_epoch = checkpoint["epoch"] + 1
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.best_pred = checkpoint["best_pred"]
                    self.best_epoch = checkpoint["best_epoch"]
                    self.records = checkpoint["records"]
            else: # inference phase
                checkpoint = torch.load(model_path)
                if "state_dict" in checkpoint.keys():
                    checkpoint = checkpoint["state_dict"]

                load_func = self.model.load_state_dict if isinstance(self.model, torch.nn.DataParallel) else self.model.module.load_state_dict

                load_func(checkpoint)

        # 第一次加载并初始化模型
        else:
            self.net_init()


    def print_network(self):
        """print network summary including module and number of parameters

        Returns:
            _type_: _description_
        """
        str_name, num_parameter = self.get_network_description(self.model)
        if isinstance(self.model, torch.nn.DataParallel):
            net_structure_str = "{} - {}".format(self.model.__class__.__name__, self.model.module.__class__.__name__)
        else:
            net_structure_str = "{}".format(self.model.__class__.__name__)

        print("===============================================================")
        print("=========> Network Summary\n")
        net_lines = []
        line_info = str_name + "\n"
        print(line_info)
        net_lines.append(line_info)
        line_info = "Network Structure: [{}], with Parameters: [{}]".format(net_structure_str, num_parameter)
        print(line_info)
        net_lines.append(line_info)

        if self.is_train:
            # 第一次保存网络结构到文件 ./experiments/network_summary.txt
            with open(os.path.join(self.experiment_root, "network_summary.txt"), "w") as file:
                file.writelines(net_lines)

        print("===============================================================")


    def net_init(self, init_type="kaiming"):
        print("========> Initializing the network using {}".format(init_type))
        init_weights(self.model, init_type=init_type)


    def feed_data(self, batch, need_GT=True):
        input_imgs = batch["input_img"]
        self.InputImgs.resize_(input_imgs.size()).copy_(input_imgs)

        if need_GT:
            target_imgs = batch["groundtrue"]
            self.GroundTrues.resize_(target_imgs.size()).copy_(target_imgs)


    def train_step(self):
        """函数风格的训练循环, 对脚本形式上作了简单的函数封装
        step 表示对一个 batch 的数据进行训练一次; epoch 表示对整个数据集所有的数据 dataset 进行训练一次

        Returns:
            _type_: _description_
        """
        self.model.train()
        # 根据反向传播的梯度计算方式可知, 需要对 tensor.grad 这个储存梯度信息的属性进行清零操作
        self.optimizer.zero_grad()

        loss_batch = 0.0
        # 为了减少对显存的需求, 将一次的 batch 数据拆分为几份更小的
        sub_batch_size = int(self.InputImgs.size(0) / self.split_batch)

        for idx in range(self.split_batch):
            loss_sub_batch = 0.0
            split_input_imgs = self.InputImgs.narrow(0, idx * sub_batch_size, sub_batch_size)
            split_ground_trues = self.GroundTrues.narrow(0, idx * sub_batch_size, sub_batch_size)

            output_imgs = self.model(split_input_imgs)
            loss_sub_batch = self.criterion_pixel(output_imgs, split_ground_trues)

            loss_sub_batch /= self.split_batch
            # 反向传播
            loss_sub_batch.backward()

            loss_batch += (loss_sub_batch.item())

        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step()
            self.last_epoch_loss = loss_batch
        else:
        # for stable training skip the batch in training phase.
            print("[Warning] Skip this batch! (Loss): {}".format(loss_batch))

        # 验证模型
        self.model.eval()
        return loss_batch
    # ------------Ending Training on Step-------------------------


    def test(self):
        self.model.eval()

        # 使用 torch.no_grad() 避免梯度记录, 也可以通过操作 model.w.data 实现避免梯度记录
        with torch.no_grad():
            forward_func = self._overlap_crop_forward if self.use_chop else self.model.forward
            if self.self_ensemble and not self.is_train:
                ReconstructedImage = self._forward_geometric_8(self.InputImgs, forward_function=forward_func)
            else:
                ReconstructedImage = forward_func(self.InputImgs)

            if isinstance(ReconstructedImage, list):
                self.ReconstructedImgs = ReconstructedImage[-1]
            else:
                self.ReconstructedImgs = ReconstructedImage

        self.model.train()
        if self.is_train:
            loss_pixel = self.criterion_pixel(self.ReconstructedImgs, self.GroundTrues)
            return loss_pixel.item()
    # ------------Ending test for Model-------------------------


    # ------------Geometric Self-ensemble for Image Super-Resolution to Model to plus-------------------------
    # Paper(EDSR and EDSR+): Enhanced Deep Residual Networks for Single Image Super-Resolution 
    # CVPR-2017: https://arxiv.org/abs/1707.02921
    # Note geometric self-ensemble is valid only for symmetric downsampling methods such as bicubic downsampling.
    # -----------------------------------------------------------------------------------------------------------
    # Paper: Seven ways to improve example-based single image super resolution
    # CVPR-2016: https://openaccess.thecvf.com/content_cvpr_2016/papers/Timofte_Seven_Ways_to_CVPR_2016_paper.pdf
    # -----------------------------------------------------------------------------------------------------------
    def _forward_geometric_8(self, x, forward_function):
        def _transform_geometric(img, operator_geometric):
            img = img.float()

            img2np = img.data.cpu().numpy()
            if operator_geometric == "vertical_flip":
                # image shape [B, C, H, W]
                geometric_img_np = img2np[:, :, :, ::-1].copy()
            elif operator_geometric == "horizontal_flip":
                geometric_img_np = img2np[:, :, ::-1, :].copy()
            elif operator_geometric == "rotating_90":
                geometric_img_np = img2np.transpose((0, 1, 3, 2)).copy()

            ret_img = self.Tensor(geometric_img_np)

            return ret_img

        low_resolution_list = [x]
        # 2^3 = 8, 
        # [img] ----> vertical_flip ----> [img, img_v]
        # [img, img_v] ----> horizontal_flip ----> [img, img_v, img_h, img_v_h]
        # [img, img_v, img_h, img_v_h] ----> rotating_90 ----> [img, img_v, img_h, img_v_h, img_t, img_v_t, img_h_t, img_v_h_t]
        for geometric_trans in "vertical_flip", "horizontal_flip", "rotating_90":
            low_resolution_list.extend([_transform_geometric(img, geometric_trans) for img in low_resolution_list])

        super_resolution_list = []
        for aug_geometric in low_resolution_list:
            super_resolution = forward_function(aug_geometric)
            if isinstance(super_resolution, list):
                super_resolution_list.append(super_resolution[-1])
            else:
                super_resolution_list.append(super_resolution)

        for idx in range(len(super_resolution_list)):
            if idx > 3:
                super_resolution_list[idx] = _transform_geometric(super_resolution_list[idx], "rotating_90")
            if idx % 4 > 1:
                super_resolution_list[idx] = _transform_geometric(super_resolution_list[idx], "horizontal_flip")
            if (idx % 4) % 2 == 1:
                super_resolution_list[idx] = _transform_geometric(super_resolution_list[idx], "vertical_flip")

        output_cat = torch.cat(super_resolution_list, dim=0)
        ouput_img_SR = output_cat.mean(dim=0, keepdim=True)

        return ouput_img_SR


    def _overlap_crop_forward(self, x, shave=10, min_size=100000, bic=None):
        """Chop for less memory comsumption during test.

        Args:
            upscale (_type_): _description_
        """
        num_GPU = 2
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w] ]

        if bic is not None: # for image super-resolution
            bic_h_size = h_size * scale
            bic_w_size = w_size * scale
            bic_h = h * scale
            bic_w = w * scale
            
            bic_list = [
                bic[:, :, 0:bic_h_size, 0:bic_w_size],
                bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
                bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
                bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w] ]

        if w_size * h_size < min_size: # patch image total pixels
            sr_list = []
            for i in range(0, 4, num_GPU):
                lr_batch = torch.cat(lr_list[i:(i + num_GPU)], dim=0)
                if bic is not None:
                    bic_batch = torch.cat(bic_list[i:(i + num_GPU)], dim=0)

                sr_batch_temp = self.model(lr_batch)

                if isinstance(sr_batch_temp, list):
                    sr_batch = sr_batch_temp[-1]
                else:
                    sr_batch = sr_batch_temp

                sr_list.extend(sr_batch.chunk(num_GPU, dim=0))
        else:
            sr_list = [self._overlap_crop_forward(patch, shave=shave, min_size=min_size) for patch in lr_list]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output


    def save_checkpoint(self, epoch, is_best):
        """Saving  the checkpoint to the experiments folder.

        Args:
            epoch (_type_): _description_
            is_best (bool): _description_

        Returns:
            _type_: _description_
        """
        file_name = os.path.join(self.checkpoint_dir, "last_ckp.pth")
        print("========> Saving last checkpoint to {} .".format(file_name))

        # 以 字典 的形式存储模型的权重以及相关信息, 为了断点续训
        ckp = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_pred": self.best_pred,
            "best_epoch": self.best_epoch,
            "records": self.records
        }
        torch.save(ckp, file_name)

        # saving the best epoch checkpoint (best denotes the metric PSNR)
        if is_best:
            print("========> Saving best checkpoint to {} ".format(file_name.replace("last_ckp", "besk_ckp")))
            torch.save(ckp, file_name.replace("last_ckp", "besk_ckp"))

        # saving the epoch checkpoint with user custom.
        if epoch % self.train_opt["save_ckp_step"] == 0:
            print("========> Saving checkpoint {} to {} ".format(epoch, file_name.replace("last_ckp", "epoch_{}_ckp".format(epoch))))
            torch.save(ckp, file_name.replace("last_ckp", "epoch_{}_ckp".format(epoch)))


    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]


    def update_learning_rate(self):
        # 根据 epoch 对学习率按照一定的比例 gamma 进行衰减
        return self.scheduler.step()


    def get_current_log(self):
        # the getter method to class encapsulation for logger.
        log = OrderedDict()
        log["epoch"] = self.current_epoch
        log["best_pred"] = self.best_pred
        log["best_epoch"] = self.best_epoch
        log["records"] = self.records

        return log

    def set_current_log(self, log):
        # the setter method to class encapsulation for logger.
        self.current_epoch = log["epoch"]
        self.best_pred = log["best_pred"]
        self.best_epoch = log["best_epoch"]
        self.records = log["records"]
    
    def save_current_log(self):
        # 利用 pandas DataFrame 存储训练日志
        data_frame = pd.DataFrame(
            data={
                "train_loss": self.records["train_loss"],
                "val_loss": self.records["val_loss"],
                "psnr": self.records["psnr"],
                "ssim": self.records["ssim"],
                "learning_rate": self.records["learning_rate"]
            },
        index=range(1, self.current_epoch + 1)
        )

        # 将 pandas DataFrame 记录的日志数据以 CSV 格式保存到硬盘
        data_frame.to_csv(os.path.join(self.records_dir, "train_records.csv"), index="epoch")


    def get_current_visual(self, need_np=True, need_GT=True):
        """return input_images, reconstructed images(ground_trues)

        Args:
            need_np (bool, optional): _description_. Defaults to True.
            need_GT (bool, optional): _description_. Defaults to True.
        """
        out_dict = OrderedDict()
        out_dict["input_imgs"] = self.InputImgs.data[0].float().cpu()
        out_dict["reconstructed_imgs"] = self.ReconstructedImgs.data[0].float().cpu()

        if need_np:
            out_dict["input_imgs"], out_dict["reconstructed_imgs"] = utiliy.tensor2np([out_dict["input_imgs"], out_dict["reconstructed_imgs"]], self.opt["rgb_range"])

        if need_GT:
            out_dict["ground_trues"] = self.GroundTrues.data[0].float().cpu()
            if need_np:
                out_dict["ground_trues"] = utiliy.tensor2np([out_dict["ground_trues"]], self.opt["rgb_range"])[0]

        return out_dict


    def save_current_visual(self, epoch, iter):
        """Saving visual results fro comparison the Reconstructed Images and GroundTrues.

        Args:
            epoch (_type_): _description_
            iter (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 每个多少个 epoch 保存主观结果
        if epoch % self.save_vis_step == 0:
            visual_list = []
            visuals = self.get_current_visual(need_np=False)
            # [B, C, H, W] ----> squeeze [C, H, W]
            visual_list.extend([utiliy.quantize_image(visuals["ground_trues"].squeeze(0), self.opt["rgb_range"]),
                                utiliy.quantize_image(visuals["reconstructed_imgs"].squeeze(0), self.opt["rgb_range"]) ])

            visual_images = torch.stack(visual_list)
            # 画布 第一行是 GT, 第二行是 重构图像, 便于主观对比
            visual_images = torchvision.utils.make_grid(visual_images, nrow=2, padding=5)
            # [C, H, W] ---> [H, W, C]
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()

            imageio.imwrite(pathlib.Path(os.path.join(self.visual_dir, "epoch_{}_img_{}.png".format(epoch, iter + 1))), visual_images)