#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 09_imageSuperResolution.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-10
@copyright Copyright (c) 2024 Wei Li
@Description: 数字图像处理+卷积神经网络: 图像超分辨率重建 SRCNN
@Paper: Image Super-Resolution Using Deep Convolutional Networks
@Link: https://arxiv.org/abs/1501.00092
'''

import os
import glob
import tqdm
import cv2 as cv
import math
import torch


class SRCNN(torch.nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            torch.nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            torch.nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = torch.nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight.data, 0.0, 
                                      math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                torch.nn.init.zeros_(module.bias.data)

        torch.nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        torch.nn.init.zeros_(self.reconstruction.bias.data)


class ImageSR():
    """利用预训练的 SRCNN 模型进行推理, 对图像进行超分辨率重建;
        计算重建后的图像 img_SR 和 img_HR 的 PSNR & SSIM;
    """
    def __init__(self, original_folder, save_folder, weights_path):
        self.m_image_list = []
        self.m_original_folder = original_folder
        self.m_save_folder = save_folder
        self.weights_path = weights_path

    def readImageFolder(self):
        """从本地磁盘文件夹中读取图像文件到列表,
            支持的图像格式后缀: png, jpg, jpeg, bmp, tiff, gif, webp

        Args:
            original_folder (string): 文件夹路径
            image_list: 图像文件列表
        """
        for extension in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            self.m_image_list += glob.glob(os.path.join(self.m_original_folder, f"*.{extension}"))
        assert len(self.m_image_list), f"there is not any image in the {self.m_original_folder}!"

    def inference(self):
        # Initialize the model
        model = SRCNN()
        model = model.to(memory_format=torch.channels_last, device=torch.device("cuda", 0))
        print("[====]Build SRCNN model successfully.")

        # Load the CRNN model weights
        checkpoint = torch.load(self.weights_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"[====]Load SRCNN model weights `{self.weights_path}` successfully.")

        # Start the verification mode of the model.
        model.eval()

        # Read LR image and HR image
        lr_image = cv.imread(args.inputs_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        # Get Y channel image data
        lr_y_image = imgproc.bgr2ycbcr(lr_image, True)

        # Get Cb Cr image data from hr image
        lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, False)
        _, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, False, False).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_y_tensor = lr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, False, False)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, lr_cb_image, lr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(args.output_path, sr_image * 255.0)

        print(f"SR image save to `{args.output_path}`")


# --------------------------
if __name__ == "__main__":
    pass