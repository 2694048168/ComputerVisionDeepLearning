#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: train.py
@Python Version: 3.11.2
@Platform: PyTorch 2.0.0+cu118
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/04/04 21:26:21
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: Training phase the SRCNN model.
'''

import os
import math
import datetime
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import SRCNN
from dataset import ImageDataset, get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_on_epoch(epoch, model, training_data_loader, criterion, optimizer):
    epoch_loss = 0.0
    # for step, batch in enumerate(training_data_loader, 1):
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f"====> Epoch[{epoch}] ({iteration}/{len(training_data_loader)}): Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(training_data_loader)
    print(f"\n====> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}")


def test(model, testing_data_loader, criterion):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
    
    avg_psnr_batch = avg_psnr / len(testing_data_loader)
    print(f"====> Avg. PSNR: {avg_psnr_batch:.4f} dB")


def main(save_folder):
    torch.manual_seed(seed=42)

    print('====> Loading datasets')
    path2lr = r"D:\Datasets\DIV2K_valid_LR"
    path2hr = r"D:\Datasets\DIV2K_valid_HR"
    train_set = ImageDataset(path2lr, path2hr)
    training_data_loader = get_dataset(train_set, mode="train", batch_size=32)

    path2test_lr = r"./dataset/super_resolution/test/LR"
    path2test_hr = r"./dataset/super_resolution/test/HR"
    test_set = ImageDataset(path2test_lr, path2test_hr)
    testing_data_loader = get_dataset(test_set, mode="test", batch_size=5)

    print('====> Building model')
    model = SRCNN().to(device)

    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('====> Training model')
    num_epochs = 300
    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
        train_on_epoch(epoch, model, training_data_loader, criterion, optimizer)
        test(model, testing_data_loader, criterion)

        model_save_path = os.path.join(save_folder, f"SRCNN_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Checkpoint saved to {model_save_path}")


""" 如何保证模型的断点续训 !!!
https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
----------------------------------------------"""
if __name__ == "__main__":
    starttime = datetime.datetime.now()

    save_folder = r"./checkpoints/"
    os.makedirs(save_folder, exist_ok=True)
    main(save_folder)

    endtime = datetime.datetime.now()
    print(f"Time Consumption is: {(endtime - starttime).seconds} seconds")
    # -----------------------------------------------------------------
