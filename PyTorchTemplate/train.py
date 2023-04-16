import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# 保证随机状态一致
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True #为了提升计算速度
torch.backends.cudnn.benchmark = False # 避免因为随机性产生出差异
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    
    # 数据模块
    data_loader = config.init_obj('data_loader', module_data) #通过config中的名字来指定
    valid_data_loader = data_loader.split_validation()

    # 模型模块
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # 损失与评估模块
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 优化器模块
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())#True的就保存下来，返回列表
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params) #optimizer框架已经实现好了，不用自己写了

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)#学习率衰减策略，也不用自己写的
    
    # 训练模型
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # 可以更改json文件中的参数直接用命令的方式
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
