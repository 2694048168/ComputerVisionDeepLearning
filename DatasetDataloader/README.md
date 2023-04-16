### Dataset and Dataloader of PyTorch

> How to **organization** and **loading** data from disk via PyTorch by our custom way, Creating 'Dataset' feed into Model, e.g. the dataset pipeline and tricks. The Image Super-Resolution example, **Image Transformation Processing**, could be to promote the Low-Light Image Enhancementation, Image Deraining, Image Defogging, for creating Dataset and the whole pipeline.

**Quick Start**

> [关于Python环境的详细配置过程以及技巧](https://2694048168.github.io/blog/#/PaperMD/python_env_ai)

```shell
# create 'pytorch' env. with python 3.11.2
conda create --name pytorch python=3.11.2

# pip install all library
pip install -r requirements.txt
```

**useful link**
- [Datasets & Dataloaders Tutorials](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Dataset Class Source](https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py)
- [Dataloader Class Source](https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py)
- Effective Python V2 book and Note

**Example**

```
. Dataset_Dataloader
|—— data.py
|—— dataset.py
|—— model.py
|—— train.py
|—— inference.py
|—— dataset
|   |—— super_resolution
|   |—— |—— train
|   |—— |—— test
|   |—— classification
|   |—— |—— train
|   |—— |—— test
|—— checkpoints
|   |—— SRCNN_epoch_{epoch}.pth
|—— results
|   |—— SR_butterfly_LRBI_x4.png
|—— requirements.txt
|—— README.md
```
