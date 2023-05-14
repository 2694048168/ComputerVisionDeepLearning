![CV&DIP Logo](./logo.jpg)

> the Computer Vision with Python; Digital Image Processing with Python; the platform for Pytorch and TensorFlow2; OpenCV with Python and Cpp.

> [Wei Li Blog](https://2694048168.github.io/blog/)

### Quick Start

```shell
# step 1: clone the repo. and into the folder 'ComputerVisionDeepLearning'
git clone --recursive https://github.com/2694048168/ComputerVisionDeepLearning.git
cd ComputerVisionDeepLearning/DatasetDataloader

# 查看每一个文件夹下的 'README.md' 文件说明
# if you want to download specical folder, please using 'gitzip' tool to enter
# the folder path, such as 'https://github.com/2694048168/ComputerVisionDeepLearning/tree/main/DatasetDataloader'

# create 'pytorch' env. with python 3.11.2
conda create --name pytorch python=3.11.2

conda activate pytorch

# pip install all library
pip install -r requirements.txt
# python --version
# python train.py

conda deactivate
```

> [gitzip](http://kinolien.github.io/gitzip/)

### Overview
```
. Project_Name
|—— DatasetDataloader
|   |—— datast
|   |—— data.py
|   |—— dataset.py
|   |—— train.py
|   |—— inference.py
|   |—— model.py
|   |—— requirements.tet
|   |—— README.md
|—— ImageTransformation
|   |—— README.md
|—— PyTorchTemplate
|   |—— README.md
|—— DigitalImageProcessing
|   |—— cpp
|   |—— python
|   |—— image
|   |—— README.md
|—— CS231N
|   |—— images
|   |—— README.md
|—— Learning_Pytorch
|—— Learning_TensorFlow2
|—— Logo.png
|—— gitee_init.png
|—— github_init.png
|—— LICENSE
|—— README.md
```


**乍一看到某个问题，你会觉得很简单，其实你并没有理解其复杂性。当你把问题搞清楚之后，又会发现真的很复杂，于是你就拿出一套复杂的方案来。实际上，你的工作只做了一半，大多数人也都会到此为让......。但是，真正伟大的人还会继续向前，直至找到问题的关键和深层次原因，然后再拿出一个优雅的、堪称完美的有效方案。**

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;—— 乔布斯

**“When you start looking at a problem and it seems really simple, you don't really understand the complexity of the problem. Then you get into the problem, and you see that it's really complicated, and you come up with all these convoluted solutions. That's sort of the middle, and that's where most people stop.... But the really great person will keep on going and find the key, the underlying principle of the problem—and come up with an elegant, really beautiful solution that works.”**

	— Steve Jobs (quoted in Insanely Great: The Life and Times of Macintosh, the Computer that Changed Everything by Steven Levy)
