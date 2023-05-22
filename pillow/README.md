## Pillow Image Processing Library with Python

> PIL(Python Imaging Library)是 Python 的第三方图像处理库, 由于其功能丰富, API 简洁易用; 在 PIL 库的基础上开发了一个支持 Python3 版本的图像处理库, Pillow.

### Pillow 库特点
1. 支持广泛的文件格式("jpeg"，"png"，"bmp"，"gif"，"ppm"，"tiff")以及相互转换
2. 图像归档, 包括创建缩略图、生成预览图像、图像批量处理等功能
3. 图像处理, 包括调整图像大小、裁剪图像、像素点处理、添加滤镜、图像颜色处理等功能
4. 图像添加水印、合成 GIF 动态效果图等复杂图像处理操作
5. Pillow 库可以配合 Python GUI 一起使用

### quick start
```shell
# https://pypi.org/project/Pillow/#files
pip install Pillow

# 导入Image类，该类是pillow中用于图像处理的重要类
from PIL import Image
# 注意,使用 PIL 导入,实际上使用的是 Pillow,这里的 PIL 可以看做是 Pillow 库的简称
```

### Organization of files
```
. pillow
|—— images
|   |—— image_01153.jpg
|   |—— image_01155.jpg
|   |—— split_merge.jpg
|   |—— sunflower.jpg
|   |—— panda.jpg
|—— basic_operators.py
|—— image_resize.py
|—— img_split_merge.py
|—— img_transformer.py
|—— image_filter.py
|—— image_color.py
|—— img_watermark.py
|—— image_gif.py
|—— README.md
```
