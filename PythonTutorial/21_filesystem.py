#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 21_filesystem.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-06
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 文件、路径以及文件系统相关操作
"""

import os
import shutil
from shutil import copytree, ignore_patterns
from pathlib import Path


def testPath():
    # 从字符串创建
    p = Path("/home/user/documents/report.csv")
    print(f"The path is: {p}")

    # 从多个片段创建（自动连接）
    p = Path("home", "user", "documents", "report.csv")
    print(f"The path is: {p}")

    # 当前工作目录
    cwd = Path.cwd()
    print(f"The current path is: {cwd}")

    # 用户主目录
    home = Path.home()
    print(f"The home path is: {home}")

    # 相对路径
    p = Path("data/output/results.csv")
    print(f"The relative path is: {p}")

    # ------ 路径属性：属性访问替代字符串解析
    p = Path("/home/user/project/src/main.py")

    print(p.parent)  # /home/user/project/src   （父目录）
    print(p.parents[0])  # /home/user/project       （上一级）
    print(p.parents[1])  # /home/user               （上两级）
    print(p.name)  # main.py                  （文件名）
    print(p.stem)  # main                     （不带后缀的文件名）
    print(p.suffix)  # .py                      （后缀）
    print(p.suffixes)  # ['.tar', '.gz']          （多后缀，如 archive.tar.gz）
    print(p.parts)  # ('/', 'home', 'user', 'project', 'src', 'main.py')

    # ----------- 路径检查与解析
    p = Path("data/config.yaml")

    # 存在性检查
    print(f"the path exists: {p.exists()}")  # 路径是否存在
    print(f"the path is file: {p.is_file()}")  # 是否为文件
    print(f"the path is dir: {p.is_dir()}")  # 是否为目录
    print(f"the path is symlink: {p.is_symlink()}")  # 是否为符号链接

    # 转换为绝对路径
    absolute_path = p.resolve()  # 解析为绝对路径（处理 .. 和符号链接）
    print(f"the absolute path is: {absolute_path}")  # 是否为符号链接

    # 判断路径类型
    print(f"the path is absolute: {p.is_absolute()}")  # 是否为绝对路径

    # --------- 路径拼接与修改
    base = Path("/data/projects")
    print(f"the base path: {base}")

    # 使用 / 运算符拼接
    config = base / "myapp" / "config" / "settings.yaml"
    print(f"the base/ config path: {config}")

    # 修改后缀
    update_path = config.with_suffix(
        ".json"
    )  # /data/projects/myapp/config/settings.json
    print(f"the config file suffix: {config}")
    print(f"the config file suffix update: {update_path}")

    # 替换文件名
    update_file = config.with_name("app.yaml")  # /data/projects/myapp/config/app.yaml
    print(f"the config filename update: {config}")
    print(f"the config filename: {update_file}")

    # 转换为字符串（仅在必须传字符串的旧 API 中使用）
    str(config)


def testFileOP():
    # 文件读写：with 语句是绝对底线,文件 I/O 的第一原则：永远使用 with 语句
    # 正确：with 自动关闭，即使发生异常
    with open("data.txt", "w", encoding="utf-8") as f:
        f.write("自动保存的数据")

    # 小文件：一次性读取
    with open("21_filesystem.py", "r", encoding="utf-8") as f:
        content = f.read()
        print(f"the content: {content}")

    # 大文件：逐行迭代，内存友好
    error_count = 0
    with open("21_filesystem.py", "r", encoding="utf-8") as f:
        for line in f:
            if "ERROR" in line:
                error_count += 1

    # 注意：read_text()/write_text() 适合中小文件
    # 大文件仍应使用 with open(...) 逐行处理
    p = Path("21_filesystem.py")

    # 读取文本
    content = p.read_text(encoding="utf-8")

    # 写入文本
    # p.write_text("新内容", encoding="utf-8")

    # 读取字节（二进制）
    data = p.read_bytes()
    print(f"the data : {data}")

    # 写入字节
    # p.write_bytes(b"\x00\x01\x02")

    # ----- 处理图片、音频、视频、压缩包等非文本文件时，必须使用二进制模式
    # 读取二进制文件
    with open("image/oil_stain.png", "rb") as f:
        data = f.read()

    # 写入二进制文件
    with open("copy.png", "wb") as f:
        f.write(data)

    # 使用 shutil 直接复制二进制文件更简单
    shutil.copy("image/oil_stain.png", "image/image.png")


def testFilemanager():
    # ----- shutil 是标准库中的“文件管理神器”，封装了复制、移动、删除、归档等高级操作

    # ============== 文件复制 ==============
    # 复制文件，保留权限
    shutil.copy("photo.jpg", "backup/photo.jpg")

    # 复制文件，保留权限 + 元数据（创建时间、修改时间等）
    shutil.copy2("photo.jpg", "backup/photo_copy.jpg")

    # 只复制内容，不保留任何元数据
    shutil.copyfile("photo.jpg", "backup/photo_plain.jpg")

    # 复制文件到已打开的文件对象
    with open("backup/photo.jpg", "wb") as f:
        shutil.copyfileobj(open("photo.jpg", "rb"), f)

    # ============== 目录操作 ==============
    # 递归复制整个目录树
    shutil.copytree("my_project", "backup/my_project_backup")

    # 递归删除整个目录（危险操作，确认路径！）
    shutil.rmtree("backup/old_project")

    # 移动文件或目录（跨文件系统也能工作，会自动回退为复制+删除）
    shutil.move("downloads/vacation.jpg", "photos/vacation.jpg")

    # ============== 忽略模式复制 ==============
    # 复制目录，但跳过 .pyc 文件和以 tmp 开头的文件/目录
    copytree("src", "dist", ignore=ignore_patterns("*.pyc", "tmp*"))

    # ============== 归档操作 ==============
    # make_archive 支持 zip、tar、gztar、bztar、xztar 等格式
    # 创建 zip 归档
    shutil.make_archive("backup", "zip", "my_project")

    # 解压归档
    shutil.unpack_archive("backup.zip", "restored_project")


# -------------------------
if __name__ == "__main__":
    # 处理文件路径的首选方案是 pathlib，而不是传统的 os.path,保持操作系统行为一致性
    # os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'output')
    data_dir = Path(__file__).parent / "data" / "output"
    print(f"the data floder: {data_dir}")

    testPath()

    testFileOP()

    # ========== 磁盘空间检查 ==========
    usage = shutil.disk_usage("/")
    print(f"总空间: {usage.total / 1e9:.1f} GB")
    print(f"已用: {usage.used / 1e9:.1f} GB")
    print(f"剩余: {usage.free / 1e9:.1f} GB")

    # ============ 文件系统遍历：os.walk 与 Path.rglob
    # os.walk() 递归遍历目录树，
    # 每次迭代返回一个三元组 (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk("./"):
        # dirpath: 当前目录的路径字符串
        # dirnames: 当前目录下的子目录名列表（可修改以控制遍历行为）
        # filenames: 当前目录下的文件名列表
        dirnames[:] = [d for d in dirnames if d != ".git"]  # 原地修改，阻止递归进入

        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                print(full_path)

    # =========== Path.rglob：pathlib 的现代替代 
    # 使用 pathlib，用 rglob 模式匹配更自然
    data_dir = Path('./')

    # 递归查找所有 .py 文件
    for py_file in data_dir.rglob('*.py'):
        print(py_file)

    # 只查找当前目录（不递归）
    for py_file in data_dir.glob('*.py'):
        print(py_file)

    # 同时匹配多种模式
    for f in data_dir.rglob('*.{py,txt,png}'):
        print(f)
