#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 26_thread_process.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-12
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 现代 Python 多进程和多线程
"""

import threading
import time
from multiprocessing import Pool


def fib(n):
    """纯 CPU 计算：第 n 个斐波那契数"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ---- 单线程 ----
def single_thread():
    start = time.perf_counter()
    for _ in range(4):
        fib(300000)
    return time.perf_counter() - start


# ---- 多线程 ----
def multi_thread():
    start = time.perf_counter()
    threads = [threading.Thread(target=fib, args=(300000,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


# ---- 多进程 ----
def multi_process():
    start = time.perf_counter()
    with Pool(4) as p:
        p.map(fib, [300000] * 4)
    return time.perf_counter() - start


def process_image(image_id):
    """模拟图像处理（CPU 密集型）"""
    time.sleep(1)
    return f"图片 {image_id} 处理完成"


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def task(n):
    time.sleep(0.5)
    return n * n


# -------------------------
if __name__ == "__main__":
    print(f"单线程: {single_thread():.2f}s")
    print(f"多线程: {multi_thread():.2f}s")
    print(f"多进程: {multi_process():.2f}s")

    print("---------------------------------")
    with Pool(8) as pool:  # 最多 8 个进程
        results = pool.map(process_image, range(8))
    for r in results:
        print(r)

    # concurrent.futures 统一接口
    # I/O 密集型 → 用线程池
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(task, range(8)))
        print("线程池:", results)

    # CPU 密集型 → 用进程池
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(task, range(8)))
        print("进程池:", results)
