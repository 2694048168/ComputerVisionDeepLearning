#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 30_merge_pdf.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-15
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description:
将一系列 PDF 文件拼接为单个 PDF,并可自由指定拼接顺序。
功能特性：
1. 自动扫描目录下（含子目录）的所有 .pdf 文件；
2. 提供多种指定拼接顺序的方式：
   a. 交互式：运行时在终端按编号选择顺序；
   b. 命令行直接传文件（按传入顺序拼接）；
   c. 通过顺序文件 / 拖入列表指定；
3. 支持重复文件（同一文件可拼接多次）；
4. 输出单一合并后的 PDF 文件。

依赖: pypdf
安装: pip install pypdf

用法示例：
  # 方式一：交互式选择顺序
  python3 merge_pdfs.py

  # 方式二：命令行按顺序传文件（顺序即拼接顺序）
  python3 merge_pdfs.py a.pdf b.pdf c.pdf

  # 方式三：指定输入目录 + 交互选择
  python3 merge_pdfs.py --dir ./pdfs --output merged.pdf

  # 方式四：从顺序文件读取（每行一个文件名）
  python3 merge_pdfs.py --order order.txt
"""

import os
import re
import sys
import argparse
from pypdf import PdfWriter, PdfReader


def natural_sort_key(name: str):
    """自然排序：把文件名中的数字按数值大小排序，而不是按字典序。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def discover_pdfs(directory: str) -> list:
    """递归扫描目录，返回按自然顺序排序的 PDF 文件绝对路径列表。"""
    if not os.path.isdir(directory):
        sys.exit(f"目录不存在：{directory}")
    pdfs = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return sorted(pdfs, key=lambda p: natural_sort_key(os.path.basename(p)))


def read_order_file(order_file: str, pdf_dir: str = ".") -> list:
    """从顺序文件读取拼接顺序：每行一个文件名或路径，空行与注释(#)忽略。"""
    if not os.path.isfile(order_file):
        sys.exit(f"顺序文件不存在：{order_file}")
    items = []
    with open(order_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 去掉行内注释
            line = re.split(r"[#\t]", line)[0].strip()
            if not line:
                continue
            p = line
            if not os.path.isabs(p):
                p = os.path.join(pdf_dir, p)
            items.append(p)
    return items


def pick_order_interactively(files: list) -> list:
    """交互式：让用户在终端按编号选择拼接顺序。"""
    print("\n发现以下 PDF 文件（已按自然顺序编号）：\n")
    for i, f in enumerate(files, 1):
        print(f"  {i:>3}. {os.path.basename(f)}")

    print(
        "\n请输入拼接顺序（用逗号或空格分隔编号，例如：3,1,2 表示先拼第3个、再拼第1个、再拼第2个）。"
    )
    print("输入 q 退出，输入 a 则全部按当前编号顺序拼接。")

    while True:
        raw = input("顺序 > ").strip().lower()
        if raw in ("q", "quit", "exit"):
            sys.exit("已取消。")
        if raw in ("a", "all", ""):
            return files[:]

        tokens = re.split(r"[,\s，、;；]+", raw)
        indices = []
        ok = True
        for t in tokens:
            if not t:
                continue
            try:
                n = int(t)
            except ValueError:
                print(f"  无效编号：{t!r}")
                ok = False
                break
            if not (1 <= n <= len(files)):
                print(f"  编号越界：{n}（应在 1~{len(files)} 之间）")
                ok = False
                break
            indices.append(n)
        if not ok or not indices:
            print("  请重新输入。")
            continue

        # 允许重复编号，实现同一文件拼接多次
        return [files[n - 1] for n in indices]


def merge_pdfs(paths: list, output: str) -> None:
    """按 paths 顺序合并 PDF，输出到 output。"""
    if not paths:
        sys.exit("没有可合并的 PDF 文件。")

    writer = PdfWriter()
    total_pages = 0
    seen = set()
    dup_warned = False

    for idx, p in enumerate(paths, 1):
        if not os.path.isfile(p):
            sys.exit(f"文件不存在：{p}")
        abs_p = os.path.abspath(p)
        if abs_p in seen and not dup_warned:
            print("提示：检测到同一文件被多次选中，将按选择拼接多次。")
            dup_warned = True
        seen.add(abs_p)

        try:
            reader = PdfReader(p)
            n = len(reader.pages)
            for page in reader.pages:
                writer.add_page(page)
            total_pages += n
            print(f"  [{idx}/{len(paths)}] 已加入：{os.path.basename(p)}（{n} 页）")
        except Exception as e:
            print(f"  警告：无法读取 {p}，已跳过（{e}）")

    if writer.get_num_pages() == 0:
        sys.exit("合并失败：没有任何页面被加入。")

    with open(output, "wb") as fh:
        writer.write(fh)

    print(f"\n合并完成！共 {len(paths)} 个文件、{total_pages} 页。")
    print(f"输出文件：{os.path.abspath(output)}")


def main():
    parser = argparse.ArgumentParser(
        description="将一系列 PDF 文件拼接为一个 PDF，可指定拼接顺序。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="命令行直接传入的 PDF 文件，按传入顺序拼接。",
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="扫描目录（含子目录），默认当前目录。",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="merged.pdf",
        help="输出文件名，默认 merged.pdf。",
    )
    parser.add_argument(
        "--order",
        default=None,
        help="从指定顺序文件读取拼接顺序（每行一个文件名）。",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="不进入交互选择：命令行未传文件时，直接按自然顺序合并全部。",
    )
    args = parser.parse_args()

    # 确定要合并的文件列表（按优先级）
    if args.files:
        selected = [os.path.abspath(f) for f in args.files]
        print(f"使用命令行传入的 {len(selected)} 个文件（按传入顺序拼接）。")
    elif args.order:
        selected = read_order_file(args.order, args.dir)
        print(f"从顺序文件读取 {len(selected)} 个文件。")
    else:
        all_pdfs = discover_pdfs(args.dir)
        if not all_pdfs:
            sys.exit(f"在目录 {args.dir} 中未发现任何 PDF 文件。")
        if args.no_interactive or len(all_pdfs) == 1:
            selected = all_pdfs
            print(f"按自然顺序合并 {len(all_pdfs)} 个文件。")
        else:
            selected = pick_order_interactively(all_pdfs)

    merge_pdfs(selected, args.output)


# --------------------------
if __name__ == "__main__":
    main()
