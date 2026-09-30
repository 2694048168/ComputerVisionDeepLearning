#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 16_function.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-09-30
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 现代Python函数封装和功能模块封装
例如，把“计算销售额”从脚本中抽出来，而不是散落在主流程里
"""

# 配合 from __future__ import annotations，可延迟注解求值，减少运行时负担
from __future__ import annotations
from collections.abc import Callable, Iterable
from functools import wraps
from typing import ParamSpec, TypeVar
import time

# 函数封装最终要落到模块。使用 __all__ 明确公开 API
__all__ = ["total_sales", "retry", "parse_amount"]


def total_sales(
    items: Iterable[tuple[str, float, int]],
    *,  # 对“必须显式传入”的参数使用 * 强制关键字
    tax_rate: float = 0.0,
    discount: float = 0.0,
) -> float:
    """计算商品总价。

    Args:
        items: (商品名, 单价, 数量) 的可迭代对象。
        tax_rate: 税率，例如 0.06。
        discount: 折扣率，例如 0.1。

    Returns:
        税后、折扣后的总金额，保留两位小数。
    """
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax_rate 必须在 0 到 1 之间")
    if not 0 <= discount <= 1:
        raise ValueError("discount 必须在 0 到 1 之间")

    subtotal = sum(price * qty for _, price, qty in items)
    after_discount = subtotal * (1 - discount)
    return round(after_discount * (1 + tax_rate), 2)


# ----------------------------
# 并保留可执行入口
if __name__ == "__main__":
    print(total_sales([("book", 59.9, 2)], tax_rate=0.06))
