#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 17_class.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-09-30
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 现代Python 类封装
例如，一个账户对象必须保证：余额不能为负、存款必须大于 0、取款不能超过余额
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
import time


@dataclass(slots=True)
class Task:
    title: str
    done: bool = False
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def complete(self) -> None:
        self.done = True


class Account:
    bank_name: str = "Modern Bank"

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner = owner
        self._balance = 0.0
        if balance:
            self.deposit(balance)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Account:
        return cls(
            owner=str(data["owner"]),
            balance=float(data.get("balance", 0.0)),
        )

    @property
    def balance(self) -> float:
        return self._balance

    @staticmethod
    def _validate_amount(amount: float) -> None:
        if amount <= 0:
            raise ValueError("金额必须大于 0")

    def deposit(self, amount: float) -> None:
        self._validate_amount(amount)
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        self._validate_amount(amount)
        if amount > self._balance:
            raise ValueError("余额不足")
        self._balance -= amount

    def __repr__(self) -> str:
        return f"Account(owner={self.owner!r}, balance={self.balance:.2f})"


# ===========================================
class Notifier(Protocol):
    def send(self, message: str) -> None: ...


class EmailNotifier:
    def __init__(self, address: str) -> None:
        self.address = address

    def send(self, message: str) -> None:
        print(f"Email to {self.address}: {message}")


class UserService:
    def __init__(self, notifier: Notifier) -> None:
        self.notifier = notifier

    def welcome(self, name: str) -> None:
        self.notifier.send(f"Welcome, {name}!")


class Timer:
    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: object | None,
        exc: object | None,
        tb: object | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start


# -----------------------------
if __name__ == "__main__":
    task = Task("学习 Python 类封装", tags=["python", "oop"])
    task.complete()
    print(task)

    acc = Account.from_dict({"owner": "Bob", "balance": 200})

    # ===========================================
    service = UserService(EmailNotifier("a@example.com"))
    service.welcome("Alice")

    with Timer() as t:
        time.sleep(0.1)
    print(t.elapsed)
