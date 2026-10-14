#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 28_client.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-14
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: TCP Echo 客户端
"""

import socket

HOST = "127.0.0.1"
PORT = 9999


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    print(f"[Client] {sock.recv(1024).decode('utf-8')}")

    try:
        while True:
            msg = input("请输入消息: ").strip()
            if not msg:
                continue
            sock.send(msg.encode("utf-8"))
            if msg.lower() == "exit":
                break
            print(f"[Client] {sock.recv(1024).decode('utf-8')}")
    finally:
        sock.close()
        print("[Client] 连接已关闭")


# -------------------------
if __name__ == "__main__":
    main()
