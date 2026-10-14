#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 28_server.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-14
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 多线程 TCP Echo 服务器
"""

import socket
import threading

HOST = "127.0.0.1"
PORT = 9999


def tcplink(sock: socket.socket, addr: tuple) -> None:
    """处理单个客户端连接"""
    print(f"[Server] 接受来自 {addr} 的连接")
    sock.send(b'Welcome! Type "exit" to quit.')
    try:
        while True:
            data = sock.recv(1024)
            if not data or data.decode("utf-8").strip().lower() == "exit":
                break
            reply = f"[Echo] {data.decode('utf-8')}".encode("utf-8")
            sock.send(reply)
    finally:
        sock.close()
        print(f"[Server] {addr} 已断开")


def main() -> None:
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(5)
    print(f"[Server] 监听 {HOST}:{PORT} ...")

    try:
        while True:
            sock, addr = server_sock.accept()
            t = threading.Thread(target=tcplink, args=(sock, addr))
            t.daemon = True
            t.start()
    except KeyboardInterrupt:
        print("\n[Server] 关闭中...")
    finally:
        server_sock.close()


# ---------------------------
if __name__ == "__main__":
    main()
