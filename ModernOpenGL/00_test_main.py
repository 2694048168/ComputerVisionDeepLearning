#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 00_test_main.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-08-27.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

import moderngl as mgl
import glfw
import numpy as np


# viewport callback function
def viewportResizeCallback(window, width, height):
    ctx.viewport = (0, 0, width, height)


# key-board event callback function
def process_input(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)


# 着色器程序
vertex_shader = """
#version 330 core

in vec3 in_vert;

void main()
{
    gl_Position = vec4(in_vert.x, in_vert.y, in_vert.z, 1.0);
}
"""

fragment_shader = """
#version 330 core

out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
"""

# --------------------------
if __name__ == "__main__":
    # 检测 GLFW 是否成功初始化
    if not glfw.init():
        raise Exception("GLFW初始化失败")

    # 创建窗口
    window = glfw.create_window(800, 600, "LearnOpenGL", None, None)
    # 检查是否成功创建窗口
    if not window:
        glfw.terminate()
        raise Exception("GLFW创建窗口失败")

    # 创建OpenGL上下文
    glfw.make_context_current(window=window)
    ctx = mgl.create_context()
    prog = ctx.program(vertex_shader, fragment_shader)

    # 顶点数据
    vertices = np.array([-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0], dtype="f4")

    vbo = ctx.buffer(vertices.tobytes())

    # 顶点数组对象
    vao = ctx.vertex_array(prog, vbo, "in_vert")

    # 视口函数注册
    glfw.set_framebuffer_size_callback(window=window, cbfun=viewportResizeCallback)

    # =============== 循环渲染 ===============
    while not glfw.window_should_close(window=window):
        # 输入处理
        process_input(window)

        # # 渲染指令
        ctx.clear(0.2, 0.3, 0.3)
        vao.render(mgl.TRIANGLES)

        # 处理事件、交换缓冲
        glfw.poll_events()
        glfw.swap_buffers(window=window)

    # 清理资源
    glfw.terminate()
