#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Tensor Data Structure of TensorFlow2
@Brief: https://tensorflow.google.cn/api_docs/python/tf/GradientTape
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-21
"""

"""
Tensorflow 底层最核心的概念是张量(tensor)，计算图(Compute Graph)以及自动微分(autograd)
有三种计算图的构建方式: 静态计算图(static compute graph), 动态计算图(dynamic compute graph), 以及 Autograph

Hinton. paper: <<Learning representations by back-propagating errors>> published in Nature

神经网络通常依赖反向传播求梯度来更新网络参数(Hinton. ), 求梯度过程通常是一件非常复杂而容易出错的事情
而深度学习框架可以帮助我们自动地完成这种求梯度运算, Tensorflow 一般使用梯度磁带 tf.GradientTape 来记录正向运算过程，然后反播磁带自动得到梯度值，这种利用 tf.GradientTape 求微分的方法叫做 TensorFlow 的自动微分机制
"""

import tensorflow as tf

# ============================================================
# tf.GradientTape is used to calculate function derivative
# ============================================================

# f(x) = a*x**2 + b*x + c
x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape.gradient(y, x)
print("==== \033[1;33;40m The Compute Gradient in TensorFlow 2.0 with AutoGrad \033[0m ====")
print(f"The derivative of this function F is : {dy_dx}")

# 对常量张量也可以求导，需要增加 watch
with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
print("==== \033[1;33;40m The Compute Gradient for constant with AutoGrad \033[0m ====")
print(dy_da)
print(dy_dc)


print("==== \033[1;33;40m The Compute Gradient in autograph with AutoGrad \033[0m ====")

@tf.function
def func_autograph(x):   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a * tf.pow(x, 2) + b * x + c

    dy_dx = tape.gradient(y, x) 

    return (dy_dx, y)

tf.print(func_autograph(tf.constant(0.0)))
tf.print(func_autograph(tf.constant(1.0)))


# =======================================================================================
# tf.GradientTape and optimizer are used to calculate optimal solution of loss function
# =======================================================================================
# 求 f(x) = a*x**2 + b*x + c 的最小值, 使用 optimizer.apply_gradients
x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c

    dy_dx = tape.gradient(y, x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])

print("==== \033[1;33;40m Calculate Optimal Solution with AutoGrad and SGD \033[0m ====")
tf.print(f"y = {y}, x = {x.numpy()}")


# ==================================================================
# 求 f(x) = a*x**2 + b*x + c 的最小值 使用 optimizer.minimize
# optimizer.minimize 相当于先用 tape 求 gradient,再 apply_gradient
x = tf.Variable(0.0, name="x", dtype=tf.float32)
def func():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c

    return y

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   
for _ in range(1000):
    optimizer.minimize(func, [x])

print("==== \033[1;33;40m Calculate Optimal Solution with AutoGrad and SGD \033[0m ====")
print(f"y = {y}, x = {x.numpy()}")


# ==================================================================
# 在 autograph 中完成最小值求解, 使用 optimizer.apply_gradients
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizer_autograph():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    for _ in tf.range(1000): # tf.range in autograph, not range
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])

    y = a * tf.pow(x, 2) + b * x + c

    return y

print("==== \033[1;33;40m Calculate Optimal Solution with AutoGrad and SGD \033[0m ====")
tf.print(f"y = {minimizer_autograph()}")
tf.print(f"x = {x}")


# ==================================================================
# 在 autograph 中完成最小值求解, 使用 optimizer.minimize
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def func_min_autograph():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c

    return y

@tf.function
def train(epochs):
    for _ in tf.range(epochs):
        optimizer.minimize(func_min_autograph, [x])
    
    return func_min_autograph()

print("==== \033[1;33;40m Calculate Optimal Solution with AutoGrad and SGD \033[0m ====")
tf.print(f"y = {train(1000)}")
tf.print(f"x = {x}")