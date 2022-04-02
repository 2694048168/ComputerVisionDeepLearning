#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Tensor Data Structure of TensorFlow2
@Brief: https://tensorflow.google.cn/api_docs/python/tf/Tensor
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-21
"""

"""
TensorFlow 是一个采用数据流图(data flow graphs), 用于数值计算的开源软件库.
节点(Nodes)在图中表示数学操作, 图中的线(edges)则表示在节点间相互联系的多维数据数组, 即张量(tensor).
灵活的架构可以在多种平台上展开计算,CPU, GPU, 服务器, 移动设备等等.
TensorFlow 最初由Google大脑小组(隶属于Google机器智能研究机构)的研究员和工程师们开发出来, 
用于机器学习和深度神经网络方面的研究, 但这个系统的通用性使其也可广泛用于其他计算领域.
TensorFlow的主要优点:
1. 灵活性: 支持底层数值计算, C++ 自定义操作符
2. 可移植性: 从服务器到PC到手机, 从CPU到GPU到TPU
3. 分布式计算: 分布式并行计算, 可指定操作符对应计算设备
4. Tensorflow 底层最核心的概念是张量(tensor)，计算图(Compute Graph)以及自动微分(autograd)
"""

import numpy as np
import tensorflow as tf

# Program = Data Structure + Algorithm
# TensorFlow program = Tensor data structure + Computational Graph Algorithm language
# Tensor of TensorFLow or PyTorch is similar to ndarray of Numpy
# 从行为特性来看, 两种类型的张量，常量 constant 和变量 Variable.
# 常量的值在计算图中不可以被重新赋值, 变量可以在计算图中用 assign 等算子重新赋值

print("================== The Constant Tensor of TensorFlow ==================")
int16_tensor_const = tf.constant(42, dtype=tf.int16)
int32_tensor_const = tf.constant(42, dtype=tf.int32) # default
int64_tensor_const = tf.constant(42, dtype=tf.int64)

float16_tensor_const = tf.constant(3.14, dtype=tf.float16)
float32_tensor_const = tf.constant(3.14, dtype=tf.float32) # default
float64_tensor_const = tf.constant(3.14, dtype=tf.float64)

double_tensor_const = tf.constant(3.14, dtype=tf.double)

string_tensor_const = tf.constant("Wei Li", dtype=tf.string) # default
bool_tensor_const = tf.constant(True)

print(f"The value of int16_tensor_const = {int16_tensor_const}")
print(f"The value of int32_tensor_const = {int32_tensor_const}")
print(f"The value of int64_tensor_const = {int64_tensor_const}")

print(f"The value of float16_tensor_const = {float16_tensor_const}")
print(f"The value of float32_tensor_const = {float32_tensor_const}")
print(f"The value of float64_tensor_const = {float64_tensor_const}")

print(f"The value of double_tensor_const = {double_tensor_const}")
print(f"The value of bool_tensor_const = {bool_tensor_const}")

print("================== The Data Type of TensorFlow and Numpy ==================")
print(f"the equal of type of TensorFlow and Numpy: {tf.int32 == np.int32}")
print(f"the equal of type of TensorFlow and Numpy: {tf.int64 == np.int64}")
print(f"the equal of type of TensorFlow and Numpy: {tf.double == np.double}")
print(f"the equal of type of TensorFlow and Numpy: {tf.float32 == np.float32}")
print(f"the equal of type of TensorFlow and Numpy: {tf.float64 == np.float64}")
print(f"the equal of type of TensorFlow and Numpy: {tf.string == np.compat.unicode}")

print("================== The Tensor of TensorFlow ==================")
scalar_const = tf.constant(True)
print(f"The rank of Scalar tensor is = {tf.rank(scalar_const)}")
print(f"The rank of Scalar tensor is = {scalar_const.numpy().ndim}")

vector_const = tf.constant([1.0, 2.0, 3.0, 4.0])
print(f"The rank of Vector tensor is = {tf.rank(vector_const)}")
print(f"The rank of Vector tensor is = {vector_const.numpy().ndim}")

matrix_const = tf.constant([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]])
print(f"The rank of Matrix tensor is = {tf.rank(matrix_const)}")
print(f"The rank of Matrix tensor is = {np.ndim(matrix_const)}")

tensor_3dims_const = tf.constant([[[1.0, 2.0], [1.0, 2.0]],
                                [[1.0, 2.0],[1.0, 2.0]]])
print(f"The rank of 3dims tensor is = {tf.rank(tensor_3dims_const)}")
print(f"The shape of 3dims tensor is = {tensor_3dims_const.shape}")
print(f"The value of 3dims tensor is = \n{tensor_3dims_const.numpy()}")

# 可以用 tf.cast 改变张量的数据类型
# 可以用 numpy 方法将 tensorflow 中的 tensor 张量转化成 numpy 中的 ndarray 多维数组
# 可以用 shape 方法查看张量的尺寸
int_const = tf.constant([123,456], dtype=tf.int32)
float_const = tf.cast(int_const, tf.float32)
print(f"the type of int_constis : {int_const.dtype}")
print(f"the type of float_const : {float_const.dtype}")

print("================== The Variable Tensor of TensorFlow ==================")
# 常量值不可以改变，常量的重新赋值相当于开辟新的内存空间
const_tensor = tf.constant([1.0, 2.0])
print(const_tensor)
print(f"The address of const_tensor is : {id(const_tensor)}")
const_tensor = const_tensor + tf.constant([1.0, 1.0])
print(const_tensor)
print(f"The address of const_tensor is : {id(const_tensor)}") 

# 模型中需要被训练的参数一般被设置成变量类型的张量
# 变量的值可以改变，可以通过 assign, assign_add 等方法给变量重新赋值
variable_tensor = tf.Variable([1.0, 2.0], name="var")
print(variable_tensor)
print(f"The address of variable_tensor is : {id(variable_tensor)}") 
variable_tensor.assign_add([1.0, 1.0])
print(variable_tensor)
print(f"The address of variable_tensor is : {id(variable_tensor)}")