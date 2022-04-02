#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The low-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-26
"""

""" 
AutoGraph的使用规范:
有三种计算图的构建方式: 静态计算图, 动态计算图, 以及Autograph
TensorFlow 2.0 主要使用的是动态计算图和Autograph
动态计算图易于调试, 编码效率较高, 但执行效率偏低
静态计算图执行效率很高, 但较难调试
Autograph 机制可以将动态图转换成静态计算图, 兼收执行效率和编码效率之利
Autograph机制能够转换的代码并不是没有任何约束的, 有一些编码规范需要遵循, 否则可能会转换失败或者不符合预期

Autograph 的编码规范和 Autograph 转换成静态图的原理
使用 tf.Module 来更好地构建 Autograph
Autograph编码规范总结:
1. @tf.function 修饰的函数应尽可能使用 TensorFlow 中的函数而不是 Python 中的其他函数. 例如使用 tf.print 而不是 print, 使用tf.range 而不是 range, 使用 tf.constant(True) 而不是 True.
2. 避免在 @tf.function 修饰的函数内部定义 tf.Variable.
3. @tf.function 修饰的函数不可修改该函数外部的 Python 列表或字典等数据结构变量
"""

import os, pathlib
import datetime
import numpy as np
import tensorflow as tf

# Step 1. @tf.function 修饰的函数应尽量使用 TensorFlow 中的函数而不是 Python 中的其他函数
@tf.function
def np_random():
    ndarray_1 = np.random.randn(3, 3)
    tf.print(ndarray_1)

@tf.function
def tf_random():
    tensor_1 = tf.random.normal((3, 3))
    tf.print(tensor_1)

# np_random 每次执行的结果都是一样的, tf_random 每次执行都会有重新生成新的随机数
np_random()
np_random()
tf_random()
tf_random()


# Step 2. 避免在 @tf.function 修饰的函数内部定义 tf.Variable
tensor_2 = tf.Variable(1.0, dtype=tf.float32)

@tf.function
def outer_var():
    tensor_2.assign_add(1.0)
    tf.print(tensor_2)
    return tensor_2

@tf.function
def inner_var():
    tensor_3 = tf.Variable(1.0, dtype=tf.float32)
    tensor_3.assign_add(1.0)
    tf.print(tensor_3)
    return tensor_3

outer_var()
outer_var()

# ValueError: tf.function only supports singleton tf.Variables created on the first call. 
# Make sure the tf.Variable is only created once or created outside tf.function.
# inner_var()
# inner_var()


# Step 3. @tf.function 修饰的函数不可修改该函数外部的 Python 列表或字典等结构类型变量
tensor_list = []

# @tf.function # 加上这一行切换成 Autograph 结果将不符合预期
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)


# =======================
# Autograph 的机制原理
# =======================
# 使用 @tf.function 装饰一个函数的时候, 后面到底发生了什么呢?
# 后面什么都没有发生, 仅仅是在 Python 堆栈中记录了这样一个函数的签名 function signatures
@tf.function(autograph=True)
def my_add(x, y):
    for i in tf.range(3):
        tf.print(i)
    
    z = x + y
    print("tracing")
    return z

# 当第一次调用这个被 @tf.function 装饰的函数时, 后面到底发生了什么?
my_add(tf.constant("hello"), tf.constant("world"))

# 发生了2件事情, 第一件事情是创建计算图, 即创建一个静态计算图, 跟踪执行一遍函数体中的 Python 代码, 
# 确定各个变量的 Tensor 类型, 并根据执行顺序将算子添加到计算图中, 
# 在这个过程中, 如果开启 autograph=True(默认开启), 会将 Python 控制流转换成 TensorFlow图内控制流, 
# 主要是将 if 语句转换成 tf.cond 算子表达, 将 while 和 for 循环语句转换成 tf.while_loop 算子表达, 
# 并在必要的时候添加 tf.control_dependencies 指定执行顺序依赖关系.
# 相当于在 tensorflow 1.0 执行了类似下面的语句：
"""
g = tf.Graph()
with g.as_default():
    a = tf.placeholder(shape=[], dtype=tf.string)
    b = tf.placeholder(shape=[], dtype=tf.string)
    cond = lambda i: i<tf.constant(3)
    def body(i):
        tf.print(i)
        return(i+1)
    loop = tf.while_loop(cond, body, loop_vars=[0])
    loop
    with tf.control_dependencies(loop):
        c = tf.strings.join([a,b])
    print("tracing")
"""
# 第二件事情是执行计算图, 相当于在 tensorflow1.0 中执行了下面的语句：
"""
with tf.Session(graph=g) as sess:
    sess.run(c, feed_dict={a:tf.constant("hello"), b:tf.constant("world")})
"""
# 因此先看到的是第一个步骤的结果: 即 Python 调用标准输出流打印 "tracing" 语句
# 然后看到第二个步骤的结果: TensorFlow调用标准输出流打印 1,2,3

# 当再次用相同的输入参数类型调用这个被 @tf.function 装饰的函数时, 后面到底发生了什么?
my_add(tf.constant("good"), tf.constant("morning"))
# 只会发生一件事情, 那就是上面步骤的第二步, 执行计算图
# 所以这一次没有看到打印 "tracing" 的结果

# 当再次用不同的的输入参数类型调用这个被 @tf.function 装饰的函数时, 后面到底发生了什么?
my_add(tf.constant(1), tf.constant(2))
# 由于输入参数的类型已经发生变化, 已经创建的计算图不能够再次使用
# 需要重新做 2 件事情: 创建新的计算图, 执行计算图.
# 所以又会先看到的是第一个步骤的结果: 即 Python 调用标准输出流打印 "tracing" 语句
# 然后再看到第二个步骤的结果: TensorFlow调用标准输出流打印 1,2,3

# 需要注意的是, 如果调用被 @tf.function 装饰的函数时输入的参数不是 Tensor 类型, 则每次都会重新创建计算图.
# 因此, 一般建议调用 @tf.function 时应传入 Tensor 类型
my_add("hello", "world")
my_add("good", "morning")


# ============================================================================================================
# 重新理解 Autograph 的编码规范. 了解 Autograph 的机制原理, 就能够理解 Autograph 编码规范的 3 条建议

# 1，@tf.function 修饰的函数应尽量使用 TensorFlow 中的函数而不是 Python 中的其他函数. 
# 例如使用 tf.print 而不是 print.
#   解释: Python 中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用, 普通 Python 函数是无法嵌入到静态计算图中的, 
# 所以在计算图构建好之后再次调用的时候, 这些 Python 函数并没有被计算, 而 TensorFlow 中的函数则可以嵌入到计算图中. 
# 使用普通的 Python 函数会导致被 @tf.function 修饰前 [eager执行] 和被 @tf.function 修饰后 [静态图执行] 的输出不一致.

# 2. 避免在 @tf.function 修饰的函数内部定义 tf.Variable
#   解释: 如果函数内部定义了 tf.Variable, 那么在 [eager执行]时, 这种创建 tf.Variable 的行为在每次函数调用时候都会发生. 
# 但是在 [静态图执行] 时, 这种创建 tf.Variable 的行为只会发生在第一步跟踪 Python 代码逻辑创建计算图时, 
# 这会导致被 @tf.function 修饰前 [eager执行] 和被 @tf.function 修饰后 [静态图执行] 的输出不一致, 
# 实际上, TensorFlow 在这种情况下一般会报错.

# 3. @tf.function 修饰的函数不可修改该函数外部的 Python 列表或字典等数据结构变量
#   解释: 静态计算图是被编译成 C++ 代码在 TensorFlow Core 内核中执行的. 
# Python 中的列表和字典等数据结构变量是无法嵌入到计算图中, 它们仅仅能够在创建计算图时被读取, 
# 在执行计算图时是无法修改 Python 中的列表或字典这样的数据结构变量的
# ============================================================================================================


# ============================================================================================================
# Autograph 和 tf.Module 概述
# Autograph 的编码规范时提到构建 Autograph 时应该避免在 @tf.function 修饰的函数内部定义 tf.Variable
# 但是如果在函数外部定义 tf.Variable 的话, 又会显得这个函数有外部变量依赖, 封装不够完美
# 一种简单的思路是定义类, 并将相关的 tf.Variable 创建放在类的初始化方法中, 将函数的逻辑放在其他方法中
# TensorFlow 提供了一个基类 tf.Module, 通过继承它构建子类, 不仅可以获得以上的自然而然的封装, 而且可以非常方便地管理变量, 
# 还可以非常方便地管理它引用的其它 Module, 最重要的是能够利用 tf.saved_model 保存模型并实现跨平台部署使用
# 实际上, tf.keras.models.Model, tf.keras.layers.Layer 都是继承自 tf.Module, 
# 提供了方便的变量管理和所引用的子模块管理的功能, 因此利用 tf.Module 提供的封装, 再结合 TensoFlow 丰富的低阶 API, 
# 实际上能够基于 TensorFlow 开发任意机器学习模型(而非仅仅是神经网络模型), 并实现跨平台部署使用
# ============================================================================================================
# 应用 tf.Module 封装 Autograph
tensor_var = tf.Variable(1.0, dtype=tf.float32)
# tf.function 中用 input_signature 限定输入张量的签名类型: shape and dtype
@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def add_print(tensor):
    tensor_var.assign_add(tensor)
    tf.print(tensor_var)
    return tensor_var

print("\033[1;31;47m The result of pure function with Autograph is \033[0m ")
add_print(tf.constant(3.0))
# add_print(tf.constant(3)) # 输入不符合张量签名的参数将报错 ValueError: Python inputs incompatible with input_signature:


# 利用 tf.Module 的子类化将其封装一下
class DemoModule(tf.Module):
    def __init__(self, init_value=tf.constant(0.0), name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope: # 相当于 with tf.name_scope("demo_module")
            self.tensor_var = tf.Variable(init_value, dtype=tf.float32, trainable=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def add_print(self, tensor):
        with self.name_scope:
            self.tensor_var.assign_add(tensor)
            tf.print(self.tensor_var)
            return self.tensor_var

print("\033[1;31;47m The result of tf.Module function with Autograph is \033[0m ")
demo = DemoModule(init_value=tf.constant(1.0))
demo.add_print(tf.constant(41.0))

# 查看模块中的全部变量和全部可训练变量
print(f"The all Variables in the Module :\n {demo.variables}")
print(f"The all Trainable Variables in the Module :\n {demo.trainable_variables}")

# 查看模块中的全部子模块
print(f"The all Submodules in the Module :\n {demo.submodules}")


# 使用 tf.saved_model 保存模型, 并指定需要跨平台部署的方法
path2model_save = r"./Models/Saving"
os.makedirs(path2model_save, exist_ok=True)
tf.saved_model.save(demo, str(pathlib.Path(path2model_save)), signatures={"serving_default": demo.add_print})

# 加载模型
demo_load = tf.saved_model.load(str(pathlib.Path(path2model_save)))
demo_load.add_print(tf.constant(5.0))

# ---------------------------------------------------------------------
# 查看模型文件相关信息, 其中的输出信息在模型部署和跨平台使用时有可能会用到
# Command line with following the command
# saved_model_cli show --dir ./Models/Saving --all
# ---------------------------------------------------------------------


# -------------------------------------------------------------------------------
# tensorboard 中查看计算图, 模块会被添加模块名 demo_module, 方便层次化呈现计算图结构
# 这个可以解释为什么需要 name 这个属性以及 模块名的命名空间 scope
# --------------------------------Error due to the Version of TF------------------
# # step 1. 创建日志
# log_folder = r"./tensorboard/autograph"
# os.makedirs(log_folder, exist_ok=True)

# stamp_time = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
# tensorboard_log_folder = str(pathlib.Path(os.path.join(log_folder, stamp_time))) 

# writer_tensorboard = tf.summary.create_file_writer(tensorboard_log_folder)

# # step 2. 开启 autograph 跟踪
# tf.summary.trace_on(graph=True, profiler=True)

# # step 3. 执行 autograph
# demo_3 = DemoModule(init_value=tf.constant(0.0))
# result = demo.add_print(tf.constant(5.0))

# # step 4. 将计算图信息写入日志
# with writer_tensorboard.as_default():
#     tf.summary.trace_export(
#         name="demomodule",
#         step=0,
#         profiler_outdir=tensorboard_log_folder)

# ------------------------------------------------------
# TensorBoard
# https://tensorflow.google.cn/tensorboard/get_started
# ------------------------------------------------------
# %tensorboard --logdir ./tensorboard/autograph # in notebook
# tensorboard --logdir ./tensorboard/autograph # in command
# ------------------------------------------------------


# 利用 tf.Module 的子类化实现封装, 也可以通过给 tf.Module 添加属性的方法进行封装
my_module = tf.Module()
my_module.x = tf.Variable(0.0)

@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])  
def addprint(tensor):
    my_module.x.assign_add(tensor)
    tf.print(my_module.x)
    return my_module.x

my_module.addprint = addprint

my_module.addprint(tf.constant(1.0))
# 查看模块中的全部变量和全部可训练变量
print(f"The all Variables in the Module :\n {my_module.variables}")
print(f"The all Trainable Variables in the Module :\n {my_module.trainable_variables}")
# 查看模块中的全部子模块
print(f"The all Submodules in the Module :\n {my_module.submodules}")
# 使用 tf.saved_model 保存模型, 并指定需要跨平台部署的方法
# 加载模型


# --------------------------------------------------------------------------------
# tf.Module to tf.keras.Model, tf.keras.layers.Layer
# tf.keras 中的模型和层都是继承 tf.Module 实现的, 也具有变量管理和子模块管理功能
print(f"The tf.keras.Model whether is the subclass of tf.Module : {issubclass(tf.keras.Model, tf.Module)}")
print(f"The tf.keras.layers.Layer whether is the subclass of tf.Module : {issubclass(tf.keras.layers.Layer, tf.Module)}")
print(f"The tf.keras.Model whether is the subclass of tf.keras.layers.Layer : {issubclass(tf.keras.Model, tf.keras.layers.Layer)}")


tf.keras.backend.clear_session() 
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(4, input_shape=(10, )))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Dense(1))
model.summary()

print(f"\nThe all Variables in the Model :\n {model.variables}")

model.layers[0].trainable = False # 冻结第 0 层的变量, 使其不可训练
print(f"\nThe all Trainable Variables in the Model :\n {model.trainable_variables}")
print(f"\nThe all Submodules in the Model :\n {model.submodules}")
print(f"\nThe all Layers in the Model :\n {model.layers}")
print(f"\nThe name of Model is : {model.name}\n")
print(f"\nThe name scope of Model is : {model.name_scope()}\n")