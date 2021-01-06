---
typora-root-url: 【tensorflow-00】学习记录-1
---

# 【Tensorflow-00】学习记录-0

[TOC]

## 1. 基本概念

常量、计算图、会话

![image-20201121101036018](/image/image-20201121101036018.png)

`Tensor("add:0", shape=(2,), dtype=float32)`

其中，

`add:0` ： 节点名：第0个输出

`shape=(2,)`：维度

`dtype`：数据类型

这是一个张量，但没有实际运算。**计算图只描述计算过程，不计算结算结果。**



**计算图：** 搭建神经网络的计算过程，只搭建，不运算。

**会话(Session)：** 执行计算图中的节点运算

```python
with tf.Session() as sess:
	print sess.run(y)
```

 **参数：**

参数，即![image-20201121102217134](/image/image-20201121102217134.png)，用变量表示，随机给初值。

一般来说：

`w = tf.Variable(tf.random_normal([2,3], stddev=2, mean=0, seed=1))`

`tf.random_normal([2,3])`: 正态分布

`stddev`：标准差为2

`mean`：均值为0

`seed`：随机种子

![image-20201121102911779](/image/image-20201121102911779.png)



## 2. 前向传播

前向传播：搭建模型，实现推理：

![image-20201121103654303](/image/image-20201121103654303.png)

![image-20201121103853608](/image/image-20201121103853608.png)



变量初始化、计算图节点运算都要用会话实现 `sess.run()`

变量初始化： 在`sess.run`中使用`tf.global_variables_initializer()`

```python
init_op = tf.global_variables_initializer()
sess.run(init_op)
```

 计算图节点运算：在`sess.run`中写入待运算节点

```
sess.run(y)
```

使用tf.placeholder占位，在`sess.run`中使用`feed_dict`喂数据

```python
# 喂一组数据
x = tf.placeholder(tf.float32, shape=(1,2))
sess.run(y, feed_dict={x:[[0.5, 0.6]]})

# 喂多组数据
x = tf.placeholder(tf.float32, shape=(None, 2))
sess.run(y, feed_dict={x:[[0.1, 0.2], [0.2, 0.3]]})
```

![image-20201121104711005](/image/image-20201121104711005.png)

![image-20201121105439872](/image/image-20201121105439872.png)

## 3. 反向传播

![image-20201121105953620](/image/image-20201121105953620.png)