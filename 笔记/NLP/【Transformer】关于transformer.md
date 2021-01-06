---
typora-root-url: 【Transformer】关于transformer
---

# Transformer - Encoder

[TOC]

来源： https://github.com/aespresso/a_journey_into_math_of_ml/blob/master/03_transformer_tutorial_1st_part/transformer_1.ipynb

论文：[Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)



**因为我是一个不跟着敲点什么就记不住的人，所以这只是一个梳理笔记，最好看看原教程和论文**



## 1. 理论部分 - transformer编码器

#### 0 transformer模型的直觉，建立直观

与`LSTM`的最大区别：`LSTM`是迭代的，一个字过完`LSTM`单元，才可以进下一个字。

而`transformer`的训练是并行的，所有字同时训练；而且使用位置嵌入来理解语言的顺序，使用`self-attention`和全连接层来进行计算。

**`transformer`**：

- `encoder`

  负责把自然语言序列映射为隐藏层，含有自然语言序列的数学表达。

- `decoder`

  把隐藏层映射为自然语言序列，用来解决下游任务。

  过程，以翻译任务为例：

  ![intuition](/image/intuition.jpg)

  - 输入自然语言序列 Why do we work
  - 编码器输出到隐藏层，再输入到解码器
  - 输入<start>符号到解码器
  - 得到第一个字“为”
  - 将得到的“为”再次输入解码器
  - 得到第二个字“什”
  - 将第二个字再次输入解码器，直到解码器输出<end>，即序列生成完成。



### **`Encoder 编码器`**:

![encoder](/image/encoder.jpg)

上图的1，2，3，4依次对应下文。

Inputx 输入： X $\text{[batch size, sequence length]}$， batch size一般为句子个数， sequence length为句子长度。

输出：$X_{hidden}: \text{[batch size, sequence length, embedding dimension]}$

即为编码器处理后的隐藏层。

输入Input Embedding，得到 $X_{Embeddinglookup}:\text{[batch size, sequence length, embedding dimension]}$

embedding为字向量。

#### 1 positional encoding 位置嵌入

由于transformer无rnn的迭代操作，所以必须提供每个字的位置信息给transformer，以此识别语言中的顺序关系。

positional encoding的维度为$\text{[max sequence length, embedding dimension]}$，**嵌入的维度同词向量的维度**。

一般来说，以字为单位训练transformer模型，所以不用进行分词操作，可以初始化字向量为$\text{[vocab size, embedding dimension]}$。



论文使用$sin$和$cosine$函数的线性变换来提供位置信息

![image-20201107104403596](/image/image-20201107104403596.png)

其中：

`pos`指句中字的位置，取值范围为$[0, maxsequence length)$

`i`值的是词向量的维度，取值范围为$[0, embeddingdimension)$

$d_{model}$为待生成的向量维度



模型能够学到位置之间的依赖关系和自然语言的时序特性。



得到位置表示`PE`之后，会与之前初步形成的词向量相加，形成新的表示。

此时的表示 $X_{embedding}=EmbeddingLookup(X)+PositionalEncoding,X \in R^{batch size * seq.len * embed. dim}$

#### **2 Multi-Head Attention** - self attention mechanism

下一步， 为了学到多重含义的表达，对$X_{embedding}$做线性映射，分配三个权重:$W_Q,W_k,W_V \in \mathbb{R}^{embed.dim * embed.dim}$，线性映射（矩阵乘）之后，形成三个矩阵$Q，K，V$， 和线性变换之前的维度一致。
$$
Q = Linear(X_{embedding} = X_{embedding}W_Q) \\
K = Linear(X_{embedding} = X_{embedding}W_K) \\
V = Linear(X_{embedding} = X_{embedding}W_V) \\
$$
之后进行多头注意力机制：

需要定义一个超参数h（head的数量），embedding dimension必须整除于h，因为需要把embedding dimension分割为h份。

将embedding dimension分割为h份后：

$Q,K,V$的维度为： $\text{[batch size, sequence length, h, embedding dimension / h]}$

将$Q,K,V$的`sequence length, h`进行转置，方便后续计算，转置后的维度为 $\text{[batch size, h, sequence length, embedding dimension/h]}$

以一组head为例：

分割后的$Q,K,V$，它们的维度都是$\text{[sequence length, embedding dimension/h]}$

首先计算$Q$和$K$的转置的点积，两个向量越相似，点积越大。这个就是**注意力矩阵**。注意力矩阵的维度为：$\text{[batch size, h, len, len]}$， 其实就是每个字和其余字的相关程度（以点积形式计算）。

 ![image-20201107111956229](/image/image-20201107111956229.png)


$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
上式为自注意力机制，求$QK^T$后（即得到注意力矩阵），使用注意力矩阵给$V$加权，$\sqrt{d_k}$是为了把注意力矩阵变成标准正态分布，使$softmax$归一化之后更加稳定。$\sqrt{d_k}$是当前的维度，即$embed.dim/h$。

得到注意力矩阵，用$softmax$归一化，目的是让每个字与其他所有字权重和相加为1，即每一行的**和为1**。注意力矩阵的作用就是一个注意力权重的概率分布。

注意力机制的操作使**每个字向量都含有当前句子内所有字向量的信息**。

$V$的维度仍然是$\text{[batch size, h, sequence length, embedding dimension/h]}$



**Attention Mask**

![image-20201107150640633](/image/image-20201107150640633.png)

注意的问题，在所有矩阵使用的`batch size`，一个批次中的所有句子很有可能不是定长的句子。

`padding`: 在一个batch中，按最长的句长对其余句子进行补齐长度，一般使用0来填充（如果是需要填充，注意词汇表中0最好就代表[pad]）。

但这会使softmax产生问题，$\sigma(Z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k}e^{z_j}}$，其中$e^0=1$，是有值的，会让padding部分参与运算，等于让无效部分参与了运算，这时需要做一个$mask$来让无效区域不参与运算，一般采用给无效区域加很大的负数偏置：
$$
z_{illegal} = z_{illegal} + bias_{illegal} \\
bias_{illegal} \rightarrow -\infty \\
e^{z_{illegal}} \rightarrow 0 
$$
这样就使得无效区域经过softmax计算之后趋近于0。



#### 3 Layer Normalization和残差连接

- **残差连接**

得到经过注意力矩阵加权后的$V$，也就是$Attention(Q,K,V)$，对其转置，使它和$X_ {embedding}$一致，也即是$\text{[batch size, sequence length, embedding dimension]}$。

之后： $X_{embedding} + Attention(Q,K,V)$

在之后的运算里，每经过一个模块的运算，都要把运算之前的值和运算之后的值相加，从而得到残差连接，训练的时候可以使梯度直接走捷径反传到最初始层：
$$
X+SubLayer(X)
$$


- **LayerNorm**

`LayerNormaliaztion`的左右是把神经网络中的隐藏层归一化为标准正态分布：
$$
\mu_i = \frac{1}{m}\sum_{i=1}^{m}x_{i,j} \\
\sigma^2_j=\frac{1}{m}\sum_{i=1}^m(x_{i,j}-\mu_j)^2 \\
LayerNorm(x)=\alpha \odot \frac{x_{i,j}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}+\beta
$$
上述第一个公式以矩阵行为单位求均值，第二个式子以行为单位求方差，之后用**每一行**的每一个元素减去这行的均值，除以这行的标准差，得到归一化之后的数值，$\epsilon$是为了防止除0。

$\alpha$和$\beta$是为了弥补归一化的过程中损失的信息，是可训练参数，一般初始化$\alpha$为全1，而$\beta$为全0。$\odot$表示逐元素相乘而不是点积。



#### 4 transformer encoder整体结构

- 字向量与位置编码
  $$
  X = EmbeddingLookup(X) + PositionalEncoding \\
  X \in \mathbb{R}^{batchsize*seq.len*embed.dim} 
  $$

- 自注意力机制
  $$
  Q = Linear(X) = XW_Q \\
  K = Linear(X) = XW_K \\
  V = Linear(x) = XW_V \\
  X_{attention} = SelfAttention(Q,K,V)
  $$

- 残差连接与LayerNormalization：
  $$
  X_{attention} = X+X_{attention} \\
  X_{attention} = LayerNorm(X_{attention})
  $$

- 第四部分： **Feed Forward**，两层线性映射并使用激活函数激活，比如$ReLU$:

  $X_{hidden} = Activate(Linear(Linear(X_{attention})))$

- 重复3）：
  $$
  X_{hidden} = X_{attention} + X_{hidden} \\
  X_{hidden} = LayerNorm(X_{hidden}) \\
  X_{hidden} \in \mathbb{R}^{batchsize*seq.len*embed.dim}
  $$
  

Bert在每句话的句头加特殊字符[CLS]，句末加[SEP]，这样模型与训练完毕之后，就可以使用句头的特殊字符$hidden state$来进行分类任务了。

