# 2019 - [ACL] - Complex Question Decomposition for Semantic Parsing

[TOC]

~~无相关工作内容~~

[https://www.aclweb.org/anthology/P19-1440/](https://www.aclweb.org/anthology/P19-1440/)

## 摘要

解决任务： 复杂问题语义解析（基于[ComplexWebQ](https://www.tau-nlp.org/compwebq)）。提出层次语义解析方法。

三个阶段：

1. question decomposition - 问题分解：将复杂问题分为子问题序列。
2. information extraction - 信息提取：提取问题类型和谓词信息（关系信息）。
3. semantic parsing - 将前两个阶段的信息进行整合，为complex question生成一个逻辑形式。



## 1. 介绍

![image-20201012160552154](C:\Users\xjf_njust\Desktop\Learning\论文阅读\【KBQA】2019 - [ACL] - Complex Question Decomposition for Semantic Parsing\image\image-20201012160552154.png)



对复杂问题的分解存在问题：

1. 人工分解规则：需要专家，难以扩展（就是传统人力制造规则的缺陷）。
2. 使用pointer network生成分裂点，形成简单问题序列。这个方法在Figure 1中的问题就不work。缺陷：可能无法找到最佳分裂方式、因此会丢失很多信息。

论文中提出的模型是直接生成完整的子问题。

层次语义解析 `HSP` 是一个 `SeqtoSeq` 结构，其潜在的思想即是分解和集成。



## 2. 相关工作

## 3. 模型

`Input` ：复杂问题输入

`Output` ： 逻辑形式

![Model Overview](C:\Users\xjf_njust\Desktop\Learning\论文阅读\【KBQA】2019 - [ACL] - Complex Question Decomposition for Semantic Parsing\image\image-20201012162132026.png)



### 3.1 模型概述：

复杂问题输入：$x = \{x_1,...,x_{|x|}\}$

逻辑形式： $y = \{y_1, ... , y_{|y|}\}$

为了产生更好的逻辑形式，模型使用两种类型的中间表示：

DR: 分解表示，表示为 $z = \{z_1, ... , z_{|z|} \}$ 

SR: 语义表示，表示为 $w = \{w_1, ..., w_{|w|}\}$

每个训练样本表示为 **$<x, y, z, w>$** 四元组。

![image-20201012163340462](C:\Users\xjf_njust\Desktop\Learning\论文阅读\【KBQA】2019 - [ACL] - Complex Question Decomposition for Semantic Parsing\image\image-20201012163340462.png)



### 3.2 基础结构：

模型的基础结构：

解析单元，解析单元包括编码网络和解码网络，是基于 Transformer 的多头注意力机制的编解码器。

- 输入：输入序列 $a = \{a_1, ... , a_{|a|}\}$ + 额外信息 $e = \{e_1, ... , e_{|e|}\}, e_i \in R^n$

- 输出： 解析目标序列 $o = \{o_1, ... , o_{|o|}\}$



**编码：**

将输入序列 $a$  编码到上下文表示 $h = \{ h_1, ... , h_{|a|}\}, h_i \in R^m$， 然后使用 L 层 `Transformer` 编码器生成输出：
$$
h = f_{enc}(a) = f_{enc}^{proc}(f_{enc}^{emb}(a))
$$
**解码：**

解码器得到 $h$ 和输入的额外信息 $e$ ，连接形成表示 $[h, e]$。

在解码器时间步 t ， 对 $[h, e]$以及之前所有输出 $o_{<t} = \{o_1,...,o_{t-1}\}$，解码器计算条件概率 $P(o_t|o_{<t}, [h,e])$

第一个解码嵌入函数 $f_{dec}^{emb}$ 将之前输出$o_{<t}$映射到词向量，增加位置编码，得到词表示。

解码器也堆叠L个相同的层，单词表示和 $[h, e]$一起被馈送到这些层。

将第 $l$ 层，位置 $j$ 的输出向量表示为$k_j^l$ ，将第 $l$ 层的第 $j$ 个位置前的输出向量表示为 $k_{<=j}^l = \{k_1^l, ..., k_j^l\}$ ，解码层的输出为$k_j^l = Layer(k_{<=j}^{l-1}, [h, e])$

给出最后一层的输出 $k_j^L$， 对当前词 $P_{vocab}^j(w)$的概率计算如下式，其中，$V = \{w_1,...,w_{|v|}\}$，$W_o, b_o$为参数：
$$
P_{vocab}^j (w) = Softmax(W_o \cdot a_j^L + b_o)
$$
解码过程由`[BOS]`触发，由`[EOS]` token结束。



**复制机制**

为了解决OOV单词（不在词汇表中的单词）产生的问题，在解码器中应用了复制机制。

- 在解码时间步 $t$ ，首先计算源序列 $a$ 上的注意力分布，使用最后一层解码层输出的 $k_t^L$ ， 和编码输出 $h$的双线性点积。如下式所示：

$$
u_t^i = k_t^LW_qh_i\\
\alpha_t = Softmax(u_t)
$$

- 之后计算复制概率 $P_{copy}^t \in [0,1]$，如下式所示， $W_q, W_g, b_g$是可学习的参数：
  $$
  P_{copy}^t = \sigma(W_g\cdot [k_t^L, h, e] + b_g)
  $$

- 使用$P_{copy}^t$计算复制概率的加权和，生成概率得到扩展词汇表$V + X$ 的最终预测概率。其中，$X$ 是源序列$a$的OOV词集：
  $$
  P_t(w) = (1-P_{copy}^t)P_{vocab}(w)+P_{copy}^t\sum_{i:w_i=w}\alpha_t^i
  $$

- 解码过程由下述公式计算得出，其中，使用 $f_{dec}^t$ 表示具有复制机制解码器的一个时间步。
  $$
  o_t = f_{dec}^t (f_{dec}^{emb}(o_{<t}, [h ,e])) \\
  o = f_{dec}([h, e])
  $$

- 损失函数： $L(\theta) = \frac{1}{T} \sum_{t=1}^{T} - logP(o_t=o_t^*|a,e,o_{<t})$

### 3.3 HSP

输入：$x$

输出的逻辑形式：$y$

训练目标： 上述损失函数的训练目标使条件概率$P(y|x)$和目标序列的真实概率$P(y^*)$的交叉熵最小。

HSP机制通过将目标转化为多个条件概率的乘积，将这个过程分为多阶段任务。对HSP模型，目标是$P(y|x,z,w)P(w|x,z)P(z|x)$，z和w表示分解表示和语义表示。

**问题分解：**

HSP的第一阶段。将复杂问题->简单问题序列

输入：复杂问题$x$

输出：分解表示$z$

- 模型首先将$x$映射到上下文感知表示（context aware representation）$h$。
  - $ h = f_{enc_1}(x)$
  - 此阶段未给出额外信息，所以融合表示就是$h$。
- 分解解码，分解后的表示预测为：$z = f_{dec_1}(h)$

图2中的解码过程可以分为多个时间步，对每个时间步，之前的输出被右移并输入解码层，蓝色线的起点是解码器的融合表示，对问题分解来说，这个融合表示就是问题嵌入（question embedding）。

**信息提取**

HSP的第二阶段从复杂问题（复杂问题本身和第一阶段分解出的简单问题序列）中提取关键信息。

输入：分解表示（上一阶段的输出结果$z$）和问题嵌入。

输出：语义表示 $w$

- 编码使用子问题编码将分解表示编码为$h^z = f_{enc2}(z)$
- $[h, h^z]$会被馈送到语义解码得到语义表示$w = f_{dec}([h,h^z])$

图2中的⊕表示融合过程，也就是concat，直接连接。

**语义解析**

HSP的最后一个阶段，它收到复杂问题的上下文感知嵌入，分解表示和语义表示序列，输出最后的逻辑形式。

- 编码，将语义表示编码为$h^w = f_{enc_3}(w)$，连接三个表示 $[h, h^w, h^z]$。
- 逻辑形式$y=f_{dec}([h, h^w, h^z])$

损失函数：
$$
L_{HSP}(\theta) = \lambda_1\cdot L_1 + \lambda_2 \cdot L_2 + L_3 \\
L_1 = -logP(z|x)\\
L_2 = -logP(w|x,z) \\
L_3 = -logP(y|x,z,w)
$$
其中，$\lambda_1,\lambda_2$为超参数。

在学习过程中，模型使用三阶段学习过程。

1. 预测分解表示  $\hat{z} = argmax_z P(z|x)$
2. 预测语义表示  $\hat{w} = argmax_w P(w|x,z)$
3. 预测逻辑形式  $\hat{y} = argmax_yP(y|x, z,w)$

获得的每个序列都是使用贪心搜索方法，比如beam search。



## 4. 实验

### 4.1 设置

建立了一个包括复杂问题，所有中间表示和逻辑形式的词汇表。词汇表由语料中出现多余4次的词组成，所有的OOV单词表示为UNK。

使用Glove词向量，如果是[UNK]、[BOS]、[EOS]，则是随机值。在训练中，所有的词向量也会随之更新。

HPS在编码嵌入时，使用预训练好的StanfordCoreNLP POS模型。使用分类词性标注，并将词性映射到30维向量，POS向量由均匀分布U(-0.1, 0.1)随即生成，在之后的训练过程中随之更新。POS向量与词向量连接形成词表示。

编码和解码的所有hidden layer设置为300。所有编码层和解码层由相同的6层堆叠而成。优化器 Adam（$\beta_1=0.9,\beta_2=0.98,\epsilon=10-9$），使用动态学习率。

正则化：dropout=0.2和label smoothing=0.1

### 4.2 数据集

[ComplexWebQuestions](https://www.tau-nlp.org/compwebq)

train-27734 dev-3480 test-3475













