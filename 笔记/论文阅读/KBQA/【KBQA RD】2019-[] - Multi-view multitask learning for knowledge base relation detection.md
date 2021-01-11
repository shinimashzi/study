# Multi-view multitask learning for knowledge base relation detection

limitations

How

[TOC]



# 1 背景/limitations/动机

提出利用相同关系对应的问题相似性来提升KBQA RD。

limitations：

KBQA RD的主要挑战：词汇鸿沟

表示学习将关系和问题都向量化，并使用余弦相似度来评估语义相关性，但是如果采用完全不同的关系表示时，表示学习就无法捕捉到相关性。比如：who are the senators of New Jersey，对应的关系是：representatives..office_holder（..表示存在关系链）。



>  论文主要注意到了两点：

1. **除了关系本身，该关系对应的问题也可能具有相似的表达。**
2. 问句Wh（英文情况下），通常表示尾部实体的类型。

多任务学习框架，优化关系检测模型和释义检测模型。


序列相似性度量的方法：

1. Siamese architecture，编码器，将序列编码为向量进行比较
2. 比较-聚合方法旨在通过细粒度级别对序列进行匹配和比较。



# 2 具体实现

![image-20210105162429066](https://cdn.jsdelivr.net/gh/shinimashzi/study/img/image-20210105162429066.png)

模型目标：

- 传统的RD：$\hat r= argmax_r RD(r|q)$

- 引入问题-问题视图后：$\hat r = argmax_r \{(1-\lambda)RD(r|q) + \lambda \frac{1}{|Q_r|\sum_{q'\in Q_r}}PD(q'|q)\}$

关系检测RD模型分为如下四部分：关系表示、问题表示、序列相关特征提取和输出层。

问题释义检测PD模型由问题表示、序列相关特征提取和输出层组成。

RD和PD的前两个模块共享。

### 2.1 RD模型

**关系表示**

一个关系路径同时以关系级和词级表示，单个关系被视为一个token，所以给定一个$M_2$ hop关系路径，包括$M_1$个词，关系embedding为 $r = [w_1,...,w_{M_1}, r_1^s, ..., r_{M_2}^s]$，词嵌入维度相同。

之后使用SRU产生上下文表示： $H_r = Bi-SRU(r)$



**问题表示**

$q = [w_1, ... , w_{|q|}]$

$H_q = Bi-SRU(q)$



**序列相关特征提取**

这个模块由RD和PD模块共享。该模块使用比较-聚合方法。

模块输入为：$H_x \in R^{d\times |X|}, H_y \in R^{d\times |Y|}$

- 首先在两个序列之间执行软对齐，即是两个序列的表示的相似矩阵由 $S=H_X^TW_0H_Y$计算，其中$W_0\in R^{d\times d}$是一个可学习的矩阵。

- 通过逐列Softmax操作从S计算注意力矩阵A：$A_{:t}=Softmax(S_{:t})$，其中，$A_{:t}\in R^{|X|}$是A中的t列。

    - 对序列Y中的每个单词$y_j$，X中单词的加权和为 $\hat{x_j}=\sum_{k=1}^{|X|}a_{k,j}x_k$，$a_{i,j}$是A中的元素，$x_k$是$H_x$的第k列。所以$\hat{x_j}$对X和Y中的第j个词进行匹配，$\hat{x_j}$和$y_j$连接成为：

        ![image-20210106093507887](https://cdn.jsdelivr.net/gh/shinimashzi/study/img/image-20210106093507887.png)

- $F_{XY}=CNN(M_{XY})$

- max-over-time池化操作从结果中得到最重要的特征$f_{XY}\in R^{d_f}$，$d_f$为filter数量。

    - $f_{XY} = \overrightarrow{F}(H_X, H_Y)$
    - 在RD中，得到$f_{rq}=\overrightarrow{F}(H_r, H_q)$

**输出层：**

给定前一模块提取出的特征，输出层估计问题q和关系r的概率$RD(r|q) = \sigma (v_0^Tf_{rq})$

其中$v0\in R^{d_f}$，$\sigma$是Sigmoid函数。

### 2.2 PD模型

给定两个问题q和$q'$，PD模型估计它们的概率，这个模型共享问题表示层和序列相关特征提取层，因此唯一不同的部分是输出层。

- 问题表示模块将两个问题映射到向量序列$H_q \in R^{2d_{sru}\times |q|}$，$H_{q'}\in R^{2d_{sru}\times |q'|}$
- 双向相关特征提取：![image-20210106095557717](https://cdn.jsdelivr.net/gh/shinimashzi/study/img/image-20210106095557717.png)
- 最后输出层估计两个问题属于同一关系的概率。



### 2.3 模型训练

Pairwise ranking loss

- 对RD模型，给出tuple(q, r+, r-)，$Loss_{RD}=max(0, γ_{RD}-RD(r^+|q) + RD(r^-|q))$
- 对PD模型，$Loss_{PD}=max(0, γ_{PD}-PD(q'^+|q) + PD(q'^-|q))$



### 2.4 实验部分

![image-20210106100324253](https://cdn.jsdelivr.net/gh/shinimashzi/study/img/image-20210106100324253.png)

在网络优化部分：首先使用粗粒度网格搜索，之后对每个参数定义一个最佳搜索空间，在验证集上残生最好效果的则被应用。

采用Adadelta优化器。

在推理过程中，$|Q_r|$被限制为10，即是选择10个句子。

# 3 觉得好的句子

One of the main challenges for relation detection lies in the lexical chasm between the relation expressions in user queries and KB.



Given two questions q and q′, the paraphrase detection model aims to estimate the probability that q and q′ query the same KB relation, denoted as PD(q′|q). 