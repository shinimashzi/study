---
typora-root-url: 2015-ICLR - [NLP Attention] Neural Machine Translation by Jointly Learning to Align and Translate
---

# Neural Machine Translation by Jointly Learning to Align and Translate

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

关于NLP的注意力机制

研究背景：

1. 解决的问题
2. 为什么这么做
3. 研究意义

精读：

1. 模型精讲
2. 实验分析和讨论
3. 论文总结： 关键点、创新点、启发点



## 1. 研究背景

基于神经网络的机器翻译的目的是构建可以联合调整以最大化翻译性能的单个神经网络。



**神经机器翻译**是一种新型的机器翻译方法，与传统的基于短语的翻译系统不同，神经机器翻译构建一个单一的大型网络。

最近的模型结构都是编码-解码结构，将源句子编码进固定长度的向量，解码器根据向量生成翻译。由于模型要将不同长度的句子压缩为固定长度的向量，导致这些模型**难以处理好长句子**，尤其是未出现过的长句子。

论文认为使用固定编码长度，会成为上述模型的瓶颈，建议让模型自动搜索源句子与目标词相关的部分。作者提出了一种encoder-decoder的扩展模型，该模型联合学习对齐和翻译。所提出的模型每次在翻译中生成单词时，都会搜索源句子中相关信息最集中的一组位置，模型基于与这些源位置和所有先前生成的目标词关联的上下文向量来预测目标词。

1. **解决的问题 / limitations** 

   - 编码-解码结构的瓶颈在于将所有的输入都固定为同一长度向量，所以提出模型解决这一问题。

   是不是意味着：编码进定长向量会损失一些信息？



**神经机器翻译：**

在概率上，翻译：找到句子y，$argmax_yp(y|x)$



## 2. 具体模型

#### 基本的编码结构 RNN-encoder-decoder

编码，编码的目的是将输入`x`编码进向量$c$：
$$
h_t=f(x_t, h_{t-1}) \\
c = q(\{h_1, ... , h_{T_x}\})
$$
其中，在`t`时刻，$h_t \in R^n$，`c`是根据隐藏序列生成的向量，`f`和`q`是非线性函数。

有人使用LSTM作为一个实例。

解码，decoder，输出y：

给定上下文向量`c`和之前预测过的所有词$\{y_1, ...,y_{t;-1}\}$，训练解码器预测下一个单词$y_t'$。
$$
p(y) = \prod_{t=1}^Tp(y_t|\{y_1, ..., y_{t-1}, c\}) \\
y = (y1,...,y_{t-1}, c) = g(y_{t-1}, s_t, c)
$$
其中，`g`是输出$y_t$概率的非线性函数，可能有多层结构。$s_t$是`RNN`的隐藏层。也有人使用混合结构。

#### 模型具体结构

**encoder - Bi-RNN：**

基本模型中使用的RNN无法与后面单词产生联系，模型使用双向RNN作为编码器。

双向RNN包括前向和后向RNN，前向的RNN $\rightarrow{f}$按顺序读入序列x，计算前向序列隐藏层state$(\mathop{h_1}\limits ^{\rightarrow},...,\mathop{h_{T_x}}\limits ^{\leftarrow})$。后向只是与前向顺序相反。

我们通过连接前向隐藏state和后向隐藏state，得到$x_j$的向量，即$h_j = [(\mathop{h_j}\limits ^{\rightarrow})^T;(\mathop{h_{j}}\limits ^{\leftarrow})^T]^T$。$h_j$将会集中在$x_j$周围的单词上。



**decoder：**

![image-20201130163258635](/image/image-20201130163258635.png)

在这个新模型结构中，我们定义条件概率：
$$
p(y_i|y_1,...,y_{i-1},x) = g(y_{i-1}, s_i, c_i) \\
$$
其中，$s_i$由下式计算得出：
$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$
$c_i$由下式计算得出：
$$
c_i = \sum_{j=1}^{T_x}\alpha_{i,j}h_j
$$
$h_j$的每个权重$\alpha_{ij}$：
$$
\alpha_{ij}=\frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})} \\
e_{ij} = a(s_{i-1}, h_j)
$$
~~其实就是softmax？~~

$e_{ij}$是一个对齐模型，对位置$j$周围和第$i$处输出进行匹配得分。

$\alpha_{ij}$可视为目标词$y_i$和源词$x_j$对齐\从中翻译的概率，第i个上下文向量$c_i$是所有概率$\alpha_{ij}$的期望。



## 实验部分

包含的语料库：

WMT ’14: [Europarl](http://www.statmt.org/wmt14/translation-task.html)、新闻评论和UN。

使用Aexlrod（http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/）的方法将合并数据集大小减少到348M。

使用`news-test-2014`作为test。

使用每种语言最常用的30,000个单词来训练模型，未在词汇表中的词会被标记为`[UNK]`。

作者训练了两种模型，一种是RNN编码-解码器，另一种可以称为RNNsearch。

RNNencodec的编码器和解码器都有1000个隐藏单元，RNNsearch的前向和后向传播的编码都有1000个隐藏单元，其解码器也有1000个解码单元。在这两种情况下，我们都使用一个具有单个maxout隐藏层的多层网络来计算每个目标词的条件概率。

使用SGD和Adadelta一起训练。batch_size=80。

训练完成后，使用beam search找到近似条件最大的翻译结果。



