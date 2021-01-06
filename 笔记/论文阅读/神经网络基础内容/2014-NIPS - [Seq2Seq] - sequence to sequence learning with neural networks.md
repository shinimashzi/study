---
typora-root-url: 2014-NIPS - [Seq2Seq] - sequence to sequence learning with neural networks
---

# sequence to sequence learning with neural networks

[TOC]

## 研究背景

1. 解决的问题

   Deep Neural Networks的输入是固定长度的，而一般的序列问题（比如翻译）都不定长度，无法用于将序列映射到序列的工作上。所以提出一种End-to-End的序列学习方法。其实也就是encoder-decoder结构。

   使用LSTM将输入维映射到固定维度向量，然后使用另一个LSTM解码。

2. 为什么这么做

   - why lstm？

3. 研究意义

   序列到序列，提升翻译任务指标。

其外，论文发现反转源句子中单词顺序可以提高LSTM性能->其实就是需要使用双向LSTM。



## 具体结构

![image-20201130203046985](/image/image-20201130203046985.png)



## 实验

