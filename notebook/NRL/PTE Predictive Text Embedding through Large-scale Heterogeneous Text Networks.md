## PTE

1. Abstract

   无监督文本映射方法比如 skip-gram 等等效果并不是非常好的另一个主要的原因在于没有考虑标签的信息，因此提出了半监督的 PTE 方法通过构建大规模的异构文本网络(标注信息和共现矩阵信息)

2. Introduction

   高效的文本表示是一件非常有意义的事情，传统的 word 表示成独立的词向量，并且 doc 表示成词袋模型。一个有效的方法就是 skip-gram 。

   文章在 LINE 的基础上利用异构的网络的信息获得了高效的低维文本映射。异构网络中有 word and word(无监督信息), word and doc(无监督信息), word and label(有监督信息) 的各种信息。是一种半监督的方法。网络中聚合有监督信息和无监督的信息。

   传统的无监督的方法比如 skip-gram 有监督的方法利用了大流量的标注信息构建的 CNN 和 RNN 

   ![](.\photo\2.png)

3. Problem Definition

   论文的实验目的在于使用 PTE 训练好的嵌入向量进行文本分类任务检验新方法的嵌入效果。

   1. definition of word-word network

      $G_{ww}=(\mathcal{V},E_{ww})$ 其中 $\mathcal{V}$ 是字典的大小，$E_{ww}$ 是单词的共现矩阵

   2. definition of word-doc network

      $G_{wd}=(\mathcal{V}\cup\mathcal{D},E_{wd})$ $\mathcal{V}$ 是词典 $\mathcal{D}$ 是文档集合，$E_{wd}$ 是文档和词典的边，整体构成了一个二部图，$w_{ij}$ 定义为 word 在 doc 中的出现次数。

   3. definition of word-label network

      $G_{wl}=(\mathcal{V}\cup \mathcal{L},E_{wl})$ 也是一个二部图，其中 $\mathcal{L}$ 是类别信息，边权值定义为 $w_{ij}=\sum_{(d:l_d=j)}n_{d_i}$ 其中是对所有的类别是 $j$ 的文档进行累计,$n_{d_i}$ 是单词 $v_i$ 在文档 $d$ 的出现的频率。  

   4. definition of heterogeneous text network

      网络是由上述的 3 中信息的节点构建的异构网络。这里的异构信息可以随意的扩充(比如word-sentence等等)主要是为了得到 word 的嵌入表示，其他的比如 doc 等表示可以用过 word 的表示聚合得到。

4. PTE

   1. 二部图映射

      参考自 LINE 我们对上述的 3 中二部图进行映射(word-word 可以看作是 source word 和 target word 的二部图，无向边拆解成是有向边)，嵌入的 loss 优化过程也是和 LINE 的一样，通过定义二阶相似度和实际的权重概率，利用 KL 散度进行优化。

   2. 异构文本网络映射

      一个直觉的优化思想就是直接对上面的 3 种二部图的 loss 联合优化
      $$
      O_{pte}=O_{ww} + O_{wd} + O_{wl}\\
      O_{ww} =-\sum_{(i,j)\in E_{ww}}w_{ij}\log p(v_i|v_j)\\
      O_{wd} =-\sum_{(i,j)\in E_{wd}}w_{ij}\log p(v_i|d_j)\\
      O_{wl} =-\sum_{(i,j)\in E_{wl}}w_{ij}\log p(v_i|l_j)\\
      $$
      上式的优化可以有两种形式，直接联合优化或者是分别预训练之后 fine-tune

      ![](.\photo\3.png)

   3. text embedding

      算法计算的是 word embedding，doc embedding 可以聚合 word embedding 得到

      $d=w_1w_2...w_n,d=\frac{1}{n}\sum_{i=1}^nu_i$ 其中 $u_i$ 是 word embedding of $w_i$ 

      实际上实验中使用的是 $O=\sum_{i=1}^n l(u_i,d)$ 其中 $l$ 是 sigmoid 函数调整 $d$ 的 doc embedding