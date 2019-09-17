## CopyNet

[CopyNet](https://ziyaochen.github.io/2018/07/11/Copy-Mechanism/)



## S2SPMN: A Simple and Effective Framework for Response Generation with Relevant Information



## Incorporating the Structure of the Belief State in End-to-End Task-Oriented Dialogue Systems
1. 目前的端到端人物导向的对话系统中 DST 部分使用的是景点的分类方法，但是这种方法并不利于针对 OOV 

2. 之前的 sequcity 的方法可以解决 OOV 的问题，但是 DST 部分的 value 是没有结构的，换句话说如果存在两个 slot 的 value 重复很高的话，这样的方法只能决定 value 是什么没有办法决定是哪个 slot 的 value 并且在DST 部分生成的 belief span 是随机的，这样直接对随机的顺序进行 encoder 的话容易产生不好的归纳偏置

3. 新的方法中每一个 slot 存在一个 decoder 但是为了保持 sequcity 中参数共享的有点，decoder 的权重共享降低了模型的参数量利于学习。其中 information slot 是 decoder，request slot 是多个二分类器，其中新家了 request slot decoder 生成在最后的回复中包含的 slot 也是二分类器

4. 新的方法

   ![1547205492679](C:\Users\GMFTBY\AppData\Roaming\Typora\typora-user-images\1547205492679.png)

   包含有 5 个组件， input encoder, belief state tracker, KB, response slot classifier, response decoder

   1. input encoder

      GRU 将 $A_{t-1},B_{t-1}, U_t$ 映射成对应的 hidden vector $h_l^E$, $l$ 指序列长度 

   2. belief state tracker

      information slot 中是参数绑定的多头 GRU decoder ，输入都是 $h_l^E$ 但是起始符号不同最后生成一个序列的 slot 和 value 值 $\{k^I\}$ 并且使用了 sequicty 中提到的 copy 机制 (copy input $A_{t-1},B_{t-1},U_t$ 中的 token) 使用交叉熵作为 loss 优化。

      request slot 使用了 attention 机制的 one-step GRU 并 sigmoid 生成最后的二分类结果，也使用交叉熵作为最后的评测指标

   3. KB

      使用 information slot 的输出结果作为查询条件，最后返还一个 5 维 one-hot 向量表示查询的结果的数目是什么。

      >一个新的 idea 如果并不是最后只需要一个结果作为返还呢，如果用户就是想要订阅多个票的话，只使用一个并不是一个很好的结果，输出多个结果才是理想中正确的。

   4. response slot classifier

      one-step GRU 使用了 Attention 机制，并使用了 KB 生成的 result one-hot 交叉熵损失函数

   5. response decoder

      copy 机制的 GRU decoders



## Key-Value Retrieval Networks for Task-Oriented Dialogue

1. 端到端的人物导向对话系统通常都在尽力的平滑和外部知识库之间的距离，本文使用 key-value 抽取架构解决这个问题，模型是端到端可谓的并且不需要显式建模 DST 

2. 提出的模型可以有效的聚合外部知识库的信息，并利用 RNN 语言建模的优势

3. 模型架构

   ![1547271170359](C:\Users\GMFTBY\AppData\Roaming\Typora\typora-user-images\1547271170359.png)

   * 主框架式 seq2seq 架构
   * 每一次 decode 的时候计算一下所有 entry 的 attention 分值输出的不仅仅式单词的概率还是一个对所有 entry 的 value 的 placeholder 出现的概率。

## Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems

[参考笔记](https://blog.csdn.net/weixin_40533355/article/details/83064795)