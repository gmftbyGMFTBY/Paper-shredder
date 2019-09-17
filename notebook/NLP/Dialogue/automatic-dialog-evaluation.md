## RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems

>AAAI 2018 yan rui

* 开放领域对话系统的评测非常困难，因为涉及到语义丰富度等信息，在任务领域对话系统的评测指标更多关注的是任务的完成结果，对于开放领域借鉴意义不大。

* 之前的工作提出过使用人工标注的数据进行预训练，但是受限于人工标注数据很难扩展

* 本文提出针对开放领域对话系统的无监督评测指标

  1. 是基于 embedding 的评测指标，其中计算了和 ground truth 的语句的相似性，使用了针对 word vector 的池化保证相似性计算是对开放领域合适的 (BLEU 等太严格了)
  2. 另外计算查询和回复之间的信息的相似性，使用了负采样尽心计算不需要人工标注数据，这个分数衡量了上下文信息
  3. 结合上述的两种信息

* ![1553651940013](C:\Users\GMFTBY\AppData\Roaming\Typora\typora-user-images\1553651940013.png)

* referenced scorer 使用了余弦相似度计算 embedding (embedding 使用 max pooling and min pooling)

* unreferenced scorer 使用 RNN 抽特征合并计算上下文相似度，负采样训练

  ![1553652271651](C:\Users\GMFTBY\AppData\Roaming\Typora\typora-user-images\1553652271651.png)

* 使用简单的启发式的混合方法是合理的 (min, max, avg, ...)

* 方法设计是针对与单轮对话但是可以通过设计 unreferenced scorer 评价的网络可以有效的扩展



## Fix 

* accuracy question

  ```
  refer: the restaurant is a nice place in [expensive] serves [chinese] food.
  respo: the restaurant is a great restaurant serves [chinese] food and in [expensive] price range.
  ```

  中心词敏感的判别器，修改 referenced scorer (×) / unreferenced scorer (√)

* 修改为 context aware，修改 unreferenced scorer，encoder 足够了

* `[!]` GAN is better than this method in this setup.

  GAN 不需要重新训练迁移到其他数据集上的可能性 ？

* PMI