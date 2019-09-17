## CopyNet

### 1. 背景: Seq2Seq with Attention

1. 输入 $X=[x_1, ...,x_{T_s}]$ 被 RNN 转换成固定维度的向量保存输入序列的信息
   $$
   h_t = f(x_t, h_{t-1}),c=\phi(\{h_1,...,h_{T_s}\})
   $$
   

   其中 $h_t$ 成为状态，$c$ 称为 context vector，$f$ 是非线性函数

2. 之后 decoder 部分将 $c$ 展开成为新的序列并计算对应的损失函数
   $$
   s_t = f(y_{t-1},s_{t-1},c)\\
   p(y_t|y_{<t},X)=g(y_{t-1}, s_t, c)
   $$
   

3. Attention 通过修改固定的 context vector $c$ 成为动态计算的 context vetor，明显的提高了模型的性能

   最常见的 attention 就是对 encoder output 使用的 weighted sum
   $$
   c_t = \sum_{\tau=1}^{T_s} \alpha_{t\tau}h_{\tau}; \alpha_{t\tau}=\frac{e^{\eta(s_{t-1},h_{\tau})}}{\sum_{\tau'}e^{\eta(s_{t-1},h_{\tau'})}}
   $$
   一般 $\eta$ 函数通常采用后接激活函数的 MLP

### 2. CopyNet

![1552143871065](C:\Users\GMFTBY\AppData\Roaming\Typora\typora-user-images\1552143871065.png)

1. copy 机制更贴近于自然语言的方式，但是因为严格的的运算过程很难融入到 seq2seq 的框架中，本文的 copynet 实现了 copy 机制的端到端可训练性

2. 仍然是 Encoder-Decoder 主框架的形式

   * Encoder 使用 BiRNN 生成 encoder outputs $M= \{h_1,...,h_{T_s}\}$ 
   * Decoder 使用 RNN 通过不断的查询 $M$ 生成对应的 target sequencce 
     1. 预测单词的概率中使用了两种模式，分别是生成模型和 copy 模式
     2. 时间步 $t$ 状态更新中不仅仅使用了之前标准的 seq2seq 中的 $y_{t-1}$ 的 word embedding 还使用了对应的 $M$ 矩阵中的 $y_{t-1}$ 的位置信息
     3. 读取 $M$ 矩阵的过程使用了 selective read 的方法加强

3. 使用 copy 和 generation 进行单词预测

   使用标准单词库 $V= \{v_1,...,v_N\}$ 和 `unk` 作为 OOV 单词的表示，$X$ 单词集表示输入的序列中出现的单词 $\{x_1,..,x_{T_s}\}$ ，因为 $X$ 中可能包含着 OOV 单词，使用 $X$ 单词可以保证 decoder 可以输出输入序列中的 OOV 单词。总此表的是三部分的并集。

   单词的输出概率的计算公式是
   $$
   p(y_t|s_t,y_{t-1}, c_t,M)=p(y_t,g|s_t,y_{t-1},c_t,M) + p(y_t,c|s_t,y_{t-1},c_t,M)
   $$
   其中 $g$ 是生成模式，$c$ 是 copy 模式

   ![1552144492467](C:\Users\GMFTBY\AppData\Roaming\Typora\typora-user-images\1552144492467.png)

   其中 $\phi_c,\phi_g$ 是对应模式的 score function，其中的归一化因子 $Z$ 
   $$
   Z = \sum_{v\in V \bigcup \{UNK\}} e^{\phi_g(v)}+\sum_{x\in X}e^{\phi_c(x)}
   $$
   ![1552144692552](C:\Users\GMFTBY\AppData\Roaming\Typora\typora-user-images\1552144692552.png)

   最后的单词输出的 score function 看上去很奇怪，针对 $\phi_g$ 的打分函数是单词的 one-hot 和隐状态的双线性变换，$\phi_c$ 的打分函数是一样的但是多加了非线性变换

4. 状态更新

   传统的状态更新 $s_t$ 需要的仅仅是 $y_{t-1}$ 但是再 copynet 中需要做出一点小的改变，使用的是 $[e(y_{t-1});\zeta(y_{t-1})]^T$ 其中 $e$ 是 embedding 的 lookup 矩阵，但是 $\zeta$ 代表的是，这部分的功效是构成了 selective read 机制，这部分的工作原理和 attention 类似，但是提供了对位置的更加精确的描述，用来锁定 $y_{t-1}$ 单词再 source 序列中的位置信息，为了可以 copy source 序列中的一串子序列而不是一个 token 为下一个要 copy 的 token 做出指示
   $$
   \zeta(y_{t-1})=\sum_{\tau=1}^{T_s}\rho_{t\tau}h_{\tau}\\
   \rho_{t\tau}=\frac{p(x_{\tau},c|s_{t-1},M)}{K},x_{\tau}=y_{t-1}\ otherwise\ 0
   $$
   $K$ 归一化因子 $\sum_{\tau':x_{\tau'}=y_{t-1}}p(x_{\tau'},c|s_{t-1},M)$ 

5. 针对 $M$ 的混合寻址

   文本信息和位置信息的 $M$ 混合寻址，这两种策略都在 decoder 中被进行管理，这两种机制分别是 attention read 和 selective read，也就是决定何时进入 copy mode

   输入 token 的语义和位置信息都被编码成了 $M$ ，在没有进入 copy mode 之前主要是 attention 发挥语义信息的作用，但是一旦进入 copy mode 主要是 selective read 进入了位置信息的控制阶段。

   在 copy mode 下，位置信息的流动是这样的

   $\zeta(y_{t-1})\longrightarrow s_t \longrightarrow y_t \longrightarrow \zeta(y_t) $

### 3. Learning

虽然 copy 机制是硬逻辑，但是 copynet 是完全可微的端到端架构
$$
L=-\frac{1}{N} \sum_{k=1}^N\sum_{t=1}^T \log[p(y_t^{(k)}|y_{<t}^{(k)},X^{(k)})]
$$
对于生成模式和 copy 模式都是概率上的，通过数据进行学习，例如，如果一个 $y_t$ 发现在 source 中的话， copy mode 将会对最后的模型生成产生贡献，反向传播会或多或少的鼓励这种模式