## GAN

---

### 1. Introduction

* Conditional GAN: 输入条件(可控向量)，输出对应的目标

* discriminator: 输出 scalar 决定数据是来自真实分布的概率

* Algorithm

  * Init G and D

  * In each training iteration

    1. Fix G, update D k times
       $$
       V=\frac{1}{m}\sum_{i=1}^m\log D(x_i) + \frac{1}{m}\sum_{i=1}^m \log (1-D(\overrightarrow{x_i}))\\
       \theta_d \leftarrow \theta_d + \eta \nabla V(\theta_d)
       $$

    2. Fix D, update G one time
       $$
       V=\frac{1}{m}\sum_{i=1}^m\log (D(G(z_i)))\\
       \theta_g\leftarrow \theta_g - \eta \nabla V(\theta_g)
       $$

* 结构化数据学习

  1. 结构化数据: image, sent, video, ...
  2. 结构化学习的核心在于如何 plan (从大局出发)
  3. GAN 就是结构化学习方法

* 为什么需要 discriminator

  之前的生成模型常用的是 Auto-Encoder 但是 AE 泛化能力不高，这一点可以通过 VAE 的一解决

  但是不论是 AE 还是 VAE 都是尝试让生成模型和 Input 之间越接近越好(MSE)但是这样的效果并不好,MSE 并不能有效的表示出相似读(从人的角度来看)，在结构化学习中欧给你，component 之间的关系也非常重要，这一点 MSE 无法衡量，需要算法有一定的大局观。

  GAN = VAE + Multi Hidden Layer

* 使用 discriminator 生成数据是否可以

  使用 discriminator 生成数据的方式是穷举所有可能的输入数据并根据 scalar 选择最好的那一个数据，但是这一点不可能实现。所有使用 generator 进行生成，discriminator 将 generator 的数据当作负采样的数据进行学习。

* GAN 直观上的效果

  通过 generator 生成的 x (输入空间中) 在 discriminator 的分布中不断的压低这些生成的数据的分之，突出真实图片的分值，从而将不可能是真实分布的输入拒绝接受可能是真实分布的输入数据，从而拟合真是数据的分布。

* G and D

  1. G: 优点是生成数据简单，缺点是难以学习组建之间的关系，没有大局观(MSE导致的)
  2. D: 优点是有大局观，缺点是难以生成

  G 是在 D 的引导下生成更像真实分布的数据的

* VAE 稳定但是 GAN 效果可能更好

---

### 2. Conditional GAN

* 可以控制输出结果的 GAN

* 传统的 Text2Image 效果不好的原因在于输出的结果过于平均化

* G 的输入是 train data (text) + normal distribution $$x = G(train, noise)$$

  D 的输入是 $$x$$ ，输出是 scalar

  这样的 discriminator 的缺点是只要生成的数据像 data 中的数据即可，和控制的信息没有关系

* 改良后的 D 的输入是 $$x$$ 和 $$train$$ 目的是让判别器要判断一下2点

  $$x$$ 需要转化成一个向量，$$train$$ 也是

  * 图像是否真实
  * 图像是否匹配

* 算法

  1. D

     1. sample m pair $$\{(c^1,x^1),(c^2,x^2),...,(c^m,x^m)\}$$from training dataset

     2. sample m noise $$\{z^1,z^2,...,z^m\}$$ from normal distribution and get the generate data $$\{\overrightarrow{x^1},\overrightarrow{x^2},...,\overrightarrow{x^m}\},\overrightarrow{x^i}=G(c^i,z^i)$$

     3. sample m objects $$\{x'^1,x'^2,...,x'^m\}$$

     4. update

        最后一项是为了强化 D 识别匹配的能力
        $$
        V=\frac{1}{m}\sum_{i=1}^m\log D(c^i, x^i) + \frac{1}{m}\sum_{i=1}^m\log (1-D(c^i,\overrightarrow{x^i})) + \frac{1}{m}\sum_{i=1}^m\log (1-D(c^i,x'^i))\\
        \theta_d \leftarrow \theta_d + \eta \nabla V(\theta_d)
        $$

  2. G

     1. sample m noise $$\{z^1,z^2,...,z^m\}$$ from a distribution

     2. sample m conditions $$\{c^1,c^2,...,c^m\}$$ from training dataset

     3. update
        $$
        V=\frac{1}{m}\sum_{i=1}^m\log (D(G(c^i,z^i)))\\
        \theta_g\leftarrow \theta_g - \eta \nabla V(\theta_g)
        $$

* discriminator 显示展开最后一项的新架构

  x -- embedding --> NN --> y --> score for real

  c & y --------------> score for match

---

### 3. GAN 的理论

* $$P_{data}$$ 是我们需要寻找的真实数据的概率分布，GAN 的目的是生成尽可能一样的分布

* 之前的实现让 $$P_g \rightarrow P_{data}$$ 的方法是 MLE

  1. $$P_{data}(x)$$ sample to get

  2. $$P_G(x;\theta)$$ 带有参数的模型，目的是找打一个参数 $$\theta$$ 使得 $$P_G(x;\theta)\rightarrow P_{data}(x)$$

  3. sample data(for example, image) $$\{x^1,x^2,...,x^m\}$$ for $$P_{data}(x),P_{data}(x)=1$$ and compute the $$P_G(x^i;\theta)$$

  4. make $$L=\Pi_{i=1}^mP_G(x^i;\theta)$$ , 因为 $$x^i$$ 来自真实数据分布，让这部分概率最大，使用梯度下降
     $$
     \begin{equation*}
     \begin{split}
     \theta^*&=\arg\max_{\theta}\Pi_{i=1}^mP_G(x^i;\theta)\\
     &=\arg\max_{\theta}\sum_{i=1}^m\log P_G(x^i;\theta)\\
     &\approx\arg\max_{\theta}\mathbb{E}_{x\backsim P_{data}}[\log P_G(x;\theta)]\\
     &=\arg\max_{\theta}\int_xP_{data}(x)\log P_G(x;\theta) dx - \int_xP_{data}(x)\log P_{data}(x)dx\\
     &=\arg\max_{\theta}\int_xP_{data}(x)[\log P_G(x;\theta)-\log P_{data}(x)]dx\\
     &=\arg\min_{\theta}KL(P_{data}||P_G)
     \end{split}
     \end{equation*}
     $$

     $$
     D_{KL}(P||Q)=-\sum_i P(i)\log \frac{Q(i)}{P(i)}\\
     D_{KL}(P||Q)=\int_{-\infty}^{+\infty}P(x)\log \frac{Q(x)}{P(x)}\\
     JS(P||Q)=\frac{1}{2}KL(P||\frac{P+Q}{2}) + \frac{1}{2}KL(Q||\frac{P+Q}{2})
     $$

     MLE 的本质在于最小化生成概率分布和真实分布的 KL 散度，$$P_G$$ 中的 $$\theta$$ 参数可以是任何的参数，也可以是高斯分布等等，但是其他的未知的分布不如已知的分布(高斯分布)好求。

* 神经网络本质上通过一个函数将输入的数据(normal 随机的分布)的分布转化成了一个新的概率分布 $$P_G$$ (高斯分布的表现太差)

* 目的让 $$P_G\rightarrow P_{data}$$
  $$
  G^*=\arg\min_GDiv(P_G,P_{data})
  $$

  1. MLE 的 Div 是 KL 散度
  2. GAN是其他的散度 (JSD)

  因为 $$P_G$$ 和 $$P_{data}$$ 都是未知的，无法计算散度，通过对抗学习可以近似实现。

* GAN

  虽然 $$P_G,P_{data}$$ 都是未知的，但是可以 sample 得到， GAN 利用 discriminator 计算散度
  $$
  V(G,D)=\mathbb{E}_{x\backsim P_{data}}[\log D(x)] + \mathbb{E}_{x\backsim P_G}[\log (1-D(x))]\\
  D^*=\arg\max_D V(D,G)
  $$

  $$
  V=\mathbb{E}_{x\backsim P_{data}}[\log D(x)] + \mathbb{E}_{x\backsim P_G}[\log(1-D(x))]=\int_x[P_{data}(x)\log D(x) + P_G(x)\log(1-D(x))]dx
  $$

  最大化上式可以得到理想的 $$D$$ ，这个 $$D$$ 可以有效的计算出两个分布的差异(散度)

  可以抽象上式为 $$a\log D + b\log(1-D)$$
  $$
  \frac{\partial V}{\partial D}=\frac{a}{D} - \frac{b}{1-D}
  $$
  可得最大的 $$V$$ 对应的 $$D=\frac{a}{a+b}=\frac{P_{data}(x)}{P_{data}(x) + P_G(x)}$$，局部最优

  带入 $$D$$ 至 $$V$$ 可得最优的 $$V(G,D^*)=-2\log2 + 2JSD(P_{data} || P_G)$$，最大化 $$V(G,D)$$ 等价于求解一个 JSD 距离

  求到这里我们得到了计算两个概率分布距离的一种方式，所以对目标函数改造
  $$
  G^*=\arg\min_G\max_DV(D,G)
  $$
  可以理解为，加入我现在有几个候选的 G ($$G_1,G_2,G_3$$) 首先计算每一个 G 对应的最大的 V 函数(通过调节 D 实现，目的是找到最好区分两个分布的 D)，然后在里面选一个最小的作为下一个 G (目的让G的分布更靠近真是分布)

  本质上 GAN 就是在求解上式，max 为了找到更好区分两种分布的方式，min是减少距离，但是只有在 G 变化不大的时候才是在求解 JSD 在论文中，为了加快 G 的收敛速度一般都会改为
  $$
  V=\mathbb{E}_{x\backsim P_G}[-\log D(x)]
  $$


---

### 4. WGAN and fGAN

* 当 $$P_G,P_{data}$$ 的重叠区域很小的时候，使用 sample 的方法 $$P_G,P_{data}$$ 几乎不会重叠，如果概率分布不重叠的话，D可以轻易的区分两个分布因为G的变化不大，所以放弃了很多的尝试，在坏的 G 左右摇摆，使得传统的 GAN 难以训练。为了加快收敛速度， D 输出可以采用 linear 函数 + 梯度裁剪，保留 G 的梯度

* 使用 wasserstein 距离替代 JSD
  $$
  V(G,D)=\max_{D\in 1-Lipsclitz}\{\mathbb{E}_{x\backsim P_{data}}D(x) - \mathbb{E}_{x\backsim P_G}D(x)\}
  $$
  其中 1-Lipsclitz 函数是平滑的函数 $$||f(x_1)-f(x_2)||\leq k||x_1-x_2||$$平滑的函数很难找，一般使用梯度裁剪实现