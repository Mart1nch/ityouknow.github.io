---
layout: post
title: Tensorflow学习笔记
category: tensorflow
tags: [tensorflow]
---

ps：看这个之前需要知道一些基本的tf的东西，看不懂或者我的疏忽导致的用词错误还请指正，文章中的图片代码部分来自于《TensorFlow实战Google深度学习框架》，侵权即删。

## 学习纲要

1、深度学习和深层神经网络的概念，深层神经网络可以解决部分浅层神经网络解决不了的问题。

    （1）线性变换存在的问题，为什么在深度学习的定义中强调多层和非线性。
    
    （2）如何去线性化及其tf程序实现。
    
    （3）深层神经网络可以解决很多浅层不能解决的问题。
    
2、设定神经网络的优化目标--损失函数；如何通过对损失函数的设置使神经网络优化的目标接近实际。

3、反向传播算法。

4、神经网络优化经常遇到的问题和解决。

## 深度学习和深层网络

- 深度学习：通过多层非线性变换对高复杂性数据进行建模算法的合集。

- 深层神经网络：实现多层非线性变换最常用的方法，可以认为深度学习就是深层神经网络的代名词。

### 1、线性模型的局限性

任意线性模型的组合仍然是线性模型，所以多层神经网络如果都是线性变换其实跟一层没什么区别。

如下图，两个输出特征x1,x2，我们分别计算没有隐层和有一层隐层时的输出，很明显可以看到如果只有线性变换的话，多了隐层其实没什么太大的所用。可以看作不管有多少隐层最后的输出y和输入x1,x2仍然只是线性关系在三维空间中只能拟合出一条直线，而我们实际应用中比方说预测房价/水位，拟合出来的不可能是一条直线（数学方面的东西不用讲了吧）

![](http://www.itmind.net/assets/images/2019/tensorflow/linenohidel.jpg)

![](http://www.itmind.net/assets/images/2019/tensorflow/linewithhidel.jpg)

### 2、偏执bias

仍然是上面的例子仅仅是x1,x2,y组成的三维空间中，那么拟合出来的直线一定只能是过原点的，加上偏置b可以让这条直线移动来达到我们预想的拟合直线。大家可以拿一次函数y=kx+b类比一下。

### 3、使用激活函数去线性化

常见的激活函数如下

![](http://www.itmind.net/assets/images/2019/tensorflow/activationfun.jpg)

加上偏置b和激活函数，最后应该是图3这样的一个模型（下图只是一个没有隐层的简单示例模型，f就是上面的sigmoid函数）。理论上来说，加上偏置和激活函数f后，再经过多层的神经网络我们可以拟合出任意一条需要的曲线。

![](http://www.itmind.net/assets/images/2019/tensorflow/demo1.jpg)

``` python
    # 实现图3输出的代码实现
    y = tf.sigmoid(tf.matmul(x, w) + b)
```

## 损失函数

我们知道神经网络训练的过程就是不断优化模型的各个参数，使我们的输出y尽量去拟合训练数据的过程。那么w怎么调整优化呢？总不致于让我们手动一个一个试吧，那还真的是‘人工智能’哈哈哈哈。所以这里就要介绍损失函数了：损失函数刻画的是标准输出和我们预测的输出的差距，是所有相对应的预测数据和标准输出间的差距和。

### 1、经典损失函数

分类问题和回归问题是监督学习的两大种类。

分类问题希望解决的是将不同的样本分到事先定义好的类别中。通过神经网络解决多分类问题最常用的办法是设置n个输出节点，n为类别的个数（比如识别手写体0-9的数字，n一般就会定义成10）。对于每一个样例，神经网络可以得到一个n维数组作为输出结果，数组中的每个维度对应一个类别。交叉熵是分类问题中使用比较广泛的一种损失函数。但是，交叉熵刻画的是两个概率分布之间的距离，但是神经网络的输出不一定是一个概率分布（概率分布的定义可以去看一下概率论，这里只简单说明：比如说n个事件，每个事件的发生率都在[0,1]，并要求n个事件概率总和为1。而神经网络的输出可能只是[0 0 0 80 10 10]这样一个‘可能性分布’，需要转化成[0 0 0 0.8 0.1 0.1]的形式）。如何将神经网络前向传播的结果变成概率分布呢？--答案是Softmax回归。Softmax回归本身可以作为一个学习算法来优化分类结果，但在tf中它的参数被去掉了只是被当作一层额外的处理层，将神经网络的输出变成一个概率分布而已。用这个函数求概率的主要原因是这个函数的求导比较方便，而我们优化损失函数的时候是需要对其进行求导的。

``` python
# y_为正确答案，y为预测答案
# 交叉熵的实现
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# softmax回归后的交叉熵
tf.nn.softmax_cross_entropy_with_logits(y, y_)
```

回归问题希望解决的是对具体数值的预测，比如水位预测房价预测销售预测。这些问题预测的不是事先定义好的类别，而是一个任意实数，因此解决回归问题的神经网络一般只有一个输出节点，即预测值。对于回归问题最常用的损失函数是均方误差。

``` python
mse = tf.reduce_mean(tf.square(y_ - y))
```

### 2、自定义损失函数

自定义损失函数可以让神经网络优化的结果更加接近实际问题的需求。比如销量预测中，一个商品成本1元，利润10元，那么多预测一个亏损1元，少预测一个少挣10元。此时，如果神经网络模型刻画的损失函数是均方误差，那么这么模型可能无法最大化预期的利润(只能让训练结果离标准答案接近而不能使利润最大化，这里所谓的标准答案就是刚好卖光的情况)。所以要将损失函数和利润直接联系起来。损失函数定义的是损失，所以要将利润最大化，定义的损失函数要刻画成本或者代价。

``` python
定义预测多以及预测少的成本
less = 10
more = 1
loss = tf.reduce_sum(tf.select(tf.greater(y, y_), (y - y_) * more, (y_ - y) * less))# y>y_时取第二个参数，y<=y_时取第三个参数
```

## 神经网络优化算法

通过反向传播算法（backpropagation）和梯度下降算法(gradient decent)调整神经网络中的参数。

梯度下降算法主要用于优化单个参数的取值，反向传播算法给出了一个高效的方式在所有参数上使用梯度下降算法，从而使神经网络模型在训练数据上的损失函数尽可能小。反向传播算法是训练神经网络的核心算法，可以根据定义好的损失函数优化神经网络中参数的取值，从而使神经网络模型在训练数据集上的损失函数达到一个较小值。

![](http://www.itmind.net/assets/images/2019/tensorflow/backpropagation.png)

图中的J(θ)是损失函数，θ表示的是需要优化的所有参数（所有的权重w等），梯度为∂J(θ)/∂θ，学习率ƞ可以近似认为每次更新参数的幅度。所以参数的更新公式为：θn+1 = θn - ƞ(∂J(θ)/∂θ)

所以神经网络的优化过程就很明朗了：第一个阶段先通过前向传播算法计算得到预测值，并将预测值和真实值比较得出两者之间的差距，通过差距刻画损失函数；然后在第二个阶段通过反向传播算法计算损失函数对每一个参数的梯度，再根据梯度和学习率使用梯度下降算法更新每一个参数。

需要注意的是，梯度下降算法不能保证被优化的函数得到全局最优解，只能是局部最优解（比如多个波谷的情况），也由此可见参数的初始值很大程度影响最后结果。只有当损失函数为凸函数时，梯度下降算法才能保证达到全局最优解。第二个问题，梯度下降算法计算时间太长了。损失函数是在所有训练数据上的损失和。在每一轮迭代中都需要重新计算在全部训练数据上的损失函数，为了加速训练过程，可以使用随机梯度下降算法（stochastic gradient descent）这个算法在每一轮迭代中随机优化某一条训练数据上的损失函数，所以它的问题也很明显：某一条上损失函数更小不代表全部数据上的损失函数更小，用这个算法甚至连局部最优也有可能无法得到。

所以要综合两种算法：每次计算一个batchsize大小的训练数据batch的损失函数，batchsize人为定义的常量，batchsize越小损就越接近随机梯度下降算法，越大越接近梯度下降算法。通过矩阵运算，每次在一个batch上优化参数不会比单个数据慢太多；每次使用一个batch可以减小收敛所需要的迭代次数，同时可以使收敛到的结果更加接近梯度下降的效果。下面代码包含了自定义损失函数和随机梯度下降算法。


``` python
 batch_size = 8
    # 两个输入节点
    x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
    # 回归问题一般只有一个输出节点
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
 
    # 定义了一个单层的神经网络前向传播过程，这里就是简单的加权和
    w1 = tf.Variable(tf.random_normal([2, 1], seed=1, stddev=1))
    y = tf.matmul(x, w1)
 
    # 定义预测多了和预测少了的成本
    loss_less = 10
    loss_more = 1
    # 更新 tf.select->tf.where
    loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                                   (y - y_) * loss_more,
                                   (y_ - y) * loss_less))
    learning_rate = 0.001
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
 
    #随机数生成模拟数据集
    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)
    # 设置回归的正确值是两个输入的和加上一个随机量。之所以加上一个随机量是为了
    # 加入不可预测的噪音， 否则不同损失函数的意义就不大了，因为不同损失函数都会在能
    # 完全预测正确的时候最低。一般来说噪音为一个均值为0的小量，所以这里的噪音设置为
    # -0.05 ~ 0.05的随机数
    Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]
 
    # 训练神经网络
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 5001
        # 0-5000
        for i in range(STEPS):
            start = (i * batch_size) % dataset_size
            end = min(start+batch_size, dataset_size)
            sess.run(train_step,
                     feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 1000 == 0:
                print('第', i, '次循环后的权重：', sess.run(w1))
                '''
                i=5000时，得到的W1是[1.01934695,1.04280889]
                也就是说预测函数是1.02x1 + 1.04x2，实际正确答案是x+y
                因为指定了预测大了损失小，所以结果让我们稍微生产过量
                如果指定预测大了损失大，那么训练出来的W应该会小于1
                这里用均方误差只能更接近正确值，而不能利润最大化
                '''
```

## 神经网络进一步优化

### 1、学习率的设置

通过指数衰减的学习率可以让模型在训练前期快速接近较优解，又可以保证模型在训练后期不会有太大波动。

tf.train.exponential_decay(staircase=False)//实现了指数级减小学习率，staircase默认false时是平滑的连续下降，当设置成true时会变成阶梯下降。

![](http://www.itmind.net/assets/images/2019/tensorflow/gradientdes.jpg)

阶梯下降的常用情景是：每完整过完一遍训练数据，学习率就减小一次，可以使训练数据集中所有数据对模型训练有相等作用。

``` python
global_step = tf.Variable(0)
# 初始学习率0.1，每训练100轮学习率乘以0.96，设置True后为梯度下降
learning_rate = tf.train.exponential_decay(
        0.1, global_step, 100, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
```
### 2、过拟合问题

当训练数据不多或者 overtraining（导致模型特征过于复杂）时就会导致过拟合。

上面讲述了如何通过训练数据优化一个给定的损失函数从而得到优化的模型。然而在真实的应用中想要的并不是让模型尽量拟合训练数据的行为，而是希望通过训练出来的模型对未知的数据进行判断（预测/修复），所以我们只需拟合一个数据的趋势即可。理论上来说，如果一个模型中的参数比总的训练数据还多，那么只要训练数据不冲突，就可以使损失函数优化到0（方程n个式子解n-1个参数的思想），但是这样训练出来的模型或许“记忆”了所有的随机噪音，而没有学习训练数据的趋势，这样就导致了预测时候的差距。试想一下用下图中过拟合的曲线去预测未来的数据好还是合理模型去预测好？毕竟过拟合中包含了不必要的很多‘偶然点’。

![](http://www.itmind.net/assets/images/2019/tensorflow/overfitting.jpg)

避免过拟合问题的常用方法是正则化。正则化的思想就是在损失函数中加入刻画模型复杂度的指标，来限制权重大小，使模型不能任意拟合训练数据中的随机噪音。此时就不是直接优化损失函数J(θ)，而是$J(θ)+λR(w)$。R(w)是模型复杂度，λ是模型复杂损失在总损失中的比例。一般来说模型复杂度只由权重w决定。刻画R(w)常用的两种方式：

L1正则化就是L1范数，L2正则化就是L2范数。

ps：L0范数是$||W_0||$，表示向量中非零元素的个数；L1范数是$||W_1||$，表示绝对值之和；L2范数是$||W_2||$，表示通常意义上的模。其实防止过拟合就是要做特征提取，让更多的$W_n=0$，即让$L_0$范数变小，同时让损失函数J变小。但是遇到了NP难的问题所以退而求其次让$L_1$范数变小，因为两者的‘意义’更加接近。所以最后希望$J+||W_1||$最小化，损失函数J后添加的额外项$||W_1||$就是$L_1$正则化。但是$L_1$正则化公式不可导，我们知道优化神经网络的时候需要对损失函数求导，L1正则化的计算复杂度就很大，这时候L2正则化就比较简洁了。

当然我们也可以两个正则化一起使用，这里就不多指点江山了。

``` python
# 之前说了最后希望J+||W1||最小化，这里我们将J和||W1||分别加入到tf的集合losses中
# 随后计算这个集合的总和就是我们最终的损失函数
 
 
# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为'losses'的集合中
def get_weight(shape, lambdaa):
    # 生成神经网络某两层之间权重张量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection函数将这个新生成变量的L2正则化损失项加入到集合
    # 这个函数的第一个参数losses是集合的名字，第二个参数是要加入这个集合的内容
    # 我们将正则化损失加入到losses后将返回权重
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambdaa)(var))
    return var
 
 
def Demo():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    batch_size = 8
    # 定义了每一层网络中节点的个数
    layer_dimension = [2, 10, 10, 10, 1]
    # 神经网络的层数
    n_layers = len(layer_dimension)
 
    # 这个变量维护前向传播时最深层的节点，开始的时候就是输入层
    cur_layer = x
    # 当前层的节点数
    in_dimension = layer_dimension[0]
 
    # 通过一个循环来生成5层全连接的神经网络结构
    for i in range(1, n_layers):
        # layer_dimension[i]为下一层的节点数
        out_dimension = layer_dimension[i]
        # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
        weight = get_weight([in_dimension, out_dimension], 0.001)
        # 生成1*10，值全为0.1矩阵
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
        # 使用ReLU激活函数
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        # 进入下一层之前将下一层的节点个数更新为当前层节点个数
        in_dimension = layer_dimension[i]
 
 
    # 在定义神经网络前向传播的同时已经将所有的L2的正则化损失函数加入了图上的集合
    # 所以这里只需要刻画模型在训练数据上表现的损失函数
    mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
 
    # 将均方误差损失函数加入损失集合
    tf.add_to_collection('losses', mse_loss)
 
    # get_collection返回一个列表，这个列表是所有这个集合中的元素
    # 这个样例中，这些元素就是损失函数的不同部分，将他们加起来就是最终的损失函数
    loss = tf.add_n(tf.get_collection_ref('losses'))
    return loss
```

### 3、滑动平均模型

采用随机梯度下降算法训练神经网络时，使用滑动平均模型可以在一定程度上提高最终模型在测试数据上的表现。

tf.train.ExponentialMovingAverage(decay, num_updates)//decay是衰减率。控制模型更新速度，ExponentialMovingAverage对每一个变量维护一个影子变量，每次运行变量更新时，影子变量的值更新为：shadow_variable=decay * shadow_variable+(1-decay) * variable，variable是待更新的变量。decay越大模型越稳定。num_updates动态设置decay的大小，如果在ema初始化时提供了这个参数，每次使用的衰减率将是min{decay, (1+num_updates)/(10+num_updates)}。滑动平均就先不说了。消化一下吧，多看几遍就疏通了。
