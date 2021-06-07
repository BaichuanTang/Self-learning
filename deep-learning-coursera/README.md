# Deep Learning Specialization 

[toc]

## Structuring Machine Learning Projects

F1 is the harmonic mean of p and r. 调和平均数

millisecond 毫秒 mɪlisekənd

wake words、trigger words唤醒词，比如siri

optimizing metric & statisfying metric 的经验：只设置一个优化目的，其他限制条件满足就行

dev set验证集

hit the bull's eye 击中靶心

But sometimes **partway** through a project 中途

pornographic 下流的

Indicator Function指示函数 

- 训练集可以和验证测试不一样的分布（反正无脑加样本就行），但是验证集和测试集必须同分布以保证效果的一致性。（我感觉现实中往往达不到）

- 不同的场景中需要定义不同的评估指标：比如说对pornographic很厌恶，那么需要将这些样本加重惩罚



![image-20210513215445312](../images/image-20210513215445312.png)

images are blurrier  模糊 blɜːri

images not well framed 没有好好拍照的

- 如果训练验证集和评价指标表现很好，但是测试集不行，需要更换指标或者更换验证集
- 吴老师的经验：尽快先选定指标和训练验证集开始工作。如果之后发现有偏差，再换。这样效率更高

**have a well-defined target and iterate efficiently towards improving performance**

- Bayes optimal error 理论最优准确率 （永远达不到），一群专家的准确率
- 如果要求人类准确率human level performance来代替Bayes error：就需要用表现最好的专家，而不是专家的平均水平/专家与普通人的平均水平
- 超过人类准确率后，提升就比较缓慢了
- 如果训练和理论准确率bayes optimal error的gap更大，那try bias reduction tactics，比如train a larger network, train longer, better optimization algorithms RMSprop, Adam, NN architecture, hyperparameter search
- 如果验证准确率和训练准确率的gap更大，那就reduce variance，比如regularization(l2, dropout), get a bigger training set, data augmentation

##### 业务启示

- 如果项目时间很短，来了新数据：数据量很少：①用新评价指标和新的验证测试集再次训练；②数据增强、数据合成data augmentation/data synthesis
- 如果竞争对手再False Negative Rate表现更好：即使调整指标（根据NER经验我理解：什么时候停取决于FNR最低）

![image-20210513222114025](../images/image-20210513222114025.png)

These examples may be a bit **contrived**.  预谋的; 不自然的; 人为的; 矫揉造作的; 做作的;  这是我举的不恰当的例子

bias avoidance tactics or variance avoidance tactics

There's quite a lot more **nuance** in how you factor in human level performance into how you make decisions in choosing what to focus on. 细微差别

surpassing a single radiologist, a single doctor's performance might mean the system is good enough to deploy in some **context**. 

- 机器比人类做的好的事情
- 不是三大领域的：online advertising, product recommendation, logistics(predicting transit time), loan approvals
- 比不上人类的Natural Perception Task：computer vision, speech recognition, or natural language processing 
- Medical tasks: read ECGs or diagnosing skin cancer 读心电图electrocardiogram



anomaly 异常事物；反常现象

That's what intrigued me about it  着迷，感兴趣[ɪnˈtriːɡd]

pretty like **painstaking work** 需要细心的，辛苦的，专注的（例如标注）

There is some superhumanness to it（我不知道人脑如何处理图像）

put all the research on hold 搁置所有研究

understand all the things that happen **under the hood** 在后台

The field will split into two **trajectories**.  AI的发展走向两个方向

It was implementing it myself from scratch that I felt gave me the best kind of **bang for the buck** in terms of understanding. 尽量设法获得更多的利润或价值（更多的打击for 美元）

You can work with it once you have written at something yourself **on the lowest detail**,至少也写点

I know that in machine learning, sometimes we **speak disparagingly of** hand engineering things, or using too much value insight 轻蔑

As you're **part way** through this process, sometimes you notice other categories of mistakes 一段时间后

#### 错例分析与统计

不仅可以知道哪类不行，还可以中途随时加入新的错误种类

![image-20210517010153779](../images/image-20210517010153779.png)

也可以对标注样本标错的情况提示出来

![image-20210517011846970](../images/image-20210517011846970.png)

But this quick counting procedure, can really help you make much better **prioritization decisions**, and understand how promising different approaches are to work on. 

可以选定优先方向，看到哪个方向会有前景。（这点的确对业务来说很重要！）

##### 业务启示

检查是否有标错的样本，但是事实上深度学习对标错的样本非常robust，不会刻意去学，因此没有必要。（只要是随机标错）但是系统性标错systematic errors不行，就像故意不标`local currency`一样

There is one **caveat** to this   告诫 [ˈkævi**æt**]

So I'd like to **wrap up with** just a couple of pieces of advice. 结束

accented speech 语音识别中（带有异国口音的，带有他乡腔调的）

And if sometimes the speaker **stutters** or if they use **nonsensical** phrases like oh, ah 口吃   无意义的

it might be okay to build a more complex system **from the get-go** by building on this large body of academic literature.  一开始的时候

just shoving it into the training set just to get it more training data.  **[ˈʃʌvɪŋ]** 猛推; 乱挤

 there's some subtleties 细微之处

speech activated rearview mirror 语音激活后视镜

#### 训练集和测试集可以不同

但是验证集和测试集需要一样，目的是为了探究在未知、新来数据集上的效果。例如：语音识别时用大量语料训练，但是最终落实在语音后视镜上。

也不一定非要用所有的数据。

如下图，**为了识别训练和验证集为什么准确率不一样**，解决方法为：1.在训练集内部划分出training-dev集，如果training-dev集的效果和dev集一样差，说明是模型过拟合造成准确率下降。如果本身training-dev集表现和训练集一样，则是验证集样本分布不同。

##### 业务启示

就像Coco经常反应模型效果下降，如果我们在训练的时候就已经划分出training-dev集，并且保证了模型没有过拟合的话，就是Coco选来的测试集的分布与咱们的分布不一样

![image-20210520231534244](../images/image-20210520231534244.png)

这是每一种集合的作用，用来评估不同的错误类型。如果验证集和测试集效果不一样，则有可能模型刚好overfit to dev set

![image-20210520233014497](../images/image-20210520233014497.png)

![image-20210520234346594](../images/image-20210520234346594.png)

#### 人工数据合成

注意点：不能都用同一段噪音去合成，噪音也需要用不同的，否则会对噪音过拟合。

在音频上表现很好，但是自动驾驶汽车图像上需要防止只对一部分汽车过拟合。

![image-20210520235503416](../images/image-20210520235503416.png)

And the challenge with artificial data synthesis is to the human ear, as far as your ears can tell, these 10,000 hours all sound the same as this one hour, so you might end up creating this very **impoverished** synthesized data set from a much smaller subset of the space without actually realizing it. 贫瘠的

radiology diagnosis 放射诊断

transfer learning=pre-training 除最后一层外的所有层+fine-tuning最后一层

- 什么时候用transfer-learning?
- when you have a lot of data for the problem you're transferring from and usually relatively less data for the problem you're transferring to. 

detecting curves, detecting edges

audio snippets 一小段音乐 

audio clip 音频片段，音频素材

maybe it won't hurt to include that 10 hours of data to your transfer learning, but you just wouldn't expect to get a meaningful gain. 

y is a four by one vector 是一个4×1的向量

![image-20210522191338103](../images/image-20210522191338103.png)

四个输出而非四分类的区别：四个输出是用四个logistic loss衡量，预测四个目标，y是4×1的向量；而多分类时只有一种输出的可能

##### 业务启示

- 我们做NER的时候，7个实体，并不是当作7分类去做，而是7个目标，因此矩阵有一层维数为7，分别反向传播
- 节约资源与时间，业务很关注这一点！
- transfer learning适用于标注数据少的情况，因此在合同NER场景里，如果NER已经在大样本里与训练过，那么会带来很好的效果。

吴老师还说，对于数据部分标签有缺失的情况，通过shared low level features的方式，仍然可以有效运算，算损失时只计算有标签的每个样本四个标签中的有的的标签。（虽然不知道具体是怎么实现的，tf这么智能了吗？）

Rich Carona, found many years ago was that the only times multi-task learning hurts performance compared to training separate neural networks is if your neural network isn't big enough. 所以只要足够大的网络，multi-task learning就不是问题

But I would say that on average transfer learning is used much more today than multi-task learning, but both are useful tools to have in your **arsenal**. 武器库

Again, with some sort of computer vision, object detection examples being the most notable exception. 目前大部分还是transfer learning远多于multi-task learning，但除了目标检测的情形，一个网络学很多目标

phoneme [ˈfoʊniːm] 音素

a face recognition **turnstile** 旋转门

![image-20210522210629164](../images/image-20210522210629164.png)

swipe an RFID badge 刷卡

pediatrician [ˌpiːdiəˈtrɪʃn] 儿科医生

But it's also not **panacea** [ˌpænəˈsiːə] 灵丹妙药

linguist 语言学家

Transcipt转译的时候，以前都是语言学家对音素phoneme定义好，但是端到端机器学习也许可以不按人类的思维学的更好

machine learning researchers tend to **speak disparagingly of** hand designing things. 轻蔑地

radar 雷达 lidar 激光雷达 [ai] 

steer your own car 转向 

**be mindful of** where you apply end-to-end deep learning. 考虑到

make good **prioritization** decisions in terms of how to move forward on your machine learning project 优化的决定

windshield wiper 风挡刮水器

you would be able to hear their **siren**. [ˈsaɪrən] 警报器

I was going to work one morning, and I **bumped into** Geoff Hinton 无意中遇到

helped with this **resurgence** of neural networks and deep learning. 复兴

even if you don't end up building computer vision systems **per se**,  [ˌpɜːr ˈseɪ] 本身，亲自

The convolution operation is one of the fundamental **building blocks** of a convolutional neural network. 构成要素

## CV

[OpenCV边缘检测(Sobel,Scharr,Laplace,Canny)](https://blog.csdn.net/qq_34711208/article/details/81703341)

padding的参数：`valid`：不进行padding和`same`：保持输出和输出维数相同的padding

which is why we **wound up with** this is three by three output 以...告终

by convention 按照惯例

可以这样理解：正常来说，是n+2p-f+1，但如果步长s不为1，则需要s等分，并且首位都算。并且，需要向下取整来保证如果最后一次卷积超出边界的话就不算。因此是下图的结果



![image-20210527152233053](../images/image-20210527152233053.png)

数学书上的卷积，是做上下、左右两次翻转（吴老师画错了），然后再去卷积的。因此实际我们深度学习中用的卷积是叫做互相关(*cross*-*correlation*)。

信号处理时的卷积之所以倒过来，是为了有(A\*B)\*C=A\*(B\*C)这样的性质，但对于深度学习来说，不需要这样的性质，叫做**associativity**，可结合性。

![image-20210527153817379](../images/image-20210527153817379.png)

以前对多层卷积一直理解错了：其实每个3×3×3的卷积核只会得到一个2维的4×4，如果训练两个卷积核就会得到4×4×2的输出，以此类推。

多层卷积：有几层要用**channel**不要用**depth**，**depth**容易和神经网络的深度混淆。

最后，卷积后真正的输出是加bias b并且用relu函数的结果

![image-20210527161745192](../images/image-20210527161745192.png)

![image-20210527163856226](../images/image-20210527163856226.png)

零是黑，白色是255

![image-20210527210354704](../images/image-20210527210354704.png)



So these is really one property of convolution neural network that **makes it less** **prone to overfitting** . 

- 网络走的越深，w宽度和h高度越窄，channels越多
- 一个或者多个卷积层后面会加一个池化层

![image-20210527181717174](../images/image-20210527181717174.png)

So we'll do that next week, but **before wrapping this week's videos** **just one last thing which is** 结束这周

#### 参数个数

吴老师图里写错了

1. 208 should be (5\*5\*3 + 1) \* 8 = 608  很重要！！！
2. 416 should be (5\*5\*8 + 1) \* 16 = 3216

3. In the FC3, 48001 should be 400\*120 + 120 = 48120, since the bias should have 120 parameters, not 1 很重要！！！

4. Similarly, in the FC4, 10081 should be 120*84 + 84 (not 1) = 10164

(Here, the bias is for the fully connected layer.  In fully connected layers, there will be one bias for each neuron, so the bias become In FC3 there were 120 neurons so 120 biases.)

5. Finally, in the softmax, 841 should be 84*10 + 10 = 850

![zsA-BM-WEemwyhIGNHHGIg_bfd232955b8aad2ed0c156bbac9f09b3_nn-example](../images/zsA-BM-WEemwyhIGNHHGIg_bfd232955b8aad2ed0c156bbac9f09b3_nn-example-1622111204245.jpg)

#### 使用卷积的好处

- parameter sharing 节约参数，每个像素只和周围一圈像素有关
- sparsity of connections 不像全连接，无论物体在图片中的任何位置，都可以用filter探测到

![image-20210527203659525](../images/image-20210527203659525.png)

$$n_H = \Bigl\lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \Bigr\rfloor +1$$
$$n_W = \Bigl\lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \Bigr\rfloor +1$$
$$n_C = \text{number of filters used in the convolution}$$



**卷积的四维矩阵的含义**

平时一直没有关注过，如果是一个两行三列的矩阵，平时我们会说2×3，但换算成维度第0维是2，第一维是3。

但如果是四维的输入，如下图`A_prev = np.random.randn(2, 5, 7, 4)`。第一维度2代表样本数m，第二维度5代表高度h，第三维度7代表宽度w，第四维度4代表channel数为4。之所以弄混的原因是，平时我们说矩形都说长×宽，但是图片是分高度height和宽度width，但宽度其实是长度。如果一个图片是2*3，则意味着高度h为2，宽度w为3。

```python
np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
```

![image-20210527225317370](../images/image-20210527225317370.png)

 Lucent Technologies was **spun off**. 独立出来

starting to **encroach** significantly into even other fields 侵占(某人的时间);侵犯(某人的权利) ；扰乱(某人的生活等);                                                                                                                                                

And then the fact that the company is not **obsessively compulsive** about IP as some other companies are makes it much easier to collaborate with universities and have arrangements by which a person can have a foot in industry and a foot in academia. 过分的；难以制止的； compulsory强行的

onerous [ˈoʊnərəs]费力的

#### 使用tf.data.Dataset.from_tensor_slices五步加载数据集

[使用tf2做mnist（kaggle）的代码](https://github.com/Rainweic/tensorflow2-mnist)

思路

Step0: 准备要加载的numpy数据
Step1: 使用 tf.data.Dataset.from_tensor_slices() 函数进行加载
Step2: 使用 shuffle() 打乱数据
Step3: 使用 map() 函数进行预处理
Step4: 使用 batch() 函数设置 batch size 值
Step5: 根据需要 使用 repeat() 设置是否循环迭代数据集

##### 代码

```python
import tensorflow as tf
from tensorflow import keras

def load_dataset():
	# Step0 准备数据集, 可以是自己动手丰衣足食, 也可以从 tf.keras.datasets 加载需要的数据集(获取到的是numpy数据) 
	# 这里以 mnist 为例
	(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
	
	# Step1 使用 tf.data.Dataset.from_tensor_slices 进行加载
	db_train = tf.data.Dataset.from_tensor_slices((x, y)）
	db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	
	# Step2 打乱数据
	db_train.shuffle(1000)
	db_test.shuffle(1000)
	
	# Step3 预处理 (预处理函数在下面)
	db_train.map(preprocess)
	db_test.map(preprocess)

	# Step4 设置 batch size 一次喂入64个数据
	db_train.batch(64)
	db_test.batch(64)

	# Step5 设置迭代次数(迭代2次) test数据集不需要emmm
	db_train.repeat(2)

	return db_train, db_test

def preprocess(labels, images):
	'''
	最简单的预处理函数:
		转numpy为Tensor、分类问题需要处理label为one_hot编码、处理训练数据
	'''
	# 把numpy数据转为Tensor
	labels = tf.cast(labels, dtype=tf.int32)
	# labels 转为one_hot编码
	labels = tf.one_hot(labels, depth=10)
	# 顺手归一化
	images = tf.cast(images, dtype=tf.float32) / 255
	return labels, images

```



Even if you end up not working computer vision yourself, I think you find a lot of the ideas from some of these exampkes, such as ResNet Inception network, many of these ideas are **cross-fertilizing** on **making their way into other disciplines**. 杂交

#### VGG16

![image-20210606005837417](../images/image-20210606005837417.png)

特点：doubling on every stack of conv-layers 越来越小，越来越长

### ResNet

![image-20210606010030539](../images/image-20210606010030539.png)

特点：第一个卷积层是正常的，但是会同时连接到第二个卷积的卷积后，ReLu函数前。

The **short cut** / **Residual Connection** / **Skip Connection** is actually added before the ReLu non-linearity.

![image-20210606010712360](../images/image-20210606010712360.png)

理论上来说，网络越深，训练误差会越少。但实际上，训练误差反而会增加。

But what happens with ResNet is that even as the number of layers gets deeper, you can have the performance of the training error going down.

通过short cut的方式，helps with vanishing and exploding gradient problems and allows you to train much deeper neural networks without really **appreciable** loss in performance, and will plateau at some point. 明显的

#### Inception

特点：

- 1x1卷积，`bottleneck`， 可以帮助节约参数，解决网络层数过多的问题，并且并不影响效果。在两个地方用到了，一个是不同卷积方式拼接后层数过多；一个是在卷积前使用减少参数（相当于承接上一个botttleneck传来的数据）

![image-20210606010911489](../images/image-20210606010911489.png)

- 可以一次性把所有的卷积和池化放进去，由模型自己选。（会带来层数过多的问题，但已经可以通过1x1卷积解决）需要保证卷积和池化均为'same'，下图的池化池化完仍然是192层，需要1x1卷积调整成32层。

![image-20210606011022658](../images/image-20210606011022658.png)

- 在中间部分使用和末尾部分相同的结构输出结果，如下图绿线所示，可以防止过拟合。And this appears to have a regularizing effecr on the inception network and helps prevent this network from overfitting. 
- 让特征更加好 Ensure features computed, even in heading units, even at intermediate layers, that are not too bad for predicting the output class of a Image. 
- 个人感觉有点像dropout，强行让参数学的更好。

![image-20210606011353001](../images/image-20210606011353001.png)

#### MobileNet

- 普通卷积

![image-20210606222703276](../images/image-20210606222703276.png)

- depthwise卷积

![image-20210606222638557](../images/image-20210606222638557.png)

![image-20210606223140261](../images/image-20210606223140261.png)

- 参数个数 2160>>432+240
- 按经验来说，输出的channel数一般为512，f为卷积核的大小，因此在实际中一般是1：10的节约

![image-20210606223502809](../images/image-20210606223502809.png)

![image-20210606223239801](../images/image-20210606223239801.png)

特点：

- depthwise-separable convolutions+pointwise convolution 每个卷积只有一层，去分别卷积输入图的所有层。此时输出层数一定和输出层数相同。如果要调整输出的通道层数，通过pointwise 用n个1x1卷积把三层的结果转换为n层。（这里吴老师的图片没讲清楚）

#### Depthwise Separable Convolution

Depthwise Separable Convolution是将一个完整的卷积运算分解为两步进行，即Depthwise Convolution与Pointwise Convolution。

##### Depthwise Convolution

不同于常规卷积操作，Depthwise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积。上面所提到的常规卷积每个卷积核是同时操作输入图片的每个通道。
同样是对于一张5×5像素、三通道彩色输入图片（shape为5×5×3），Depthwise Convolution首先经过第一次卷积运算，不同于上面的常规卷积，DW完全是在二维平面内进行。卷积核的数量与上一层的通道数相同（通道和卷积核一一对应）。所以一个三通道的图像经过运算后生成了3个Feature map(如果有same padding则尺寸与输入层相同为5×5)，如下图所示。

![img](../images/企业微信截图_16229992964716.png)

Depthwise Convolution完成后的Feature map数量与输入层的通道数相同，无法扩展Feature map。而且这种运算对输入层的每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的feature信息。因此需要Pointwise Convolution来将这些Feature map进行组合生成新的Feature map。

##### Pointwise Convolution

Pointwise Convolution的运算与常规卷积运算非常相似，它的卷积核的尺寸为 1×1×M，M为上一层的通道数。所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature map。有几个卷积核就有几个输出Feature map。如下图所示。

![img](../images/20180812163629103)

- This turns out to perform well while being much less computationally expensive than earlier algorithms that used a normal convolutional operation.

![image-20210606212741665](../images/image-20210606212741665.png)

MobileNet v2的结构也叫bottleneck，相对于v1，多了一步用1x1卷积expansion和residual的步骤

在下图，它先用1x1卷积Expansion增维（Inception是用1x1降维），然后对每一层卷积，最后再用Pointwise降维。

![image-20210606213151004](../images/image-20210606213151004.png)

I'm using this **blue glow** here to denote maybe high resolution image. 蓝光

#### EfficientNet

help you to choose a good trade-off between r, d, and w.

![image-20210606224351013](../images/image-20210606224351013.png)

### 数据增强

- color shifting
- PCA color Augmentation

![image-20210607001807908](../images/image-20210607001807908.png)

If your image is mainly purple, if it mainly has red and blue **tints**, and very little green, then PCA Color Augmentation will add and subtract a lot to red and blue, but relatively little to  the greens, so kind of keeps the overall color of the **tint** the same. 浅色调

So, if you look across **a broad spectrum of** machine learning problems, you see on average that when you have a lot of data you tend to find people getting away with using simpler algorithms as well as less hand-engineering.

#### 如何赢下竞赛

- Ensambling

- Multi-crop (这个我没见过) 比如说10-crop

  就是说，在测试集的图片上，先对图片mirroring，再分别对两张图片随机裁剪左上角，右上角等大部分区域，进行预测。对预测结果取平均

  10-crop就是对正反两张图的 正、左上右上等四个方向共10图crop

![image-20210607003851940](../images/image-20210607003851940.png)

For multi-crop I guess at least you keep just one network around, so it doesn't **suck up** as much memory, but it still slows down your run time quite a bit 讨好

you can use an open source implementation if possible, because the open source implementation might have figured out all the **finicky** details like the learning rate, case scheduler, and other hyper parameters. 难讨好的，难取悦的, 需细心的，需注意细节的



## NLP

referring to the **genre** of music you want to generate or maybe the first few notes of the piece of music you want [ˈʒɑːn] 音乐的类型

Jazz **improvisation** with LSTM 即兴创作

