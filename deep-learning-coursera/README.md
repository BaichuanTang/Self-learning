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

#### ResNet

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

### 图片增强与迁移学习

具体可见W2A2中用MobileNet实现增强与迁移学习的例子，感觉这是实战中最常用的手段了。

#### 用AutoTune设定每次读取文件的个数

You may have encountered `dataset.prefetch` in a previous TensorFlow assignment, as an important extra step in data preprocessing. 

Using `prefetch()` prevents a memory bottleneck that can occur when reading from disk. It sets aside some data and keeps it ready for when it's needed, by creating a source dataset from your input data, applying a transformation to preprocess it, then iterating over the dataset one element at a time. Because the iteration is streaming, the data doesn't need to fit into memory.

You can set the number of elements to prefetch manually, or you can use `tf.data.experimental.AUTOTUNE` to choose the parameters automatically. Autotune prompts `tf.data` to tune that value dynamically at runtime, by tracking the time spent in each operation and feeding those times into an optimization algorithm. The optimization algorithm tries to find the best allocation of its CPU budget across all tunable operations. 

To increase diversity in the training set and help your model learn the data better, it's standard practice to augment the images by transforming them, 

i.e., randomly flipping and rotating them. Keras' Sequential API offers a straightforward method for these kinds of data augmentations, with built-in, customizable preprocessing layers. These layers are saved with the rest of your model and can be re-used later.  Ahh, so convenient! 

As always, you're invited to read the official docs, which you can find for data augmentation [here](https://www.tensorflow.org/tutorials/../images/data_augmentation).

#### 多轮训练

If you do not set initial_epoch and you train for 2 epochs, then rerun the fit_generator with epochs=4 it will train for 4 **more** epochs starting from the state preserved at the end of the second epoch (provided you use the built in optimizers). Again the history object state is NOT preserved from the initial training and only contains the data for the last 4 epochs. I noticed this because I plot the validation loss versus epochs.

说白了就是给tensorboard画图用的，

如果继续训练时，设置nitial_epoch为0，epochs为500，tensorboard会在同一张图上重叠显示两段结果；
如果initial_epoch设置为500，epochs应该设置为1000，tensorboard会合并成0-1000代的整段图。

因此只是用于显示，实际上再训练的时候还会接着训练

![image-20210612184252252](../images/image-20210612184252252.png)

## Object Detection

### Object Localization

![image-20210612202905073](../images/image-20210612202905073.png)

Localization：知道这是一个汽车的图像（已经识别出来），在图中把汽车定位出来

Detection：里面有很多汽车，我都要找出来

So, in particular, you can have the neural network output four more numbers, and I'm going to call them bx, by, bh, and bw. And these four numbers parameterized the bounding box of the detected object

![image-20210612210859171](../images/image-20210612210859171.png)

y的输出形式，第一个 $p_c$表示是否有我们想分类的1、2、3三种类别。之后的四个角是他的四个坐标。最后$c_1$ $c_2$ $c_3$ 是最终的分类类别，这三个数只能取其一为1。

**很重要**：对于损失函数来说，$c_1$ $c_2$ $c_3$ 应当使用log likelihood+Softmax，因为它们是多分类。b1b2b3b4应当用L2损失，因为它们相当于回归。$p_c$是二分类，应当用Logistic Regression Loss。



![image-20210612211537174](../images/image-20210612211537174.png)

#### 业务启示

它的损失函数就是L2损失，但是如果$p_c$为0，也就是图中没有物体要检测，那么损失只看$\hat{y_1}$和$y_1$就行了，也就是有没有分对图中是否有物体要检测这一点。

这一点在业务中我曾经问过马凯：结构话模型需不需要加负样本进去，例如：tff模板，有些图片是第二页，只含有少量的文字，没有要提取的字段。马凯说不用加，因为有前置分类模型让它进入不了tff这一块。但我现在觉得，如果有负样本，并不会让模型学到任何东西，因为没有损失发生，因此这个样本就没有反向传播这一流程。就像这里一样。

![image-20210612212610697](../images/image-20210612212610697.png)

### Landmark Detection

landmark 地标

![image-20210612215205676](../images/image-20210612215205676.png)

**for the sake of argument**, let's say 64 points or 64 landmarks on the face. 为了论证

define the **jaw line**by selecting a number of landmarks and generating a label training sets that contains all of these landmarks, you can then have the neural network to tell you where are all the key positions or the key landmarks on a face 下颌线

用AR augmented reality时，ins抖音，如何探测你的脸并把妆画上去

Being able to detect these landmarks on the face, there's also a key building block for the computer graphics effects that warp the face or drawing various special effects like putting a crown or a hat on the person. 

where someone will have had to go through and laboriously annotate all of these landmarks.

**laboriously [ləˈbɔriəsli] 辛苦地**

One last example, if you are interested in people pose detection, you could also define a few key positions like the midpoint of the chest, the left shoulder, left **elbow**, the **wrist**, and so on, and just have a neural network to annotate key positions in the person's pose as well and by having a neural network output,

elbow 肘部 wrist [rɪst] 手腕

### Sliding Window Detection

![image-20210612222200257](../images/image-20210612222200257.png)

方法：用一个小块去扫描全局，捕获汽车。逐渐换更大的window再去扫描

input into this cofinite a small rectangular region. 

 if you use a very coarse stride, then that will reduce the number of windows you need to pass through the convnet, but that coarser **granularity** may hurt performance. Whereas high computational cost.

在以前，都是人工构造的特征，因此sliding window的方式很好用，因为计算快，都是linear function

now running a single classification task is much more expensive and sliding windows this way **is infeasibily slow**.

###  A convolutional implementation of sliding windows object detection

用卷积可以大量减少参数，

之前的和卷积形式的实现效果是相同的，如下：

![image-20210612224524390](../images/image-20210612224524390.png)

![image-20210612225311640](../images/image-20210612225311640.png)

可以快速知道哪一块里有汽车，而不是一步一步重复计算，一个神经网络就搞定。

缺点：不够准确

![image-20210612225437356](../images/image-20210612225437356.png)

### YOLO

划分成3x3=9个区域（实际中会划分成19*19=361个小块），以物体的中心点在哪个区域来判断物体属于哪个区域，采取最开始的方法定义y。因此输出为3x3x8。

![image-20210612234006467](../images/image-20210612234006467.png)

我觉得训练还是很好理解的，标注好左上角坐标和height,width（如果h,w出现横跨小块就会大于1），划分好小块就可以训练，这就好解释了。即使一个物体横跨多个区域，也只分进一个区域。

这里可能会有一个误解：并不是分别对这9个区域做9次卷积，而是只做一次卷积得到3x3x8的输出，所有卷积都共享参数。看起来并没有告诉卷积核：哪些初始区域对应了最终3x3=9个的输出，但是模型会去学对应关系。

预测时，不需要考虑一个物体横跨多个区域，也只分进一个区域这一特点，因为已经不是训练了。

![image-20210612234813920](../images/image-20210612234813920.png)

##### 个人评价

- YOLO吸取了Object Localization的优点，保留了一样的输出方式。

- 没有用传统的landmark的方法，当时课程项目还是用haar cascade特征做的。
- 采用了conv sliding window类似的策略，可以共享参数，一次卷积。但conv sliding window对不同大小的输入图片会得到不同的输出，且无法处理物体横跨的情况，只能定死地看每一个固定小块内是否有汽车，且没法适应物体的大小程度。
- 但是YOLO可以处理跨小块的物体，且用回归的方式返回坐标。并且每个物体只会被分进一个小块内，所以物体都一定能被检测到，且横跨的问题也解决了。
- 但有人肯定会问：我为什么要3x3的小块呢？直接对汽车的四个坐标做回归不就完事了？根本原因是因为你不知道图中有几个汽车，因此划分成小块后，只需要对每个小块做一次分类就行了。
- 肯定有人又要问：这不就是削弱版的sliding window吗？首先，sliding window只负责判断该小块内有没有汽车，sliding window当然可以用图片单分类的方式做8维输出回归，但重复学了很多仅仅平移就学到的知识，没有必要。其次，对于跨块的样本，YOLO中直接用大于1的w和h就解决了这个问题。sliding window当然可以学YOLO一样很多步stride走完整张图片，但是相对于YOLO，它其实做了很多没必要的步骤，就好像打比赛的时候用Ensabmble模型，但是真正迎来效果大提升的永远是创新的方法。
- 说白了，就是sliding window的简版，在原来只有分类的版本上加了回归，最亮眼的是设计的xywh坐标的回归，并保证不会重复看，减少了大量的计算负担。

### 注意点：

there are some more complicated parameterizations involving sigmoid functions to make sure this is between 0 and 1. 

左上角的坐标的输出会加sigmoid激活函数，w和h会用exp激活函数保证是非负的

In the meantime, if you want, you can take a look at YOLO paper reference at the bottom of these past couple slides I use. Although, just one warning, if you take a look at these papers which is the YOLO paper is one of the harder papers to read. I remember, when I was reading this paper for the first time, I had a really hard time figuring out what was going on. And I wound up asking a couple of my friends, very good researchers to help me figure it out, and even they had a hard time understanding some of the details of the paper. So, if you look at the paper, it's okay if you have a hard time figuring it out. 

I wish it was more uncommon, but it's not that uncommon, sadly, for even senior researchers, that review research papers and have a hard time figuring out the details. And have to look at open source code, or contact the authors, or something else to figure out the details of these outcomes. But don't let me stop you from taking a look at the paper yourself though if you wish, but this is one of the harder ones. 

### Intersection Over Union

IoU=Intersection/Union

Correct if IoU>0.5

If you want to be more **stringent**, you can judge an answer as correct, only if the IoU is greater than equal to 0.6 or some other number. [ˈstrɪndʒənt] 严格的

### Non-max Supression

解决对一个物体重复画框多次的问题

non-max means that you're going to output your maximal probabilities classifications but suppress the **close-by** ones that are non-maximal. 离xx很近

![image-20210613011935571](../images/image-20210613011935571.png)

如果多分类就分别做n次

#### 业务启示

当时看小韩老师的输出，有一个`4.`还有一个`4. xxxxxx` 于是我很好奇，问小韩老师：为什么会这样呢？这不应该啊，有两个框输出了相同的一部分，一个框的内容在另一个框以内？小韩老师说：确实有这种情况，并让我算一下IoU卡阈值。我当时觉得太麻烦就没算。

现在想来，应该是小韩老师做的non-max supression里，对IoU卡的阈值为0.5，但这个例子可能只有0.4，因此没卡掉。当时考虑到可能是有字重叠的情况（例如印章），要加强识别的能力，因此就像目标检测一样这样做。

然而，在ocr里还有一种情况，如果正常文本的话，不会有重叠的，因此可以直接卡IoU为一个非常低的值。

### Anchor Boxes

解决一个在一个小块内探测多个目标的问题，用到了anchor box的思想：

- 预先定义好两种形状的anchor box
- y的输出维度由3x3x8变成3x3x16，上下两个区域分别对应两种anchor box的输出
- 预测时：探测在这两种anchor box形状下的输出
- 训练时：对每一个物体，定好中心，用两个anchor box分别去套，哪一个IoU大，就归进哪个anchor box

![image-20210613133703472](../images/image-20210613133703472.png)

![image-20210613134556782](../images/image-20210613134556782.png)

#### 缺点

1. 如果一个小块中由三个物体，但只用了两个anchor box，那么就没有办法检测第三个。但实际中我们划分成19x19的小块，已经很少会出现一个小块三个物体的情况了
2. 如果两个物体在同一个grid cell里，且拥有一样的anchor box（例如两个人），也就是有一样的形状，那么也没法处理

better results that anchor boxes gives you is it allows your learning algorithm to specialize better. 

this allows your learning algorithm to specialize so that some of the outputs can specialize in detecting white, fat objects like cars, and some of the output units can specialize in detecting tall, skinny objects like pedestrians. 

在之后的YOLO论文中，用kmeans分类出两种你所需要的anchor box形状

And then to use that to select a set of anchor boxes that this most **stereotypically representative** of the maybe multiple 典型代表性          

In the next video, let's take everything we've seen and **tie it back together into** the YOLO algorithm.   向回串联

### YOLO算法合集

#### 训练预测

![image-20210613140406404](../images/image-20210613140406404.png)

![image-20210613140820571](../images/image-20210613140820571.png)

#### non-max supression

- 先处理anchor box的信息，对每一个grid cell输出的两个anchor box，过滤到低概率的anchor box （注意，在这一步中，可能有的grid cell的左上角xy不在该grid cell内）
- 然后就不管anchor box和grid cell了，直接对每一个分出来的类别进行non-max supression，不断找到一个类别中最高概率的输出，去掉所有IoU>0.5的框，再往下找次高的输出，再去掉。。。

![image-20210613141040671](../images/image-20210613141040671.png)

### Region Proposal R-CNN

Regions with CNNs, 两步走

1. propose regions
2. classifiers

#### segmentation algorithm

 you find maybe 2000 blobs and place bounding boxes around about 2000 blobs and run your classifier on just those 2000 blobs, and this can be a much smaller number of positions on which to run your convnet classifier, then if you have to run it at every single position throughout the image

![image-20210613143317330](../images/image-20210613143317330.png)

- 缺点是R-CNN很慢
- 虽然是用CNN再去探测一遍，但是R-CNN仍然会返回坐标，就像之前的输出一样

#### 演变

![image-20210613144226282](../images/image-20210613144226282.png)

2013年的R-CNN是用神经网络一个一个移动，再去卷积的；

2015年的Fast R-CNN是用sliding window的方式，用卷积代替手工移动，有共享参数的优势，因此更快。

2016年的Faster R-CNN是何凯明大神提出的，用卷积神经网络去propose region，解决了原有的segmentation algorithm计算慢的问题（但还是比不上YOLO快）

uses a convolutional neural network instead of one of the more traditional segmentation algorithms to propose a blob on those regions, and that wound up running quite a bit faster than the fast R-CNN algorithm. 

#### 业务启示

我们当时用的ocr分定位识别两步，实际上用的mrcnn作为定位模型，且没有要求定位模型提供输出，只要求它提供bounding box，再用识别模型去识别它，因为带transformer的识别模型效果一定比直接识别文字的效果来得更好，但是牺牲了很多效率，增加了很多computational cost。

#### 个人评价

R-CNN是人类的正常思维方式，yolo简直是神想出来的。你去思考object detection算法的时候，你肯定希望分这两步走：1.先判断这里是不是会有一个东西 2.这是个什么东西。但经常人看到一样东西就会直接反应出：哦，这是一个杯子，再看到它在桌子上（定位它的坐标）。而不是先观察到桌子上有一个东西，再看出它是一个杯子

R-CNN慢的原因是：它仍然需要两步走，先propose region看自由分割出的区域内是否有物体，再对region识别。Yolo跳过了propose region的一步，直接一次卷积后输出所有结果。

But that's my personal opinion and not necessary the opinion of the whole computer vision research committee. So feel free to **take that with a grain of salt**, 

take sth. with a grain of salt 这个习语的字面意思是“和一撮盐一起吃下去”，为什么要与盐一起吃呢？

据说这个习语要追溯到罗马时代，罗马将军庞培曾发现一种解毒剂，必须和着一小把盐才服得下去。解毒剂难咽，加了盐也许好咽些，于是这句习语用于描述对一些不靠谱的，值得怀疑的东西，得“和着盐”才能勉强接受。

现在，对某件事情或某人说的话有所保留，将信将疑，持怀疑态度，就可以说 take it with a grain of salt.

### Semantic Segmentation with U-Net

![image-20210613151214438](../images/image-20210613151214438.png)

segment out in the image exactly which pixels correspond to certain parts of the patient's **anatomy**. 解剖出的人体

![image-20210613153207150](../images/image-20210613153207150.png)

### Transpose Convolution

![image-20210613154044164](../images/image-20210613154044164.png)

![image-20210613154539652](../images/image-20210613154539652.png)



如果重叠，就对两个区域内的值相加

![image-20210613154748781](../images/image-20210613154748781.png)

![image-20210613154849153](../images/image-20210613154849153.png)

![image-20210613154923137](../images/image-20210613154923137.png)

#### 网络结构

![image-20210613161340136](../images/image-20210613161340136.png)

skip connection既可以学到 lower resolution, but high level spatial, high level contextual information（也就是原本从U-Net中一步一步传过来的信息），也可以学到high resolution, the low level, more detailed texture like information

What the skip connection does is it allows the neural network to take this very high resolution, low level feature information where it could capture for every pixel position, how much fairy stuff is there in this pixel? 

![image-20210613164058997](../images/image-20210613164058997.png)

往下走时，channel数不断增加，但是h,w是变小的，每一步池化都对应了红色箭头。

注意点：白色箭头是skip connection，还记得残差连接吗？就是把原有的输入**加**到现有的channel后面，组成一个更深的网络。但这里是**拼接**（图中深蓝和浅蓝），再用卷积调整channel数，一个类似Inception的方法。

最终输出是$n_{classes}$层，再通过argmax函数画出我们实际看到的segmentation map

一些需要理解的细节：是怎么argmax生成segmentation map的，为什么不直接用残差而是要像Inception一样拼接？残差是用Add()函数，拼接用什么函数？

向上走的路径：upsampling，decoder，expanding path

向下走的路径：downsampling, encoder, contracting path

### 人脸识别

#### Siamese Network

![image-20210707233338918](../images/image-20210707233338918.png)

#### Triplet Loss

用Anchor Image和Positive、Negative Image对比

定义优化目标如下，但是会遇到一个问题，如果我让他们所有人的向量完全重合，那么也能满足目标。此时需要让每张图片的encoding都互相不同。第二点，需要≤0-α，其中α是一个非常小的数，同时和SVM里的margin一样，这里的α移到等式左边也是margin，即两个向量最少的间隔距离。

![image-20210707235115046](../images/image-20210707235115046.png)

最后还用Loss=max(dis,0)的原因是：类似于铰链损失函数，只要距离<0，就当作损失为0（因为损失不能为负数吧）

最后，如何选择样本？如果随机选择的话，损失函数的这一条件很容易就满足了。因此要选择本身就相似的样本，且可以节约计算成本。

#### north of sth

 used to say that an [amount](https://dictionary.cambridge.org/zhs/词典/英语/amount) is more than the [stated](https://dictionary.cambridge.org/zhs/词典/英语/state) [amount](https://dictionary.cambridge.org/zhs/词典/英语/amount):*The [share](https://dictionary.cambridge.org/zhs/词典/英语/share) [price](https://dictionary.cambridge.org/zhs/词典/英语/price) is [expected](https://dictionary.cambridge.org/zhs/词典/英语/expected) to [rise](https://dictionary.cambridge.org/zhs/词典/英语/rise) north of $20.* “SMART 词汇”：相关单词和短语

Datasets **north** of a million images is not uncommon, some companies are using **north** of 10 million images and some companies have **north** of 100 million images with which to try to train these systems.

预测时，训练一个logistic unit来判断他俩是否为同一个人（1 或者 0），也可以用chi square similarity

上下两个网络的参数均是一样的

![image-20210708002010794](../images/image-20210708002010794.png)

![image-20210708002603919](../images/image-20210708002603919.png)

还有一个trick，每一个人的输出向量其实是可以pre-compute的，预测时只需要对新图像进行预测。

### 神经风格迁移

Content+Style=Generated Imaghe

待补充，没怎么看懂























































## NLP

### RNN

referring to the **genre** of music you want to generate or maybe the first few notes of the piece of music you want [ˈʒɑːn] 音乐的类型

Jazz **improvisation** with LSTM 即兴创作

![image-20210709215658153](../images/image-20210709215658153.png)

为什么不能用普通的网络来预测？

 And to **kick off** the whole thing, we'll also have some either made-up activation at time zero, 开始



![image-20210709235604376](../images/image-20210709235604376.png)

Let's keep **fleshing out** this graph. 充实

![image-20210710101743643](../images/image-20210710101743643.png)

the input x could be maybe just an integer, telling it what **genre** of music you want or what is the first note of the music you want, and if you don't want to input anything, x could be a null input, could always be the vector zeroes as well. 

例如机器翻译，也是many to many，但是输入输出的长度不一样

![image-20210710102200118](../images/image-20210710102200118.png)

![image-20210710102259396](../images/image-20210710102259396.png)

So, now you know most of the building blocks, the building are pretty much all of these neural networks except that there are some **subtleties** with sequence generation, which is what we'll discuss in the next video. 微妙之处

#### RNN损失函数

![image-20210710120726843](../images/image-20210710120726843.png)

#### GRU

如果遇到exploding gradients，其实用gradient clipping直接截断就行。但是vanishing gradient没有好办法处理，这就有了GRU。Gated recurrent Unit

![image-20210710172630027](../images/image-20210710172630027.png)

因为大多数情况下$Γ_u$都是非常接近0，因此$C^t$可以一直保存下去，因此不会有梯度消失的问题。

![image-20210710174613122](../images/image-20210710174613122.png)

 the GRU is one of the most commonly used versions that researchers have converged to and found as robust and useful for many different problems.

![image-20210710174828421](../images/image-20210710174828421.png)

But GRUs and LSTMs are two specific **instantiations** of this set of ideas that are most commonly used. 实例

![image-20210710175005633](../images/image-20210710175005633.png)

#### LSTM

u=update; f=forget; o=output

![image-20210710181523561](../images/image-20210710181523561.png)

![image-20210710181903103](../images/image-20210710181903103.png)

![image-20210710191611762](../images/image-20210710191611762.png)

*Acyclic* Graph 无环图

![image-20210710214536395](../images/image-20210710214536395.png)

So, for example, if you're building a speech recognition system, then the BRNN will let you take into account the entire speech utterance but if you use this straightforward implementation, you need to wait for the person to stop talking to get the entire **utterance** before you can actually process it and make a speech recognition prediction. 话

![image-20210710215057006](../images/image-20210710215057006.png)

### Word Embedding

you'll see how to debias word embeddings. That's to reduce undesirable gender or **ethnicity** or other types of bias that learning algorithms can sometimes pick up 种族

Well, man and woman doesn't **connotes** much about age.  意味，暗示

![image-20210712124811265](../images/image-20210712124811265.png)

300维的向量，每一维其实是一种特征

What if you see Robert Lin is a **durian cultivator**? 榴莲栽培机

huge amounts of unlabeled text that you can suck down essentially for free off the Internet to figure out that orange, apple, and durian are fruits. 吸进

But the terms encoding and embedding are used somewhat **interchangeably**. 可互换地

#### Analogy Reasoning

Let's go on to the next video, where you'll see how word embeddings can help with **reasoning about analogies**.类比推理

![image-20210712132003973](../images/image-20210712132003973.png)

![image-20210712133017095](../images/image-20210712133017095.png)

So, that's why the embedding matrix E times this one-hot vector here **winds up** selecting out this 300-dimensional column corresponding to the word Orange. /waɪnd/ 结束

![image-20210714002057904](../images/image-20210714002057904.png)

别忘了推导softmax

**缺点**：Softmax步骤非常computationally expensive

**注意点**：需要用较低频的词才会有效果，如果是高频的词，如is, are, I，那么模型的梯度更新将会学习这些没用的东西，而且学这些词出来的效果也肯定不好。

word2vec有两种，一种是skip gram，也就是用中间词预测两边的词。

![img](../images/v2-514a2e9c173fbb3bc5ce7eb67197825d_1440w.jpg)


**这样我们的神经网络模型在遍历整个语料库的时候（比如维基百科文本）将看到寻多个(center: orange, context: juice)样本（相对于(orange, Ak47)样本来说），同理也会看到许多个(apple, juice)样本。**
**对应第2小点的内容，这样使得模型不管看到”orange”还是看到”apple”都要预测出”juice”，就”逼”着它让它学习让它认为这两个单词是高度相似的，即我们最终得到的”orange”和”apple”词向量是高度相似的。**
而CBOW则与之相反，用的是上下文词来预测中心词。但是与skip-gram有一点差别的是，它获得的训练样本并非是直观以为的(context: juice, center: orange)。而是将窗口内的所有上下文词放在一起来预测中心词，即(context: [glass, of, juice], center: orange),即：

![img](../images/v2-7a9d78d1dadc2314f23ce63986fea750_1440w.jpg)





#### Negative Sampling

解决softmax需要从所有词中计算昂贵的问题

把softmax问题转化为n个逻辑回归问题，用逻辑回归区分开这些随机挑选的单词。

#### GloVe

global vectors for word representations

![image-20210714230854589](../images/image-20210714230854589.png)

$X_{ij}$定义为 #times i appears in context of j，因此$X_{ij}$为0时，就变成log$X_{ij}$就没有意义，因此用$f(X_{ij})$作为权重加在前面。还有一个原因，对于有些频繁出现的词，例如this is of a 来说，他们相对于不常出现的词（例如durian）理应被赋予更大的权重

the weighting factor can be a function that gives a meaningful amount of computation, even to the less frequent words like durion, and gives more weight but not an **unduly** large amount of weight to words like, this, is, of, a, which just appear a lot in language. 不适当地

And so, there are various heuristics for choosing this weighting function F that need or gives these words too much weight nor gives the infrequent words too little weight. 

![image-20210714233544274](../images/image-20210714233544274.png)

 当真正构建出词向量时，不能保证每一个维度都是有意义的。![image-20210714233351224](../images/image-20210714233351224.png)



#### Debiasing

对于bias的方向也可以用奇异值分解SVC来做，类似PCA的方法

![image-20210717022931122](../images/image-20210717022931122.png)

And really, in the more general case, you might want words like doctor or babysitter to be **ethnicity neutral** or **sexual orientation neutral**, and so on, but we'll just use gender as the illustrating example here.

So for words like doctor and babysitter, let's just project them onto this axis to **eliminate their component in the bias direction**. 

第三步：Equalize：确保每组词在遇到第三个带性别的词的时候不会有偏差，例如，grandpa和grandma遇到babysister是相同的 

第二步：Neutralize，如何确定哪些词需要被中性，grandpa和grandma肯定不需要放进第二部里确定bias的方向。因此这一步用一个逻辑回归筛选出不需要进入Neutralize步骤的词

#### Beam Search

为什么不用greedy search？因为对每一个词挑下一个最优的词而言，真实来说不如直接挑全局最优的词好。同时，当句子长时，也不可能遍历所有可能性的结果。所以用approximate search algorithm

But the set of all English sentences of a certain length is too large to **exhaustively** enumerate. So, we have to **resort to** a search algorithm.  耗尽一切地；  使用

Beam Search的参数B代表beam width，同时考虑三个结果

![image-20210717174542913](../images/image-20210717174542913.png)

- 第一轮：挑出三个词
- 第二轮，对这三个词，分别考虑10000个词典中的所有可能，因此是30000种可能，从中跳出来概率最高的三条路径。（注意：此时第一轮中可能有词已经被扔掉）
- 接下来每一轮都是挑概率最高的三条路径，也就是30000中选三个结果

Just want to notice that because of beam width is equal to three, every step you **instantiate** three copies of the network to evaluate these partial sentence fragments and the output. 

如果beam width退化成1，就变成了greedy search

But when B gets very large, there is often diminishing returns.

##### Length Normalization

对于概率值，如果连乘，就会遭遇result in **numerical underflow**. Meaning that it's too small for the floating part representation in your computer to store accurately. 

因此采用取log的方法

So by taking logs, you end up with a more **numerically stable algorithm** that is **less prone to** rounding errors, numerical rounding errors, or to really numerical underflow.

So in most implementations, you keep track of **the sum of logs of the probabilities** rather than the product of probabilities. 

而且，模型还有喜欢造短句的问题

And so this objective function has **an undesirable effect**, that maybe it **unnaturally tends to prefer** very short translations. 

对结果除以词的数量的0.7次方，这是一个更soft的方式，是在full nomalizatio和no normalization之间

There isn't a great theoretical justification for it, but people have found this works well. People have found that it works well in practice, so many groups will do this. And you can try out different values of alpha and see which one gives you the best result.

最后，和DFS和BFS相比，不一定是全局最好的结果，但是快，效率高，效果也还可以。

beam search is an **approximate search** **algorithm**, also called a **heuristic search algorithm**. 

##### 业务启示

当时小韩老师的ocr置信度分数也做了normalization，避免ocr框里出现长词，NLP里的分数也会有这样的问题，

##### Error Analysis

![image-20210717182744962](../images/image-20210717182744962.png)

如果本来$P(y^*|x)$ >$P(\hat{y}|x)$ ，说明Beam Search挑的时候有问题，因为Beam Search挑的时候每一轮挑下一个y都是挑能使$P(y|x)$最大的y，而此时检查发现不符合这个情况，因此需要增加Beam Width来使Beam Search挑出这个结果。如果$P(y^*|x)$ <$P(\hat{y}|x)$ ，说明Beam Search的结果是对的，说明RNN没有给出正确的预测，从而RNN需要重训。

#### Bleu Score

为了解决同时有多个好结果（人类会有多种翻译方式），怎么衡量的问题。

bilingual evaluation understudy

![image-20210718000911087](../images/image-20210718000911087.png)

Modified precision：分子是reference中一句话里出现过某单词（此处是unigram）最多的次数，分母是MT（Machine Translation）中某单词出现的次数。

![](../images/image-20210718001442969.png)

$Count_{clip}$是在reference中该bigram在每个句子里出现过最多的次数。

![image-20210718001756014](../images/image-20210718001756014.png)

个人觉得的缺点：没考虑顺序

##### Combined Bleu Score

exponentiation is strictly **monotonically** **increasing** operation 单调递增x

还要外加BP（**brevity penalty**）简洁性惩罚，因为一个句子很短的话，总是能达到更高的precision，而我们并不希望得到的句子很短，![ymrbYcnyEemGTBIZaklqgg_31d054bb3a5b00444b14a5c86a7e6fd7_Screen-Shot-2019-08-28-at-5.19.56-PM](../images/ymrbYcnyEemGTBIZaklqgg_31d054bb3a5b00444b14a5c86a7e6fd7_Screen-Shot-2019-08-28-at-5.19.56-PM-1626541692996.png)

所以最终的式子是 $BP*exp(1/4\sum{p_n})$

### Attention

为了解决长句子bleu score下降的问题，因为模型记不住这些东西

![image-20210718014522168](../images/image-20210718014522168.png)

This is really a very influential, I think very **seminal** paper in the deep learning literature. 影响深远的

![](../images/image-20210718015635278.png)



![image-20210718021246823](../images/image-20210718021246823.png)

使用softmax来保证对每个t而言，$a^{<t,t^`>}$   加起来为1。

![image-20210718022400392](../images/image-20210718022400392.png)

![image-20210718022746872](../images/image-20210718022746872.png)

#### CTC Cost

Connection Temporal Classification

![image-20210718025233864](../images/image-20210718025233864.png)

### Transformer

#### Self-Attention

![image-20210718171008212](../images/image-20210718171008212.png)

![image-20210718171437191](../images/image-20210718171437191.png)

![image-20210718171506440](../images/image-20210718171506440.png)

![image-20210718171810438](../images/image-20210718171810438.png)

#### Multi-head Attention

所有的三种W变成$W_1$，这样可以学到第二种模式，例如：第一种是what，第二种是when

![image-20210718173352207](../images/image-20210718173352207.png)

h表示# heads，每一个head中都会有一个词被highlight

![image-20210718181443809](../images/image-20210718181443809.png)

which lets you ask multiple questions for every single word and learn a much richer, much better representation for every word.























# References

## CV

------

### **Week 1:**

- [The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model) (TensorFlow Documentation)
- [The Functional API](https://www.tensorflow.org/guide/keras/functional) (TensorFlow Documentation)

### **Week 2:**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren & Sun, 2015)
- [deep-learning-models/resnet50.py/](https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py) (GitHub: fchollet)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, & Adam, 2017)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (Sandler, Howard, Zhu, Zhmoginov &Chen, 2018)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)

### **Week 3:**

- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (Redmon, Divvala, Girshick & Farhadi, 2015)
- [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (Redmon & Farhadi, 2016)
- [YAD2K](https://github.com/allanzelener/YAD2K) (GitHub: allanzelener)
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Fully Convolutional Architectures for Multi-Class Segmentation in Chest Radiographs](https://arxiv.org/abs/1701.08816) (Novikov, Lenis, Major, Hladůvka, Wimmer & Bühler, 2017)
- [Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks](https://arxiv.org/abs/1705.03820) (Dong, Yang, Liu, Mo & Guo, 2017)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger, Fischer & Brox, 2015)

### **Week 4:**

- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf) (Schroff, Kalenichenko & Philbin, 2015)
- [DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) (Taigman, Yang, Ranzato & Wolf)
- [facenet](https://github.com/davidsandberg/facenet) (GitHub: davidsandberg)
- [How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) (Jason Brownlee, 2019)
- [keras-facenet/notebook/tf_to_keras.ipynb](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb) (GitHub: nyoki-mtl)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys, Ecker & Bethge, 2015)
- [Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
- [TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
- [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) (Simonyan & Zisserman, 2015)
- [Pretrained models](https://www.vlfeat.org/matconvnet/pretrained/) (MatConvNet)

## NLP

------

### **Week 1:**

- [Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy ](https://gist.github.com/karpathy/d4dee566867f8291f086)(GitHub: karpathy)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Andrej Karpathy blog, 2015)
- [deepjazz](https://github.com/jisungk/deepjazz) (GitHub: jisungk)
- [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf) (Gillick, Tang & Keller, 2010)
- [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07 Proceedings/SMC07 Paper 55.pdf) (Keller & Morrison, 2007)
- [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf) (Pachet, 1999)

### **Week 2:**

- [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) (Bolukbasi, Chang, Zou, Saligrama & Kalai, 2016)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) (Pennington, Socher & Manning, 2014)
- [Woebot](https://woebothealth.com/).

### **Week 4:**

- [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing?) (by [DeepLearning.AI](https://www.deeplearning.ai/))
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, 2017)
