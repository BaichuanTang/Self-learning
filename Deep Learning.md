# Deep Learning Specialization 

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



![image-20210513215445312](images/image-20210513215445312.png)

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
- 如果项目时间很短，来了新数据：数据量很少：①用新评价指标和新的验证测试集再次训练；②数据增强、数据合成data augmentation/data synthesis
- 如果竞争对手再False Negative Rate表现更好：即使调整指标（根据NER经验我理解：什么时候停取决于FNR最低）

![image-20210513222114025](images/image-20210513222114025.png)

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

![image-20210517010153779](images/image-20210517010153779.png)

也可以对标注样本标错的情况提示出来

![image-20210517011846970](images/image-20210517011846970.png)

But this quick counting procedure, can really help you make much better **prioritization decisions**, and understand how promising different approaches are to work on. 

可以选定优先方向，看到哪个方向会有前景。（这点的确对业务来说很重要！）

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

今天继续学习