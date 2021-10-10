# AI for Medicine

[toc]



## AI for Medical Diagnosis

medical diagnostic task 医疗诊断案例

edema  [ɪˈdiːmə] 水肿

fluid 积液

radiology 放射医疗

pneumonia [njuːˈməʊniə] 肺炎

dermatology [ˌdɜːməˈtɒlədʒi] 皮肤病学

dermatologist 皮肤病学家

determine whether a **mole** is a skin cancer or not 色素痣

ophthalmology [ˌɒfθælˈmɒlədʒi] 眼科学

retinal fundus photos 视网膜眼底图片

diabetic retinopathy 糖尿病型视网膜突变 

![image-20210505131743074](images/image-20210505131743074.png)

histopathology [ˌhɪstəʊpəˈθɒlədʒi] 组织病理学

pathologist 病理学家

lymph node 淋巴结

![image-20210505132137215](images/image-20210505132137215.png)

patch 色斑，小块

病理学图片都很大，因此每次都选出不同的patch进行训练

![image-20210505132404728](images/image-20210505132404728.png)

brain tumor segmentation 脑肿瘤分割

Chest X-Ray 可以检测many diseases，比如pneumonia, lung cancer, ，每年会新增2 billion图片。

分为Mass和Normal两种。pulmonary mass肺结节（肿块）

skin lesion [ˈliːʒn]  皮肤病变

diameter  [daɪˈæmɪtə(r)] 直径

Mass的定义：damage of tissue seen on a chest X-ray as greater than 3 centimeters in diameter

![image-20210505133944062](images/image-20210505133944062.png)

### 医疗领域的三大难题 3 Key Challenges

Class Imbalance; Multi-Task; Dataset Size

#### Class Imbalance

##### Binary Cross-entropy Loss

$$
\begin{aligned}

&{L(X, y})=\left\{\begin{array}{ll}
-\log P(Y=1 \mid X) & \text { if } y=1 \\
-\log P(Y=0 \mid X) & \text { if } y=0
\end{array}\right.
\end{aligned}
$$
对于损失函数来说，只要控制大部分y=0就可以让损失降到最小，因此改进损失函数。通常$w_{p}$分母都是总数，分子取对方的数量来做平衡。用分数形式的好处是：Loss可以和之前没有加权重时可以比较。


$$
L(X, y)=\left\{\begin{array}{ll}
w_{p} \times-\log P(Y=1 \mid X) & \text { if } y=1 \\
w_{n} \times-\log P(Y=0 \mid X) & \text { if } y=0
\end{array}\right.
$$

##### Resampling

上采样和下采样

#### Multi-Task

可以有三个模型分别解决三个问题，但是用一个模型更加好，而且可以拥有共享的特征。

Mass vs No Mass

Pneumonia vs No Pneumonia

Edema vs No Edema

##### Multi-label / Multi-task Loss

$$
L\left(X, y_{\text {mass }}\right)+L\left(X, y_{\text {pneumonia }}\right)+L\left(X, y_{\text {edema }}\right)
$$

![image-20210505151138313](images/image-20210505151138313.png)

结合样本数量的平衡，我们可以给出对于分辨Mass任务的损失函数：
$$
\begin{array}{c}
\text { Multi-Task } \quad L\left(X, y_{\text {mass }}\right)+L\left(X, y_{\text {pneumonia }}\right)+L\left(X, y_{\text {edema }}\right) \\
L\left(X, y_{\text {mass }}\right)=\left\{\begin{array}{ll}
-w_{\mathrm{p}, \text { mass }} \log P(Y=1 \mid X) & \text { if } y=1 \\
-w_{\mathrm{n}, \text { mass }} \log P(Y=0 \mid X) & \text { if } y=0
\end{array}\right.
\end{array}
$$

#### Dataset Size

需要从大样本中学习，模型很大

##### ransfer Learning迁移学习

可以先从其他图像中预训练，前面的layer都是学习边缘特征。然后，固定前面的Layer，只对后面的Layer进行训练。

##### Data Augmentation数据增强

增强点：

不同的图片对比度/明暗程度不一样；

心脏一般在左边，但是不能进行图片的翻转，否则心脏就在右边dextrocadia了

对于Skin Cancer：Rotate+flip

对于组织病理学Histopathology：图片中有不同的阴影shade：可以通过Rotate+Crop+Color Noise制造不同的shade来帮助泛化。

![image-20210505155412796](images/image-20210505155412796.png)

### 样本选取和训练集划分三大难题

#### Patient Overlap

- 要保持训练集和测试集独立。如果有一个带着项链的人的X-Ray一个在训练集一个在测试集，那么模型肯定会记住这个人的标签

反思：在第四范式的实际的项目中，当划分训练和验证样本时：是把份拆开按页划分的，而且同一种类别的样式也没有分开，因此不能保证训练集和验证集独立。

#### Set Sampling

- 小样本且样本不平衡时，要保证测试集中也要有正样本，一般可以设置一个至少 X%，或者在验证集中直接选取50%的正负样本
- 因此采样的顺序应当是：先Test Set再Validation Set最后才是Train Set

#### Ground Truth

inter-observer disagreement 各个专家意见不一：类似我们和业务认为的标注标准不一致。

- Consensus Voting

- 借助外力，比如说Mass的Chest X-ray还可以用CT打标签，比如皮肤病学Dermatology里，标签是由皮肤病变活组织检查skin lesion biopsy决定的

![image-20210505163814365](images/image-20210505163814365.png)

### 评价指标

### Sensitivity Specificity

![image-20210505170257170](images/image-20210505170257170.png)

![image-20210505170336772](images/image-20210505170336772.png)

![image-20210505170402279](images/image-20210505170402279.png)

#### PPV NPV

以上的条件概率都是基于已知真实label，来推算预测结果为真的情况。此时如果知道预测结果，想看真是结果时：

PPV=Positive Predictive Value=P(disease|+)

NPV=Negative Predictive Value=P(normal|-)

其实PPV就是Precision，Sensitivity就是Recall

![image-20210505171433484](images/image-20210505171433484.png)

#### Confusion Matrix, ROC Curve, AUC

![image-20210505171620715](images/image-20210505171620715.png)

如果我把Threshold设为0，那么所有的样本都会被预测成+，Sensitivity会变成1，Specificity会变成0

如果我把Threshold设为1，那么所有的样本都会被预测成-，Sensitivity会变成0，Specificity会变成1

- 果然和马凯说的一样，在画ROC曲线前，先把所有的概率值从小到大排列，并带着他们的Label（红：+，蓝色：-）

![image-20210505173212970](images/image-20210505173212970.png)

### 置信区间

误解：95%置信区间跟可能性没有关系！There is a nuanced difference细微的差别：

![image-20210505173811802](images/image-20210505173811802.png)

置信区间是你每一次抽样得到的区间，因此再做一次抽样会得到不同的置信区间

如下图：这是6次抽样带来的置信区间。因此，真正的解释是：95%的抽样里的置信区间会包含真实的总体值：p

![image-20210505174230072](images/image-20210505174230072.png)

### 概率校准 Calibration

https://zhuanlan.zhihu.com/p/90479183

校准是你预测有多少人的比例处于某个区间内(x)，相对于实际有多少人处于某个区间内(y)
这个问题在风控和ctr领域比较常见，风控对模型预测的真实概率是比较看重的。例如：

一些银行或者P2P公司会建立机器学习模型来估计拒绝借贷的概率，通过设置阈值以自动过滤一批借贷请求。假设我们对10000个预测概率为0.2左右的用户进行放款，我们通过预测概率为0.2认为这10000个用户中将来有8000个左右能够按时还款付息，而剩下的2000个可能违约赖账成为死账户，这个时候金融机构，尤其是银行，就要根据这种情况进行一些坏账准备，同时根据用户被预测出来的概率，金融机构还会对不同概率区间的用户设置不同的利率，利率的计算建立在用户的预测概率上。但是如果模型根本没法反应真实的概率，这些决策都会受到很大的影响，所以我们需要进行概率校准以使得模型的预测概率和真实概率对应起来

from sklearn.calibration import calibration_curve

### Shap包  模型可解释性

https://github.com/slundberg/shap

### MRI data

核磁共振

特点：多个连续的3D图像

处理方法：随机选取N（这里是3）个截面，就像是图片的3个channel一样进行输入，并把三个图片合成一张。

困难：这N张图不一定是正好对齐的Not Aligned，比如说有旋转，那么这些图片对应的位置会对不上

解决方法：[图像配准](https://www.cnblogs.com/carsonzhu/p/11188574.html)，通过[Image Registration](https://www.sicara.ai/blog/2019-07-16-image-registration-deep-learning)把图片摆正

![image-20210505182708444](images/image-20210505182708444.png)

### Segmentation

2D的像素：pixel 

3D的像素：voxel

cortex information：大脑皮层信息

silver lining：困境中的一线希望

#### 2D方法

一张一张图片预测，然后再回到3D的分割块。但有可能失去3D信息 3D context。比如一个肿瘤在一张图片里，那么很可能旁边的图片里也有。但由于一张一张图片传，很可能学不到这个信息。

#### 3D方法

由于一次性放入3D太占内存，因此把3D图片切成小块，最后再拼起来。但也有可能丢失Spatial Context。

#### U-Net

包括Contracting（收缩） path和Expanding Path

只能用2D的方法训练

3D U-Net 所有操作都变成三维

#### Segmentation

把一张图像分成很多个小块进行预测，然后输出每个正方形小块的概率值。

#### Soft Dice Loss

还记得深度学习的课上，吴老师详细讲了U-Net，但是没有讲损失函数，这里反而讲了它的损失函数

减少Loss等于增加Overlap

p对应预测概率，g对应真实值

Loss越小，希望分子越大分母越小。因此可以突出g=1的情况，不会受到样本不平衡的影响。

如果p在g=0的地方都是1，虽然分子没有优化（都是0），但是分母会压缩，因此也会照顾到为0样本。

**引申**：[Dice Loss](https://zhuanlan.zhihu.com/p/86704421)即取绝对值的版本

[图像语义分割中的 loss function](https://zhuanlan.zhihu.com/p/101773544) cross entropy loss; weighted loss; focal loss; dice soft loss; soft iou loss

![image-20210505195215698](images/image-20210505195215698.png)

#### 局限性

TB test: tuberculosis 肺结核

retrospective data 历史数据

clinician 临床医师

- 标准不同：清晰度不一样

![image-20210505201231957](images/image-20210505201231957.png)



- 解决方法：可以用旧模型在新数据上，取一小部分用于调参

- 真实世界里还需要进行预处理，clean数据集。
- 训练数据都是Frontal X-ray，真实应用也有可能是侧面Lateral X-ray
- 真实世界要考虑Age, Sex, Socioeconomic Status
- clinician需要可解释性

## AI for Medical Prognosis

(对病情的)预断，预后; 预测; 预言; 展望;

Machine learning is a powerful tool for prognosis, and can provide a tremendous boost to this branch of medicine by using many different types of medical data to make accurate predictions about a patient's future health.

multiple examples of prognostic tasks, including a few examples where prognosis using risk calculations is part of **routine clinical practice**. 常规临床做法

mortality 死亡率

**Without further ado,** let's dive in. 毫不迟延; 干脆; 立即

there are blood tests that are used to estimate the risk of developing breast and **ovarian cancer**. 乳腺癌、卵巢癌

An example of this is **cancer staging**, which gives an estimate of the survival time for patients with that particular cancer. 癌症分期（晚期）

Another example is the six-month mortality risk. This is used for patients with terminal conditions that have become advanced and **uncurable** and is used to determine who should receive **end-of-life care**. 不能治愈的 临终关怀

estimating the 10-year **cardiovascular** risk of an individual. 心血管的;

Profile might also include physical exam findings such as **vital signs** including temperature and blood pressure. 生命体征

Atrial fibrillation 心房颤动（简称房颤）

Atrial fibrillation is a common abnormal **heart rhythm** that puts the patient at a risk of stroke. 心率 

A 70-year-old male diagnosed with Atrial fibrillation has **hypertension** and diabetes. 高血压

congestive heart failure CHF 充血性心脏衰竭; 心力衰竭; 

Liver Disease Mortality 肝病死亡率

This is the **model for end-stage liver disease** which produces what is called the MELD Score。肝病终期

**take the natural log** of the values before **plugging** them **into** the model. 

Now this makes sense because HDL-C is high-density **lipoprotein cholesterol** and that's often called good cholesterol. And thus we might expect would lower the risk of heart disease.脂蛋白 胆固醇

The apple in my left hand already looks **stale** and will expire in two days, 不新鲜的

#### *concordant* pairs

同序对（*concordant* pairs）和异序对（*discordant* pairs）

如果分类变量与数值预测结果方向相同，就叫concordant。如果数值预测两者一样，就叫risk tie

要形成permissible pair，就需要两个gt不同的人，也就是一定要一个死一个没死。

![image-20210607235705188](images/image-20210607235705188.png)

![image-20210607235651995](images/image-20210607235651995.png)

#### C-Index

注意：C-index和acc并没有直接的关系！

![image-20210607235852164](images/image-20210607235852164.png)

![image-20210607235938891](images/image-20210607235938891.png)

#### 用正态拟合数据

```python
from scipy.stats import norm
data = np.random.normal(50,12, 5000)
fitting_params = norm.fit(data)
norm_dist_fitted = norm(*fitting_params)
t = np.linspace(0,100, 100)
plt.hist(data, bins=60, density=True)
plt.plot(t, norm_dist_fitted.pdf(t))
plt.title('Example of Normally Distributed Data')
plt.show()
```

![image-20210608001756585](images/image-20210608001756585.png)

 *Diabetic Retinopathy* 糖尿病视网膜病变 

Often machine learning models are considered black boxes due to their complex inner workings, but in medicine, the ability to explain and interpret a model may be critical for human acceptance and trust. 

![image-20210610223445613](images/image-20210610223445613.png)

Systolic Blood Pressure 收缩压

![image-20210610233325093](images/image-20210610233325093.png)

We can see that there is a **spike** on the older people between 65 and 75. 尖刺

缺失值的启示：删除缺失值有可能会让模型的训练数据的分布变化，从而导致效果不一致。由上图，old test里年轻人很少，很可能是医生不记录年轻人的血压，但他死了。由于空，我们删了这条数据，最终导致上线效果不一致。

imputation, **impute** the missing value 估算    注意：插值是interpolation

### mean imputation 和 regression imputation

我以为是在不同的类别内，分别对0和1的数据进行回归/取平均，然后预测出缺失值。其实不然，它是一视同仁，直接对所有数据进行回归/取平均。测试集也用训练集的数据

![image-20210613204304411](images/image-20210613204304411.png)

![image-20210613205321502](images/image-20210613205321502.png)

![image-20210613204328200](images/image-20210613204328200.png)

![image-20210613203709224](images/image-20210613203709224.png)

Cardiovascular disease event prediction / CVD prediction 心血管疾病

systolic blood pressure心脏收缩压

无论如何都要填充空值，即使空值不准的目的是：空值可能是数据在某些特定原因下形成的，为了保持数据原有的分布，所以要填充空值。

### 生存模型

![image-20210614184726885](images/image-20210614184726885.png)

censoring 统计学借以描述对某些个体不可能观察到特定点事件发生的现象。在本例中，被观察者在14个月后中途退出了，就写成14+，代表censoring

开始时间是做手术的时间，只要是没有发病，或者退出实验，都会写出xx+

![image-20210614185101350](images/image-20210614185101350.png)

![image-20210614185347173](images/image-20210614185347173.png)

Right censoring：在last contact后总会发生，而不是一直都不会发生。right就是在右边的意思

![image-20210614185519597](images/image-20210614185519597.png)

![image-20210614190105261](images/image-20210614190105261.png)

die immediately：实际是每个人立刻死，这时候的存活率，就是存活到这个时间点的人数

Never die相比于die immediately的差别是：分子加上了censoring的部分

![image-20210614200003170](images/image-20210614200003170.png)

![image-20210614200123434](images/image-20210614200123434.png)

Kaplan Meier 估计 /maier/

![image-20210614194502850](images/image-20210614194502850.png)

![image-20210614195329338](images/image-20210614195329338.png)

学会了如何推导：S(t)=1-F(T)，F(T)就是T的累计密度函数。

λ(t)

=P(T=t)/P(T>=t)  根据条件概率展开

=f(T)/S(T)  课件中S(t)的形式应当写成>=的，不过在连续的情况下其实都一样

=-S'(t)/S(T)  然后再回到exp的形式

![image-20210614195459945](images/image-20210614195459945.png)

![image-20210614195605843](images/image-20210614195605843.png)

![image-20210614200836876](images/image-20210614200836876.png)

![image-20210614201040395](images/image-20210614201040395.png)

![image-20210614201437328](images/image-20210614201437328.png)

![image-20210614201949386](images/image-20210614201949386.png)

![image-20210614202438008](images/image-20210614202438008.png)

**个人总结**：survival是在起始点看，将来每个时间点还能活下多少人。hazard是将来到了哪个时间点，在那个时间点上你死掉的概率。cumulative hazard是hazard的积分

![image-20210614204651621](images/image-20210614204651621.png)

用exp是保证它乘的系数大于0.在这个例子中，价格is_smoker和age都取0，那么exp(0)=1，就相当于不变。用exp还有一个好处，做系数解释比较不同X的效果时，直接是exp(coef*ΔX)

![image-20210614210700137](images/image-20210614210700137.png)

![image-20210614221738187](images/image-20210614221738187.png)

![image-20210614222113149](images/image-20210614222113149.png)

这就是Cox proportional hazard model

chemotherapy [ˌkiːmoʊ]化疗

Nelson Aalen estimator

![image-20210614232300451](images/image-20210614232300451.png)

之前的Caplan Meire是估计Survival S的，这里的Nelson Aalen是估计cumulative Hazard的

之前是连乘，这里是连加

![image-20210614232824961](images/image-20210614232824961.png)

死亡率得分 mortality score

![image-20210614233536456](images/image-20210614233536456.png)

Harrell's C-index

![image-20210614234057564](images/image-20210614234057564.png)

感觉这里的逻辑有点奇怪：他说，正因为这个人只活了20个月，那么他更加惨，那么他的risk score更高。可是我记得之前算hazard时，是有一个先降后升的过程

![image-20210614235013402](images/image-20210614235013402.png)

这种也算concordant

![image-20210614235034363](images/image-20210614235034363.png)

![image-20210614235053467](images/image-20210614235053467.png)

这两种都算risk tie，无论是T相同还是risk score相同。这是因为他们之间不完全错，所以都叫risk tie

以前的permissible pair是一定要一个人死一个人没死，现在的即使两个人T相同也可以

![image-20210614234707987](images/image-20210614234707987.png)

B outlived A B比A活得长

![image-20210614234907837](images/image-20210614234907837.png)

![image-20210614234946958](images/image-20210614234946958.png)

只有这两种是non-permissible

![image-20210615000300526](images/image-20210615000300526.png)

这里面的risk score既可以来源于mortality score，也可以来源于Cox proportional hazard model

## AI for Medical Treatment

We will **switch gears** and cover information extraction. 换别的事情做

control = placebo=没有施加效应=no treatment

一般都说treatment vs control, control就是没有施加效应

![image-20210628222345243](images/image-20210628222345243.png)

Now the NNT is simply the **reciprocal** of the ARR. 互惠的；相互的；倒数的，彼此相反的，**相反数**

![image-20210628223750198](images/image-20210628223750198.png)

Obeserved=Factual  unobserved=counterfactual

If the data does not come from randomized control trials, then we often cannot apply the following idea without making strong assumptions about how the data is generated. 

The answer is reduced blood **glucose** levels. 葡萄糖

![image-20210629231229735](images/image-20210629231229735.png)

non-contextualized 只由词向量决定，与上下文无关

为什么一定要用矩阵的方式存储？而非在start里直接挑出来最高，在end里也挑出来最高，然后就行了？是因为start必须要在end前面，因此这矩阵相当于遍历所有。

![image-20210629232237365](images/image-20210629232237365.png)

However, this would be time and cost intensive, requiring radiologists to interpret each one of thousands of these images.

thesaurus同义词词典

![image-20210630235820194](images/image-20210630235820194.png)

macro 宏观 对所有类别

micro 微观 对所有样本

##### Drop Column Method

哪一列去掉后效果降低越多，那一列就越重要

##### permutation method

对测试集某一列的值排序后，其他列不变，进行预测。如果效果下降不多，说明这个变量不重要

特点：不需要重新训练模型

![image-20210701002627157](images/image-20210701002627157.png)

![image-20210701004517931](images/image-20210701004517931.png)

### Shapley Value

使用Shapley Value的原因是针对相关性强的两个字段而言的，如果直接看贡献程度会很低，但是Shapley Value会把所有其他变量的排列组合都试一遍，最后求平均，这样算出来的贡献程度就很公平。

Shapley Value是针对**每一条样本**算的，也就是该条样本的预估值在有/无某些列的时候训练出来的模型之间的预测值的差，在这个场景中是0-1之间的概率。如果把测试集中每一条样本，在不同模型下，都算出一个预估值，那么就可以算该条样本的一个Shapley Value。某个变量的Shapley Value是测试集中所有样本的该条字段的Shapley Value的平均数。

![image-20210701005005895](images/image-20210701005005895.png)

#### 业务启示

在三井AutoML的可解释性报告里，为什么所有的变量的Feature Importance都在0-1之间，就是这个原因。明天需要做加入利率看模型是否会改进的实验，马凯想通过Feature Importance看重不重要，但是得保证测试集和验证集一致，才能对特征重要性有一个公正的评价。马凯不了解其中的原理，因此没有考虑到这一点。

![image-20210701200451515](images/image-20210701200451515.png)

In this chest x-ray, the patient has an enlarged heart also called **cardiomegaly**. 心肥大

![image-20210701220157929](images/image-20210701220157929.png)

### GradCam & Localization Map

![image-20210701220625331](images/image-20210701220625331.png)

那ak是怎么算？

一共k层，求每一层的梯度，是通过y和它输出值的导数决定的。最后再取平均，就得到每一层的梯度。

ak相当于权重，最后对所有层和他们的权重相加，得到输出的Localization Map

![image-20210701220858220](images/image-20210701220858220.png)

![image-20210701220935660](images/image-20210701220935660.png)

![image-20210701221126972](images/image-20210701221126972.png)

#### Heat Map

对上一步的输出结果加一层ReLu，最终把数字转图像就成了HeatMap，越靠近0就是越黑，越靠近1就是越Bright

![image-20210701221301297](images/image-20210701221301297.png)

![image-20210701221351341](images/image-20210701221351341.png)

但是像素很低，因为只对输出层做了操作。这时候需要你用interpolation，变成更清晰的图像。再加一点transparency，就可以覆盖在原图上。

![image-20210701221443096](images/image-20210701221443096.png)

如果是多class分类，就用y_c的导数作为权重，

![image-20210701221758432](images/image-20210701221758432.png)

Missing data is a common occurrence in data analysis, that can be due to a variety of reasons, such as measuring **instrument malfunction**, respondents not willing or not able to supply information, and errors in the data collection process. 仪表失灵

## 作业相关

### risk-models-using-tree-based-models

#### 可视化决策树

![4266f09e109980060df0cd7f44ac67a](images/4266f09e109980060df0cd7f44ac67a.png)

![f4e8c2e1fd7774f32f33abb1d98f18f](images/f4e8c2e1fd7774f32f33abb1d98f18f.png)

#### 超参数查找

```python
def holdout_grid_search(clf, X_train_hp, y_train_hp, X_val_hp, y_val_hp, hyperparams, fixed_hyperparams={}):
    '''
    Conduct hyperparameter grid search on hold out validation set. Use holdout validation.
    Hyperparameters are input as a dictionary mapping each hyperparameter name to the
    range of values they should iterate over. Use the cindex function as your evaluation
    function.

    Input:
        clf: sklearn classifier
        X_train_hp (dataframe): dataframe for training set input variables
        y_train_hp (dataframe): dataframe for training set targets
        X_val_hp (dataframe): dataframe for validation set input variables
        y_val_hp (dataframe): dataframe for validation set targets
        hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                            names to range of values for grid search
        fixed_hyperparams (dict): dictionary of fixed hyperparameters that
                                  are not included in the grid search

    Output:
        best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                             validation set
        best_hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                                 names to values in best_estimator
    '''
    best_estimator = None
    best_hyperparams = {}
    
    # hold best running score
    best_score = 0.0

    # get list of param values
    lists = hyperparams.values()
    
    # get all param combinations
    param_combinations = list(itertools.product(*lists))
    total_param_combinations = len(param_combinations)

    # iterate through param combinations
    for i, params in enumerate(param_combinations, 1):
        # fill param dict with params
        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
            
        # create estimator with specified params
        estimator = clf(**param_dict, **fixed_hyperparams)

        # fit estimator
        estimator.fit(X_train_hp, y_train_hp)
        
        # get predictions on validation set
        preds = estimator.predict_proba(X_val_hp)
        
        # compute cindex for predictions
        estimator_score = cindex(y_val_hp, preds[:,1])

        print(f'[{i}/{total_param_combinations}] {param_dict}')
        print(f'Val C-Index: {estimator_score}\n')

        # if new high score, update high score, best estimator
        # and best params 
        if estimator_score >= best_score:
                best_score = estimator_score
                best_estimator = estimator
                best_hyperparams = param_dict

    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return best_estimator, best_hyperparams
```

代码亮点：通过itertools.product生成所有超参数的笛卡儿积（两两组合）

#### 对每一列画缺失值和全集的分布

目的是看缺失值会不会引起bias，看缺失值的分布是否是随机的

![image-20210703190559079](images/image-20210703190559079.png)

```python
dropped_rows = X_train[X_train.isnull().any(axis=1)]

columns_except_Systolic_BP = [col for col in X_train.columns if col not in ['Systolic BP']]

for col in columns_except_Systolic_BP:
    sns.distplot(X_train.loc[:, col], norm_hist=True, kde=False, label='full data')
    sns.distplot(dropped_rows.loc[:, col], norm_hist=True, kde=False, label='without missing data')
    plt.legend()

    plt.show()
```

Most of the covariates are distributed similarly whether or not we have discarded rows with missing data. In other words missingness of the data is independent of these covariates.

If this had been true across *all* covariates, then the data would have been said to be **missing completely at random (MCAR)**.

But when considering the age covariate, we see that much more data tends to be missing for patients over 65. The reason could be that blood pressure was measured less frequently for old people to avoid placing additional burden on them.

As missingness is related to one or more covariates, the missing data is said to be **missing at random (MAR)**.

Based on the information we have, there is however no reason to believe that the _values_ of the missing data — or specifically the values of the missing systolic blood pressures — are related to the age of the patients. 
If this was the case, then this data would be said to be **missing not at random (MNAR)**.

##### 总结

最容易混淆的是missing at random (MAR)和missing not at random (MNAR)。他俩的根本区别在于。missing at random (MAR) 的着重点是 probability of missingness on x ，而missing not at random (MNAR)的着重点是 the values of x。也就是说，对于MAR，即使是血压缺失，对于65岁以上的人来说，血压的分布是与缺失值无关的，因为在MAR中，**缺失本身就是一个概率的随机问题**，仍有其他记录被记录下来，无论是否缺失，并不改变65岁以上的人的血压的分布情况，去掉这些行不会有事。而对于MNAR来说，65岁以上的人的血压在缺失和不缺失的人群中，分布是不一样的。去掉他们就会改变原有的分布，这就是为什么他会说the missing systolic blood pressures — are related to the age of the patients。其实对于MAR来说，缺失值看起来也与年龄有关，比如说年龄大的人越容易缺失血压，但对于这一些年龄大又缺失血压的人来说，我们还可以找到一群年龄大但是有血压数据的人，这叫missingness is related to age，也就是MAR。而MNAR，缺失血压的这一部分老年人的本应有的血压和另一群有血压数据的老年人的血压，分布是不一致的，这叫 the _**values**_ of the missing data is related to age。

#### 用其他列预测有缺失值的列

Next, we will apply another imputation strategy, known as **multivariate feature imputation**, using scikit-learn's `IterativeImputer` class (see the [documentation](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer)).

With this strategy, for each feature that is missing values, a regression model is trained to predict observed values based on all of the other features, and the missing values are inferred using this model.
As a single iteration across all features may not be enough to impute all missing values, several iterations may be performed, hence the name of the class `IterativeImputer`.

In the next cell, use `IterativeImputer` to perform multivariate feature imputation.

> Note that the first time the cell is run, `imputer.fit(X_train)` may fail with the message `LinAlgError: SVD did not converge`: simply re-run the cell.

##### Impute using regression on other covariates
```python
imputer = IterativeImputer(random_state=0, sample_posterior=False, max_iter=1, min_value=0)
imputer.fit(X_train)
X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
```

#### Shap包 可解释性

取测试集中的一个样本，计算每个变量的Shapley Value，然后画图

How to read this chart:
- The red sections on the left are features which push the model towards the final prediction in the positive direction (i.e. a higher Age increases the predicted risk).
- The blue sections on the right are features that push the model towards the final prediction in the negative direction (if an increase in a feature leads to a lower risk, it will be shown in blue).
- Note that the exact output of your chart will differ depending on the hyper-parameters that you choose for your model.

We can also use SHAP values to understand the model output in aggregate. Run the next cell to initialize the SHAP values (this may take a few minutes).

![image-20210703200739338](images/image-20210703200739338.png)

```python
explainer = shap.TreeExplainer(rf_imputed)
i = 0
shap_value = explainer.shap_values(X_test.loc[X_test_risk.index[i], :])[1]
shap.force_plot(explainer.expected_value[1], shap_value, feature_names=X_test.columns, matplotlib=True)
```

### survival-estimates-that-varies-with-time

#### pd.get_dummies，对df指定要转换的列名，直接转df

![image-20210703231513226](images/image-20210703231513226.png)

### 3D U-Net

The MRI scan is one of the most common image modalities that we encounter in the radiology field.
Other data modalities include:

- [Computer Tomography (CT)](https://en.wikipedia.org/wiki/CT_scan),
- [Ultrasound](https://en.wikipedia.org/wiki/Ultrasound)
- [X-Rays](https://en.wikipedia.org/wiki/X-ray).



Implementations of dependency parsers are very complex, but luckily there are some great **off-the-shelf** tools to do this. 开箱即用

This increases our performance given that biomedical text is full of **acronyms**, **nomenclature** and medical jargon. 缩略词 命名法 

### 调用预训练模型，改最后几层输出；自定义损失函数

![image-20210705003859361](images/image-20210705003859361.png)



#### 论文来源

(1)DenseNet:《Densely connected convolutional networks》https://zhuanlan.zhihu.com/p/37189203

下载:https://arxiv.org/pdf/1608.06993.pdf

(2)深度学习识别皮肤癌《Dermatologist-level classi_cation of skin cancer withdeep neural networks》

下载：http://on-demand.gputechconf.com/gtc/2017/presentation/s7822-andre-esteva-dermatologiest-level-classification-of-skin-cancer.pdf

(3)深度学习CT出血检测《Radiologist level accuracy using deep learning for hemorrhagedetection in ct scans》

(4)糖尿病视网膜病变的深度学习检测《Development and validation of a deep learning algorithmfor detection of diabetic retinopathy in retinal fundus photographs.》

下载：https://pdfs.semanticscholar.org/1ea4/41b6f99a70d05d66b0cd331568507e7f822d.pdf

(5)Batch normalization:《 BatchNormalization: Accelerating Deep Network Training by Reducing InternalCovariate Shift》

下载：http://proceedings.mlr.press/v37/ioffe15.pdf

(6)CAM:《Learning deep features for discriminative localization.》

下载：https://www.cvfoundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf

(7)ChestX-ray8数据集：https://nihcc.app.box.com/v/ChestXray-NIHCC

(8)ChestX-ray14数据集：https://arxiv.org/pdf/1705.02315.pdf



We see, for example, that the model predicts Mass (abnormal spot or area in the lungs that are more than 3 centimeters) with high probability. Indeed, this patient was diagnosed with mass. However, we don't know where the model is looking when it's making its own diagnosis. To gain more insight into what the model is looking at, we can use GradCAMs.

GradCAM is a technique to visualize the impact of each region of an image on a specific output for a Convolutional Neural Network model. Through GradCAM, we can generate a heatmap by computing gradients of the specific class scores we are interested in visualizing.



#### Keras提取中间层输出

![image-20210705223845161](images/image-20210705223845161.png)

#### 获取中间层输出，heatmap，cv2 插值填充



```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def grad_cam(input_model, image, category_index, layer_name):
    """
    GradCAM method for visualizing input saliency.
    
    Args:
        input_model (Keras.model): model to compute cam for
        image (tensor): input to model, shape (1, H, W, 3)
        cls (int): class to compute cam with respect to
        layer_name (str): relevant layer in model
        H (int): input height
        W (int): input width
    Return:
        cam ()
    """
    cam = None
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # 1. Get placeholders for class output and last layer
    # Get the model's output
    output_with_batch_dim = model.output
    
    # Remove the batch dimension
    output_all_categories = output_with_batch_dim[0]
    
    # Retrieve only the disease category at the given category index
    y_c = output_all_categories[category_index]
    
    # Get the input model's layer specified by layer_name, and retrive the layer's output tensor
    spatial_map_layer = model.get_layer(layer_name).output
    # 2. Get gradients of last layer with respect to output

    # get the gradients of y_c with respect to the spatial map layer (it's a list of length 1)
    grads_l = K.gradients(y_c,spatial_map_layer)
    
    # Get the gradient at index 0 of the list
    grads = grads_l[0]
        
    # 3. Get hook for the selected layer and its gradient, based on given model's input
    # Hint: Use the variables produced by the previous two lines of code
    spatial_map_and_gradient_function = K.function([model.input],[spatial_map_layer,grads])
    
    # Put in the image to calculate the values of the spatial_maps (selected layer) and values of the gradients
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])

    # Reshape activations and gradient to remove the batch dimension
    # Shape goes from (B, H, W, C) to (H, W, C)
    # B: Batch. H: Height. W: Width. C: Channel    
    # Reshape spatial map output to remove the batch dimension
    spatial_map_val = spatial_map_all_dims[0]
    
    # Reshape gradients to remove the batch dimension
    grads_val = grads_val_all_dims[0]
    
    # 4. Compute weights using global average pooling on gradient 
    # grads_val has shape (Height, Width, Channels) (H,W,C)
    # Take the mean across the height and also width, for each channel
    # Make sure weights have shape (C)
    weights = np.mean(grads_val,axis=(0,1))
    
    # 5. Compute dot product of spatial map values with the weights
    cam = spatial_map_val@weights

    ### END CODE HERE ###
    
    # We'll take care of the postprocessing.
    H, W = image.shape[1], image.shape[2]
    cam = np.maximum(cam, 0) # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()
 # !!!!cv2插值填充
    return cam
```



You should see age as by far the best prediction of near term mortality, as one might expect. Next is sex, followed by diastolic blood pressure. Interestingly, the poverty index also has a large impact, despite the fact that it is not directly related to an individual's health. This **alludes to** the importance of social determinants of health in our model. 影射 暗指































