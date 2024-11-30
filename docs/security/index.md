## 对抗攻防

### Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection[CVPR'22][JHU]
- 提出了SAC检测器用于防御目标检测的贴纸攻击，SAC检测器分为两部分，分割部分和补全部分。
- 分割部分使用了将贴纸检测视为图像分割任务，使用UNet进行自对抗训练，找到贴纸大致位置。
- 补全部分使用汉明距离作为预测和GT的度量，补全的是所有在gama重叠范围内的交集。
- 改进了APRICOT数据集，增加了Mask标注。

### Defending Physical Adversarial Attack on Object Detection via Adversarial Patch-Feature Energy[MM'22][South Korea]
- 提出了APE(Adversarial Patch-Feature Energy Masking)模块防御类别为人的目标检测的攻击，APE模块可分为两部分，APE-masking 和 APE-refinement。
- APE-masking部分负责解析出攻击对应的mask，具体做法是通过objectness loss反向传播的L1范式的平方获得FE ，根据干净样本和对抗样本的分布差异确定阈值，大于阈值会被认为是proposal patch，然后通过上采样累加形成最终mask。
- APE-refinement部分根据APE-masking解析出的mask进行加固，具体是根据干净样本的分布均值的比例进行clip（这里似乎假设对抗样本的分布均值比干净的大？），对应位置上的adv patch即为clip平滑后的值。
- 属于图像信号处理的一类工作，有点意思，但套路有点老。

### Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame[TIP'22][KAIST]

### Role of Spatial Context in Adversarial Robustness for Object Detection[CVPR'20][UMBC]


### Physically Adversarial Attacks and Defenses in Computer Vision: A Survey[arXiv'22][Beihang]
- 额外文档，见[计算机视觉领域的物理对抗攻防综述](计算机视觉领域的物理对抗攻防综述.md)

### A survey on hardware security of DNN models and accelerators[arXiv'22]
- 额外文档，见[DNN模型和加速器的硬件安全综述](DNN模型和加速器的硬件安全综述.md)

## 模型后门和数据投毒

### Trojan attack on neural networks[NDSS'18][purdue]
- 神经网络的后门攻击
- 相比BadNet添加了神经网络反传时的优化信息，取代了badnet中任意像素块的触发器


### Latent Backdoor Attacks on Deep Neural Networks[CCS'19][UChicago]
- 后门攻击，将后门触发器隐藏在模型隐特征中，当用户下载预训练模型后便可触发
- 训练教师网络模型使其包含目标攻击类别y_t,然后使用target和非target数据进行微调
- 根据训练好的教师模型使用优化算法生成触发器
- 使用投毒数据注入模型后门，训练数据，固定最后一层，只训练模型特征
- 将模型的最后一层去除，隐藏y_t

### Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks[ISP'19][UChicago]
- 后门防御，通过优化的方法构建逆向触发器，基于指定后门标签的触发器相比其他标签的小


### Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering[AAAI'19][IBM]
- 后门检测与防御方法，通过特征图的
- 使用投毒数据，取模型最后一层前的特征图数据，压平
- 对所有激活进行降维、聚类成2类
- 使用异常分析算法分析是否具有投毒样本，主要有1. 阈值法 2. 平均度法，相对大小比对 3. 聚类边缘值

