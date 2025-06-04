
## 图像预训练

### ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale[ICLR'21][Google]
- 使用了纯transformer去解决Image分类问题，提出了用transformr解决问题的方法
- 将图像分为了14\*14的patch，每个patch的大小为768，使用Linear Embedding将图像转换为196\*768大小的输入，增加了cls patch作为分类的标志，输入是197\*768
- 将位置作为1D编码加到原有的embdding中，不同编码方式对结果影响不大

 
 
### BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension[ACL'20][Facebook]
- 同时参考了bert的encoder和gpt的decoder，使用全量的transformer架构
- bart对输入加入了各种噪声，在输出时将其还原，加入的噪声包括，1. 随机替换token为mask 2. 随机删除token 3. 将连续span替换为[mask]，4. 将doc中的句子打乱 5. 随机选择一个句子作为开头
- 对于ABCDE的输入，首先在AC之间加0-3的噪声，再将CD进行掩码，得到输出是ABCDE
- 下游微调任务为文本分类，句子标注，句子生成和翻译


### DeiT: Training data-efficient image transformers & distillation through attention[ICML'21][Facebook]
- 针对vit训练慢的问题，使用了超参数设置，知识蒸馏，数据增广等方式加速模型训练
- 使用教师模型的软标签结果，在transformer后添加蒸馏token，连同原有的分类的token的损失一起组成loss
- 超参数设置上，使用了截断正态分布，使用cosine学习调节方法
- 使用了多种数据增广方法。随机擦除，cutmix，随机增强等

### BEiT: 


## 行为检测
### MS-TCT: Multi-Scale Temporal ConvTransformer for Action Detection[CVPR'22][Inria]
- 额外文档，见[MS-TCT](MSTCT.md)

## 目标检测
### YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications[meituan]
- 见额外文档，[yolov6](yolov6.md)


### YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors[arxiv'22][IIS]
- 见额外文档，[yolov7](yolov7.md)


### ViTDet: Exploring Plain Vision Transformer Backbones for Object Detection[ECCV'22][META]
- 提供一个另外的思路使用vit来构建下游目标检测任务，即使用纯预训练的vit模型，不改变模型结构
- 使用了简单的特征金字塔，类似于ssd,将最后一层特征的上采样和下采样得到不同尺度特征，特征不做融合
- 不使用shift窗口的局部卷积，在每个subset后面加一个全局注意力层学习全局信息，达到60+的AP


### Swin Transformer: Hierarchical Vision Transformer using Shifted Windows[ICCV'21][MSRA]
- 为了解决1. 模型的通用骨架问题，图像的token不像语言token是基本元素 2. 之前的vit是固定scale的token,对高分辨率不友好，全局attention计算复杂度高 
- 首先将h\*w\*c大小的图像分割为4块，每块的大小为h/4\*w/4\*48，在每一个块中做滑动窗口block运算，在每次进入到attention block前，都要进行一次patch merging，类似于池化操作，将相邻patch做合并，将分辨率大小缩小1/2，将通道数增加一倍；而block中则是做了两次，一次是常规的局部的transformer,一次是滑动窗口的transformer，所以模型block内层数是2的倍数
- mask机制：移动窗口会带来额外的不规则的窗口，每个窗口内的个数不同，做法是将窗口右下移动，在计算attention时将不相邻的patch做掩码，最后一次计算就能得到不同窗口的掩码


### DETR: End-to-End Object Detection with Transformers[ECCV'20][Facebook]
- 使用transformer做目标检测，使用二分图匹配计算损失去除了nms过程，得到Faster-RCNN近似的AP
- 首先将原图像经过R50，得到850\*256大小的特征，然后送入到Encoder结构中，作为Decoder的输出，在Decoder输入中加入Learnable Query，最终得到100\*256大小的输出
- 对于输出，通过二分匹配的方式计算损失，而在推理阶段则使用阈值的方式得到输出框，阈值为0.7