
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

### DETR: