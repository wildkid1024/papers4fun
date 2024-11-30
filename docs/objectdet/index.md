
## 行为检测
### MS-TCT: Multi-Scale Temporal ConvTransformer for Action Detection[CVPR'22][Inria]
- 额外文档，见[MS-TCT](MSTCT.md)


## 目标检测
### YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications[meituan]
- 见额外文档，[yolov6][yolov6.md]


### YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors[arxiv'22][IIS]
- 见额外文档，[yolov7](yolov7.md)


## OCR 识别

### SVTR: Scene Text Recognition with a Single Visual Model[IJCAI'22][BJTU,Baidu]
- 介绍了场景文本识别的几种主流结构，分别为：1. CNN+RNN 2. CNN + 自回归 3. CNN + MHA fusion 
- 首先将图像经过一个Patch Embedding，通常是CNN，在这里是2个3\*3 stride为2的CBR结构
- 然后经过3个stage，每个stage包含了mixing block和merging的网络结构，最终图像大小从`H\*W\*3 -> 1 \* (W/4) * D_3`
- mixing block使用了global attention结构来提取字符和字符间的特征，使用local attention提取笔画间的特征，merging操作是一个CB块，将高度减半，相应地channel的维度翻倍
- 在paddleocr中，使用了两个mobilenet block提取视觉特征，svtrblock的stage设置为2，后面的GAP被替换成了conv1x1