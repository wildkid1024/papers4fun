# 每周论文合集


## 信息检索、搜索

### Möbius Transformation for Fast Inner Product Search on Graph[NIPS'19][Baidu]
- 现有的图上的ANN检索算法依赖于在metric measure(L2) 构建出的类德劳内图，但在no-metric measure(IP) MIPS 上无此性质，效果不佳（// TODO 效果不佳如何证明）
- 提出了 mobius 变换：$y_i= x_i/‖x_i‖^2$ ，将数据集从 IP space 映射到 L2 sapce，在此基础上建图
- 多个公开数据集上，同等算力下的Recall 精度有明显提升（20% ～ 30%）

### Embedding-based Retrieval in Facebook Search[KDD'20][Facebook]


### Transformer Memory as a Differentiable Search Index[NIPS'22][Google]
- 将Transformer(T5模型)应用到信息检索，实现从问题到docid的映射。
- 主要有几个问题：1. 文档表示：原始文本或词袋 2. 文本id表示：直接表示为整数、非结构化自动化编号以及结构化语义编号。3. 索引方法：问题到文档id(q,j)或者文本到文档id(d,j)。
- 探索了Inputs2target、Targets2Inputs、Bidirectional、Span Corruption等几种索引方法，直接索引、集合索引、逆索引等文档表示方法，以及非结构化自动表示、结构化字符表示、语义结构化表示等几种不同的docid表示，实验验证了组合的表示方法。


### A Neural Corpus Indexer for Document Retrieval[NIPS'22][MSRA]

- 提出了一种端到端的基于Transformer的信息检索架构。
- 传统的信息检索分为两类，一类是基于term的，建立倒排表，根据倒排表进行索引，缺点：无法召回语义相关的文档；另外一类是基于语义的，建立双塔模型，使用ANN进行检索，缺点：在精确匹配上表现较差，二是ANN有欧拉空间的强假设，模型无法合并文档交互？
- 在预处理部分加入了kmeans,使文档ID具有分类层次化的特点，然后借助DocT5Query和Document As Query对文档生成query，将生成或真实的<query, docid>对送入到encoder中，使用了前缀可感知的共享权重decode进行解码，最后使用同一query间的输入尽可能接近，不同query间尽可能大进行对比学习，使得算法更加稳定。
- 缺点：在v100机器上，最大的吞吐只有50左右，在实际场景中是不够的。当出现新的文档的时候，docid会发生变化，模型要进行重训练。


## 推荐系统
### Wide & Deep Learning for Recommender Systems[DLRS'16][Google]
- 提出了一种wide和deep结合的推荐系统方案，wide部分使用LR模型，deep部分使用全连接网络模型，分别提取低阶和高阶的特征
- wide部分采用的是稀疏+稠密的特征工程，deep部分则直接对特征进行embedding，然后输入到深度网络中，两部分分别训练
- 线上测试能提高3.9的收益


### DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[IJCAI'17][Huawei]
- 与wide&deep一样都使用了wide和deep模型，但将wide和deep模型的训练融合在一起
- wide部分由LR换成了FM，wide和deep部分都采用了embedding稠密特征，无需进行特征工程。

### Deep & Cross Network for Ad Click Predictions(DCN)[arXiv'17][google]
- 对wide&deep网络的wide侧进行了修改，借鉴了残差网络，设计了交叉网络学习交叉特征
- 交叉网络实际上就是通过xl和cross weight的乘积来得到当前特征与x0特征的权重，然后使用该权重与x0的乘积作为残差来学习
- 是element粒度的特征交叉


### Deep Interest Network for Click-Through Rate Prediction(DIN)[KDD'18][Ali]
- 是对原始的深度推荐模型的改进，主要针对原始模型中sum pooling没有考虑item之间权重不同的问题
- 文章有3点贡献： 1. 提出了DIN，借助了NMT中的Attention结构学习用户行为和候选Item之间的权重信息，在文中成为activation Unit 2. 提出dice激活函数，相当于prelu的泛化版本 3. 提出了稀疏训练的方法，在L2惩罚项更新的时候，只更新参数不为0的部分，对参数为0的部分不更新。


## 系统、编译器设计、优化

### TVM: an automated end-to-end optimizing compiler for deep learning[OSDI'18][UW]
- TVM是一个能够将上层计算图表示编译转化为后端IR的工具。首先是高层次计算优化，将计算图分解为Tensor和计算，然后使用自动优化对低级的Tensor和计算针对特定的硬件进行优化以达到最佳性能。
- 高层次计算图级的优化：算子融合(融合相邻的算子)、常数折叠(提前计算静态值)、静态内存预分配(预分配内存)、数据存储转换(改变数据分布以利用缓存和SIMD特性)。
- 低层次Tensor级优化：利用Halide原则将规划和计算分开，首先引入了领域专用语言Tensor expression表示计算，然后写Schedule进行优化，转化为TVM IR对应着特定硬件表示。具体的Schedule为：带共享内存的循环并行、向量化以利用硬件的SIMD和向量运算、访问执行分离隐藏延迟。
- 自动化优化：使用了Xgboost根据配置进行性能预测，使用真实的测试数据作为训练数据，使用模拟退火的方法进行配置更新；并提供了一个可以交叉编译的分布式远程调用。
- 开创性的工作，不过TVM现在还在开发当中，有些组件还不太稳定，另外还不够用户友好。

### Ansor: Generating High-Performance Tensor Programs for Deep Learning[OSDI'20][Cal]
- 是一个张量生成框架，有三个特色，一是有更大的搜索空间，二是在更大的搜索空间内进行高效地搜索，三是识别并优化子图以获得更好的端到端的性能
- 如何选取更大的搜索空间？首先将原图划分为一系列子图，使用一些规则进行约束，从后往前枚举DAG图，论文列举了6种，1. 直接跳过 2. 严格inline 3. 数据重用时进行Tiling 3. 数据重用且可Fuse时 4. 数据重用但不可Fuse，使用Cache 5. 可Reduce并行
- 如何高效地进行搜索，根据上一步的搜索空间得到的草稿，随机从搜索空间中选取一些点(tile size,并行化外部循环、向量化内部循环、unroll部分内循环)，然后使用cost model进行迭代微调
- 使用xgboost作为cost model，每次选择一个特征空间，然后使用进化算法随机再变异一些基因，进行多次迭代，最后在硬件平台测量实际效果最后更新cost model
- 限制：没有考虑dynamic shape，不支持稀疏计算，只考虑高层级的与硬件无关的代码优化，没有考虑诸如tensor的硬件相关的代码优化

### Relax: Composable Abstractions for End-to-End Dynamic Machine Learning[Arxiv'23][UW]
- 主要解决TVM推导过程中的动态形状问题
- 提出了一种跟踪全局动态Tensor shape关系和调用的程序抽象
- 跨层级的抽象优化，能同时使用tvm本身和其他外部的lib
- 语法声明：相比与以？声明的TVM动态形状，改为以n、m的sym_var()形式声明，这样诸如reshape类的操作可以保留形状信息，提前申请内存，动态形状也是一等公民
- 组合优化：跨层级的动态shape算子融合，减少了跨函数调用
- 组合优化：预先内存分配，通过预留的形状信息，可有效减少内存分配大小
- 组合优化：分步骤地进行lowwer，可以同时使用TensorIR和cutlass

### LightSeq: A High Performance Inference Library for Transformers[NAACL'21][ByteDance]
- 主要针对transformer的优化，有3点贡献
- 1. 将粗粒度的节点融合转化为细粒度的节点融合，以避免频繁的kernel启动，例如手写layer norm kernel可以节省内存启动和保存中间结果。
- 2. 层次的自回归搜索，采用了检索和重排的思想。
- 3. 动态的GPU内存重用方案，将前后依赖的结果存在相同的内存。

### LightSeq2: Accelerated Training for Transformer-based Models on GPUs[SC'22][ByteDance]
- 主要针对transformer的优化，有4点贡献
- 将粗粒度节点转化为手写的细粒度节点并进行融合，具体来讲就是将GEMM部分使用cuBLAS来实现，其他的元素级操作(Dropout, ReLU, Reshape)和约减操作(LayerNorm and Softmax)用手写的kernel来代替，主要对Transformer、Embedding、Criterion、Layer-batched cross Attention层进行了分析。
- 有依赖的约减操作的拆分，对LayerNorm层的梯度计算进行拆分，分别并行计算两部分乘法运算以进行加速。对Softmax层实现了不同的模板，并进行参数调节以适应不同大小和不同形状的softmax计算。
- 加速混合精度梯度更新计算。把所有要更新的参数放到一整片内存区，以避免每次更新的时候都要启动kernel去加载和卸载内存，同时可以节省一些内存。
- 悬空张量的内存管理。具体来讲就是将内存分为永久内存和暂时内存，并将训练参数和更新参数要用的最大内存提前分配好，并进行内存复用，可以节省一部分频繁加载卸载的消耗。
- 整体还是偏工程的工作，作为学术的novelty并不那么fancy，不过对于实现还是有些启发的。

### Bring Your Own Codegen to Deep Learning Compiler[Arivx'21][AWS]
- 为了解决不同模型在不同编译器上的部署问题，提出了一个统一的编译器划分框架
- 首先将编译模型分为Host端和加速器端，Host端调用通用的函数，加速器端则使用抵用依赖的指令
- 执行三步操作对图进行划分：1. 基于pattern的划分模式 2. 对划分好的块进行注释 3. 按照执行量的阈值进行划分 
- 针对加速的设计主要考虑两点：量化和NCHW转换；针对codegen 使用了3种方式，json、c和特定格式；
- 在runtime时对模型输入输出权重进行管理，可以利用内存重用和cache engine的一些方法

### Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems[Arixv'23][CMU]
 - 见论文[LLM_Serving_Survey](LLM_Serving_Survey.md)

### Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation[OSDI'24][MS]
- 基于3点观察：1. 越来越多的量化类型 2. 硬件对量化支持并不丰富 3. 低精度计算并不高效
- 提出了tType和tTile分别表示数据类型和分片，将tTile作为最小的计算单位，可以表示任意位数
- 解耦计算和存储，pipeline分为load、conmpute和store三个阶段，有Slice、Map、Pad、Convert等几种变换
- 引入了更大的搜索空间，1. 根据硬件带宽分配作为提示 2. 使用现有的tvm调度方法 3. 添加变换

### Welder: Scheduling Deep Learning Memory Access via Tile-graph
- 基于几点观察：1. 模型推理瓶颈在内存访问 2. 内存访问间有重用和提高带宽的机会 3. 这种重用机会是可以配置来进行tradeoff的
- 提出了基于tile的内存访问调度方法，首先把相邻间的算子计算，分为了大小不同的tile,然后去遍历不同的fuse模式，找到内存访问最小的那种模式
- 举了Conv+Relu+MaxPooling的例子，加载[4,4,c]到L1中，加载[3,3]到L0中，在L0级别计算Conv+Relu Fuse，在L1时累计[2,2]个元素，然后再调度到L0中进行maxpooling计算

 
## 模型优化
### FastFormers: Highly Efficient Transformer Models for Natural Language Understanding[arxiv'20][MSRA]
- msra文章，但是只是单纯做了模型裁剪、蒸馏和量化，是一篇纯实验结果堆的文章 
- https://github.com/microsoft/fastformers



## OCR 文字识别

### DBNet: Real-time Scene Text Detection with Differentiable Binarization[AAAI'20]


### SVTR: Scene Text Recognition with a Single Visual Model[IJCAI'22][BJTU,Baidu]
- 介绍了场景文本识别的几种主流结构，分别为：1. CNN+RNN 2. CNN + 自回归 3. CNN + MHA fusion 
- 首先将图像经过一个Patch Embedding，通常是CNN，在这里是2个3\*3 stride为2的CBR结构
- 然后经过3个stage，每个stage包含了mixing block和merging的网络结构，最终图像大小从`H\*W\*3 -> 1 \* (W/4) * D_3`
- mixing block使用了global attention结构来提取字符和字符间的特征，使用local attention提取笔画间的特征，merging操作是一个CB块，将高度减半，相应地channel的维度翻倍
- 在paddleocr中，使用了两个mobilenet block提取视觉特征，svtrblock的stage设置为2，后面的GAP被替换成了conv1x1


### General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model[arxiv'24][StepFun]
- 包含了3部分，分别是image encoder层，linear适应层和输出解码层，其中encoder层使用的ViDet模型,解码器使用的Qwen-0.5B模型
- 训练分为3步，第一步训练encoder,使用纯字符识别任务，使用opt-125M进行解码，第二步连接了Qwen-0.5B解码器，使用更大量的数据进行联合训练，第三步冻结encoder,训练decoder以适应细粒度的任务
- 数据输入为1024\*1024\*3,输出的token数量为256\*1024，linear层的权重为1024\*1024,decoder部分与Qwen-0.5B对齐，
- 在预训练encoder阶段，使用5M数据，3M的场景识别和2M的文档识别，3M的场景数据使用paddleocr识别工具，分割为整图级别和单行级别，2M的文档数据使用fitiz提取


### Nougat: Neural Optical Understanding for Academic Documents[ICLR'24][Meta]
- 一个端到端的文档识别模型，使用的encoder和decoder的模式，其中encoder使用的swin transformer,decoder使用的是mbart
- 输入大小为896\*672,Decoder采用的是10层的架构，包含了自注意层和跨注意层，最大长度为4096,所有模型均采用预训练模型
- 数据增强部分：使用了Albumentations库进行增强，在输入patch中加入扰动
- 模型是一页一页进行预测的，预测整篇文档时，会先预测全部，然后使用一个TF-IDF向量机的模型进行模糊分页和fuzzy比对精确分页


### Dount: 

### LayoutLM: