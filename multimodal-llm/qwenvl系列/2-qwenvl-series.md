

# 1.引言

在梳理Qwen-VL系列前，先以更通用的视角简单了解下基于LLM的多模态模型的设计框架，方便我们先了解下业界通用的做法，也了解一些基本的概念。


在MM-LLMs综述一文中，总结了多模态大语言模型的通用模型框架和每个模块的一些实现方法，如图1 所示：
![alt text](./assets/2/1.png)

从上图中可以看到，在通用的MM-LLM（Multi-Modality LLM）框架里，共有五个模块，整体以LLM为核心主干，分别在前后有一个输入、输出的投影模块（Projector），投影模块主要是用于桥接不同模态输入和输出。输入投影模块（Input Projector）用于将模态编码器处理的不同模态特征映射到文本特征空间，以便输入给LLM；输出投影模块（Output Projector）用于将文本特征空间结果映射到模态生成器的输入空间，以引导模态生成器生成多模态结果。五个模块按数据流顺序，具体描述如下：

- **模态编码器（Modality Encoder）**：将多模态的数据编码成向量空间特征，该模块通常是单独进行预训练的，典型的方法有基于CNN的ResNET，基于Transformer的ViT等。
- **输入投影层（Input Projector）**：将模态编码器的输出映射到LLM的输入特征空间的适配层，一般模型结构比较简单，不同的多模态模型一般是随机初始化该模块的参数做冷启训练。典型的网络层：MLP，Cross-Attention等
- **LLM主干网络（LLM Backbone**）：LLM是经过预训练的模型，一般还要串联多个模块继续做Post-Pretrain和微调，使得模型能识别多模态的特殊token和多模态的特征输入。
- **输出投影层（Output Projector）**：将LLM生成的数据，映射成Modality - Generator 可理解的特征空间，一般是简单的Transformer层或MLP层。
- **模态生成器（Modality Generator）**：多模态的生成器，最终输出多模态的结果如图像、语音、视频等。模型基本都是基于LDM（Latent Diffusion Models）的衍生模型，如图片领域的Stable Diffusion方法。

以上介绍完通用的MM-LLM的框架。本文梳理的Qwen-VL模型是一系列视觉+文本多模态理解模型，即LVLM(Large-scale Vision-Language Model)，主要处理文本和视觉特征，输入Text、Image、Video，输出Text。

## 二、QWEN-VL

Qwen-VL 是以Qwen-7B Base为主干模型，通过引入视觉感知器（Visual receptor）来增强视觉特征的感知能力。视觉感知器包括一个跟语言模型对齐视觉编码器（visual encoder）和一个位置感知的适配器（position- aware adapter）。套用上面的通用多模态框架，Qwen-VL包括了典型的前3个模块：

- **模态编码器（Modality Encoder）** ： 视觉编码器（visual encoder），只用来编码图片视觉特征
- **输入投影层（Input Projector）**：位置感知的适配器（position-aware adapter）
- **LLM主干网络（LLM Backbone）**： Qwen-7B Base 模型

### 2.1. 视觉编码器（Visual Encoder）
Qwen-VL的视觉编码器使用的是ViT架构（Vision Transformer），ViT的网络设置和初始化参数使用了OpenCLIP预训练好的ViT-bigG模型。

>OpenCLIP是laion.ai组织的一个开源项目，是对OpenAI's的CLIP（Contrastive Language-image Pre-training）的开源实现。laion.ai发布了一系列基于CLIP框架训练的不同size模型，同时他也为CV领域贡献了大量的开源数据，ViT-bigG是经过了2B的训练数据训出来的ViT模型。

Qwen-VL使用的ViT(ViT-bigG)是基于CLIP框架训练的，CLIP是通过**Contrastive Learning**的方式来学习Vision和文本的表征。

如下图（左图）所示，对于一个Batch的数据，以样本集中原始图文pair $<I_i, T_i>$为正例pair，Batch内与其他样本的$I_x, T_x$组成为负例pair：$<I_i, T_x>, <I_x, T_i>$其中$i ≠ x$ 。模型训练采用了对比损失函数，通过最大化正例Pair的相似度，同时最小化负例Pair的相似度来训练模型。通过这种方式，能学习到视觉特征和文本特征的对齐关系。最后将训练好的Image Encoder模型(即ViT)参数保存下来，以供其他下游任务热启使用。

![alt text](image.png)


在Qwen-VL中采用的是标准的ViT框架，ViT的原理比较简单：将图片分割成多个图像块（Patch），然后针对每个Patch通过线性映射转化成token，再将所有token拼接成序列，最终将一张图片从$(H, W, C)$格式转换成$(S, H)$，格式的序列特征。

在标准的ViT实现上，输入图片会先被调整成长宽比1：1的正方形，然后再分割成固定的图像块。

因此这种标准的ViT框架的设计，只能接收固定分辨率的图片，同时Patch的大小也是模型在训练期间使用的一个固定size。ViT处理过程如图3所示：

![alt text](image-1.png)

ViT核心处理就几行代码，如下：

```python
class VisionTransformer(nn.Module):
   def __init__(...):
       self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

   def forward(self, x: torch.Tensor):
        # 注释1：通过卷积核将一张图片从[H，W，C]=[448, 448, 3] 映射成 [width, grid, grid] = [1664, 32, 32]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # 注释2：一张图片按行展开，[width, grid, grid] 映射成 [grid * grid, width]二维序列
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 注释3：增加位置编码输入transformer模型
        x = x + get_abs_pos(self.positional_embedding, x.size(1))
        x = self.transformer(x)
```

代码注释1的处理过程：一张图片做卷积操作，处理成 [width, grid, grid] = [1664, 32, 32]的数据，如图4所示。
![alt text](image-2.png)

代码注释2的处理过程：按行优先展开，处理成一个二维格式的数据[sequence_len, hidden_size] = [1024, 1664]（类似与一条文本处理后的序列）。如图下5所示。
![alt text](image-3.png)