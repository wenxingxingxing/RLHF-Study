# Lesson 17：SFT 有监督微调（Supervised Fine-Tuning）

> **CS336 面试导向学习指南** — 对齐阶段的第一站：把「续写网页」的基座模型，变成「听从指令、给出有用回答」的助手模型。

---

## 一、概念（Concepts）

### 1.1 大模型训练的三阶段全景：Pretraining → SFT → RLHF

在工业界与学术界的常见叙事中，**通用大语言模型**从「能写」到「好用」往往经历三个层次（第三层有时可省略，或用 **DPO / GRPO** 等替代经典 RLHF）：

| 阶段 | 主要数据形态 | 核心目标 | 典型规模与特点 |
|------|----------------|----------|------------------|
| **预训练（Pretraining）** | 大规模无标注文本（网页、书籍、代码等） | 学习语言与世界的统计规律，**下一词预测（NTP）** | 数据量极大、算力消耗最高；模型学会语法、常识与广泛知识 |
| **有监督微调 SFT** | 指令–回答对（多轮对话亦可结构化） | 让模型学会**遵循指令**、**对话格式**与**任务模式** | 数据量远小于预训练，但格式与质量要求高 |
| **偏好对齐 RLHF / DPO / GRPO 等** | 人类偏好、排序、或成对比较；或可验证奖励 | 在 SFT 基础上进一步**符合人类偏好**（有用、诚实、无害等）或优化任务奖励 | 常依赖奖励模型或偏好损失；可与 SFT 迭代 |

**直观理解**：

- **预训练**：模型像读遍图书馆，学会「接龙写文章」。
- **SFT**：用大量「用户怎么说、助手该怎么答」的示范，把行为从「续写」扭转为「按指令完成任务」。
- **RLHF / DPO / GRPO**：在「已经会听指令」的前提下，用偏好信号或可验证奖励细调语气、安全性和任务表现。

三者**不是互斥替代关系**：SFT 往往是对预训练权重的**继续训练**（通常学习率更小、数据更 curated）；RLHF/DPO 则常在 SFT checkpoint 之上进行。**面试常考**：能画出这条流水线，并说明每一阶段的**数据形态、损失函数、与上下游接口**（例如 reference 模型从哪来）。

---

### 1.2 什么是 SFT（Supervised Fine-Tuning）？

**SFT** 指在**有标注的（指令，回答）**数据上，用监督学习（通常是条件语言建模损失）对**已预训练**的模型进行微调。

**核心目标**：

1. **指令遵循（Instruction Following）**：用户给任务描述，模型按要求输出（翻译、摘要、代码、推理步骤等）。
2. **对话与角色**：多轮上下文、系统提示（system）下的稳定行为。
3. **格式与工具占位**（视数据而定）：如 JSON、特定标签、`\boxed{}` 数学答案格式等，为后续工具调用、RAG 或 RL 阶段铺路。

与预训练相比，SFT 更强调 **「谁在说话、要完成什么」** 的结构化交互，而不仅是裸露的文本续写。业界常把这一阶段称为 **Instruction Tuning** 或 **Chat Fine-Tuning**。

---

### 1.3 SFT 与预训练的根本差异

| 维度 | 预训练 | SFT |
|------|--------|-----|
| **数据** | 原始文档流，无显式「指令」边界 | **指令 + 回答**（常含 system / user / assistant 角色） |
| **损失形式** | 对整段文本做 NTP（或经掩码的变体） | 通常 **仅对 assistant 回复部分** 计算 token 级交叉熵（见下文 masking） |
| **目的** | 通用表征与知识 | **行为对齐到任务接口**（instruction-following） |
| **学习率** | 相对较大（量级依规模与 schedule 而定） | **通常更小**，如 `1e-5`～`5e-5`，避免破坏预训练知识 |
| **数据量** | TB 级常见 | 千条到百万条级皆常见，更重 **质量与多样性** |

一句话：**预训练学「语言与知识」，SFT 学「按人类交互方式使用这些知识」。**

---

### 1.4 指令数据格式：System + User + Assistant

#### 三角色结构

- **System**：全局规则、人设、安全策略、输出格式要求（可选但工业界很常用）。
- **User**：用户任务或问题。
- **Assistant**：模型应学习的标准回答（**SFT 的监督信号主要来自这里**）。

多轮对话可重复 user/assistant 轮次；**损失仍通常只打在需要模型生成的部分**（assistant 内容）。

#### Chat 模板与常见格式

不同模型使用不同的 **chat template**（对话模板），把结构化字段渲染成**单一 token 序列**，再送进 Transformer。**训练与推理必须使用同一模板**，否则分布严重偏移。

**（1）ChatML 风格（概念示意）**

每条消息用角色标签包裹，例如：

```text
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
把下面句子翻译成英文：……<|im_end|>
<|im_start|>assistant
Here is the translation: ...<|im_end|>
```

特点：边界清晰，便于解析与 **只对 assistant 段计算 loss**。

**（2）Alpaca 格式（指令微调经典格式）**

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

当无额外 input 时，Input 可省略或置空。该格式**可读性强**，许多开源数据与脚本仍沿用或兼容。

**面试要点**：无论哪种格式，**tokenizer 加 chat template 后的字符串**才是最终训练序列；不同开源模型（Llama、Qwen、Mistral 等）的 special token 与模板不同，**混用模板会导致分布偏移**。

---

### 1.5 指令数据构建：人工标注、Self-Instruct、Evol-Instruct 与数据质量

**（1）人工标注**

- **优点**：质量高、可控性强、可覆盖安全与边界案例。
- **缺点**：成本高、扩展慢。
- **适用**：安全红队样本、高难度推理、品牌话术、合规话术等。

**（2）Self-Instruct**

- 用已有强模型**自举**生成大量「指令 + 回答」，再经规则/模型过滤、去重。
- **优点**：规模化快、成本低。
- **缺点**：分布受 teacher 能力限制，可能放大偏见、幻觉或错误模式。

**（3）Evol-Instruct（及同类演化方法）**

- 对指令进行**演化**：加深难度、增加约束、改写领域或场景，以扩增**多样性**与**难度曲线**。
- **优点**：覆盖更广、难例更多。
- **缺点**：需质量控制，否则噪声与矛盾指令会累积。

**（4）从更强模型蒸馏**

- 用更大/更强的教师模型生成回复，训练较小学生模型。
- **优点**：以小博大，改善小模型表现。
- **缺点**：依赖教师分布；需注意许可与合规。

**（5）数据质量：面试与工程的核心**

- **正确性**：错误答案、自相矛盾会直接教坏模型。
- **多样性**：任务类型、领域、语言、长度、难度需均衡，避免过拟合到单一风格或题型。
- **一致性**：同一任务类型应用统一的输出格式（尤其数学、代码、JSON）。
- 实践中常 **混合**：高质量种子 + 规模化合成 + 规则/模型过滤 + 持续去重。

**（6）常见公开数据集（了解即可）**

| 名称 | 备注 |
|------|------|
| **Stanford Alpaca** | 早期指令微调标杆，格式经典 |
| **ShareGPT** | 用户分享的对话风格数据，多轮多 |
| **OpenAssistant** | 众包对话与质量信号 |
| **LIMA** | 强调**少量高质量**指令数据也能对齐得很好 |

---

### 1.6 训练细节：对哪些 token 算 loss、Padding 与 Packing

#### 仅对 Assistant 回复计算损失（Loss Masking）

在 SFT 中，标准做法是：**仅对 assistant 回复（及多轮里模型应生成的部分）的 token 参与交叉熵**，对 system、user、以及模板中的固定前缀 token **mask 掉 loss**（常见实现：`labels` 在这些位置设为 `-100` 或等价 `ignore_index`）。

**原因简述**：

1. **训练目标对齐**：要学的是「在给定上文条件下如何**生成**回答」，而不是拟合用户问题的 token 分布。
2. **梯度效率**：避免在用户措辞上过拟合，把容量用在「如何答」上。
3. **与推理一致**：推理时模型只看到前文，不会「预测用户下一句」。

**数学上**，若 \(m_t \in \{0,1\}\) 表示位置 \(t\) 是否参与监督，常写作：

\[
\mathcal{L}_{\text{SFT}} = - \frac{1}{\sum_t m_t} \sum_{t} m_t \log p_\theta(x_t \mid x_{<t})
\]

实现上需注意：**多轮对话**中每一轮 assistant 段都要计入 loss；若使用工具调用等特殊格式，团队需统一规则（哪些 token 算模型责任）。

#### Padding

- 同一 batch 内序列长度不同，需 **padding** 到 `max_length`（或按 batch 内最长序列动态 pad）。
- **关键点**：padding 位置的 `labels` 必须设为 `ignore_index`，**不参与 loss**；attention mask 需屏蔽 pad token，避免注意力关注到无效位置。
- **标签与 logits 对齐**：Causal LM 通常对 `logits[..., :-1, :]` 与 `labels[..., 1:]` 做移位，mask 需同步移位。

#### Sequence Packing（序列打包）

- 将多条短样本**拼进同一最大长度窗口**，用 **attention mask**（或 FlashAttention 的 varlen / cu_seqlens）隔离不同样本，减少 padding 浪费，提高 GPU 吞吐。
- **必须正确处理**：**position id**（常按段重置）、**样本间不可互看**（否则标签泄漏）、以及 **每条样本仅在自身 assistant 段累计 loss**。
- 工业训练（如部分 Llama 系 recipe）广泛使用 packing；作业实现时需对照官方对 mask 的单元测试。

---

### 1.7 参数高效微调：LoRA 与 QLoRA

#### LoRA 数学：\(W = W_0 + BA\)，秩 \(r\) 与 \(\alpha\) 缩放

对某线性层原权重 \(W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\)（实现中常等价讨论转置），LoRA **冻结** \(W_0\)，仅训练低秩增量：

\[
W = W_0 + \Delta W,\quad \Delta W = B A
\]

其中 \(B \in \mathbb{R}^{d_{\text{out}} \times r}\)，\(A \in \mathbb{R}^{r \times d_{\text{in}}}\)，**秩 \(r \ll \min(d_{\text{in}}, d_{\text{out}})\)**。

前向（以输入 \(x\) 为例，忽略 bias）：

\[
y = W_0 x + \frac{\alpha}{r} \cdot B A x
\]

**\(\alpha\)** 为 LoRA 缩放超参（与 \(r\) 常一起调）：\(\alpha/r\) 使在改变 \(r\) 时保持**更新幅度的大致可比性**（不同框架命名可能为 `lora_alpha`，实现细节以所用库为准）。

**Rank \(r\)**：越大容量越大，可训练参数约 \(r(d_{\text{in}}+d_{\text{out}})\)；过大可能过拟合，常见 8、16、32、64。

**施加在哪些层**：常见对 **注意力层的 \(W_q, W_k, W_v, W_o\)**（及有时 FFN）加 LoRA；**全层 LoRA** 更强但更贵。面试可答：**先 attention，再视任务扩到 FFN**。

#### 为什么 LoRA 往往有效：低秩与内在维度

**直观解释**：大量经验表明，**特定任务上的有效权重更新**往往落在**低维子空间**内——即「微调需要的方向」不必填满整个高维权重矩阵。用 \(BA\) 低秩分解，用较少参数近似该子空间中的主要更新方向，从而**省显存、省存储、减轻灾难性遗忘**（相对全参而言）。

**补充**：这与「**内在维度（intrinsic dimension）**」相关文献一致：许多下游适配可用远小于全参的自由度描述。**并非**声称所有能力都低秩，而是**任务相关的偏移**常可低秩近似。

#### QLoRA：4-bit 量化 + LoRA

**QLoRA**（典型实现：bitsandbytes + PEFT）将**基座权重以 4-bit 量化**（如 **NF4** 数据类型 + **双量化**进一步压存储）加载到显存，**前向/反向中按需反量化**参与计算；**LoRA 适配器**仍以 FP16/BF16 等较高精度训练。

**优势**：

- **显存**：显著降低，使单卡或多卡上微调更大模型成为可能。
- **效果**：在不少设置下接近 **全精度 LoRA** 或全参微调（依任务与实现而定）。

**注意**：需关注量化内核、梯度稳定性、与不同 GPU 的兼容性；超参（如 `r`、`alpha`、学习率）可能需略调。

---

### 1.8 全量微调 vs LoRA vs QLoRA 对比

| 维度 | Full Fine-Tuning | LoRA | QLoRA |
|------|------------------|------|-------|
| **更新对象** | 全部权重 | 冻结 \(W_0\)，训 \(A,B\) | 同 LoRA，基座 4-bit |
| **显存 / 优化器** | 最高（全参 Adam 状态） | 较低 | **最低**（基座量化） |
| **表达能力上限** | 最高 | 受 \(r\) 与层选择限制 | 同 LoRA（数值上受量化影响） |
| **Checkpoint** | 全量大文件 | 小适配器权重 | 小适配器 + 可选合并脚本 |
| **灾难性遗忘** | 相对更易「改写」基座 | 通常较轻 | 通常较轻 |
| **典型场景** | 数据足、需深度改基座 | 默认 PEFT、多任务多适配器 | **单卡大模型**、资源紧 |

**选型口诀**：资源紧、多租户适配器 → **LoRA/QLoRA**；数据极大且需重塑广泛行为 → 考虑 **Full** 或 **更大 r + 更多层 LoRA**；**QLoRA** 优先在显存硬约束下使用。

---

### 1.9 灾难性遗忘（Catastrophic Forgetting）与缓解策略

**含义**：在下游任务或窄分布 SFT 数据上训练后，模型在**未在该阶段充分覆盖的任务或分布**上性能明显下降（例如通用知识、其他语种、代码能力）。

**缓解思路**：

1. **混合数据**：SFT 中保留一定比例 **通用指令 / 预训练风格** 数据，维持广度。
2. **较小学习率、较少 epoch**：减轻对基座的大幅偏移。
3. **正则与约束**：RLHF/DPO 中常见的 **KL 到 reference**（常为 SFT 模型）；纯 SFT 也可从直觉上理解「别偏离原模型太远」。
4. **PEFT**：只动少量参数，基座知识相对保留更好（非绝对）。
5. **多阶段 / 回放**：重要任务数据周期性回放；或分阶段先宽后窄。

---

### 1.10 SFT 评估：MMLU、HumanEval、MT-Bench

SFT 质量**不能**只看训练 loss，需**多维基准**（与业务任务对齐）：

| 基准 | 测什么 | 备注 |
|------|--------|------|
| **MMLU** | 57 个学科的多选题**知识与推理** | 考察广度与「像考试」的闭卷能力；SFT 后常提升指令格式下的表现，但需注意与预训练知识重叠 |
| **HumanEval** | **Python 代码**从 docstring 补全 | 测代码能力；对是否混入代码数据敏感 |
| **MBPP** 等 | 基础 Python 题 | 与 HumanEval 互补 |
| **MT-Bench** | **多轮对话**、多任务，强模型作裁判打分 | 贴近 chat 体验；注意裁判偏差与版本 |
| **AlpacaEval / Arena** | 与强基线对比胜率或 Elo | 指令跟随与风格 |

**面试表述**：**MMLU / HumanEval** 偏**客观任务**；**MT-Bench** 偏**对话综合**；上线前常辅以**人工评估**与**线上 A/B**。避免单一排行榜过拟合。

---

### 1.11 CS336 Assignment 5 中的 SFT 组件

Stanford **CS336 Assignment 5（Alignment）** 在课程叙事中把 **SFT、RL（如 GRPO）、可选 DPO** 串成对齐链路。就 **SFT 子任务** 而言，与 [Lesson 19：Assignment 5 对齐实战](./19-Assignment5对齐实战.md) 一致，通常包括：

1. **数据**：数学推理等场景下的 **instruction–response**（常含 **思维链 CoT** 与可解析答案格式，如 `\boxed{}`）。
2. **损失**：标准 **Causal LM 交叉熵**，**仅对 assistant 完成部分** 累计；`labels` 在 user/system/padding 处 **ignore**。
3. **训练**：学习率、epoch、精度（BF16 等）、梯度裁剪；可选 **LoRA/QLoRA** 以降低资源占用。
4. **接口**：SFT 产出的 checkpoint 常作为 **RL 阶段的初始策略** 与 **冻结的 reference 模型** \(\pi_{\text{ref}}\)，用于 **KL 惩罚** 或优势基线。

**公式对齐**（与 Assignment 5 文档一致）：

\[
\mathcal{L}_{\text{SFT}} = - \frac{1}{\sum_t m_t} \sum_{t} m_t \log p_\theta(x_t \mid x_{<t})
\]

其中 \(m_t\) 仅在 **模型应生成的 token** 上为 1。具体文件名与测试以**当年官方仓库**为准。

---

## 二、代码（Code）

### 2.1 使用 Chat Template 构造训练样本

以下展示 **Hugging Face Transformers** 常见用法思路（具体 API 随版本略有差异，以文档为准）：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model-name", trust_remote_code=True)

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "用三句话解释什么是 LoRA。"},
    {"role": "assistant", "content": "LoRA 是一种参数高效微调方法……"},
]

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,  # 若 tokenizer 支持
)

input_ids = encoded["input_ids"]
labels = [
    tid if m else -100
    for tid, m in zip(input_ids, encoded.get("assistant_tokens_mask", [False] * len(input_ids)))
]
```

若 `assistant_tokens_mask` 不可用，则需**手动**根据模板中 assistant 起始 special token 位置切分并构造 mask。

### 2.2 只对 response 求交叉熵（PyTorch）

```python
import torch
import torch.nn.functional as F

def masked_ce_loss(logits, labels, ignore_index=-100):
    # logits: (B, T, V), labels: (B, T)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

**要点**：`labels` 在 user/system/pad 段为 `ignore_index`，**仅 assistant 段**为真实 token id；与 **causal LM 的移位**对齐。

### 2.3 LoRA 线性层（教学用极简实现）

```python
import torch.nn as nn
import torch

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x, base_linear):
        return base_linear(x) + self.scaling * self.lora_b(self.lora_a(x))
```

生产环境应使用 **`peft`** 或框架内置 LoRA，以正确处理保存、合并与推理。

### 2.4 完整 SFT 训练示例（Hugging Face Trainer + PEFT LoRA）

下面给出一条可改造的**端到端骨架**：**加载模型 → LoRA → 数据集 map → Trainer**。依赖：`transformers`, `datasets`, `peft`, `torch`。

```python
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # 示例；按权限与显存替换

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

raw = [
    {
        "messages": [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "2"},
        ]
    },
]

def preprocess(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(text, max_length=512, truncation=True)
    input_ids = enc["input_ids"]
    # 简化：若 tokenizer 支持 assistant mask，应在此填 labels；否则用手动区间
    labels = input_ids.copy()
    # 占位：真实项目必须用 assistant_tokens_mask 或定位 assistant 起止
    enc["labels"] = labels
    return enc

ds = Dataset.from_list(raw)
ds = ds.map(preprocess, remove_columns=["messages"])

args = TrainingArguments(
    output_dir="./sft-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,  # LoRA 常用略高于全参 SFT；全参常 1e-5~5e-5
    bf16=True,
    logging_steps=10,
    save_steps=200,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
)

trainer.train()
model.save_pretrained("./sft-lora-adapter")
tokenizer.save_pretrained("./sft-lora-adapter")
```

**说明**：

- **`labels` 构造**：上例为骨架；**生产级**必须用 `return_assistant_tokens_mask=True` 或解析模板，将非 assistant 位置置 `-100`。
- **QLoRA**：将 `from_pretrained` 换为 `BitsAndBytesConfig` 加载 4-bit，其余 LoRA 类似（见 `peft` 与 `bitsandbytes` 文档）。
- **学习率**：LoRA 有时用 `1e-4`～`3e-4`；**全参 SFT** 更保守；以验证集为准。

---

## 三、面试要点（Interview points）

1. **三阶段**：预训练 → SFT（指令监督）→ RLHF/DPO/GRPO（偏好或奖励）；各阶段**数据形态、损失、参考模型角色**不同。
2. **SFT 目标**：指令遵循与对话行为，不是裸续写；常称 **instruction tuning**。
3. **格式**：ChatML / Alpaca 等；**训练与推理同一 chat template**。
4. **Loss**：**只对 assistant 生成段**做 NTP；system/user/pad **mask**。
5. **Padding / Packing**：pad 不参与 loss；packing 需防跨样本注意力与错误 position。
6. **数据**：人工 + Self-Instruct + Evol-Instruct + 蒸馏；**质量、多样性、一致性**。
7. **LoRA**：\(W=W_0+BA\)，\(\alpha/r\) 缩放，\(r\) 控容量；**低秩有效**与任务子空间直觉。
8. **QLoRA**：4-bit 基座 + LoRA 适配器，**省显存**。
9. **遗忘**：混合数据、小 LR、少 epoch、PEFT、KL、回放。
10. **评测**：**MMLU**（知识）、**HumanEval**（代码）、**MT-Bench**（多轮对话）+ 人工。
11. **Assignment 5**：SFT 提供 **policy 初始化** 与 **reference**；**loss masking** 必会考。
12. **LIMA**：少量高质量数据可对齐得很好 — **策展**重要性。

---

## 四、面试高频题（详解 10+ 道）

### Q1：SFT 和预训练的区别？

**答**：**数据与目标不同**。预训练用海量无标注文本做**下一词预测**，学通用语言与知识；SFT 用 **（指令，回答）** 或对话形式的数据，把行为对齐到**遵循指令、按角色输出**。**损失上**，SFT 常只对 **assistant 回复** 计交叉熵，而非整段文档。**超参上**，SFT 学习率通常更小、轮数更少（全参场景），以免破坏预训练能力。两者是同一套 Transformer 架构上的**不同阶段**。

---

### Q2：指令数据如何构建？

**答**：常见组合包括：**（1）人工标注** — 高质量、高成本；**（2）Self-Instruct** — 强模型自举再过滤；**（3）Evol-Instruct** — 演化增难与增广；**（4）蒸馏** — 教师生成伪标签。工程上要做 **去重、毒性过滤、长度与难度分层、多语言与多任务混合**。**核心原则**：宁可少一些，也要避免系统性错误与单一风格占主导。

---

### Q3：为什么只对 response 部分计算 loss？

**答**：监督信号要教的是：**在给定 system/user 上下文后，如何生成正确 assistant 回复**。对用户问题 token 算 loss 会迫使模型拟合「用户会怎么说」，与目标不符。**多轮**中每一轮 assistant 都应计入。实现上用 **labels mask**（`-100`）忽略非生成段。

---

### Q4：LoRA 的公式是什么？\(\alpha\) 和 \(r\) 起什么作用？

**答**：\(\Delta W = BA\)，\(B\in\mathbb{R}^{d_{\text{out}}\times r}\)，\(A\in\mathbb{R}^{r\times d_{\text{in}}}\)，前向常写 \(y = W_0 x + \frac{\alpha}{r} BAx\)。**\(r\)** 控制秩与容量；**\(\alpha\)** 与 **\(\alpha/r\)** 调节 LoRA 分支幅度，便于在改变 \(r\) 时保持尺度可比。实际常用 **PEFT** 实现，超参需在小验证集上扫。

---

### Q5：为什么说权重更新具有低秩性？LoRA 为什么有效？

**答**：经验与「内在维度」研究表明，许多**任务特定微调**的有效更新可集中在**低维子空间**，不必填满整个权重矩阵。LoRA 用 \(BA\) **参数化该子空间中的主要方向**，从而**大幅减少可训练参数与显存**，并常减轻对基座的全局改写。**注意**：不是断言所有现象都低秩，而是**适配偏移**常可低秩近似。

---

### Q6：QLoRA 是什么？相比 LoRA 多做了什么？

**答**：**QLoRA** 将基座权重以 **4-bit（如 NF4）** 加载，显著降低显存；**LoRA 适配器**仍以浮点训练。相比 LoRA，多的是**量化加载与反量化计算**；优势是**同等硬件可训更大模型或更大 batch**。需关注实现细节与数值稳定性。

---

### Q7：全量微调、LoRA、QLoRA 怎么选？

**答**：**全量**：数据足、需深度改行为、资源够。**LoRA**：默认 PEFT、多任务多适配器、快速迭代。**QLoRA**：显存硬约束下微调大模型。面试可补一句：**评测集上对比**遗忘与任务分，再定案。

---

### Q8：什么是灾难性遗忘？SFT 中如何缓解？

**答**：在新数据上训练后，**旧分布或通用能力**下降。**缓解**：混合通用数据、小 LR、少 epoch、PEFT、RL 中 KL 锚定 reference、回放等。

---

### Q9：Padding 和 Packing 在 SFT 里分别要注意什么？

**答**：**Padding**：pad 位置 **labels 为 ignore**，attention 屏蔽 pad。**Packing**：多段拼一条时 **不能跨段注意力**，position 与 **loss 分段**必须正确，否则泄漏或错梯度。

---

### Q10：如何用 MMLU、HumanEval、MT-Bench 评价 SFT？

**答**：**MMLU** 看多学科知识与推理；**HumanEval** 看代码补全；**MT-Bench** 看多轮对话综合体验。三者侧重不同，应**组合**看，并结合业务人工评估。

---

### Q11：SFT 的学习率为什么通常比预训练小？LoRA 为何有时更大？

**答**：SFT 在强基座上做**局部修正**，过大 LR 易**遗忘**与过拟合指令集。**LoRA** 只训少量参数，有效步长分布不同，实践中常见 **略高于全参 SFT** 的 LR，但仍需**验证集**与梯度稳定性。

---

### Q12：LIMA 对数据策略有什么启示？

**答**：**少量、高质量、多样化**的指令数据也能得到强指令跟随，强调**策展**与覆盖关键能力，而非盲目堆量（具体以论文实验为准）。

---

### Q13：CS336 Assignment 5 里 SFT 和后面 RL（如 GRPO）如何衔接？

**答**：SFT 提供**会按格式输出**的初始策略，并常作为 **reference**；RL 阶段用 **KL** 约束偏离，避免为刷奖励而崩坏。数据上数学场景常含 **CoT 与可验证答案格式**，与 **规则奖励** 对接。

---

## 五、练习（Practice）

1. **模板一致性**：对同一条多轮对话分别用 **Alpaca 手写拼接** 与 **`apply_chat_template`**，对比 token 序列与 **assistant 区间**，思考对 loss 的影响。
2. **Mask 实现**：不使用 `assistant_tokens_mask` 时，用 special token 位置**手动**构造 `labels`，小批量验证 `ignore_index`。
3. **LoRA 消融**：固定数据与 epoch，扫 **rank ∈ {4,8,16,32}** 与 **alpha**，记录验证 loss 与小型指令集评分。
4. **QLoRA 对照**：在单卡上对比 **bf16 LoRA** 与 **4-bit QLoRA** 的峰值显存与下游 50 条样例表现。
5. **遗忘粗测**：SFT 前后在同一 **通用知识问答集**上评测；尝试混入 10% 通用指令数据是否缓解掉点。
6. **评测脚本**：各跑一次 **MMLU 子集 / HumanEval / MT-Bench**（或官方子集），记录 SFT 前后变化（资源不足可缩小规模并注明）。
7. **阅读**：LIMA、Self-Instruct、Evol-Instruct 的摘要各一页，写出各自**适用边界**。
8. **（CS336）**：阅读 [Assignment 5 对齐实战](./19-Assignment5对齐实战.md)，标出 SFT 阶段张量形状与 **reference model** 在 GRPO/DPO 中的用法。

---

## 六、导航（Navigation）

| 上一课 | 下一课 |
|--------|--------|
| [16-Assignment3-4实战指南.md](./16-Assignment3-4实战指南.md) | [18-RLHF-DPO-GRPO对齐技术.md](./18-RLHF-DPO-GRPO对齐技术.md) |

---

**本节小结**：SFT 是把预训练模型变成「听得懂指令的助手」的关键一步；**模板、loss mask、padding/packing、数据质量、LoRA/QLoRA 与评测（MMLU / HumanEval / MT-Bench）** 是面试与工程中的反复考点。完成本节后，建议进入 **Lesson 18** 学习 RLHF/DPO/GRPO，并结合 **Lesson 19** 完成 Assignment 5 的端到端对齐实践。

*文档版本：Lesson 17 · SFT 有监督微调 · 与 CS336 对齐叙事及本仓库 Assignment 5 文档一致；作业细则以当年官方 PDF 为准。*
