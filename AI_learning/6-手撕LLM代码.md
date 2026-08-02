

# 1. 写 top-p 和 top-k

Top-p 和 Top-k 采样

概念讲解：

在自回归文本生成中，模型每一步会输出一个概率分布（logits 经过 softmax），我们需要从中采样下一个 token。直接使用整个词汇表采样（即 temperature 缩放后的随机采样）可能导致生成低概率 token，使结果不连贯。Top-k 采样 和 Top-p 采样 是两种常用的截断采样方法，用于限制候选 token 集合，提高生成质量。

- Top-k 采样：
  - 做法：只保留概率最高的 k 个词，把剩下的词概率强制设为 0，然后重新归一化（让剩下的概率和为 1），再从中采样。
  - 作用：直接砍掉长尾的低概率词，防止生成生僻字或乱码。
  - 缺点：k 是固定的。如果模型很自信（某个词概率 90%），k 太大也会采样到噪音；如果模型很犹豫（概率很平），k 太小会限制多样性。
- Top-p (Nucleus) 采样：
  - 做法：将词按概率从大到小排序，依次累加概率，直到累加和超过 p (比如 0.9)。保留这些词，剩下的截断，重新归一化，再采样。
  - 作用：动态调整候选词数量。模型自信时候选词少，模型犹豫时候选词多。
  - 现状：目前 LLM 推理中，Top-p 比 Top-k 更常用，或者两者结合。
两种方法可以结合使用（如先取 top-k 再取 top-p），但通常分别实现。

```python
import torch
import torch.nn.functional as F

def custom_softmax_torch(logits, dim=-1):
    """
    数值稳定的 PyTorch 版 Softmax
    """
    # 1. 找到指定维度上的最大值
    # keepdim=True 极其重要：为了保持维度数不变，否则后续减法无法触发自动广播机制
    max_logits, _ = torch.max(logits, dim=dim, keepdim=True)

    # 2. 减去最大值，然后求指数 (防止溢出)
    exp_logits = torch.exp(logits - max_logits)

    # 3. 求和
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)

    # 4. 归一化得出概率分布
    return exp_logits / sum_exp

def top_k_top_p_sampling(logits, top_k=0, top_p=1.0, temperature=1.0):
    """
    手撕 Top-K 和 Top-P 采样策略
    
    参数:
        logits: shape为 [batch_size, vocab_size] 的未归一化张量
        top_k: int, 保留概率最高的前K个词 (0表示不启用)
        top_p: float, 保留累加概率达到P的词集合 (1.0表示不启用)
        temperature: float, 温度系数，控制分布的平滑度
    返回:
        sampled_tokens: 采样出的 token 索引，shape 为 [batch_size, 1]
    """
    # 1. 应用温度系数 (Temperature scaling)
    # T > 1 会让分布更平滑(更随机)；T < 1 会让分布更尖锐(更确定)
    logits = logits / temperature
    
    filter_value = -float('Inf')
    
    # 2. Top-K 过滤
    if top_k > 0:
        # torch.topk 会返回 (最大值组成的tensor, 对应的索引)
        # 注意：这里限制 top_k 不能超过词表大小
        top_k = min(top_k, logits.size(-1)) 
        
        top_k_values, _ = torch.topk(logits, top_k)
        # 取出每行中第 k 大的值，也就是保留名单中的底线值
        min_values = top_k_values[:, -1].unsqueeze(1)
        
        # 将小于这个底线值的 logits 替换为 -inf
        logits = torch.where(logits < min_values, 
                             torch.full_like(logits, filter_value), 
                             logits)

    # 3. Top-P (核采样) 过滤
    if top_p < 1.0:
        # 将 logits 降序排列
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # 用 softmax 算出当前的概率，并计算累加概率 (cumsum)
        # cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cumulative_probs = torch.cumsum(custom_softmax_torch(sorted_logits), -1)

        # 找出累加概率超过 p 的位置，标记为 True (准备移除)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 【核心细节】：我们需要保留刚好把累加概率顶过 p 的那个 token
        # 比如 top_p=0.9，累加到第3个词是 0.88，第4个词是 0.92。第4个词必须保留。
        # 做法是将掩码(mask)整体向右平移一位
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # 永远确保第一个 token 不会被屏蔽 (防止把所有词都过滤掉了)
        sorted_indices_to_remove[..., 0] = 0

        # 将排好序的掩码 (mask) 还原回原始 logits 的对应位置 (使用 scatter)
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        # 把要移除的位置设为 -inf
        logits = logits.masked_fill(indices_to_remove, filter_value)

    # 4. 采样
    # 经过前面设为 -inf 的操作，softmax 后被过滤的词概率会变成 0
    # probs = F.softmax(logits, dim=-1)
    
    
    # 按照计算出的概率分布，通过多项式分布 (multinomial) 抽取 1 个 token
    sampled_tokens = torch.multinomial(probs, num_samples=1)
    
    return sampled_tokens


# ==========================================
# 测试用例
# ==========================================
if __name__ == "__main__":
    # 假设 batch_size=1, vocab_size=6
    # 故意设置几个明显的高分和几个低分
    mock_logits = torch.tensor([[10.0, 9.0, 8.0, 2.0, 1.0, 0.5]])
    
    print("原始 Logits:", mock_logits)
    print("原始概率分布:", F.softmax(mock_logits, dim=-1))
    
    # 测试单纯的 Top-K (K=2)
    # 期望: 只有 10.0 和 9.0 有概率，其他变成 0
    token_k = top_k_top_p_sampling(mock_logits.clone(), top_k=2)
    print("\nTop-K 采样结果:", token_k)

    # 测试单纯的 Top-P (P=0.9)
    token_p = top_k_top_p_sampling(mock_logits.clone(), top_p=0.9)
    print("Top-P 采样结果:", token_p)
    
    # 实际应用中，通常两个连用：先选出前K个，再在这个小圈子里做Top-P
    token_kp = top_k_top_p_sampling(mock_logits.clone(), top_k=50, top_p=0.8)
```


# 2. 写 LayerNorm 和 RMSNorm。

概念讲解：

- Layer Normalization (LayerNorm)：对每个样本的每个特征层进行归一化。给定输入 x 形状 (batch, seq_len, hidden_size)，对最后一个维度（hidden_size）计算均值和方差，然后标准化：(x - mean) / sqrt(var + eps)，再乘以可学习的缩放参数 gamma 并加上偏移 beta。LayerNorm 广泛用于 Transformer，稳定训练。

- RMSNorm：Root Mean Square Layer Normalization 是 LayerNorm 的一个简化变体。它假设减去均值不是必需的，只使用均方根 (RMS) 进行归一化：x / RMS(x)，其中 RMS(x) = sqrt(mean(x^2) + eps)。同样乘以可学习的缩放参数 gamma，但没有 beta。RMSNorm 计算量更小，且在实验中性能与 LayerNorm 相当。

```python
import torch
import torch.nn as nn
 
class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    公式: y = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size) 或 (batch, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        
        # 1. 计算均值，keepdim=True 为了后续广播相减
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. 计算方差 (有偏估计，除以 N)
        # 注意：这里直接手动算，比调 API 更加无懈可击
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
        # 3. 归一化
        # 使用 torch.rsqrt 计算 1/sqrt(...) 会比先 sqrt 再取倒数更快且数值更稳定
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        
        # 4. 仿射变换
        return self.weight * x_norm + self.bias
    
class RMSNorm(nn.Module):
    """
    RMS 归一化 (Root Mean Square Layer Normalization)
    公式: y = gamma * x / RMS(x), 其中 RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size) 或 (batch, hidden_size)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm

class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 只有缩放参数 gamma，没有 beta
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        
        # 【加分细节】: 为了数值稳定性，将输入转为 fp32 计算平方和，防止溢出
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        
        # 1. 计算均方根的平方 (即 X^2 的均值)
        rms_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
        
        # 2. 计算 rsqrt 并归一化，然后转换回原数据类型
        x_normed = (x_f32 * torch.rsqrt(rms_sq + self.eps)).to(input_dtype)
        
        # 3. 乘以缩放权重
        return self.weight * x_normed
 
 
# 示例用法
if __name__ == "__main__":
    batch, seq, hidden = 2, 3, 512
    x = torch.randn(batch, seq, hidden)
 
    ln = LayerNorm(hidden)
    rms = RMSNorm(hidden)
    
    official_ln = nn.LayerNorm(512)
    print("LayerNorm 差异:", torch.max(torch.abs(ln(x) - official_ln(x))).item()) # 趋于0
 
    out_ln = ln(x)
    out_rms = rms(x)
 
    print("LayerNorm 输出形状:", out_ln.shape)
    print("RMSNorm 输出形状:", out_rms.shape)
```


# 3. 用 Torch 写 SFT 的 Loss 计算代码（注意 Shift Right）。

概念讲解：

在监督微调（Supervised Fine-Tuning, SFT）中，模型以自回归方式预测下一个 token。给定输入序列 input_ids，我们需要计算损失，使得模型预测的每个位置的 token 与真实的下一个 token 一致。

具体的，我们用第 1 到 t 个 token 预测第 t+1 个 token。因此：

- 输入 (input_ids)：[BOS] 我 喜欢 深度 学习
- 标签 (labels)：我 喜欢 深度 学习 [EOS]
即 labels 是 input_ids 左移一位（或说 input 右移一位对齐 labels）。代码实现中通常用 input_ids[:, :-1] 作为输入，input_ids[:, 1:] 作为预测目标。

实现 SFT loss 的关键操作是 Shift Right：

- 将 labels 设置为与 input_ids 相同（或者指定忽略的位置为 -100）。
- 在计算损失时，取 logits[..., :-1, :] 和 labels[..., 1:] 进行对齐。
- 或者，在输入模型时，将 input_ids 作为输入，labels 作为目标，模型内部可能会自动处理 shift（如 HuggingFace 的 transformers 库）。但手动实现时，我们需要显式 shift。

另外，通常会将填充部分（padding）的损失忽略，通过在 labels 中将填充 token 对应的位置设为 -100（因为 CrossEntropyLoss 默认忽略 -100 的目标）。

<details>
<summary>SFT Loss实现</summary>

```python
import torch
import torch.nn as nn
 
def sft_loss(logits, labels, ignore_index=-100):
    """
    计算 SFT 的交叉熵损失，自动处理 shift right 和忽略 padding。
    参数:
        logits: 模型输出的 logits，形状 (batch_size, seq_len, vocab_size)
        labels: 真实 token ids，形状 (batch_size, seq_len)，其中填充部分设为 ignore_index
    返回:
        标量损失
    """
    # shift logits 和 labels
    # logits 的最后一个位置没有对应的下一个 token，所以我们取 logits[:, :-1, :]
    # labels 的第一个位置没有对应的前一个预测，所以我们取 labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()   # (batch, seq_len-1, vocab)
    shift_labels = labels[:, 1:].contiguous()       # (batch, seq_len-1)
 
    # 展平以便计算交叉熵
    loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss
 

if __name__ == "__main__":
    batch_size, seq_len, vocab_size = 2, 5, 10
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # 模拟 padding: 将序列后半部分设为 ignore_index
    labels[:, 3:] = -100
 
    loss = sft_loss(logits, labels)
    print("SFT Loss:", loss.item())
```
</details>

# 4. 手撕交叉熵、Softmax 实现。

概念讲解：

这是深度学习最基础的算子，但面试常考 **数值稳定性**。

Softmax: 
- 问题：如果$z_i $很大，$e^{z_i}$会数值溢出(Infinity)
- 解决：利用softmax的平移不变性，所有 z 减去最大值max(z)。

Cross Entropy (CE): $L_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$
- 在分类任务中，y 是 one-hot，所以简化为 $-log(P_{target})$
- 结合softmax：通常不单独算softmax再算Log，而是合并为 LogSoftmax，数值更稳定
- 在以下代码中计算的“log sum exp”，是 softmax 概率的分母的 log。用原先的 logits 减去这个 log sum exp，就可以得到 log 概率（log prob）了。
- 交叉熵损失在 PyTorch 中（nn.CrossEntropyLoss）实际上等于 LogSoftmax + NLLLoss (负对数似然)。

```python
import torch
 
def stable_softmax(logits, dim=-1):
    """
    数值稳定的 Softmax 实现
    """
    # 1. 减去最大值，防止 exp 溢出
    # keepdim=True 保证形状可以广播
    max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    
    # 2. 归一化
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    probs = exp_logits / sum_exp_logits
    return probs
 
def custom_cross_entropy_torch(logits, targets):
    """
    数值稳定的 PyTorch 版交叉熵损失函数
    公式：log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    参数:
        logits: shape [batch_size, num_classes], 模型最后一层的线性输出(未激活)
        targets: shape [batch_size], 真实类别的索引 (0 到 num_classes-1)
    """
    # 1. 找到每行的最大值，用于数值稳定性 (keepdim=True 保证广播)
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    
    # 2. 减去最大值 (x_i - M)
    shifted_logits = logits - max_logits
    
    # 3. 计算 LogSumExp 项: log(sum(exp(x_i - M)))
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    
    # 4. 得到 log_softmax 的结果
    # 公式: x_i - M - log(sum(exp(x_i - M)))
    log_probs = shifted_logits - log_sum_exp
    
    # 5. 取出正确标签 (targets) 对应的 log_prob (也就是 NLLLoss)
    batch_size = logits.size(0)
    # torch.arange(batch_size) 生成 [0, 1, 2...], 配合 targets 取出对应位置的值
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    
    # 6. 取负号并求均值
    loss = -target_log_probs.mean()
    
    return loss
 
# --- 测试 Demo ---
if __name__ == "__main__":
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])
    targets = torch.tensor([0, 1]) # 第一个样本目标是类 0，第二个是类 1
    
    # 对比 PyTorch 原生实现
    torch_ce = nn.CrossEntropyLoss()
    torch_loss = torch_ce(logits, targets)
    
    # 对比手写实现
    my_loss = cross_entropy_loss(logits, targets)
    
    print(f"PyTorch Loss: {torch_loss.item():.6f}")
    print(f"My Loss:      {my_loss.item():.6f}")
    # 两者应该非常接近
```

# 5.介绍一下 QKV 的计算代码实现、手撕 Self-Attention、手撕 Multi-Head Attention (MHA)

概念讲解：

什么是 QKV？想象我们在图书馆找书：

- Query (查询 Q)：心中的问题，比如"我想找关于机器学习的书"
- Key (键 K)：每本书的标签/标题，比如"深度学习入门"、"Python编程"
- Value (值 V)：书里的实际内容

计算过程：

1. 我们的问题(Q)和所有书的标签(K)做匹配，算出相似度（注意力分数）
2. 根据相似度，决定从每本书(V)里提取多少信息
3. 最后把所有书的内容按权重加权求和，得到答案

Self-Attention 的计算流程：

1. 投影 (Projection)：输入向量 X 通过三个不同的线性层，变成Q\K\V。
2. 打分 (Score)：计算$Q K^T $。这表示“当前词”和“其他词”的匹配程度。
3. 缩放 (Scale)：除以$\sqrt{d_k} $。防止点积结果太大导致 Softmax 梯度消失。
4. 归一化 (Softmax)：把分数变成概率分布（权重和为 1）。
5. 加权求和 (Weighted Sum)：用权重乘以V。匹配度高的地方，其内容V就被更多地保留下来。

代码实现：

这段代码完全可运行，包含了 QKV 投影、缩放点积注意力、因果掩码 (Causal Mask) 和 多头机制。

<details>
<summary>MHA代码实现</summary>

```python
import torch
import torch.nn as nn


def my_softmax(logits, dim = -1):

    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]

    exp_logits = torch.exp(logits - max_logits)

    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)

    probs = exp_logits / sum_exp_logits

    return probs


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        多头注意力机制
        Args:
            hidden_size（int）: 输入特征的维度
            num_heads（int）: attention 头数
            dropout(float): dropout的prob
        """
        super(MultiHeadAttention, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 定义线性变换层，用于生成 q,k,v
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # 输出线性层
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            hidden_state: torch.Tensor,
            attn_mask: torch.Tensor = None,
            mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数
        Args:
            hidden_state(torch.tensor): 输入的hidden_state，shape为[batch_size, seq_len, hidden_size]
            attention_mask(torch.tensor, optional): 注意力掩码，用于屏蔽某些位置，shape为[batch_size, seq_len]
        :returns
            torch.Tensor： 注意力输出，形状为[batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_state.shape
        # seq_len = hidden_state.size(1)

        # 1. attention计算得到Q, K, V
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        # 2. 将Q,K,V 拆分为多头 [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # step 3: 计算注意力分数 Q @ k^T / sqrt(head_dim) [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # step 4: 应用 padding mask(如果有)
        if mask is not None:
            # mask: [batch, seq_len_k] -> [batch, 1, 1, seq_len_k]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        if attn_mask is not None:
            # attn_mask: [seq_len_q, seq_len_k] -> [1, 1, seq_len_q, seq_len_k]
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        # step 5: softmax + dropout
        attn_weights = my_softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # step 6: 加权求和 attn @ V
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, value)

        # step 7: 拼接多头
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # step8: 输出投影
        output = self.out_proj(context)

        return output, attn_weights


def test_mha():
    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 10

    mha = MultiHeadAttention(hidden_size=d_model, num_heads=n_heads)

    # 自注意力测试
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn = mha(x)

    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {attn.shape}")
    print(f"注意力权重每行和 ≈ 1: {attn[0, 0, 0].sum().item():.4f}")

    # 因果 mask 测试（decoder 用）
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    output2, attn2 = mha(x, attn_mask=causal_mask)
    print(f"\n因果 mask 后，第一行注意力只看第一个 token:")
    print(f"attn[0,0,0,:5] = {attn2[0, 0, 0, :5].detach().numpy()}")

    # 验证参数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n总参数量: {total_params:,}")
    # 理论：4 * d_model * d_model + 4 * d_model (bias)
    print(f"理论参数量: {4 * d_model * d_model + 4 * d_model:,}")


if __name__ == "__main__":
    test_mha()

```
</details>


<details>
<summary>GQA代码实现</summary>

```python
import torch
import torch.nn as nn


def my_softmax(logits, dim = -1):

    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]

    exp_logits = torch.exp(logits - max_logits)

    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)

    probs = exp_logits / sum_exp_logits

    return probs

class GroupedQueryAttention(nn.Module):

    def __init__(self, hidden_size: int, num_q_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super(GroupedQueryAttention, self).__init__()

        assert num_q_heads % num_kv_heads == 0, "Q头数必须可以被KV头数整除"

        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads # 每组kv组对应Q头数目
        self.head_dim = hidden_size // num_q_heads # 单q头维度

        # 创建 W_q\W_k\W_v
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        K/v 多头扩充，适配Q的头数
        :param x: shape = [B, num_kv_heads, seq_len, head_num]
        :return: shape = [B, num_q_heads, seq_len, head_num]
        """
        B, num_kv_heads, seq_len, head_dim = x.shape

        if self.group_size == 1:
            return x
        # 新增维度重复，reshape 恢复头维度
        x = x[:, :, None, :, :].expand(B, num_kv_heads, self.group_size, seq_len, head_dim)
        x = x.reshape(B, num_kv_heads * self.group_size, seq_len, head_dim)
        # # 极简写法，面试首选
        # return x.repeat_interleave(self.group_size, dim=1)
        return x

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                pad_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # 1.计算Q\K\V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2.分头： 拆成[batch_size, heads, seq, head_dim]
        q = q.view(batch_size, seq_len_q, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3.GQA核心：KV头复制对齐Q
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # 4.attention 计算： Q @ K^T，与MHA一致
        attn_score = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5)

        # 5. 应用pad_mask和casual_mask
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
            attn_score = attn_score.mask_fill(pad_mask, float('-inf'))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_score = attn_score + attn_mask

        # 6. softmax + dropout
        attn_weight_raw = my_softmax(attn_score)
        attn_weight = self.dropout(attn_weight_raw)

        # 7. 加权求和 attn @ v
        context = torch.matmul(attn_weight, v)

        # 8.多头拼接
        context = context.transpose(1,2).contiguous().reshape(batch_size, seq_len_q, self.hidden_size)
        out = self.out_proj(context)
        return out, attn_weight_raw


def test_gqa():
    batch = 2
    d_model = 512
    n_q = 8
    n_kv = 2
    seq_len = 16

    gqa = GroupedQueryAttention(hidden_size=d_model, num_q_heads=n_q, num_kv_heads=n_kv)
    x = torch.randn(batch, seq_len, d_model)
    out, attn = gqa(x, x, x)

    print("输出shape:", out.shape)  # torch.Size([2, 16, 512])
    print("注意力权重shape:", attn.shape)  # torch.Size([2, 8, 16, 16])

    # attn：[B, head, q_seq, k_seq]
    # 取 batch0，head0，q位置0，所有k：正确下标
    row_sum = attn[0, 0, 0, :].sum().item()
    print(f"注意力一行总和：{row_sum:.4f}")

    # 因果掩码
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * -1e9, diagonal=1)
    out2, attn2 = gqa(x, x, x, attn_mask=causal_mask)
    print("因果掩码首行前3个权重：", attn2[0, 0, 0, :3].detach().numpy())


if __name__ == "__main__":
    test_gqa()
```
</details>


面试加分点 (Tips)：

1. 维度变换：一定要口述清楚 transpose(1, 2) 是为了把 num_heads 放到第二维，方便并行计算。
2. Mask 的作用：在 LLM 预训练（Decoder-only）中，必须使用 Causal Mask 防止信息泄露（即预测第 t 个词时不能看到 t+1 及之后的词）。
3. 为什么除以$\sqrt{d_k} $：防止点积过大，导致 Softmax 进入饱和区（梯度接近 0），训练变慢。
4. Flash Attention：如果面试官问优化，可以提一句标准的 Attention 是 $O(N^2) $显存占用，而 FlashAttention 通过**分块计算（Tiling）和重计算（Recomputation）**减少了 HBM 访问，是推理加速的关键。




# 6. RoPE (Rotary Positional Embedding)

概念讲解：

为什么需要位置编码？Self-Attention 是位置无关的（permutation invariant）：

- 输入 "人咬狗" 和 "狗咬人" 对 Attention 来说是一样的
- 但语义完全不同！需要让模型知道词的顺序

传统方法 vs 旋转位置编码 (Rotary Position Embedding, RoPE)：
|方法|原理|缺点|
|---|---|---|
|绝对位置编码 (原版Transformer)|每个位置学一个向量，加到词嵌入上|长度外推性差|
|RoPE (旋转位置编码)|用旋转矩阵编码相对位置|内积自然体现相对距离，外推性好，现在是主流|

代码实现：

下面用 NumPy 实现 RoPE 的前向计算，并给出一个简单的测试。

<details>
<summary>ROPE前向计算</summary>

```python
import numpy as np
 
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    预计算旋转角度的 cos 和 sin 值（复数形式常称为 cis，即 cos + i sin）
    dim: 特征维度，必须是偶数
    max_seq_len: 最大序列长度
    theta: base 频率参数，通常为 10000
    返回:
        cos: (max_seq_len, dim//2)
        sin: (max_seq_len, dim//2)
    """
    assert dim % 2 == 0, "维度必须为偶数"
    # 计算每个维度的频率 theta_i
    i = np.arange(0, dim, 2)  # i = 0, 2, 4, ..., dim-2
    freqs = 1.0 / (theta ** (i / dim))  # (dim//2,)
 
    # 生成位置序列
    t = np.arange(max_seq_len)  # (max_seq_len,)
    # 计算角度 t * freqs (广播)
    angles = np.outer(t, freqs)  # (max_seq_len, dim//2)
 
    cos = np.cos(angles)  # (max_seq_len, dim//2)
    sin = np.sin(angles)  # (max_seq_len, dim//2)
    return cos, sin
 
def apply_rotary_emb(x, cos, sin):
    """
    对输入张量 x 应用旋转位置编码
    x: (seq_len, dim) 或 (batch_size, seq_len, dim)，dim 必须为偶数
    cos, sin: (seq_len, dim//2) 预计算好的值
    返回: 编码后的张量，形状与 x 相同
    """
    # 将 x 按最后一维分成两半，reshape 成 (..., seq_len, dim//2, 2)
    # 这样最后一维的两个元素分别是 x_{2i} 和 x_{2i+1}
    orig_shape = x.shape
    seq_len = x.shape[-2]  # 倒数第二维是序列长度
    d = x.shape[-1]
    assert d % 2 == 0
    # 将 x 重塑为 (..., seq_len, d//2, 2)
    x_reshaped = x.reshape(*orig_shape[:-1], d//2, 2)
 
    # 分别取出偶数索引和奇数索引部分（即两个分量）
    x_even = x_reshaped[..., 0]  # (..., seq_len, d//2)
    x_odd  = x_reshaped[..., 1]  # (..., seq_len, d//2)
 
    # 应用旋转公式：
    # x'_{2i} = x_{2i} * cos - x_{2i+1} * sin
    # x'_{2i+1} = x_{2i} * sin + x_{2i+1} * cos
    # 注意 cos, sin 的形状为 (seq_len, d//2)，需要广播到与 x_even 一致
    # 如果 x 有 batch 维度，cos/sin 会自动广播到 batch 维（只要维度对齐）
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd  = x_even * sin + x_odd * cos
 
    # 将两部分堆叠回 (..., seq_len, d//2, 2) 形状
    rotated = np.stack([rotated_even, rotated_odd], axis=-1)  # (..., seq_len, d//2, 2)
    # 重塑回原始形状 (..., seq_len, d)
    return rotated.reshape(*orig_shape)
 
# ========== 示例运行 ==========
if __name__ == "__main__":
    # 参数
    dim = 8
    seq_len = 4
    batch_size = 2
 
    # 预计算 cos 和 sin
    cos, sin = precompute_freqs_cis(dim, seq_len)
    print("cos 形状:", cos.shape)  # (4, 4) 因为 dim//2 = 4
    print("sin 形状:", sin.shape)
 
    # 随机生成 Query 和 Key（假设无 batch 或带 batch）
    # 示例 1: 单序列
    q = np.random.randn(seq_len, dim)
    k = np.random.randn(seq_len, dim)
    q_rot = apply_rotary_emb(q, cos, sin)
    k_rot = apply_rotary_emb(k, cos, sin)
    print("\n单序列 RoPE 后 Q 形状:", q_rot.shape)
 
    # 示例 2: 带 batch 的序列
    q_batch = np.random.randn(batch_size, seq_len, dim)
    k_batch = np.random.randn(batch_size, seq_len, dim)
    q_batch_rot = apply_rotary_emb(q_batch, cos, sin)
    k_batch_rot = apply_rotary_emb(k_batch, cos, sin)
    print("带 batch 的 RoPE 后 Q 形状:", q_batch_rot.shape)
 
    # 验证内积是否只与相对位置有关（理论性质，这里简单打印差值）
    # 取位置 1 和 2 的 Q/K 计算内积，并与位置 2 和 3 的比较（未严格验证，仅供演示）
    attn1 = np.dot(q_rot[1], k_rot[2])
    attn2 = np.dot(q_rot[2], k_rot[3])
    print(f"内积 (pos1 Q, pos2 K): {attn1:.4f}")
    print(f"内积 (pos2 Q, pos3 K): {attn2:.4f}")

```
</details>

说明：

- precompute_freqs_cis 根据公式生成每个位置、每个维度对的 cos 和 sin 值。
- apply_rotary_emb 将输入张量（最后两维为序列长度和特征维度）应用旋转。它支持任意形状，只要最后两维是 (..., seq_len, dim)，且 dim 为偶数。
- 在示例中，我们分别对单序列和 batch 数据进行了测试。
- 最后简单验证了两个不同相对位置的内积值，实际 RoPE 的特性是它们应依赖于相对位置差，但这里仅为演示代码可用性。

面试加分点 (Tips)：

1. 为什么只加在 Q 和 K 上？
- Attention 的核心是 $Q K^T $。RoPE 的性质保证了 $(Q_m K_n) $的内积只依赖于m-n（相对位置）。V 代表内容，不需要位置信息来参与匹配。

2. RoPE 的外推性 (Extrapolation)：
相比绝对位置编码，RoPE 在推理时如果序列长度超过训练长度，表现衰减更慢。因为它学习的是相对关系，而不是死记硬背位置 ID。
3. Theta 参数：
Llama 2/3 使用了 10000，但为了支持更长上下文，Llama 3 或某些变体调整了$\theta $或者使用了 NTK-Aware Scaled RoPE。如果面试官问“如何支持 100k 上下文”，我们可以提到修改 $\theta $ 或插值方法。

一个问题：为什么位置编码要同时使用 sin 和 cos，而不是只使用其中一个？

因为 RoPE（旋转位置编码）的本质是在复数平面（或二维空间）上对词向量进行旋转（Rotation）。在数学上，要完成一次保范数（Norm-preserving）的二维旋转，必须同时依赖正弦和余弦的配合；如果只用其中一个，就变成了破坏向量长度的单纯缩放（Scaling）。

写 PPO DPO GRPO 的 loss function。
写 PPO 伪代码、PPO 的 GAE。
写 ReAct 伪代码，并添加 tool call。
