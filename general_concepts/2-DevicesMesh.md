# DeviceMesh 入门讲解

## 1. 什么是 DeviceMesh？

**DeviceMesh** 是 PyTorch FSDP2 (Fully Sharded Data Parallel) 引入的一个 API，用于**逻辑上组织和管理多个 GPU**。

简单理解：它把你的多张 GPU 定义成一个"网格"，然后告诉 FSDP "这些 GPU 怎么分组、怎么协作"。

## 2. 核心概念

### 一维 DeviceMesh（纯 FSDP）

```python
device_mesh = init_device_mesh(device_name, mesh_shape=(8,), mesh_dim_names=["fsdp"])
```

```
GPU: [0] [1] [2] [3] [4] [5] [6] [7]
Group: |_________ 8 GPUs 共享同一份模型权重 _________|
```

- 所有 GPU 在**同一个 FSDP 分片组**中
- 模型被分片存储，每张 GPU 只持有部分参数
- 适合：单节点多卡，模型较大但节点数少

### 二维 DeviceMesh（混合 DDP + FSDP）

```python
device_mesh = init_device_mesh(
    device_name,
    mesh_shape=(4, 2),        # 4个DDP组, 每组内2卡FSDP
    mesh_dim_names=["ddp", "fsdp"]
)
```

```
          GPU 0  GPU 1 | GPU 2  GPU 3 | GPU 4  GPU 5 | GPU 6  GPU 7
DDP组0:  |  FSDP组  |   |              |             |
DDP组1:  |          |   | FSDP组       |             |
DDP组2:  |          |   |              | FSDP组      |
DDP组3:  |          |   |              |             |  FSDP组
```

- **第一维 (ddp)**：数据并行 — 不同 GPU 处理不同数据 batch
- **第二维 (fsdp)**：模型分片 — GPU 组内共享模型分片

## 3. 实际场景举例

```python
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 场景A: 纯 FSDP
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        # 场景B: 混合 DDP + FSDP
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh
```

| 参数 | 效果 |
|------|------|
| `world_size=8, fsdp_size=-1` | 纯 FSDP，8卡一组（`fsdp_size < 0`） |
| `world_size=8, fsdp_size=8` | 纯 FSDP，8卡一组（`fsdp_size >= world_size`） |
| `world_size=8, fsdp_size=2` | 二维，4个 DDP 组 × 每组 2 卡 FSDP |
| `world_size=8, fsdp_size=1` | 二维，8个 DDP 组 × 每组 1 卡 FSDP（每个 GPU 独立） |

## 4. 为什么需要二维？

**扩展到多节点时**问题来了：

- 假设 32 卡，跨 4 个节点，每节点 8 卡
- 节点内通信带宽高，节点间带宽低
- 如果只用一维 FSDP，所有 32 卡通信一次，开销巨大

用二维 mesh：

```
Node0: [GPU0-7]  Node1: [GPU8-15]  Node2: [GPU16-23]  Node3: [GPU24-31]
          ↓              ↓                ↓                ↓
      FSDP组内通信   FSDP组内通信    FSDP组内通信    FSDP组内通信  (高速)
            ↓↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↓
                DDP组间同步 (低频, 可以慢)
```

- FSDP 组内频繁通信走高速节点内网络
- DDP 组间同步低频，走节点间网络也 OK

## 5. 类比理解

| 概念 | 类比 |
|------|------|
| DeviceMesh | 建筑平面图（怎么分配房间） |
| mesh_shape | 楼层规划（几行几列） |
| mesh_dim_names | 房间类型标签（厨房、卧室） |
| FSDP 维度 | 合租 — 室友共享一个房间的资源 |
| DDP 维度 | 整租 — 每个租户独立占一间房 |

## 6. 与常见并行策略对比

| 并行策略 | DeviceMesh 中的表示 | 通信模式 | 适用场景 |
|----------|---------------------|----------|----------|
| **DDP** | `mesh_shape=(N,)` 的一维 | AllReduce (高频) | 小模型、多节点 |
| **FSDP1** | 同上但分片策略不同 | AllGather + ReduceScatter | 大模型、单节点 |
| **FSDP2** | 同上，配合 `ShardingStrategy` | 更加细粒度的分片 | 中等模型、内存优化 |
| **混合 (DDP+FSDP)** | `mesh_shape=(M, N)` 的二维 | 两种通信模式结合 | 超大规模、多节点 |

### 通信模式详解

```
DDP 通信模式:
  GPU0 ──→ AllReduce ──→ GPU1
  GPU1 ──→ AllReduce ──→ GPU2
  GPU2 ──→ AllReduce ──→ GPU3
  (每个 step 同步梯度，所有 GPU 都要参与)

FSDP 通信模式:
  Step1: GPU0 ──ReduceScatter──→ 只保留 1/N 参数
  Step2: AllGather ──获取完整参数──→ 所有 GPU
  (只传输分片，节省带宽)

混合通信:
  FSDP 维度: 组内 AllGather/ReduceScatter (高频，节点内)
  DDP 维度: 组间 AllReduce (相对低频，节点间也可接受)
```

## 7. API 详解

### 7.1 核心 API

```python
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

# 初始化一维 DeviceMesh
device_mesh = init_device_mesh(
    device_name="cuda",           # 设备类型
    mesh_shape=(8,),              # GPU 数量和排列
    mesh_dim_names=["fsdp"]       # 维度名称
)

# 获取 mesh 信息
print(device_mesh.mesh.shape)      # torch.Size([8])
print(device_mesh.mesh_dim_names)  # ['fsdp']

# 获取特定维度的子 mesh
fsdp_mesh = device_mesh["fsdp"]    # 获取 fsdp 维度的子 mesh
```

### 7.2 与 FSDP2 结合使用

```python
from torch.distributed.fsdp import FSDP
from torch.distributed.fsdp.api import ShardingStrategy

# 方式1: 纯 FSDP
device_mesh = init_device_mesh("cuda", mesh_shape=(8,), mesh_dim_names=["fsdp"])

model = FSDP(
    model,
    device_mesh=device_mesh,           # 传入 device_mesh
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 分片策略
    use_orig_params=True,
)

# 方式2: 混合 DDP + FSDP (需要在 DDP 包装外层)
device_mesh = init_device_mesh(
    "cuda",
    mesh_shape=(4, 2),
    mesh_dim_names=["ddp", "fsdp"]
)

# DDP 负责数据并行
model = DDP(model, device_mesh=device_mesh["ddp"])

# FSDP 负责模型分片
model = FSDP(model, device_mesh=device_mesh["fsdp"])
```

### 7.3 可视化 mesh 结构

```python
def visualize_device_mesh(device_mesh):
    """可视化 DeviceMesh 结构"""
    mesh = device_mesh.mesh.cpu().numpy()
    dim_names = device_mesh.mesh_dim_names

    print(f"Mesh Shape: {mesh.shape}")
    print(f"Dimensions: {dim_names}")
    print(f"\n实际 GPU 分配:")
    print(mesh)

    # 二维 mesh 可视化
    if len(mesh.shape) == 2:
        print(f"\n矩阵形式 (行={dim_names[0]}, 列={dim_names[1]}):")
        for i, row in enumerate(mesh):
            print(f"  {row}  ← {dim_names[0]} group {i}")

# 示例
mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=["ddp", "fsdp"])
visualize_device_mesh(mesh)
# 输出:
# Mesh Shape: (4, 2)
# Dimensions: ['ddp', 'fsdp']
#
# 实际 GPU 分配:
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
#
# 矩阵形式 (行=ddp, 列=fsdp):
#   [0 1]  ← ddp group 0
#   [2 3]  ← ddp group 1
#   [4 5]  ← ddp group 2
#   [6 7]  ← ddp group 3
```

## 8. 多维 DeviceMesh（三维示例）

对于更复杂的多节点场景，可以创建三维 DeviceMesh：

```python
# 场景: 4 节点 × 每节点 8 卡 × 每卡 2 个 GPU（假设超算有更细粒度）
device_mesh = init_device_mesh(
    "cuda",
    mesh_shape=(4, 8, 2),
    mesh_dim_names=["node", "fsdp", "tensor"]
)
```

```
维度结构:
  - node: 节点间并行
  - fsdp: 节点内模型分片
  - tensor: 张量并行（最细粒度，通信最频繁）

通信优先级:
  tensor (最高频) > fsdp (高频) > node (低频)
```

## 9. 常见问题与解决方案

### Q1: 如何选择 fsdp_size？

```python
# 经验法则
def recommend_fsdp_size(world_size):
    """
    推荐 fsdp_size 的经验值
    - 模型参数量大 → 减小 fsdp_size（更多 GPU 分摊）
    - 节点内带宽高 → 可适当增加 fsdp_size
    """
    # 估算: 每卡能承载的参数（GB） / 模型参数（GB）
    param_size_per_gpu = 24  # A100 80GB 约能放下 24B 参数（带优化器状态）

    # 动态调整
    if model_size > param_size_per_gpu * 0.5:
        return max(2, world_size // 2)  # 需要更多分片
    else:
        return world_size  # 纯 FSDP 即可

# 实际建议
print(f"8卡推荐: fsdp_size={recommend_fsdp_size(8)}")
print(f"16卡推荐: fsdp_size={recommend_fsdp_size(16)}")
```

### Q2: 报错 "mesh_shape must be a valid factor of world_size"

```python
# 检查 mesh_shape 是否整除 world_size
world_size = 8
fsdp_size = 3  # 8 % 3 != 0，报错！

# 解决方案: 确保整除
valid_fsdp_sizes = [s for s in range(1, world_size+1) if world_size % s == 0]
print(f"8卡有效的 fsdp_size: {valid_fsdp_sizes}")
# [1, 2, 4, 8]
```

### Q3: 如何调试 DeviceMesh 配置？

```python
import torch.distributed as dist

def debug_device_mesh():
    """打印分布式环境信息"""
    print(f"Rank: {dist.get_rank()}")
    print(f"World Size: {dist.get_world_size()}")
    print(f"Local Rank: dist.get_local_rank()")
    print(f"CUDA Device: {torch.cuda.current_device()}")

    # 打印进程组信息
    if dist.is_initialized():
        for name, group in dist.group.WORLD.group_members.items():
            print(f"Process Group: {name}, size: {group.size()}")

# 每个 rank 都会打印自己的信息
debug_device_mesh()
```

## 10. 代码模板（MT-R1-Zero 中的实际使用）

```python
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def setup_device_mesh(world_size: int, fsdp_size: int = -1):
    """
    MT-R1-Zero 中的 DeviceMesh 初始化

    参数:
        world_size: 总 GPU 数量
        fsdp_size: FSDP 组大小，-1 表示使用纯 FSDP

    返回:
        device_mesh: 配置好的 DeviceMesh
    """
    # 根据 fsdp_size 决定 mesh 维度
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 场景1: 纯 FSDP
        # 例: world_size=8, fsdp_size=-1 → mesh_shape=(8,)
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=["fsdp"]
        )
        print(f"[Rank {dist.get_rank()}] Using pure FSDP with {world_size} GPUs")
    else:
        # 场景2: 混合 DDP + FSDP
        # 例: world_size=8, fsdp_size=2 → mesh_shape=(4, 2)
        #     4个DDP组, 每组2卡FSDP
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"]
        )
        ddp_size = world_size // fsdp_size
        print(f"[Rank {dist.get_rank()}] Using Hybrid DDP({ddp_size}) + FSDP({fsdp_size})")

    return device_mesh

# 使用示例
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    device_mesh = setup_device_mesh(world_size, fsdp_size=-1)

    # 后续用于 FSDP 初始化
    print(f"Device Mesh initialized: {device_mesh.mesh.shape}")

    dist.destroy_process_group()
```

## 11. 总结

```
DeviceMesh 的本质：
  定义"哪些 GPU 划为一组" + "组的排列方式"
         ↓
  FSDP 根据这个划分决定参数怎么分片、怎么通信
         ↓
  用户不需要手动管理通信，只需声明拓扑，框架自动优化

核心价值：
  ✅ 声明式配置 - 告诉框架拓扑，不关心实现细节
  ✅ 灵活扩展 - 轻松支持一维/二维/多维
  ✅ 硬件感知 - 配合 NCCL 自动优化通信
  ✅ 多策略组合 - DDP、FSDP、TensorParallel 自由组合
```