# torch.compile 与 MedNeXt 深监督切换问题

本文记录与当前 nnU-Net / MedNeXt 推理相关的 `torch.compile` 双层封装和 deep supervision 切换问题。

---

##### 一、torch.compile 只编译模型的 forward()，其他都不管

整个推理流程里，只有神经网络本身被编译：

```
读取 CT 文件
    ↓
预处理（resample、normalize）         ← 纯 Python/numpy，不编译
    ↓
切成 patch（sliding window）          ← 纯 Python，不编译
    ↓
┌─────────────────────────────┐
│  model.forward(patch)        │  ← 只有这里被 torch.compile 编译
│  卷积 → 激活 → 上采样 → ...  │
└─────────────────────────────┘
    ↓
把所有 patch 的输出拼回去             ← 纯 Python/numpy，不编译
    ↓
后处理（连通域分析、去小CC等）         ← 纯 Python/scipy，不编译
    ↓
保存 .nii.gz
```

原则：**只有 `torch.nn.Module` 的 `forward()` 才会被 compile**。

所以后处理的连通域分析（`scipy.ndimage.label` 等）完全不受影响，有 bug 就是普通 Python bug，和 compile 无关。

---

##### 二、先理解 forward() 和"编译"是什么

正常 Python 运行一个神经网络：

```
输入 x → Python 解释器逐行执行 forward() → 输出
```

每次推理都要重新"翻译"一遍代码，慢。

`torch.compile` 做的事：

```
第一次推理时：
  观察 forward() 的执行过程 → 生成一份优化的机器码（CUDA kernel）→ 缓存起来

第二次推理时：
  直接跑缓存的机器码，跳过 Python 解释
```

**类比**：第一次做一道菜，边做边写菜谱；以后再做，直接照菜谱，不用再想步骤。

---

##### 三、torch.compile 产生的"两层结构"

调用 `torch.compile(model)` 之后：

```
调用前：
    self.network = MedNeXtModel()        ← 就一层，就是模型本身


调用后：
    self.network = OptimizedModule       ← 外层：编译器的"外壳/管家"
                       │
                       └── ._orig_mod = MedNeXtModel()   ← 内层：真正的模型
```

###### OptimizedModule 是什么？

它是 PyTorch 自动生成的一个**管家类**，本身不包含任何神经网络参数，只负责：

1. 接收输入 x
2. 检查"缓存里有没有合适的编译好的图"
3. 有 → 直接用缓存跑
4. 没有 → 让内层的真实模型跑一遍，同时录制下来编译，存缓存

```
你调用 self.network(x)
        │
        ▼
  OptimizedModule（管家）
        │
        ├─ 检查缓存 ──有──→ 直接跑编译好的机器码 → 输出
        │
        └─ 没有 → 调用 _orig_mod.forward(x) 并录制 → 编译 → 存缓存 → 输出
```

###### _orig_mod 是什么？

`_orig_mod` = **original module** = 原始模型

就是你的 `MedNeXtModel()`，代码、结构、权重、属性全都在这里：

```
MedNeXtModel（_orig_mod）
    ├── 代码（forward() 怎么算）     ← 类定义，所有实例共享
    ├── 结构（有哪些层）             ← conv1, conv2, encoder, decoder ...
    ├── 权重（每层的参数数值）        ← 训练出来的 float 数字，占几个 GB
    └── 属性（do_ds, do_res ...）    ← 控制行为的开关
```

**唯独没有**：编译缓存。那个在 OptimizedModule 里。

###### 编译缓存里存的是什么？

不只是"代码"，是**编译好的 GPU 可执行程序 + 触发条件**：

```
编译缓存里存着：
    ├── GPU 机器码（CUDA kernel）   ← 优化后的二进制，GPU 直接执行
    ├── 计算顺序图                  ← 哪步先算、哪步后算、怎么并行
    └── Guard（触发条件）           ← 例如"当 do_ds==True 时用这份图，否则重编译"

不在缓存里的：
    └── 权重数值                    ← 权重永远住在 _orig_mod 里
                                      跑的时候去那里取，不复制进缓存
```

类比：菜谱（缓存）写"第三步加盐"，但盐本身（权重）还在厨房柜子（`_orig_mod`）里，每次做菜都去柜子取。

---

##### 四、深度监督（Deep Supervision）开关是什么

MedNeXt 训练时用多个输出头（深度监督）让浅层也学习：

```
训练时 do_ds = True：

输入 → 编码器 → 解码器 → [输出1（最终）, 输出2（中间层）, 输出3（更浅层）]
                                  ↑
                           返回一个 list，里面有多个 tensor
```

```
推理时 do_ds = False：

输入 → 编码器 → 解码器 → 输出1（只要最终结果）
                                  ↑
                           返回单个 tensor
```

---

##### 五、"缓存"（编译图）和 do_ds 的关系

torch.compile 在录制/编译时，会把 `do_ds` 的值**烘焙进编译好的图里**：

```
训练结束时，缓存里存着：
┌─────────────────────────────────────────────────┐
│  编译好的图（菜谱）                               │
│  条件：_orig_mod.do_ds == True 时使用            │
│  执行逻辑：运行 DS=True 的 forward，返回 list     │
└─────────────────────────────────────────────────┘
```

---

##### 六、Bug 发生的完整过程

###### 验证开始，nnUNet 调用：
```python
self.set_deep_supervision_enabled(False)
```

###### MedNeXt 的（有 Bug 的）实现：
```python
mod = self.network            # ← 拿到的是 OptimizedModule（管家）
mod.do_ds = False             # ← 在管家身上贴了个标签 "do_ds=False"
```

###### 内存里实际发生的：

```
self.network = OptimizedModule
    │  .do_ds = False   ← ✅ 我们设置了这里
    │
    └── ._orig_mod = MedNeXtModel
            .do_ds = True   ← ❌ 这里没变！还是 True！
```

###### 推理时：

```
你调用 self.network(x)
        │
        ▼
  OptimizedModule（管家）
        │
        └─ 查缓存：有！（训练时编译的 DS=True 图）
              │
              ▼
        直接跑 DS=True 的编译图
              │
              ▼
        返回 [tensor1, tensor2, tensor3]   ← list！
              │
              ▼
torch.flip([tensor1, tensor2, tensor3], axes)
              │
              ▼
        💥 TypeError: flip() 需要 Tensor，不是 list
```

**管家完全无视了自己身上的 `do_ds=False` 标签**，因为管家本来就不懂 `do_ds` 是什么，
它只是查缓存，而缓存里存的图是 DS=True 版本的。

---

##### 七、为什么"之前的修法"（torch._dynamo.reset()）没用

`torch._dynamo.reset()` 清空了缓存（把菜谱烧掉了）：

```
缓存清空后，第一次推理：
  OptimizedModule（管家）
        │
        └─ 查缓存：空的！没有菜谱
              │
              ▼
        调用 _orig_mod.forward(x) 重新录制
              │
              ▼
        录制时读 _orig_mod.do_ds = ???
```

此时 `_orig_mod.do_ds` 是多少？

**还是 True！** 因为我们根本没改过 `_orig_mod.do_ds`，只改了管家的。

```
重新编译的图：DS=True（因为 _orig_mod.do_ds 还是 True）
返回：list   →   还是崩！
```

只不过 reset 导致重新编译花了 3-10 分钟，看起来"在跑"，实际上根本没修好。

---

##### 八、正确的修法

对齐 base nnUNetTrainer 的做法，先解包到内层模型再设置：

```python
def set_deep_supervision_enabled(self, enabled: bool):
    from torch._dynamo import OptimizedModule
    mod = self.network                      # 先拿到管家
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod                 # 穿透管家，拿到真实模型
    mod.do_ds = enabled                     # 改真实模型的 do_ds
```

###### 修完之后内存里：

```
self.network = OptimizedModule（管家）
    │
    └── ._orig_mod = MedNeXtModel
            .do_ds = False   ← ✅ 真正改到了这里
```

###### 推理时：

```
OptimizedModule（管家）
    │
    └─ 查缓存：有 DS=True 的图，但条件是 _orig_mod.do_ds==True
          │
          └─ 检查 _orig_mod.do_ds → 现在是 False → 条件不符合！
                │
                ▼
          重新编译 DS=False 的图（很快，几秒）
                │
                ▼
          返回单个 tensor   ← ✅
                │
                ▼
    torch.flip(tensor, axes)   ← ✅ 正常！
```

---

##### 九、一句话总结

```
torch.compile 编译的是 _orig_mod（内层），
但我们的 set_deep_supervision_enabled 改的是 OptimizedModule（外层），
两个是不同的对象，改外层对内层没有任何影响。
```

base nnUNetTrainer 早就知道这个坑，在它的实现里写了 `mod = mod._orig_mod`，
MedNeXt 覆盖时忘记加这一句，才导致了这个 Bug。


---
