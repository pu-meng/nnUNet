
# 目录

[toc]

## 1.1 命令行工具

wegt 是 "world wide web get" 的缩写

## 2.1 Medical VQA 项目笔记

### 2.1.1 F.cross_entropy

$$
\text{CrossEntropy}(\mathbf{z}, y) = -\log \frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}}
$$

```python
labels = torch.arange(B, device=img_feat.device)
loss_i = F.cross_entropy(logits, labels)
loss_t = F.cross_entropy(logits.T, labels)
return (loss_i + loss_t) / 2
```

`logits`：`(B, B)`，`labels = torch.arange(B)`

例如：

\[
A = \begin{bmatrix}
a_{00} & a_{01} & \cdots & a_{0,B-1} \\
a_{10} & a_{11} & \cdots & a_{1,B-1} \\
\vdots & \vdots & \ddots & \vdots    \\
a_{B-1,0} & a_{B-1,1} & \cdots & a_{B-1,B-1}
\end{bmatrix}
\]

$$
loss\_i=-\frac{1}{B}\sum\limits_{i=0}\limits^{n-1}\log\frac{e^{a_{ii}}}{\sum\limits_{j=0}\limits^{n-1}e^{a_{ij}}}
$$

相当于先从 \(A\) 沿着列方向做 softmax，而后平均：

$$
A\rightarrow \begin{bmatrix}
b_0\\
b_1\\
\vdots\\
b_{B-1}
\end{bmatrix}
$$

$$
loss\_t=-\frac{1}{B}\sum\limits_{j=0}\limits^{n-1}\log\frac{e^{a_{ii}}}{\sum\limits_{i=0}\limits^{n-1}e^{a_{ij}}}
$$

`loss_t` 是先沿着行方向做 softmax，而后平均。

```
输入：一张图 + 一个问题句子

        图 ──→ CLIP图像编码器 ──→ img_feat (512维向量)
                                        ↓
                              CrossAttentionFusion
                           (Q=问题特征, K/V=图像特征)
       问题 ──→ CLIP文本编码器 ──→ txt_feat (512维向量)
                                        ↓
                                  fused (512维向量)
                                        ↓
                                   分类头(MLP)
                                        ↓
                                   logits (num_classes维)
```

融合方式是 **CrossAttention**，不是拼接。问题特征作为 Query，图像特征作为 Key/Value：

```python
# vqa_model.py CrossAttentionFusion.forward
q  = txt_feat.unsqueeze(1)    # (B, 1, 512)  ← Query：问题"问什么"
kv = img_feat.unsqueeze(1)    # (B, 1, 512)  ← Key/Value：图里有什么
out, _ = self.attn(q, kv, kv)
out = self.norm(q + dropout(out))  # 残差 + LayerNorm
fused = out.squeeze(1)        # (B, 512)  ← 输出仍是512维，没有拼接
```

分类头：一个普通的全连接层

```
MLP:  512维 → 512维 → num_classes维（比如1552）

输出: [0.1, 3.2, -0.5, 0.8, ..., 1.1]  ← 这就是 logits
       答案0  答案1  答案2  答案3       答案1551
```

### 2.1.2 模型结构：frozen backbone + 轻量头

#### 2.1.2.1 整体架构

| BiomedCLIP（冻结，不训练） | 可训练部分 |
|---|---|
| 图像 → `encode_image()` → 512维 | → CrossAttentionFusion |
| 文字 → `encode_text()` → 512维 | → MLPClassifier |
| | → 1552个答案类别 |

#### 2.1.2.2 真正在训练的只有两个小模块

- **CrossAttentionFusion**：让问题特征“看”图像特征
- **MLPClassifier**：把融合后的特征映射到1552个答案

#### 2.1.2.3 策略优缺点

**好处**
- BiomedCLIP 在1500万医学图文对上预训练，特征质量已经很好
- 训练参数量极少（几M级别），训练快，不容易过拟合
- 小数据集（12k样本）上尤其合适

**代价**
- BiomedCLIP 提取的特征质量直接决定上限，无法针对当前任务微调特征

> 运行训练时打印的 `Trainable params: X.XXM / Total: XXXM`，可以看出可训练参数只占总参数的很小一部分。

### 2.1.3 VQA 整体流程

**输入**：图像 + 问题文本 → 模型 → **输出**：从词库里选一个答案（分类问题）

#### 2.1.3.1 具体流程

**训练时**
```
图像 synpic41148  +  "what kind of image is this?"
         ↓  模型预测
    ["ct", "mri", "xr - plain film", "cta - ct angiography", ...]
  ← 词库（1552个词）
         ↓  对比真实答案
    正确答案 = "cta - ct angiography"  → 计算 loss → 更新参数
```

**推理时**
```
新图像 + 新问题  →  模型  →  从词库1552个候选里选概率最高的  →  输出答案
```

#### 2.1.3.2 本质是多分类问题

- 不是生成文字，是从固定词库里“选”
- 词库大小 = 训练集里出现过的所有答案（1552个）
- `yes` / `no` 只是词库里的两个普通类别，和其他答案地位相同

#### 2.1.3.3 问题不是“焊死”在图上的

同一张图可以回答多个问题：

| 问题 | 答案 |
|---|---|
| "is this a ct scan?" | "yes" |
| "was contrast given?" | "no" |
| "what modality is this?" | "ct with iv contrast" |

模型必须同时理解**图像内容** + **问题语义**，才能给出正确答案。

#### 2.1.3.4 一句话总结

> **VQA = 图像特征 + 文本特征 → 融合 → 在答案词库里做分类**

这就是本项目使用 BiomedCLIP 的原因——它能同时编码图像和文字，做融合很自然。

### 2.1.4 模型架构：图文融合分类

```
images  (B, 3, 224, 224)  →  encode_image  →  img_feat (B, 512)
                                                      ↓
tokens  (B, 77)           →  encode_text   →  txt_feat (B, 512)
                                                      ↓
                                       fusion (CrossAttention)
                                                      ↓
                                          fused (B, 512)
                                                      ↓
                                     classifier (MLP: 512 → num_classes)
                                                      ↓
                                          logits (B, num_classes)
```

`num_classes` 是训练集中出现过的不重复答案数量（即 `len(ans2idx)`，约几百个）。每个位置对应一个候选答案的原始得分，取 `argmax` 即为预测的答案类别。

## 3.1 json 文件

```python
history = [
    {"epoch": 1, "train_acc": 0.45, "val_acc": 0.42, "loss": 1.23},
    {"epoch": 2, "train_acc": 0.61, "val_acc": 0.58, "loss": 0.95},
    {"epoch": 3, "train_acc": 0.73, "val_acc": 0.70, "loss": 0.78},
]

with open('json_examples/history.json', 'w') as f:
    json.dump(history, f, indent=2)

# print(open('json_examples/history.json').read())
with open('json_examples/history.json') as f:
    print(f.read())
```

`json.dump` 的缩进 `indent` 是遇到 `{` 和 `[` 时自动缩进，遇到 `}` 和 `]` 时自动回退，`indent` 只认识括号。  
换行的规则是：一个 `{` 或 `[` 里面有多个元素，每个元素单独一行，只有一个元素不换行。

```python
data_cn = {"模型": "BiomedVQA", "准确率": 0.873}

# 默认 ensure_ascii=True，中文变成转义码
print('默认:')
print(json.dumps(data_cn, indent=2))

# ensure_ascii=False，中文正常显示
print('\ensure_ascii=False:')
print(json.dumps(data_cn, indent=2, ensure_ascii=False))
```

如果内容有中文，必须要有 `ensure_ascii=False`。

```python
# 读回 ans2idx
with open('json_examples/ans2idx_pretty.json') as f:
    loaded = json.load(f)
```

`json.load(f)` 读取文件的同时把 json 解析还原成 Python 对象。  
`f.read()` 返回类型为 `str`，不能直接用比如 `loaded[0]` 这种，需要 `json.loads(f.read())`。

## 4.1 BiomedVQA 的整体结构（vqa_model.py:59）

```
图像 → BiomedCLIP Image Encoder (冻结) → img_feat (B,512)
问题 → BiomedCLIP Text Encoder  (冻结) → txt_feat (B,512)
            ↓
     CrossAttentionFusion
       Q=txt_feat, K=V=img_feat   → fused (B,512)
            ↓
     MLPClassifier → logits (B, num_classes)
```

```
txt_feat (B, 512) --[线性投影 W_Q]--> Q (B, 512)
img_feat (B, 512) --[线性投影 W_K]--> K (B, 512)
img_feat (B, 512) --[线性投影 W_V]--> V (B, 512)

① 相似度打分：  scores = Q · Kᵀ / √512     → (B, B) 或 (B,1,1)
② softmax：     attn   = softmax(scores)
③ 加权求和：    fused  = attn · V           → (B, 512)
```


- CLIP 编码器全程冻结，只有 fusion + classifier 是可训练的
- forward 返回 `(logits, img_feat, txt_feat)`，后两个是给 CLIP 对比损失用的


``` python
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
```
这里的p是torch.nn.Parameter，本质是一个被标记为"模型参数"的tensor,

.numel()是返回tensor中元素的数量，即tensor的size的乘积

```
  p.shape = (512, 512)  →  p.numel() = 262144               
  p.shape = (512,)      →  p.numel() = 512                                     
  p.shape = (8, 64, 64) →  p.numel() = 32768
```

```
每个 batch：
  scaler.scale(loss).backward()   ← scaler 负责梯度精度
  scaler.step(optimizer)          ← optimizer 负责参数更新方向
  scaler.update()
  scheduler.step()                ← scheduler 负责学习率衰减（每epoch）
```

``` python
scaler.scale(loss).backward()     
scaler.step(optimizer)
scaler.update()
```
fp16的表示范围很小,反向传播时梯度值很容易变成0.0,

- 第一步:**scaler.scale(loss).backward()**
```
loss (float32)
  ↓ × scale_factor（比如 65536）
scaled_loss (float32)
  ↓ .backward()
各参数的 .grad  ← 全部是放大了 scale_factor 倍的梯度
```

- 第二步:**scaler.step(optimizer)**


```
① 把所有 param.grad 除以 scale_factor，还原为真实梯度
        ↓
② 检查梯度中是否有 inf 或 nan
        ↓
   有 inf/nan？──Yes──→ 跳过本次 optimizer.step()（本次 step 作废）
        ↓ No
   正常执行 optimizer.step()，用真实梯度更新参数

```
- 第三步:**scaler.update()**
根据本次step是否出现inf/nan,动态调整scale_factor

```
本次出现 inf/nan？
  ├─ Yes → scale_factor ÷ 2（缩小，减少下次溢出概率）
  └─ No  → 连续 N 次正常后，scale_factor × 2（放大，尽量保持梯度精度）
```
这个叫做动态loss scaling,让scale_factor自动收敛到一个合适的值

```
forward pass (autocast 区域，fp16计算)
        ↓
loss (fp32)
        ↓ × scale_factor
scaled_loss
        ↓ backward()
param.grad = 真实梯度 × scale_factor
        ↓ scaler.step()
   ┌────┴────┐
   检查 inf/nan
   ├─ 有 → 跳过step，参数不更新
   └─ 无 → grad /= scale_factor → optimizer.step()
        ↓ scaler.update()
   动态调整 scale_factor（放大或缩小）
```

```
scale_factor 太小 → 下溢 → 梯度=0   （不可见，坏）
      ↑ 连续正常就 ×2
   [理想区间]
      ↓ 出现inf就 ÷2
scale_factor 太大 → 上溢 → 梯度=inf （可见，可修复）
```
```
scale_factor 太小  →  梯度下溢 → 0  →  模型学不动（无明显报错，很隐蔽）
scale_factor 太大  →  梯度上溢 → inf →  被 scaler 检测到，自动处理
```
无论太小了还是太大了都考虑到了,区别是太大了被检测为inf,很好找出,自动处理;

```python
scheduler.step()  # 负责控制学习率lr,随着epochs变化
```

## yaml.safe_load

**作用**: 把 yaml 配置文件解析成 Python 字典

**安装**: pip install pyyaml（注意导入时写 import yaml）

**为什么 safe**: 防止 yaml 文件中嵌入恶意代码被执行

**最小例子**:
\```python
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
# cfg["train"]["lr"] → 0.0001
\```

**为什么用 yaml 而不是硬编码**:
参数和代码分离，换实验只换配置文件，代码不动

```
1. 这个东西解决了什么问题？   ← why
2. 最小用法是什么？           ← how  
3. 有什么坑/注意点？          ← gotcha
```

## sed
```
sed -i 's/\.jpg)/\.jpg){width=50%}/g' presentation.md
```
`sed` 是 Linux 的流编辑器，这条命令拆解如下：

```
sed  -i  's/old/new/g'  文件名
```

| 部分 | 含义 |
|------|------|
| `-i` | 直接修改文件（in-place），不加 `-i` 只是打印到屏幕 |
| `s/` | substitute，替换模式 |
| `\.jpg)` | 要找的内容：`.jpg)` （`\.` 是转义，因为`.`在正则里有特殊含义） |
| `\.jpg){width=50%}` | 替换成的内容 |
| `/g` | global，一行里有多个也全替换 |
| `presentation.md` | 操作的文件 |

---

所以这条命令的效果就是：

```
# 改之前
![图片](path/to/image.jpg)

# 改之后  
![图片](path/to/image.jpg){width=50%}
```

直接在终端里运行就行，记得先 `cd` 到文件目录。
```
s / \.jpg) / \.jpg){width=50%} / g
  ↑          ↑                  ↑
分隔符      分隔符              分隔符
```

```
s/查找内容/替换内容/标志
```

```
s|\.jpg)|\.jpg){width=50%}|g
```
这里的改\为|效果一样.