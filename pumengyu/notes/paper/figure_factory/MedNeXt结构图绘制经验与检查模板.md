# MedNeXt 结构图绘制经验与检查模板

> 定位：论文绘图规范的 MedNeXt 专项补充，用于代码核对、层级拆分和交付检查。

这份笔记总结 MedNeXt_MLA 结构图的绘制经验，作为后续新增或修改结构图的统一模板。核心原则是：**先按真实代码确定计算关系，再安排图的层级和版式；不能为了让图好看而改变代码语义。**

## 一、推荐的讲解顺序

结构图应从大到小拆成五个层次：

1. **网络总览**：输入、Input Projection、Encoder Stage、DownBlock、Bottleneck、UpBlock、Decoder Stage、输出头以及 Encoder–Decoder skip。
2. **Stage 内部**：多个 MedNeXtBlock 如何顺序连接，以及每个 block 的残差只在本 block 内闭合。
3. **Block 内部**：主分支、identity 分支、逐元素相加的位置。
4. **过渡模块**：DownBlock 和 UpBlock 的下采样/上采样主分支与 resampling shortcut。
5. **基础算子**：深度可分离卷积、倒置瓶颈、GroupNorm、GELU、逐元素 add 等。

不要在图 2(a) 重新复制整套 Encoder；总览已经说明 Stage 数量和网络位置，后续子图应回答“这一层内部具体做什么”。

## 二、读代码后必须记录的五件事

对每个模块先定位真实类和 `forward()`，再填写下面的表：

| 项目 | 必须回答的问题 |
|---|---|
| 输入 | 来自哪个模块？形状是 `[B,C,D,H,W]` 还是序列？ |
| 主分支 | 依次经过哪些真实层？stride、kernel、groups、通道数是多少？ |
| 旁路 | 是否存在 identity、projection 或 resampling shortcut？是否改变尺寸？ |
| 合并 | 是逐元素 `add`、`concat` 还是仅并联展示？合并发生在哪里？ |
| 输出 | 合并后是否还经过层？最终输出去哪里？ |

尤其不能只看类名判断残差。要同时检查：

- `__init__()` 中 `do_res` 是否被保存或覆盖；
- `forward()` 中是否真的执行 `x + res`；
- shortcut 是否使用卷积改变通道或空间尺寸；
- 上采样后是否有 `pad` 或 shape 对齐。

## 三、当前代码对应的残差关系

### 1. MedNeXtBlock

一个普通 MedNeXtBlock 有两条分支：

- 主分支：深度卷积 → GroupNorm → `1×1×1` expansion → GELU → `1×1×1` projection；
- identity 分支：输入直接保留；
- 末端：主分支与 identity **逐元素 add**。

Stage 内的多个 block 是顺序连接的。每个 block 各自完成一次 add，前一个 block 的输出才进入下一个 block；Stage 本身没有额外的跨所有 block 残差。

### 2. DownBlock / UpBlock

当前实现中，DownBlock 和 UpBlock 调用父类主分支时显式关闭父类 residual，然后在模块末尾自己完成一次 resampling residual：

- DownBlock：主分支使用 depthwise `Conv3d(stride=2)`；黄色旁路使用 `1×1×1 Conv3d(stride=2)`；最后 `main + shortcut`。
- UpBlock：主分支使用 depthwise `ConvTranspose3d(stride=2)`；黄色旁路使用 `1×1×1 ConvTranspose3d(stride=2)`；必要时先 pad 对齐，再 `main + shortcut`。

因此 f/g 子图中，黄色分支不是第二个普通 block，也不是 concat；它表示**该 DownBlock/UpBlock 内唯一的 resampling shortcut add**。

### 3. Encoder–Decoder skip

Encoder 的 Stage 输出会在进入 DownBlock 前保存为 `x_res_i`。Decoder 中先执行 UpBlock，再与对应的 `x_res_i` 做逐元素 add，之后才进入 Decoder Stage 的 MedNeXtBlocks：

`UpBlock → Encoder skip add → Decoder MedNeXtBlocks`

这个 add 与 MedNeXtBlock 内部 add、Down/UpBlock 内部 add 是三个不同层次，图中必须用位置或图注明确区分。

### 4. Bottleneck

瓶颈层是多个 MedNeXtBlock 的顺序堆叠；当前配置为 8 个 block。每个 block 内部有自己的 identity add，但 8 个 block 之间没有额外的大残差环路。

## 四、子图模板

### Stage 子图

- 顶部：`Stage input [B,C,D,H,W]`；
- 中间：`MedNeXtBlock 1 → MedNeXtBlock 2 → …`；
- 每个 block 单独画一个 identity 旁路和圆形 `+`；
- 底部：`Stage output [B,C,D,H,W]`；
- 不再额外画一个跨整个 Stage 的残差。

### Block 子图

- 主路径纵向排列；
- identity 从 Input 的右侧水平引出，绕到末端；
- identity 箭头最终指向 `+` 的边界；
- `+` 明确表示 element-wise add，不能画成 concat；
- 只在图中保留关键层名和关键维度，其余解释放图注。

### DownBlock / UpBlock 子图

- 主路径放在左侧并纵向排列；
- 黄色 shortcut 放在右侧中部，与主路径保持明显横向间距；
- Input 先水平引出，再垂直进入黄色 shortcut；
- shortcut 输出垂直向下，再水平进入底部 `+`；
- 主分支和 shortcut 在底部只合并一次；
- 不用长斜线，不让箭头穿过色块、文字或加号。

### Decoder Stage 子图

- 先画 `MedNeXt UpBlock`；
- 再画 Encoder skip 与 UpBlock 输出的 add；
- 最后画 Decoder Stage 内部的 MedNeXtBlocks；
- 不把 “residual + MedNeXt UpBlock” 写成一个模块名，因为 UpBlock 内部残差和 Encoder–Decoder skip 是两次不同的 add。

## 五、本轮反复修改中暴露的典型错误

1. **把 Stage、Block、Encoder–Decoder skip 混为同一种残差**：解决办法是按层级分别标注 add 的位置。
2. **把 DownBlock/UpBlock 画成普通 MedNeXtBlock**：忽略了 stride、转置卷积和 resampling shortcut。
3. **把 shortcut 误画成 concat 或 expansion 分支**：当前 shortcut 只是 `1×1×1` 投影/重采样后与主分支 add。
4. **在图 2(a) 重复整套 Encoder**：造成信息重复和拥挤，应改成单独的 Stage 内部结构。
5. **把小字全部塞进色块**：导致灰色文字与边框或其他色块重合；只保留模块名和一行关键维度，其余移入图注。
6. **标题贴近色块**：标题与第一个色块之间预留稳定的垂直间距。
7. **使用斜向或断开的箭头**：优先使用水平/垂直正交连线，所有箭头头部必须完整可见。
8. **黄色 shortcut 离主路径太近**：先确定主路径宽度，再把 shortcut 放到右侧中部并留出空白。
9. **没有说明色块标签是输入还是输出**：图注统一说明标签含义，图内避免重复长句。
10. **只看图猜代码**：最终必须回到类定义和 `forward()` 验证层顺序、通道、stride、padding 和 add 次数。

### 2026-09-17：箭头看似缺失的排查经验

- 图 B 的 Block 内部箭头误加了 `+2` 坐标偏移，箭头没有画在相邻模块之间；Stage 顶部各框的间距也太短。先逐段核对箭头端点与框的实际边界，再拉开框距。
- 图 D 的相邻框只留约 `0.15–0.16` 个数据坐标单位，箭身几乎不可见；加号还被按普通矩形计算上下端点。应增加纵向空间，并按圆心与半径计算进入、离开 `+` 的端点。
- `FancyArrowPatch` 默认会在两端收缩。若端点已放在色块或加号的边界，可设置 `shrinkA=0, shrinkB=0`；仍要在导出的 PNG 中检查线段和箭头头部，不能仅凭绘图代码或 SVG 文件存在判断连线正确。

### 2026-09-19：图 D 纵向串联与图 E 的 MoE 展开

- **图 D 的问题与修正：**原布局将两个 Transformer Block 各排成一整行，输出从右端绕回下一行左端，读图需要反复折返。改为同一中轴从上到下串联，残差从侧边汇入；增加画布高度，保留两个完整 Block 和各自两次残差相加。
- **加号后的箭头也要检查：**复用旧纵向布局时发现加号下方到输出框的空隙仍过短。按圆周端点重新留出间距，同时检查输入变换到 Block 1、Block 1 到 Block 2、Block 2 到最终归一化的连线是否完整。标题放在侧边，说明文字避开箭身。
- **图 E 的范围：**从图 D 的 LayerNorm₂ 输出进入 MoE-FFN，分别送往共享专家、4 个路由专家和 Router；展示 Top-2 索引、选中原始分数的 softmax、专家输出加权、shared + routed，最后还原形状并返回图 D 的 Add_moe。底部单独展开一个专家的内部层。
- **代码事实：**当前 4 个路由专家全部计算后才 gather 两个输出；均衡偏置只影响选谁。专家宽度为 `512 → 1024 → 512`，当前 Dropout 为 0。图 E 内部相加不能与图 D 外层残差混同；这些配置以后随真实实现复核。
- **本次交付：**[结构图 SVG](../../模型结构图/model_architecture.svg) 与同目录 300 DPI PNG 已更新；[生成脚本](../../../tools/model_audit/draw_model_architecture.py) 中图 D 与独立瓶颈图复用纵向 Block 绘制函数，图 E 由 `draw_moe_panel()` 生成。已核对 `MHATransformerBlock`、`MHABottleneck3D`、`MoEFFN.forward()` 和 `FFNExpert.forward()`，完成脚本编译、SVG 解析及 PNG 目视检查。

## 六、交付前检查表

- [ ] 每个色块都能在真实代码中找到对应层或操作；
- [ ] 输入、输出、通道数和空间尺寸方向正确；
- [ ] 每个 `add` 都有明确的两条输入；
- [ ] 没有把 `add` 误画成 `concat`，也没有把并联误写成残差；
- [ ] Stage、Block、Down/UpBlock、Encoder–Decoder skip 的层级边界清楚；
- [ ] shortcut 是否改变尺寸、是否需要 projection/pad 已核对；
- [ ] 箭头完整、正交、少转角，不穿过文字和色块；
- [ ] 标题、色块、加号和旁路之间有足够空白；
- [ ] 小字不过载，细节优先放图注；
- [ ] 脚本可编译并重新生成 SVG/PNG；
- [ ] 在论文实际尺寸下打开 PNG 检查可读性；
- [ ] 正文只解释原因、优点和边界，不重复抄图中每一个步骤。

## 七、推荐的最小实现模板

新画一个模块时，先在绘图脚本中按下面顺序实现：

1. 固定画布、中心线和主路径宽度；
2. 先放 Input、主路径色块和 Output；
3. 再放 `+`，确认主路径箭头能够完整进入；
4. 最后添加 identity/skip/shortcut，并使用正交折线连接；
5. 删除不影响结构判断的小字；
6. 生成 SVG/PNG，实际打开检查后再修改间距，而不是直接缩小字体；
7. 将无法放入图内的实现细节写进图注或正文。
