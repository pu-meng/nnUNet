# 目录


#### 目录
[toc]

#### 1. 概述
##### 1.1 项目简介
single_segmentation是把medseg_project和twostage_medseg当作仓库,去逼近nnunet的代码

原本的medseg_project和twostage_medseg有,但是nnunet不一样的

#### 2. single_segmentation项目
##### 2.1 项目描述
单阶段肝脏+肿瘤分割，最大程度对齐 nnUNet 3d_fullres。

##### 2.2 设计目的
验证假设：**两阶段 bbox 裁剪本身是精度瓶颈，而非增广/损失/模型的问题。**  
控制变量：只改"单阶段 vs 两阶段"，其他全部对齐 nnUNet。

##### 2.3 文件结构
```
single_segmentation/
├── transforms.py   nnUNet 对齐的在线增广
├── loss.py         Batch Dice + FocalTversky 损失
├── train.py        主训练脚本
├── eval.py         评估脚本（输出格式与两阶段一致）
└── readme.md       本文件
```

medseg_project 和 twostage_medseg 作为仓库只读，不修改。

#### 3. 与 nnUNet 的对齐项
##### 3.1 对齐项总览
nnUNet 源码参考：`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

| 项目 | nnUNet（源码行） | single（本实现） |
|------|----------------|-----------------|
| 旋转角度 | ±0.524 rad（±30°）`:472` | ±0.524 rad ✓ |
| Elastic Deformation | 关闭 `p_elastic_deform=0` `:738` | 不使用 ✓ |
| Scale | 0.7~1.4, p=0.2 `:740` | RandZoomd ✓ |
| Mirror | axes=(0,1,2) `:473/:806` | 三个 RandFlipd ✓ |
| Gaussian Noise | std=(0,0.1), p=0.1 `:749` | RandGaussianNoised ✓ |
| Gaussian Blur | sigma=(0.5,1), p=0.2 `:755` | RandGaussianSmoothd ✓ |
| Multiplicative Brightness | (0.75,1.25), p=0.15 `:763` | RandScaleIntensityd ✓ |
| **Linear Contrast** | (0.75,1.25), preserve_range, p=0.15 `:770` | RandLinearContrastd ✓ |
| **SimulateLowResolution** | scale=(0.5,1), p=0.25 `:779` | RandSimulateLowResolutiond ✓ |
| **Gamma（倒置）** | p_invert=1, range=(0.7,1.5), p=0.1 `:788` | RandAdjustContrastd invert_image=True ✓ |
| Gamma（正常） | range=(0.7,1.5), retain_stats, p=0.3 `:797` | RandAdjustContrastd ✓ |
| 每 epoch 步数 | 固定 250 步 `:151` | 固定 250 步 ✓ |
| 前景采样比例 | 33% `:149` | 33% ✓ |
| Dice 计算 | Batch Dice `:416` | Batch Dice ✓ |
| padding_mode | constant（填0）`:738` | zeros ✓ |
| spacing | [1.0, 0.757, 0.757] | [1.0, 0.757, 0.757] ✓ |
| patch size | 128³ | 128³ ✓ |
| 优化器 | SGD + Nesterov `:531` | SGD + Nesterov ✓ |
| LR 调度 | Poly power=0.9 | Poly power=0.9 ✓ |
| 数据划分 | — | seed=0, val=0.2, test=0.1（与两阶段一致） ✓ |

**粗体行**为 2025-04-21 补齐的三项（原版本缺失）。

##### 3.2 已知剩余差异
###### 3.2.1 不影响对比结论的差异
- Gaussian noise/smooth 参数：与 nnUNet 接近，非精确复制
- ContrastTransform 在 nnUNet 中无 `synchronize_channels` 区分，single 逐 channel 独立采样（仅单通道 CT，实际等价）

#### 4. 使用方式
##### 4.1 数据准备
**先修好数据目录（去掉嵌套层）：**
```bash
mv /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver/Task03_Liver/* \
   /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver/
rmdir /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver/Task03_Liver
```

##### 4.2 训练
**训练：**
```bash
cd /home/PuMengYu/MSD_LiverTumorSeg/single_segmentation

CUDA_VISIBLE_DEVICES=0 python train.py \
  --exp_name single_nnunet_align \
  --epochs 200 \
  --batch_size 2 \
  --amp
```

##### 4.3 评估
**评估：**
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/single/single_nnunet_align/train/XX/best.pt \
  --split test
```

#### 5. 预期结论
##### 5.1 结论分析
- 若 single 接近 nnUNet（0.755）→ 两阶段 bbox 裁剪是主要瓶颈，值得重构
- 若 single 仍低于 nnUNet → 差距来自其他因素（模型容量、数据量、5折集成等）

#### 6. 具体对应的nnUNet的代码
##### 6.1 旋转角度
###### 6.1.1 代码位置
nnUNetTrainer.py:472

``` python
 do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
```
180. 这里的.表示浮点数,max(patch_size) / patch_size[0]表示最大值除以第一个值,max()是对元组/列表求最大值;

如果最大值除以第一个值大于ANISO_THRESHOLD，则说明(z,x,y)中z的值很小,说明各向异性,采用180度旋转,否则采用30度旋转.一般不考虑x,y差别很大;

##### 6.2 Elastic Deformation
###### 6.2.1 代码位置
nnUNetTrainer.py:738

``` python
  transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,#关闭弹性形变
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )#type:ignore
        )
```
p_elastic_deform=0,关闭弹性形变,
Elastic Deformation:弹性形变,给图像施加随机的局部位移场,让组织像被"蹂"一下,边界变得不规则;

**为什么被关闭?**

nnUNet的作者/论文明确说过,在3d_fullers的配置下关闭elastic是实验得出的结论

- 1.计算代价高,比旋转/缩放慢很多
- 2.效果不稳定,形变过强破坏解剖结果的合理性
- 3.其他增强已经足够

位移场,是指代,原始坐标$(x,y,z)\rightarrow \text{新坐标}(x+dx,y+dy,z+dz)$,这里的dx,dy,dz是随机生成的,且dx,dy,dz是相互独立的,且服从正态分布,均值为0,方差为1,即dx,dy,dz服从N(0,1)分布;

3d_fullers是指代在原始分辨率下裁patch训练,保留最多的细节;也就是不缩放原始的spacing,直接在原的分辨率重采样;是nnUNet在大多数任务熵精度最高的配置

**注意**:医学CT的"分辨率"不是像素数量,而是每个体素代表真实空间多少毫米,就是spacing,

spacing=[1.0, 0.757, 0.757]表示每个体素=$1mm\times 0.757mm\times 0.757mm$的真实空间;
3d_fullers是指代没有为了省下空间就把spacing搞大来缩小体积大小;不同样本nnUNet统计得到目标spacing,后统一重采样到这个,再裁patch,只要没有为了节省空间故意搞大spacing,就是3d_fullers,也是这里的说的原始分辨率.

##### 6.3 Scale
###### 6.3.1 代码位置
nnUNetTrainer.py:740

``` python
      transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,#type:ignore
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            ) #type:ignore
        )

```
- p_scaling=0.2
是每个sample被施加缩放的概率,20%的概率会做缩放增广,80%的概率不缩放保持原样;
- scaling=(0.7, 1.4)
这个是缩放因子的采样范围(均匀分布),均匀采样一个比例s,
- p_synchronize_scaling_across_axes=1
这个是三个轴(x,y,z)的缩放是否同步,
值为1表示100%同步,在一次缩放中,z/y/x用同一个scale factor

我们真实每个step送进网络的是若干个patch,不是整张CT,以patch的中心位置为原点,比如patch:(128,128,128),中心是(64,64,64)

- 仿射矩阵=一个能同时表达旋转,缩放,平移,切变得矩阵;

- 在3D里面是,一个$4\times 4$矩阵,对任意一点 $(x, y, z)$，变换后的新坐标 $(x', y', z')$ 是这样算的：
$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix}
=
\begin{bmatrix}
a_{11} & a_{12} & a_{13} & t_x \\
a_{21} & a_{22} & a_{23} & t_y \\
a_{31} & a_{32} & a_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

- 左上 3×3 块控制 **旋转 + 缩放**；
- 右边一列 $(t_x, t_y, t_z)$ 控制 **平移**。

对于 Scale 增广（缩放因子 s）：

$$
M = \begin{bmatrix}
1/s & 0 & 0 & 0 \\
0 & 1/s & 0 & 0 \\
0 & 0 & 1/s & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

- `s = 1.3` → 对角线是 `1/1.3 ≈ 0.77`，意思是"输出的每一格，去原图里采第 0.77 格的内容"，相当于 **把原图拉大了 1.3 倍**。
- `s = 0.7` → 对角线是 `1/0.7 ≈ 1.43`，意思是"输出每一格，去原图第 1.43 格采"，相当于 **原图被压缩到 0.7 倍**。

**为什么用 `1/s` 而不是 `s`？** 因为这是 **反向映射**：我们是站在"输出网格"的每一格，问"我该去输入图像的哪里取值"。

####### 以 patch 中心为原点

默认情况下，图像的坐标原点在 **左上角（或左下前角）**，即 `(0,0,0)`。

但做旋转、缩放时，如果围绕 `(0,0,0)` 旋转，整个 patch 会飞出去（因为它在第一象限，会绕角落转）。

如果我们放大1.3倍,那么,输出$\rightarrow$输入,输出除以1.3;
如果是缩小0.7,也是输出$\rightarrow$输入,输出除以0.7;

**以 patch 中心为原点**，就是把坐标原点临时移到 patch 正中央（比如 128³ 的中心是 `(64,64,64)`），这样：

- 旋转 30° → 绕着中心转，肝脏还在 patch 里；
- 放大 1.3 倍 → 从中心往外放大，中心的肿瘤还是中心的肿瘤。

2D 示意：

```
原点在角上:              原点在中心:

  ┌─────────┐           ┌────┬────┐
  │ ●       │           │    │    │
  │         │    vs     ├────●────┤
  │         │           │    │    │
  └─────────┘           └────┴────┘
  旋转→飞出                 旋转→还在里面
```

###### 6.3.3 grid_sample

`grid_sample` 是 PyTorch 里的一个函数：**`torch.nn.functional.grid_sample`**。

它做的事：**给定一张输入图像 + 一个"采样网格"，按照网格里每个点指定的坐标去输入图像里取值（插值），产出一张新图像。**

####### 配合仿射矩阵的完整流程

```
1. 准备一个 128³ 的"空白输出网格"，坐标规范化到 [-1, 1]
     │
2. 把仿射矩阵 M 作用到这个网格上
   → 每一格得到一个"在原图哪个位置取值"的坐标
     │
3. grid_sample(原图, 变换后的网格)
   → 对每一格，去原图对应位置做插值采样
     │
4. 得到变换后的 patch（128³，但内容被旋转/缩放过）
```

####### 插值方式

- **图像（CT 强度）** → 三线性插值（trilinear），相邻 8 个体素加权平均，结果平滑；
- **标签（label）** → 最近邻插值（nearest），必须是 0/1/2 等整数类别，不能插出 0.5。

####### 为什么不直接 "resize 然后 crop/pad"

- 仿射 + grid_sample 可以 **一次性** 把"旋转 30° + 缩放 1.3 倍 + 轻微平移"合成到 **一个矩阵** 里，只做 **一次重采样**；
- 如果分步做（先旋转一次、再缩放一次），每次插值都会损失一点精度，累积模糊；
- 而且输出 shape 永远是 128³，显存可控。

###### 6.3.4 完整 Scale 增广流程

以 `s=1.3`（放大）为例：

1. 从 CT 里随机裁一个 128³ patch；
2. 构造仿射矩阵 M（对角 `1/1.3`，以 patch 中心为原点）；
3. 用 M 生成采样网格：输出的 `(64,64,64)` 格去采输入的 `(64,64,64)`（中心不动），输出的 `(0,0,0)` 格去采输入大约 `(64-64/1.3, ...)` 的位置；
4. `grid_sample` 按这个网格从原 patch 里三线性采样，得到一个新 128³；
5. 新 patch 的视觉效果：**肝脏/肿瘤被放大了 1.3 倍**，边缘可能有 0 填充（因为有些采样点落在原 patch 外面）。

整个过程既没改变 patch 的 shape，也保证了等比、不变形、只插值一次。

对输出的每一个格 (i, j, k)：
- 1. 通过仿射矩阵 M，算出它对应的输入坐标 (i', j', k')
- 2. (i', j', k') 一般不是整数，比如 (76.3, 64.0, 64.0)
- 3. 在输入图像中，取 (76.3, 64, 64) 周围的 8 个整数格子
- 4. 按距离加权做三线性插值，得到一个值
- 5. 把这个值填到输出的 (i, j, k)
##### 6.4 Mirror
###### 6.4.1 代码位置
nnUNetTrainer.py:473页/806页
``` python
473页
  elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
```
``` python
806页
   if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )
```
mirror_axes=(0, 1, 2)表示允许在三个轴熵进行镜像:
轴0:z轴
轴1:x轴
轴2:y轴

在你的肝脏肿瘤分割任务中，MirrorTransform特别有用，因为：

- 肝脏大致对称
- 肿瘤可能出现在任何位置
- 增强模型对不同方向扫描的适应性

这个也是为什么nnUNet在3D训练中,默认启用三个轴的镜像增强的原因.
##### 6.5 Gaussian Noise
###### 6.5.1 代码位置
nnUNetTrainer.py:742

``` python
         if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
```
- noise_variance=(0, 0.1):每次触发时,从[0,0.1]均匀采样一个方差$\sigma^2$,然后生成$N(0,\sigma^2)$噪声加到图像上,不是固定强度,每次随机.

- p_per_channel=1:1等价于100%,表示每个通道都加噪声.
- synchronize_channels=True:表示同步加噪声,多通道时,所有通道用一个采样出来的$\sigma^2$,而不是每个通道各采各的;
- apply_probability=0.1:表示有10%的概率触发这个增强.
- RandomTransform:统一管理触发概率,接口一致

##### 6.6 Gaussian Blur(755页)

``` python
755页

        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
```
- blur_sigma=(0.5, 1.):每次触发时,从[0.5,1]均匀采样一个方差$\sigma^2$,然后生成高斯模糊.
- synchronize_channels=False:表示不同通道用不同的$\sigma^2$.
- synchronize_axes=False:表示不同轴用不同的$\sigma^2$.
- p_per_channel=0.5:表示每个通道有50%的概率加噪声.
- apply_probability=0.2:表示有20%的概率触发这个增强.
- benchmark=True:用PyTorch的cudnn.benchmark模式,让CUDA自动选择最快的卷积算法,纯速度优化
- RandomTransform:统一管理触发概率,接口一致
- 高斯模糊是一个卷积操作;
我们选定好$sigma$后,卷积核的大小是:
比如,3D核:
- 单边范围=ceil($3\times \sigma$),ceil是向上取整,ceil(1.2)=2
- 核大小=2*单边范围+1
- 核的shape=核大小$\times$核大小$\times$核大小

- 每个位置的数值:
核中心是原点(0,0,0),某个位置的(x,y,z)的值:
$$G(x,y,z)=e^{-\frac{x^2+y^2+z^2}{2\sigma^2}}$$
算完所有值之后,归一化,让核的权重之和=1

##### 6.7 Multiplicative Brightness(763页)
``` python

        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),#type:ignore
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
```
- MultiplicativeBrightnessTransform:$$\text{输出}=\text{输入}\times m $$
- multiplier_range=BGContrast((0.75, 1.25)):
- BGContrast是batchgeneratorsv2的一个采样器,从[0.75,1.25]均匀采样一个值m,作为乘数.然后:
$$\text{输出像素}=\text{输入像素}\times m $$
- multiplier_range接收的是一个采样器对象,




##### 6.8 Linear Contrast(770页)
``` python
  transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),#type:ignore
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))

```
BGContrast是batchgeneratorsv2的一个采样器,从[0.75,1.25]均匀采样一个值m,返回这个值,不干任何其他工作;也就是,BGContrast返回的是一个浮点数;

$\text{output}=\mu+(input-\mu)\times m$,这里的$\mu$是该图像/通道的均值,
synchronize_channels=通道同步,如果不同步,就是每个通道独立采样$m$,各个通道用不同的值.

##### 6.9 SimulateLowResolution(779页)
###### 6.9.1 代码
``` python
     transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,#type:ignore
                allowed_channels=None,#type:ignore
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
```
- SimulateLowResolutionTransform:input先下采样(scale=s)得到小图,再上采样,插值还原(scale=1/s)得到原图,相当于对原图进行模糊.
- ignore_axes:忽略的轴,比如ignore_axes=(2,),表示忽略z轴,只对x,y轴进行下采样.
- allowed_channels:允许采样的通道,比如allowed_channels=(0,1),表示只对0,1通道进行下采样.
- allowed_channels=None:表示所有通道都进行下采样.
- p_per_channel=0.5:每个通道有50%的概率进行下采样.
- RandomTransform(...,apply_probability=0.25):有25%的概率触发这个增强
输入大$\rightarrow$输出小=下采样
输入小$\rightarrow$输出大=上采样


 ### 6.9.2 **三次样条插值公式**

设输出点 $(i,j)$ 映射到输入坐标 $(x,y)$（非整数），令整数部分和小数部分分别为：

$$x_0 = \lfloor x \rfloor, \quad y_0 = \lfloor y \rfloor, \quad d_x = x - x_0, \quad d_y = y - y_0$$
这里的$x_0$是整数,$\lfloor x\rfloor$是针对$x$向下取整.
则：

$$
\begin{aligned}
\text{output}(i,j) &= \sum_{m=-1}^{2}\sum_{n=-1}^{2} W(d_x - m) \cdot W(d_y - n) \cdot \text{input}(x_0+m,\ y_0+n)\\
&= \sum_{m=-1}^{2}\sum_{n=-1}^{2} W(x-x_0 - m) \cdot W(y-y_0 - n) \cdot \text{input}(x_0+m,\ y_0+n)
\end{aligned}
$$

权重函数：

$$W(t) = \begin{cases} 1.5|t|^3 - 2.5|t|^2 + 1 & 0 \leq |t| < 1 \\ -0.5|t|^3 + 2.5|t|^2 - 4|t| + 2 & 1 \leq |t| < 2 \\ 0 & |t| \geq 2 \end{cases}$$

---

###### 6.9.3 关键点

- $\text{input}(x_0+m, y_0+n)$ 是以 $(x_0, y_0)$ 为中心的 **4×4 邻域像素**，随 $(i,j)$ 不同而不同
- $W(d_x - m)$ 中 $d_x \in [0,1)$，$m \in \{-1,0,1,2\}$，所以 $d_x - m$ 的范围恰好覆盖 $(-2, 2)$，正好落在 $W$ 有效范围内;
为什么$input(x_0+m,y_0+n)$中的$m=-1,0,1,2$,因为$x_0\leq x<x_0+1$;

##### 6.10 Gamma(倒置)(788页)
```python
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),#type:ignore
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
```

第一步：倒置p_invert_image=1
$$inv=max(x)-x$$
这里的p_invert_image=1,表示100%执行倒置



图像明暗翻转，亮的变暗，暗的变亮。

第二步：Gamma变换:

$$output={inv}^{\gamma},\gamma∼ Uniform(0.7,1.5)$$

- $\gamma<1$:暗区变亮(提升暗部细节)
- $\gamma>1$:亮区变暗(提升亮部细节)
- $\gamma=1$:无变化

**对比度的精确定义:**


考虑输入图像中**两个相邻像素** $x$ 和 $x + \Delta x$，经过映射 $f(x) = x^\gamma$ 之后：

$$\text{输入差} = \Delta x$$
$$\text{输出差} = f(x+\Delta x) - f(x) \approx f'(x) \cdot \Delta x$$



**拉伸和压缩的精确含义**

$$\text{比值} = \frac{\text{输出差}}{\text{输入差}} = \frac{f'(x) \cdot \Delta x}{\Delta x} = f'(x)$$
所以：
$$f'(x) > 1 \Rightarrow \text{输出差} > \text{输入差} \Rightarrow \text{两像素更可分辨} \Rightarrow \text{拉伸（对比度增强）}$$
$$f'(x) < 1 \Rightarrow \text{输出差} < \text{输入差} \Rightarrow \text{两像素更难分辨} \Rightarrow \text{压缩（对比度降低）}$$



** 代入 $\gamma=0.5$ 验证**

$$f'(x) = \frac{1}{2\sqrt{x}}$$

| 区域 | $x$ | $f'(x)$ | 含义 |
|---|---|---|---|
| 暗区 | 0.1 | 1.58 | 相邻像素输出差变大，细节增强 |
| 亮区 | 0.9 | 0.53 | 相邻像素输出差变小，细节丢失 |

---

**核心**

对比度不是一个像素的绝对值，**是两个相邻像素之间的差**，$f'(x)$ 就是衡量这个差被放大还是缩小的系数。

第三步：p_retain_stats=1,

Gamma变换会该百年图像的均值和方差,retain_stats将其强制还原
$$
\text{output}=\frac{output-\mu_{after}}{\sigma_{after}}\times\sigma_{before}+\mu_{before}
$$
即做了一次均值方差对齐，保证整体亮度分布不漂移。



##### 6.11 Gamma(正常)(797页)
``` python

        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),#type:ignore
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
```
##### 6.12 每epoch步数(151页)
```python
  self.num_iterations_per_epoch = 250
```
nnUNet不是把整个数据集跑过一遍算一个epoch,而是固定每个epoch跑多少个迭代
- 每个iteration=采样一个batch,前向传播,反向传播,更新参数
- 好处是,数据集大小不影响epoch长度,学习率调度,checkpoint保存等基于epoch数,行为一致
##### 6.13 前景采样比例(149页)
```python
 self.oversample_foreground_percent = 0.33
```
文件:/home/PuMengYu/nnUNet/nnUNet/nnunetv2/preprocessing/preprocessors/default_preprocessor.py 239行
``` python
    def _sample_foreground_locations(
        seg: np.ndarray,
        classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
        seed: int = 1234,
        verbose: bool = False,
        min_num_samples=10000,
        min_percent_coverage=0.01,
    ):
```
我自己是先从.nii.gz预处理成为.pt,之后进行各种处理;nnUNet是先预处理成
##### 6.14 Dice计算(416页)
```python

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)   #type:ignore

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
```
##### 6.15 padding_mode(738页)
```python
transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,#type:ignore
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            ) #type:ignore
        )
```
##### 6.16 spacing()

##### 6.17 patch_size

##### 6.18 优化器:SGD+Nesterov(531页)
```    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,#type:ignore
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
```
##### 6.19 LR调度

#### 7. nnUNet预处理总体流程

我自己的做法：`nii.gz` → 离线预处理 → 保存为 `.pt` → 训练时直接读

nnUNet的做法完全一样，也是离线预处理保存中间文件，训练时直接读：

```
nii.gz（原始数据）
   ↓
第一步：DatasetFingerprintExtractor（整个数据集跑一次）
   ↓ 输出 dataset_fingerprint.json
第二步：DefaultPreprocessor.run_case_npy（每个case单独跑）
   ├── 2.1 transpose（轴重排）
   ├── 2.2 crop_to_nonzero（裁掉背景）
   ├── 2.3 normalize（归一化，在resample之前！）
   ├── 2.4 resample to target_spacing
   └── 2.5 _sample_foreground_locations（预存前景坐标）
   ↓ 输出 case.b2nd + case_seg.b2nd + properties
训练时读 .b2nd，直接送 DataLoader
```

**和我的 `.pt` 的区别**：

| | 我的方案 | nnUNet v2 |
|---|---|---|
| 格式 | `.pt`（`torch.save`）| `.b2nd`（blosc2）|
| 内容 | `{"image": Tensor, "label": Tensor}` | 图像/标签分开存，properties 单独存 |
| 加载方式 | `torch.load(..., mmap=True)` | `blosc2.open(..., mode='r')` |
| IO优化 | mmap 按页加载 | chunk 按 patch_size 对齐，**裁 patch 时只读对应 chunk** |

.b2nd可以节省CPU RAM,chunk_size应该和patch_size接近或相等,.b2nd开始生成一次,之后一直用,nnUNet也是离线预处理;

比如.pt整体为(100,288,288),patch:(50,100,100),实际读入(50,288,288);

.b2nd chunk_size=(50,100,100),实际读入(50,100,100),但是有时候chunk 不是和patch完美切合的,到时候需要读多个chunk,


##### 7.1 第一步：数据指纹提取（DatasetFingerprintExtractor）

文件：`nnunetv2/experiment_planning/dataset_fingerprint/fingerprint_extractor.py`

整个训练集只跑**一次**，输出 `dataset_fingerprint.json`，供后续所有 planning 使用。

##### 7.2 每个case做的事（analyze_case）

```python
#### 1.找所有通道nonzero的union bounding box,把图像和标签都裁到这个范围,返回裁剪后的data,seg,bbox
data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)
#segmentation是标签
#2.从前景区域采样强度值
foreground_intensities_per_channel, foreground_intensity_stats_per_channel = 
    DatasetFingerprintExtractor.collect_foreground_intensities(
        seg_cropped, data_cropped,
        num_samples=num_foreground_samples_per_case  # 整个数据集共采 10e7 个体素
    )
    #data_cropped:裁掉边缘空气的CT灰度数组
    #seg_cropped:用同一个bbox裁掉的标签数组
#3.记录spacing和形状信息
spacing = properties_images['spacing']
shape_before_crop = images.shape[1:]
shape_after_crop = data_cropped.shape[1:]
relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
#### relative_size_after_cropping=裁后体积/原始体积,反应这个case有多少是"有效区域",后续用这个值决定归一化方式
```
前景=seg>0的体素,也就是标注liver或tumor的地方

mask=一张和图像同形状的True/False数组,用来"圈定范围"

- `num_foreground_samples_per_case = int(10e7 / len(dataset))`，总量固定 1 亿个体素，平摊到每个 case
- 强度统计**只在前景体素**（`seg > 0`）里采样，不包含背景空气

##### 7.3 采集强度统计（collect_foreground_intensities）

```python
#### fingerprint_extractor.py:59
foreground_mask = segmentation[0] > 0   # seg>0 即前景
percentiles = np.array((0.5, 50.0, 99.5))

for i in range(len(images)):
    foreground_pixels = images[i][foreground_mask]
    # 有放回采样，防止前景体素少于 num_samples 的 case 被欠代表
    intensities_per_channel.append(
        rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else []
    )
```
- rs.choice：比如有1000万个体素,随机抽取num_samples个体素,

##### 7.4 汇总整个数据集，写出 fingerprint.json

```python
#### fingerprint_extractor.py:165
#### 把所有 case 的前景体素合并后统一算百分位
foreground_intensities_per_channel = np.concatenate([r[2][i] for r in results])
percentile_00_5, median, percentile_99_5 = np.percentile(foreground_intensities_per_channel, [0.5, 50.0, 99.5])

fingerprint = {
    "spacings": spacings,                                # 每个case的spacing，list[list]
    "shapes_after_crop": shapes_after_crop,              # 每个case crop后的shape
    "foreground_intensity_properties_per_channel": {
        0: {
            "mean": ...,   "std": ...,
            "min": ...,    "max": ...,
            "percentile_99_5": ...,
            "percentile_00_5": ...
        }
    },
    "median_relative_size_after_cropping": ...           # 所有case relative_size的中位数
}
```

**fingerprint.json 的字段含义：**

| 字段 | 含义 | 用途 |
|------|------|------|
| `spacings` | 每个case的原始spacing | 计算 target_spacing（取中位数）|
| `shapes_after_crop` | crop后的shape | 计算 median shape，决定 patch size |
| `mean/std` | 全数据集前景体素的均值/标准差 | CT 归一化时用 |
| `percentile_00_5/99_5` | 全数据集前景体素的0.5/99.5分位 | CT clip 时用 |
| `median_relative_size_after_cropping` | crop后体积占原始体积的中位数 | 判断是否需要 mask_for_norm |

---

##### 7.5 第二步：Crop（crop_to_nonzero）

文件：`nnunetv2/preprocessing/cropping/cropping.py`

```python
#### default_preprocessor.py:92
shape_before_cropping = data.shape[1:]
properties["shape_before_cropping"] = shape_before_cropping

data, seg, bbox = crop_to_nonzero(data, seg)

properties["bbox_used_for_cropping"] = bbox                         # 存bbox，推理时用于还原
properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]
```

- 找所有通道 nonzero 的 union bounding box，crop 到此范围
- **保存 bbox**，推理结束后 resample 回原 spacing，再 pad 回 crop 前的 shape
- crop 只对训练前的原始 case 做一次，不是每次 getitem 做

---

##### 7.6 第三步：归一化（normalize）

**注意顺序：normalize 在 resample 之前！**

```python
#### default_preprocessor.py:109
#### normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
#### longer fitting the images perfectly!
data = self._normalize(data, seg, configuration_manager,
                       plans_manager.foreground_intensity_properties_per_channel)
```

##### 7.7 CT 归一化（CTNormalization）

文件：`nnunetv2/preprocessing/normalization/default_normalization_schemes.py:58`

```python
def run(self, image, seg=None):
    lower_bound = self.intensityproperties['percentile_00_5']  # 来自fingerprint（全数据集）
    upper_bound = self.intensityproperties['percentile_99_5']  # 来自fingerprint（全数据集）
    mean_intensity = self.intensityproperties['mean']
    std_intensity  = self.intensityproperties['std']

    np.clip(image, lower_bound, upper_bound, out=image)   # step1: clip 去极值
    image -= mean_intensity                                # step2: z-score
    image /= max(std_intensity, 1e-8)
    return image
```

- mean/std 是**全训练集**前景体素汇总后算的（不是 per-case），来自 fingerprint.json
- clip 边界也是全训练集前景的 0.5/99.5 分位

##### 7.8 MRI 归一化（ZScoreNormalization）

文件：`nnunetv2/preprocessing/normalization/default_normalization_schemes.py:27`

```python
def run(self, image, seg=None):
    if seg is not None and self.use_mask_for_norm:
        mask = seg >= 0            # seg=-1 表示 nonzero crop 外的区域（背景）
        mean = image[mask].mean()  # 只在 mask 内（前景+roi）算均值
        std  = image[mask].std()
        image[mask] = (image[mask] - mean) / max(std, 1e-8)
        # mask 外的像素保持 0（leaves_pixels_outside_mask_at_zero=True）
    else:
        mean = image.mean()        # per-case 全图算
        std  = image.std()
        image = (image - mean) / max(std, 1e-8)
    return image
```

- `use_mask_for_norm=True` 的触发条件：`median_relative_size_after_cropping < 0.75`
  （crop 后体积不到原体积 75%，说明背景很多，只用前景区域的统计量更准）
- MRI 是 **per-case** 归一化，不用全数据集统计量

---

##### 7.9 第四步：重采样（resample）

文件：`nnunetv2/preprocessing/resampling/default_resampling.py`

##### 7.10 新 shape 计算

```python
#### default_resampling.py:29
new_shape = [int(round(old_spacing[i] / new_spacing[i] * old_shape[i])) for i in range(3)]
```

物理尺寸不变，spacing 变小 → shape 变大（体素更多更密）

##### 7.11 各向异性判断（separate_z）

```python
#### default_resampling.py:15
do_separate_z = (np.max(spacing) / np.min(spacing)) > ANISO_THRESHOLD  # ANISO_THRESHOLD=3.0
```

- 如果最大spacing/最小spacing > 3，判断为各向异性（z轴分辨率低）
- 触发 separate_z=True，z轴和xy轴用不同的插值阶数

##### 7.12 插值方式

| 数据类型 | 各向同性 | 各向异性（separate_z=True）|
|----------|----------|---------------------------|
| 图像（data）| order=3（三次样条，xy和z都用）| xy: order=3，z轴: order=0（最近邻）|
| 标签（seg）| order=0（最近邻）| order=0（始终）|

各向异性时 z 轴用 order=0 的原因：z 轴的 spacing 很大（如 5mm），相邻层之间内容差异大，高阶插值会产生模糊的中间值，反而不如最近邻。

##### 7.13 图像插值（skimage.transform.resize）

```python
#### 各向同性，图像
resize_fn = resize   # skimage
kwargs = {'mode': 'edge', 'anti_aliasing': False}
reshaped_final[c] = resize_fn(data[c], new_shape, order=3, **kwargs)
```

##### 7.14 各向异性的 separate_z 两步重采样

```python
#### 第一步：在 xy 平面做 order=3 插值（每一层 slice 单独 resize）
for slice_id in range(shape[z_axis]):
    reshaped_here[slice_id] = resize(data[c, slice_id], new_shape_2d, order=3)

#### 第二步：沿 z 轴做 order=0（最近邻）插值
reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=0, mode='nearest')
```

---

##### 7.15 第五步：采样前景坐标（_sample_foreground_locations）

文件：`nnunetv2/preprocessing/preprocessors/default_preprocessor.py:239`

在重采样**之后**执行，坐标是**重采样后**的坐标空间：

```python
#### default_preprocessor.py:152
properties["class_locations"] = self._sample_foreground_locations(
    seg, collect_for_this, verbose=self.verbose
)
```

每个 case 对每个前景类采样最多 10000 个体素坐标（`min_num_samples=10000`），存入 properties：

```python
{
    1: array([[z1,y1,x1], [z2,y2,x2], ...]),   # liver 体素坐标（≤10000个）
    2: array([[z3,y3,x3], ...])                  # tumor 体素坐标
}
```

训练时 DataLoader 的 `get_bbox` 直接用这个字典，**不需要在训练时重新 argwhere**，节省 IO 和 CPU 开销。

---

##### 7.16 保存格式（.b2nd）

```python
#### default_preprocessor.py:227
nnUNetDatasetBlosc2.save_case(
    data, seg, properties,
    output_filename_truncated,
    chunks=chunk_size_data,    # chunk 大小按 patch_size 计算，随机读 patch 时只加载对应 chunk
    blocks=block_size_data,
)
```

输出文件：
```
nnUNet_preprocessed/Dataset/nnUNetPlans_3d_fullres/
    case_001.b2nd          # 图像，float32，[C, D, H, W]
    case_001_seg.b2nd      # 标签，int16，[1, D, H, W]
    case_001.pkl           # properties，含 spacing、bbox、class_locations
```

blosc2 chunk 按 patch_size 对齐的意思：随机裁一个 `128×128×128` 的 patch，只从磁盘读那几个 chunk，而不是把整个 `512×512×300` 的 volume 加载进内存。对大体积 case 特别省内存。

**和我的 `.pt` + mmap 的本质区别**：
- `.pt` mmap：操作系统级别的 page（4KB）按需加载，粒度细但随机读一个 patch 可能触发很多 page fault
- `.b2nd` chunk：应用级别的 chunk（patch 大小），随机读 patch 时只加载恰好覆盖该 patch 的 chunk，IO 次数少


---
