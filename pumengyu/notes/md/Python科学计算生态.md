# Python 科学计算生态

---

## 数组和数值计算

| 库 | 定位 | 特点 |
|----|------|------|
| numpy | 基础，所有库的底层 | 数组运算、线性代数、傅里叶变换 |
| cupy | numpy 的 GPU 版 | API 几乎和 numpy 一样，把 np 换成 cp 就能跑在 GPU 上 |
| jax | Google 出品 | numpy API + 自动微分 + JIT 编译，学术界流行 |

---

## 统计和数学

| 库 | 定位 | 特点 |
|----|------|------|
| scipy | numpy 上层 | 统计检验、优化、信号处理、插值、积分、线性代数 |
| statsmodels | 专业统计建模 | 线性回归、时间序列、假设检验、ANOVA |
| pingouin | 友好统计库 | 比 scipy.stats 更易用，一行代码出完整统计报告 |
| sympy | 符号数学 | 推导公式、求导、积分，输出数学表达式而不是数值 |

---

## 机器学习

| 库 | 定位 | 特点 |
|----|------|------|
| scikit-learn | 传统 ML 全家桶 | SVM、随机森林、聚类、降维、交叉验证 |
| xgboost | 梯度提升树 | 表格数据竞赛标配 |
| lightgbm | 梯度提升树 | 微软出品，比 xgboost 更快 |

---

## 深度学习

| 库 | 定位 | 特点 |
|----|------|------|
| torch | Meta 出品 | 动态图，研究界主流，自动微分 + GPU |
| tensorflow | Google 出品 | 工业部署强 |
| jax + flax | 学术界新宠 | 函数式风格，XLA 编译极快 |

---

## 医学图像专用

| 库 | 定位 | 特点 |
|----|------|------|
| SimpleITK | 医学图像处理 | 读写 .nii.gz，图像配准，坐标变换 |
| nibabel | 神经影像 IO | 读写 nii/nii.gz 格式 |
| monai | 医学图像深度学习 | 基于 torch，有现成的 3D 分割模型 |
| pydicom | DICOM IO | 读写 DICOM 格式，处理元数据 |
| skimage | 通用图像处理 | 形态学操作、滤波、特征提取 |

---

## 可视化

| 库 | 定位 | 特点 |
|----|------|------|
| matplotlib | 最基础 | 几乎所有库的绘图后端 |
| seaborn | 统计图 | 比 matplotlib 好看，一行出箱线图/热力图 |
| plotly | 交互式图表 | 可以缩放拖拽，支持网页嵌入 |

---

## 本项目用到的

| 库 | 用途 |
|----|------|
| numpy | 数组运算、rank_norm、直方图计算 |
| scipy.stats | spearmanr、wasserstein_distance、mannwhitneyu、ks_2samp |
| blosc2 | 读 b2nd 预处理文件（CT 体素 + seg） |
| SimpleITK | 读原始 CT（.nii.gz） |
| torch | 训练 nnUNet，mixins 里的 tensor 操作 |
| monai | nnUNet 底层依赖之一 |
| tqdm | 进度条 |
