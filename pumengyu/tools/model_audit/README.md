# 模型审计工具

这个目录用于辅助理解真实代码中的 MedNeXt-MHA-MoE，不参与训练、不修改
checkpoint，也不改变任何实验配置。

## 一键入口

直接执行总入口：

```bash
/home/PuMengYu/model.sh
```

也可以指定命令：

```bash
/home/PuMengYu/model.sh check
```

需要 CPU shape trace 时执行：

```bash
/home/PuMengYu/model.sh trace
```

生成帮助理解网络和修改位置的结构图：

```bash
/home/PuMengYu/model.sh graph
```

会生成可放大查看的结构总览和 MHA/MoE 细节图；图中的 shape、skip/add、
Stage/Block 内部、Top-2 路由和四个实际计算的 routed experts 按当前代码绘制。
成品统一放在 [`../../notes/模型结构图/`](../../notes/模型结构图/README.md)，
与论文图片目录分开。

如果要看 case 如何经过预处理、patch/增强、网络、损失函数、滑窗推理和后处理：

```bash
/home/PuMengYu/model.sh pipeline
```

它在 `pumengyu/notes/模型结构图/` 默认只生成 `model_pipeline.svg`，不训练、不读取病例。旧路径 `output/` 是通向该目录的软链接。

## 第一版：静态结构审计

```bash
/home/PuMengYu/anaconda3/envs/medseg/bin/python \
  pumengyu/tools/model_audit/trace_mednext_mha_moe.py
```

输出：

- `pumengyu/notes/模型结构图/model_architecture.md`：适合人工阅读的结构说明和研究问题清单；
- `pumengyu/notes/模型结构图/model_architecture.json`：供后续画图和分析脚本读取的结构数据。

默认只构建网络并检查模块，不跑 GPU、不加载数据、不启动训练。

`--forward` 会在 CPU 上对一个 `128×128×128` patch 进行 shape trace，可能较慢：

```bash
/home/PuMengYu/anaconda3/envs/medseg/bin/python \
  pumengyu/tools/model_audit/trace_mednext_mha_moe.py --forward
```

## 解释边界

参数量不等于计算量；优化建议必须再用受控 forward/backward profiler 验证。
第一版把“可能的改进问题”写出来，但不把它们当成已经证实的结论。

日常绘图默认只输出 SVG；分析或论文需要 PNG 时，用 `./model.sh graph --png` 或 `./model.sh pipeline --png` 同时导出。
