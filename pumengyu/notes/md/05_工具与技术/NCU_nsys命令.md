# 安装

```bash
#### NCU（Nsight Compute）：kernel 级分析
sudo apt install nsight-compute

#### nsys（Nsight Systems）：系统级时间线
sudo apt install nsight-systems

#### 验证
which ncu && ncu --version
which nsys && nsys --version
```

两个工具在不同的包里，分别安装，不会互相冲突，很干净。

---

##### 权限设置

NCU 读取 GPU 硬件性能计数器需要 root 权限：

```bash
#### 临时降低权限限制（重启后失效）
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

#### 更可靠：直接用 sudo 跑 ncu
sudo ncu ... /完整路径/python script.py
```

**为什么 sudo 下要写 Python 完整路径**：
sudo 切换到 root，root 的 `$PATH` 不含 conda 环境，找不到 `python`。

```bash
#### 你的 conda 环境 Python 路径
/home/PuMengYu/anaconda3/envs/medseg/bin/python
```

nsys 不需要 root，直接用当前用户跑即可。

---

##### NCU 使用

###### 不需要改代码

NCU 可以直接 wrap 任何 Python 脚本，**不需要对被 profile 的代码做任何修改**。

唯一限制：**不要在被 profile 的脚本里同时用 `torch.profiler`**，两者都抢 CUPTI 接口，会冲突。

###### 常用命令

```bash
#### 基础：profile 所有 kernel，看 4 个关键指标
sudo ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_hit_rate.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

#### 查看实际计算量（FLOPs）
sudo ncu \
  --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__ops_path_tensor_src_precision_fp16.sum \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

#### 查看资源分配（寄存器/shared memory/occupancy）
sudo ncu \
  --metrics launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

#### 只 profile 特定 kernel（按名字过滤）
sudo ncu --kernel-name "flash_fwd_kernel" \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

#### 保存完整报告（含 Roofline 图，用 ncu-ui 打开）
sudo ncu --set full --export ~/nnUNet_workspace/profiling/ncu_result \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py
```

###### 关键指标速查

| 指标 | 含义 | 好/坏 |
|------|------|-------|
| `sm__throughput` | SM 利用率 | 越高越好 |
| `lts__t_hit_rate.pct` | L2 命中率 | 越高越好（4090 上 n/a） |
| `stall_long_scoreboard` | 等 HBM stall 占比 | **高 = 内存瓶颈**，优化目标 |
| `stall_math_pipe_throttle` | 等 Tensor Core stall 占比 | 高 = 在算，"好的瓶颈" |
| `op_ffma_pred_on.sum` | FP32 FFMA 指令数（×2=FLOPs） | 实际执行量 |
| `op_tensor_src_precision_fp16` | FP16 Tensor Core 操作 | 矩阵乘实际计算量 |
| `registers_per_thread` | 每线程寄存器数 | 影响 Occupancy |
| `warps_active.pct_of_peak` | Occupancy | 越高延迟隐藏越好 |

---

##### nsys 使用

nsys 是**系统级时间线**，不需要 root，速度快，适合先看全局。

**`nsys` 的子命令结构**（`profile` 是子命令，不是参数）：

```
nsys  <子命令>   [选项]            <要运行的程序>
nsys  profile    --output ...      python train.py   ← 运行程序并录制，生成 .nsys-rep
nsys  stats      file.nsys-rep     --report ...      ← 读取文件，输出文字分析
nsys  export     file.nsys-rep     --type ...        ← 转换成其他格式
```

类比：`nsys` 是录像机，`profile` 是"开始录像"按钮，`stats` 是"播放并分析录像"。

**完整选项拆解**：

```bash
nsys        profile                        \  ← 子命令：录制模式
  --output  ~/nnUNet_workspace/profiling/x \  ← 输出文件路径（.nsys-rep 自动补全）
  --trace   cuda,nvtx                      \  ← 录哪些事件（CUDA kernel + 自定义标注）
  --delay   60                             \  ← 延迟60秒后才开始录（跳过预热）
  --duration 30                            \  ← 录30秒后停止并终止程序
  python your_script.py                       ← 被录制的程序
```

两步分开执行，不能同时运行：

```bash
#### 第一步：录制（生成 .nsys-rep 文件，程序在 delay+duration 秒后被终止）
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --output ~/nnUNet_workspace/profiling/timeline \
  --trace cuda,nvtx \
  python your_script.py

#### 第二步：分析（程序终止后执行）
nsys stats ~/nnUNet_workspace/profiling/timeline.nsys-rep

#### GUI 版（需要本地有 nsight-systems 图形界面）
nsys-ui ~/nnUNet_workspace/profiling/timeline.nsys-rep
```

nsys 输出里可以看到：
- 每个 kernel 的名字和执行时间
- kernel 之间的间隔（CPU 调度开销）
- H2D/D2H 数据传输时间
- 哪些地方 GPU 在空转等 CPU

**nsys 和 ncu 的分工**：
```
nsys → 先找"哪个 kernel 最慢"（时间线视角）
ncu  → 再深入"这个 kernel 为什么慢"（硬件指标视角）
```

---

##### 不需要改代码：直接 profile nnUNet 训练

**完全不需要修改 trainer 代码**，直接 wrap 训练命令。

###### 用 nsys 看训练时间线（轻量，先用这个）

两步分开执行：第一步录制生成 `.nsys-rep` 文件（训练程序跑完或被终止后结束），
第二步再用 `nsys stats` 读取文件分析，不需要同时运行。

```bash
#### 第一步：录制（运行训练，录稳定运行阶段）
#### --delay 60   跳过前60秒的 torch.compile 编译预热（第0轮344s全是编译噪声）
#### --duration 30  预热后录30秒，约26个iteration，足够看kernel时间分布
CUDA_VISIBLE_DEVICES=1 nsys profile \
  --output ~/nnUNet_workspace/profiling/ib7 \
  --trace cuda,nvtx \
  --delay 60 \
  --duration 30 \
  nnUNetv2_train Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4
#### ↑ 程序在 delay+duration=90秒后被 nsys 终止，文件自动生成

#### 第二步：分析（程序终止后执行）
nsys stats ~/nnUNet_workspace/profiling/ib7.nsys-rep --report gpukernsum | head -30
```

###### 用 ncu 深入分析特定 kernel（重量级，针对性用）

ncu 会把每个 kernel 重放 9 次，直接 profile 全程训练会极慢。用 `--launch-count` 只 profile 前 N 个 kernel：

```bash
#### 只 profile 前 100 个 kernel launch（跳过前 500 个预热）
sudo ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
  --launch-skip 500 \
  --launch-count 100 \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_Baseline

#### 只看卷积相关 kernel（过滤名字）
sudo ncu \
  --kernel-name "ampere_fp16.*gemm\|cudnn.*conv" \
  --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_Baseline
```

###### 推荐流程

```
1. nsys profile nnUNet 训练 30 秒
   → 看哪类 kernel 占时间最多（conv？BN？loss？）

2. ncu --kernel-name 针对最慢的 kernel 类型深入
   → 看 stall_long_scoreboard 高不高（内存瓶颈）
   → 看 stall_math 高不高（计算瓶颈）

3. 根据结果判断：
   stall_long_scoreboard 高 → 考虑算子融合/减少中间结果
   stall_math 高 → 已经是最优了，加速需要换更快的 kernel 实现
   SM 利用率很低（<20%）→ Occupancy 问题，看寄存器/shared memory 分配
```

---
