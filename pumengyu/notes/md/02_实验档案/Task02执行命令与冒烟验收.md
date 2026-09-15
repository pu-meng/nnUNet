# Task02 命令清单

## 运行留档

- 之前双卡 `HCC-only，k=3` 运行未成功，目录保留但不作为正式结果：
  `/home/PuMengYu/nnUNet_workspace/results_task02/Dataset003_Liver/nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03__nnUNetPlans__3d_fullres/fold_0`
- 失败原因：数据加载后台 worker 异常退出，终端最后出现 `One or more background workers are no longer alive`；该目录中的 checkpoint 不用于本次正式结果。
- 重新训练使用 GPU0 单卡、K=3、HCC-only；结果写入 `results_task02_gpu0_single_20260913`，不使用 `--c`。

1. 冒烟（Replay，k=3）

```bash
TASK02_SMOKE_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_smoke_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_smoke.sh
```

2. HCC-only，k=1

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K01
```

3. Replay，k=1

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K01
```

4. HCC-only，k=3

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03
```

5. Replay，k=3

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03
```

6. HCC-only，k=5

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K05
```

7. Replay，k=5

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K05
```

8. HCC-only，k=10

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K10
```

9. Replay，k=10

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K10
```

10. Replay，k=70（上界）

```bash
TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 TASK02_CUDA_VISIBLE_DEVICES=0 TASK02_NUM_GPUS=1 bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K70
```

11. Replay K03：评估预检

```bash
python -m pumengyu.task02.final_evaluation --trainer nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03 --gpu 0 --model_results_root /home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913 --dry_run
```

12. Replay K03：三域评估

```bash
python -m pumengyu.task02.final_evaluation --trainer nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03 --gpu 0 --model_results_root /home/PuMengYu/nnUNet_workspace/results_task02_gpu0_single_20260913
```
