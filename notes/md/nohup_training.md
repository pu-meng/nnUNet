# 后台训练命令（nohup）

日志目录：`/home/PuMengYu/nnUNet_workspace/logs/`

```bash
mkdir -p /home/PuMengYu/nnUNet_workspace/logs
```

Python / 命令路径前缀：
- Python：`/home/PuMengYu/anaconda3/envs/medseg/bin/python`
- nnUNetv2_train：`/home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train`

---

## 离线预计算距离场（只需跑一次）

```bash

nohup /home/PuMengYu/anaconda3/envs/medseg/bin/python pumengyu/scripts/precompute_dist_fields.py \
    --dataset 3 \
    --config 3d_fullres \
    --num_workers 3 \
    > /home/PuMengYu/nnUNet_workspace/logs/precompute_dist.log 2>&1 &
echo "PID: $!"

```

---

## 标准 nnUNet 训练

### GPU 0：fold 0, 3

```bash
nohup bash -c '
    for fold in 0 3; do
        CUDA_VISIBLE_DEVICES=0 /home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train 3 3d_fullres $fold
    done
' > /home/PuMengYu/nnUNet_workspace/logs/train_gpu0.log 2>&1 &
echo "PID: $!"
```

### GPU 1：fold 1, 2

```bash
nohup bash -c '
    for fold in 1 2; do
        CUDA_VISIBLE_DEVICES=1 /home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train 3 3d_fullres $fold
    done
' > /home/PuMengYu/nnUNet_workspace/logs/train_gpu1.log 2>&1 &
echo "PID: $!"
```

### GPU 1：fold 4

```bash
nohup bash -c '
    CUDA_VISIBLE_DEVICES=1 /home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train 3 3d_fullres 4
' > /home/PuMengYu/nnUNet_workspace/logs/train_gpu1_fold4.log 2>&1 &
echo "PID: $!"
```

---

## BATseg 训练

### fold 0，GPU 0

```bash
nohup bash -c '
    CUDA_VISIBLE_DEVICES=0 /home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_BATseg
' > /home/PuMengYu/nnUNet_workspace/logs/batseg_gpu0_fold0.log 2>&1 &
echo "PID: $!"
```

### fold 4，GPU 1

```bash
nohup bash -c '
    CUDA_VISIBLE_DEVICES=1 /home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_BATseg
' > /home/PuMengYu/nnUNet_workspace/logs/batseg_gpu1_fold4.log 2>&1 &
echo "PID: $!"
```

---

## 查看日志

```bash
# 实时跟踪
tail -f /home/PuMengYu/nnUNet_workspace/logs/train_gpu0.log
tail -f /home/PuMengYu/nnUNet_workspace/logs/train_gpu1.log

# 查看所有日志文件
ls -lh /home/PuMengYu/nnUNet_workspace/logs/
```

## 管理进程

```bash
# 查看所有训练进程
ps aux | grep nnUNetv2_train

# 终止某个进程（用上面记录的 PID）
kill <PID>
```
