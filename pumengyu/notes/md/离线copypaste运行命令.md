# 离线 CopyPaste 运行命令

每条命令都是单行，逐条复制粘贴即可。

## 1. 清理上次 Killed 残留的 cp 文件

```bash
cd /home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres && rm -f *_cp*
```

```bash


rm -f /home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/splits_final_cp.json

```

## 2. 运行离线生成脚本（内存峰值约 30MB）

```bash

cd /home/PuMengYu/nnUNet && python pumengyu/tools/offline_copypaste.py --fold 4 --n_aug 3

```

## 3. 另开终端监控内存（可选）

```bash
watch -n 2 free -h
```

## 4. 脚本跑完后，启动训练

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_OfflineCopyPaste_v2
```

## 5. 如果要彻底清除离线生成的所有数据（回到干净状态）

```bash
cd /home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres && rm -f *_cp*
```

```bash
rm -f /home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/splits_final_cp.json
```

---

## 关键说明

- `splits_final.json` 原始文件**不动**，脚本只写 `splits_final_cp.json`
- 只有 `nnUNetTrainer_OfflineCopyPaste_v2` 这个 trainer 读 `splits_final_cp.json`
- 其他实验（baseline / UFL / CopyPaste_v2）完全不受影响
- 断点续跑：脚本加 `--skip_existing` 参数可跳过已生成的文件
