#!/bin/bash
# fold_0 实验：DiceCE+FocalTversky
# GPU 1: DiceCE + FocalTversky
#
# 用法: bash pumengyu/scripts/run_focaltversky_vs_baseline.sh

DATASET_ID=3
CONFIG=3d_fullres
FOLD=2
# $0是当前脚本的路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GT_DIR=${nnUNet_preprocessed:-/home/PuMengYu/nnUNet_workspace/preprocessed}/Dataset003_Liver/gt_segmentations
IMG_DIR=${nnUNet_raw:-/home/PuMengYu/nnUNet_workspace/raw}/Dataset003_Liver/imagesTr
RESULTS=${nnUNet_results:-/home/PuMengYu/nnUNet_workspace/results}

#:-是默认值语法,如果没有值就用:-后面的默认值

# ── GPU 1: FocalTversky ──────────────────────────────────────────────────
echo "===== [GPU 1] DCFocalTversky   fold=${FOLD} ====="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${REPO_ROOT} nnUNet_extTrainer=${REPO_ROOT}/trainers \
  nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
  -tr nnUNetTrainer_DCFocalTversky
#-tr是nnUNet的参数,只传类名; nnUNet_extTrainer告诉nnUNet去哪个目录找这个类
CODE_FT=$?  # $?=上一个命令的退出码,0=成功,非0=失败

# ── eval FocalTversky ────────────────────────────────────────────────────
if [ ${CODE_FT} -eq 0 ]; then
  VAL_FT="${RESULTS}/Dataset003_Liver/nnUNetTrainer_DCFocalTversky__nnUNetPlans__${CONFIG}/fold_${FOLD}/validation"
  echo "===== Eval DCFocalTversky ====="
  PYTHONPATH=${REPO_ROOT} python "${SCRIPT_DIR}/../tools/eval_fold_report.py" \
    --val_dir "${VAL_FT}" --gt_dir "${GT_DIR}" --img_dir "${IMG_DIR}" --no_vis
else
  echo "[ERROR] DCFocalTversky 训练失败 (exit ${CODE_FT})"
fi


# -eq等于,-ne不等于,-lt是小于,-le小于等于,-gt大于,-ge大于等于
# {}界定变量名边界,""保护字符串,允许变量展开,
echo "===== 全部完成 ====="
