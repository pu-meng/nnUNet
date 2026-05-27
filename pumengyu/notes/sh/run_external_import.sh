#!/bin/bash
# 外部无肿瘤 case 一键导入脚本
#
# 步骤：
#   1. IRCADb DICOM → staging nii.gz
#   2. CHAOS DICOM → staging nii.gz（CHAOS 下载后解除注释）
#   3. inject：preprocessing + 修改 splits + 写 log
#
# 回退命令：
#   python pumengyu/tools/external_data/eject.py
#
# 先试运行（不实际执行）：
#   bash pumengyu/notes/sh/run_external_import.sh --dry_run

set -e
cd /home/PuMengYu/nnUNet

DRY_RUN=""
if [[ "$1" == "--dry_run" ]]; then
    DRY_RUN="--dry_run"
    echo "[dry_run 模式] 仅打印计划，不实际执行"
fi

IRCAD_DIR=/home/PuMengYu/8T/Datasets/3Dircadb1
CHAOS_DIR=/home/PuMengYu/8T/Datasets/CHAOS/Train_Sets   # CHAOS 下载后修改此路径

STAGING_IRCAD=/home/PuMengYu/nnUNet_workspace/external_staging/ircad
STAGING_CHAOS=/home/PuMengYu/nnUNet_workspace/external_staging/chaos

echo "=========================================="
echo "外部无肿瘤 case 导入"
echo "=========================================="

# ── Step 1：IRCADb 转换 ───────────────────────────────────────────────────
echo ""
echo "[1] IRCADb 无肿瘤 case 转换（5个）..."
if [[ -n "$DRY_RUN" ]]; then
    echo "  [dry_run] 跳过 convert_ircad.py"
else
    python pumengyu/tools/external_data/convert_ircad.py \
        --ircad_dir "$IRCAD_DIR" \
        --out_dir   "$STAGING_IRCAD" \
        --cases 5 7 11 14 20
fi

# ── Step 2：CHAOS 转换（下载后解除注释）────────────────────────────────────
echo ""
echo "[2] CHAOS 转换..."
if [[ -d "$CHAOS_DIR/CT" ]]; then
    if [[ -n "$DRY_RUN" ]]; then
        echo "  [dry_run] 跳过 convert_chaos.py"
    else
        python pumengyu/tools/external_data/convert_chaos.py \
            --chaos_dir "$CHAOS_DIR" \
            --out_dir   "$STAGING_CHAOS"
    fi
    CHAOS_ARG="--staging_dir $STAGING_CHAOS"
else
    echo "  [skip] CHAOS 目录不存在，跳过（下载后重新运行即可）"
    CHAOS_ARG=""
fi

# ── Step 3：inject ──────────────────────────────────────────────────────────
echo ""
echo "[3] 注入 Dataset003_Liver..."
python pumengyu/tools/external_data/inject.py \
    --staging_dir "$STAGING_IRCAD" \
    $CHAOS_ARG \
    $DRY_RUN

echo ""
echo "=========================================="
echo "完成"
echo "验证集未动：回退命令为"
echo "  python pumengyu/tools/external_data/eject.py"
echo "=========================================="
