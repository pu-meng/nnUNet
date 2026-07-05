#!/bin/bash
# 全量重生成紧凑格式 viz（test+train 双卡并行推理）
# 用法：bash pumengyu/tools/regen_full_viz.sh
# GPU 0 → test 26 cases
# GPU 1 → train 92 cases
# val 直接用已有 validation/ 预测，无需推理

set -e
cd /home/PuMengYu/nnUNet

export nnUNet_raw="/home/PuMengYu/nnUNet_workspace/raw"
export nnUNet_preprocessed="/home/PuMengYu/nnUNet_workspace/preprocessed"
export nnUNet_results="/home/PuMengYu/nnUNet_workspace/results_v2"
export nnUNet_extTrainer="/home/PuMengYu/nnUNet/pumengyu"

TRAINER="nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4"
DATASET="Dataset003_Liver"
FOLD=0
CHECKPOINT="checkpoint_best.pth"
IMG_DIR="$nnUNet_raw/$DATASET/imagesTr"
SPLIT_JSON="$nnUNet_preprocessed/$DATASET/split_info_712.json"
FOLD_DIR="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}"

# 临时目录（export 让子进程/python 也能读到）
export TMP_TEST=$(mktemp -d /tmp/regen_test_XXXXXX)
export TMP_TRAIN=$(mktemp -d /tmp/regen_train_XXXXXX)
export TMP_IN_TEST=$(mktemp -d /tmp/regen_in_test_XXXXXX)
export TMP_IN_TRAIN=$(mktemp -d /tmp/regen_in_train_XXXXXX)

cleanup() {
    echo "清理临时目录..."
    rm -rf "$TMP_TEST" "$TMP_TRAIN" "$TMP_IN_TEST" "$TMP_IN_TRAIN"
}
trap cleanup EXIT

echo "=============================="
echo " Trainer: $TRAINER"
echo " Fold dir: $FOLD_DIR"
echo "=============================="

# ── 1. 读取 split，建 symlink 输入目录 ─────────────────────────────────────
echo "[1/4] 读取 split，创建推理输入目录..."

python3 - <<'PYEOF'
import json, os, sys
from pathlib import Path

split  = json.loads(Path(os.environ["nnUNet_preprocessed"],
                         "Dataset003_Liver", "split_info_712.json").read_text())
img_dir = Path(os.environ["nnUNet_raw"], "Dataset003_Liver", "imagesTr")
in_test  = Path(os.environ["TMP_IN_TEST"])
in_train = Path(os.environ["TMP_IN_TRAIN"])

for c in split["test"]["cases"]:
    src = img_dir / f"{c}_0000.nii.gz"
    if src.exists(): (in_test / src.name).symlink_to(src)

for c in split["train"]["cases"]:
    src = img_dir / f"{c}_0000.nii.gz"
    if src.exists(): (in_train / src.name).symlink_to(src)

print(f"  test  输入: {len(list(in_test.iterdir()))} 个 case")
print(f"  train 输入: {len(list(in_train.iterdir()))} 个 case")
PYEOF

# ── 2. 双卡并行推理 ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] 双卡并行推理（GPU0=test, GPU1=train）..."
echo "  日志: /tmp/regen_test_infer.log  /tmp/regen_train_infer.log"

CUDA_VISIBLE_DEVICES=0 conda run -n medseg nnUNetv2_predict \
    -i "$TMP_IN_TEST" -o "$TMP_TEST" \
    -d "$DATASET" -c 3d_fullres -tr "$TRAINER" -f $FOLD -chk "$CHECKPOINT" \
    > /tmp/regen_test_infer.log 2>&1 &
PID_TEST=$!

CUDA_VISIBLE_DEVICES=1 conda run -n medseg nnUNetv2_predict \
    -i "$TMP_IN_TRAIN" -o "$TMP_TRAIN" \
    -d "$DATASET" -c 3d_fullres -tr "$TRAINER" -f $FOLD -chk "$CHECKPOINT" \
    > /tmp/regen_train_infer.log 2>&1 &
PID_TRAIN=$!

echo "  GPU0 PID=$PID_TEST (test)  |  GPU1 PID=$PID_TRAIN (train)"
echo "  等待两卡推理完成..."

wait $PID_TEST
STATUS_TEST=$?
wait $PID_TRAIN
STATUS_TRAIN=$?

if [ $STATUS_TEST -ne 0 ] || [ $STATUS_TRAIN -ne 0 ]; then
    echo "[ERROR] 推理失败！test=$STATUS_TEST, train=$STATUS_TRAIN"
    echo "  查看日志: tail /tmp/regen_test_infer.log /tmp/regen_train_infer.log"
    exit 1
fi

echo "  推理完成 ✓"

# ── 3. 生成 PNG ───────────────────────────────────────────────────────────────
echo ""
echo "[3/4] 生成紧凑格式 PNG..."

conda run -n medseg python3 - <<PYEOF
import json, os, shutil, sys
from pathlib import Path

os.environ.setdefault("nnUNet_raw",          "/home/PuMengYu/nnUNet_workspace/raw")
os.environ.setdefault("nnUNet_preprocessed", "/home/PuMengYu/nnUNet_workspace/preprocessed")
os.environ.setdefault("nnUNet_results",      "/home/PuMengYu/nnUNet_workspace/results_v2")
sys.path.insert(0, "/home/PuMengYu/nnUNet")

from pumengyu.tools.gen_sizeov4_full_viz import process_split, GT_DIR, IMG_DIR

FOLD_DIR   = Path("$FOLD_DIR")
TMP_TEST   = Path("$TMP_TEST")
TMP_TRAIN  = Path("$TMP_TRAIN")
SPLIT_JSON = Path("$SPLIT_JSON")

split = json.loads(SPLIT_JSON.read_text())
grand = 0

# val（用已有 validation/ 预测）
print("  [val] 生成 val_viz ...")
n = process_split("val", split["val"]["cases"],
                  FOLD_DIR / "validation", GT_DIR, IMG_DIR,
                  FOLD_DIR / "val_viz", min_voxel=1, clean_slices=3)
grand += n

# test
print("  [test] 生成 test_viz ...")
n = process_split("test", split["test"]["cases"],
                  TMP_TEST, GT_DIR, IMG_DIR,
                  FOLD_DIR / "test_viz", min_voxel=1, clean_slices=3)
grand += n

# train
print("  [train] 生成 train_viz ...")
n = process_split("train", split["train"]["cases"],
                  TMP_TRAIN, GT_DIR, IMG_DIR,
                  FOLD_DIR / "train_viz", min_voxel=1, clean_slices=3)
grand += n

print(f"\n  共生成 {grand} 张 PNG")
for name in ("val_viz", "test_viz", "train_viz"):
    d = FOLD_DIR / name
    nc = sum(1 for p in d.iterdir() if p.is_dir())
    np_ = sum(1 for _ in d.rglob("*.png"))
    print(f"    {name}/: {nc} cases, {np_} PNG")
PYEOF

echo ""
echo "[4/4] 完成！"
echo "  train_viz: $FOLD_DIR/train_viz"
echo "  val_viz:   $FOLD_DIR/val_viz"
echo "  test_viz:  $FOLD_DIR/test_viz"
