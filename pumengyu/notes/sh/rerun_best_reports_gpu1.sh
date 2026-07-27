#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/PuMengYu/nnUNet"
BEST_ROOT="/home/PuMengYu/nnUNet_workspace/results_v2_best"
GPU="${GPU:-1}"

export nnUNet_raw="/home/PuMengYu/nnUNet_workspace/raw"
export nnUNet_preprocessed="/home/PuMengYu/nnUNet_workspace/preprocessed"
export nnUNet_results="/home/PuMengYu/nnUNet_workspace/results_v2"

METHODS=(
  "Baseline:nnUNetTrainer_Baseline"
  "SizeOV2:nnUNetTrainer_SizeOversampleV2"
  "SizeOV3:nnUNetTrainer_SizeOversampleV3"
  "MLAUNet:nnUNetTrainer_MLAUNet"
  "MoE_SizeOV5:nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5"
  "SwinUNETR:nnUNetTrainer_SwinUNETR"
  "nnFormer:nnUNetTrainer_nnFormer"
  "DeepPlainResGN:nnUNetTrainer_DeepPlainResGN"
  "DeepResGN_MLA:nnUNetTrainer_DeepResGN_MLA"
  "DeepDWIBResGN:nnUNetTrainer_DeepDWIBResGN"
  "DeepDWIBMedConfig:nnUNetTrainer_DeepDWIBMedConfig"
  "MedNeXt:nnUNetTrainer_MedNeXt"
  "MedNeXt_SizeOV4:nnUNetTrainer_MedNeXt_SizeOV4"
  "MedNeXt_MLA_MoE:nnUNetTrainer_MedNeXt_MLA_MoE"
  "MedNeXt_MLA_MoE_SizeOV4:nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4"
)

cd "$REPO_ROOT"
mkdir -p "$BEST_ROOT"

for item in "${METHODS[@]}"; do
  method="${item%%:*}"
  trainer="${item#*:}"
  echo
  echo "========================================================================"
  echo "[BEST] method=$method trainer=$trainer gpu=$GPU"
  echo "========================================================================"

  if ! python pumengyu/tools/run_internal_test_best_report.py \
      --trainer "$trainer" \
      --method "$method" \
      --gpu "$GPU" \
      --result_root "$BEST_ROOT"; then
    echo "[SKIP] $method: internal best checkpoint/artifacts failed validation; external reports are not generated"
    continue
  fi

  python pumengyu/ext_val/03_gen_method_report.py \
    --method "$method" \
    --predict \
    --trainer "$trainer" \
    --gpu "$GPU" \
    --checkpoint checkpoint_best.pth \
    --result_root "$BEST_ROOT/IRCADb/source_only"

  python pumengyu/ext_val/05_gen_hcc_test_report.py \
    --method "$method" \
    --predict \
    --trainer "$trainer" \
    --gpu "$GPU" \
    --checkpoint checkpoint_best.pth \
    --result_root "$BEST_ROOT/Dataset013_HCCReferencedCT/source_only"
done

echo
echo "[done] best-only reports written under $BEST_ROOT"
