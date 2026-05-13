#!/usr/bin/env bash
set -euo pipefail

SRC="/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres"
DST="${SRC}_dist"

mkdir -p "$DST"

count=$(ls "$SRC"/*_dist.npz 2>/dev/null | wc -l)
echo "Found $count _dist.npz files to move"

mv "$SRC"/*_dist.npz "$DST"/

echo "Done. Files in $DST: $(ls "$DST" | wc -l)"
