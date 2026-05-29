"""
外部无肿瘤验证集 FP 率评估
用法：python 02_eval_fp_rate.py

读取 external_val/predictions/<trainer>/ 下的预测结果，
统计每个方法在 25 个无肿瘤 case 上的误报率和误报体素数。
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path

PRED_ROOT = Path("/home/PuMengYu/nnUNet_workspace/external_val/predictions")
INPUT_DIR = Path("/home/PuMengYu/nnUNet_workspace/external_val/input")

# 已见过这些case的方法（训练时包含了外部数据）
SEEN_TRAINERS = {
    "nnUNetTrainer_SizeOversampleV2_Ext25_fold4",
    "nnUNetTrainer_SizeOversampleV2_NTFP_Ext25_fold4",
    "nnUNetTrainer_Ext25_fold4",
}


def get_all_cases():
    cases = sorted([f.stem.replace("_0000", "") for f in INPUT_DIR.glob("*_0000.nii.gz")])
    ircad_cases = [c for c in cases if c.startswith("ircad")]
    chaos_cases = [c for c in cases if c.startswith("chaos")]
    return cases, ircad_cases, chaos_cases


def eval_one_method(pred_dir: Path, cases: list) -> dict:
    """评估一个方法在所有case上的FP情况"""
    results = {}
    for case in cases:
        pred_file = pred_dir / f"{case}.nii.gz"
        if not pred_file.exists():
            results[case] = None
            continue

        img = nib.load(str(pred_file))
        data = np.asarray(img.dataobj)

        # nnUNet 输出: 0=背景, 1=肝脏, 2=肿瘤
        tumor_voxels = int(np.sum(data == 2))
        is_fp = tumor_voxels > 0
        results[case] = {"fp": is_fp, "tumor_voxels": tumor_voxels}

    return results


def format_size(n_voxels):
    if n_voxels == 0:
        return "0"
    elif n_voxels < 1000:
        return f"{n_voxels}"
    else:
        return f"{n_voxels:,}"


def main():
    all_cases, ircad_cases, chaos_cases = get_all_cases()
    print(f"外部无肿瘤验证集：{len(all_cases)} cases")
    print(f"  IRCADb: {len(ircad_cases)} cases | CHAOS CT: {len(chaos_cases)} cases")
    print()

    pred_dirs = sorted([d for d in PRED_ROOT.iterdir() if d.is_dir()])
    if not pred_dirs:
        print(f"[ERROR] 未找到预测结果目录：{PRED_ROOT}")
        print("请先运行 bash 01_run_inference.sh")
        return

    print("=" * 80)
    print(f"{'方法':<45} {'全部FP率':>8} {'IRCADb FP':>10} {'CHAOS FP':>10} {'备注':>6}")
    print("=" * 80)

    all_results = {}

    for pred_dir in pred_dirs:
        method_name = pred_dir.name
        results = eval_one_method(pred_dir, all_cases)

        valid = {k: v for k, v in results.items() if v is not None}
        if not valid:
            print(f"{method_name:<45} [无预测结果]")
            continue

        fp_cases = [k for k, v in valid.items() if v["fp"]]
        ircad_fp = [k for k in fp_cases if k.startswith("ircad")]
        chaos_fp = [k for k in fp_cases if k.startswith("chaos")]

        total = len(valid)
        fp_rate = f"{len(fp_cases)}/{total}"
        ircad_rate = f"{len(ircad_fp)}/{len([c for c in ircad_cases if c in valid])}"
        chaos_rate = f"{len(chaos_fp)}/{len([c for c in chaos_cases if c in valid])}"
        note = "[SEEN]" if method_name in SEEN_TRAINERS else ""

        print(f"{method_name:<45} {fp_rate:>8} {ircad_rate:>10} {chaos_rate:>10} {note:>6}")
        all_results[method_name] = results

    print("=" * 80)
    print("[SEEN] = 这些case在训练时已被模型见过，结果不代表真实泛化能力")
    print()

    # 详细的 per-case 报告
    print("\n" + "=" * 80)
    print("Per-Case 详细误报报告（仅显示有误报的case）")
    print("=" * 80)

    for method_name, results in all_results.items():
        fp_cases = [(k, v) for k, v in results.items() if v is not None and v["fp"]]
        if not fp_cases:
            continue
        print(f"\n[{method_name}]  FP case 列表:")
        for case, info in sorted(fp_cases):
            print(f"  {case:<25} 误报体素数: {format_size(info['tumor_voxels'])}")

    # 汇总：哪些 case 被最多方法误报
    print("\n" + "=" * 80)
    print("容易被误报的 case 排名（被多少个方法误报）")
    print("=" * 80)
    case_fp_count = {}
    for method_name, results in all_results.items():
        if method_name in SEEN_TRAINERS:
            continue
        for case, info in results.items():
            if info is not None and info["fp"]:
                case_fp_count[case] = case_fp_count.get(case, 0) + 1

    methods_without_seen = len([m for m in all_results if m not in SEEN_TRAINERS])
    for case, count in sorted(case_fp_count.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {case:<25} {count}/{methods_without_seen} 个方法误报")


if __name__ == "__main__":
    main()
