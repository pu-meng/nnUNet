"""
将外部无肿瘤 case 注入 Dataset003_Liver

流程：
  1. 读取 staging 目录里的 nii.gz（*_0000.nii.gz + *.nii.gz）
  2. 对每个 case 运行 nnUNet preprocessing（resampling + normalization）
  3. 将 .b2nd + .pkl 写入 preprocessed/nnUNetPlans_3d_fullres/
  4. 备份 splits_final.json → splits_final.json.bak_<timestamp>
  5. 修改 splits_final.json：外部 case 加入所有 fold 的 train，val 绝对不动
  6. 写入 external_cases_log.json（记录来源 + 时间，用于回退）

用法：
  python pumengyu/tools/external_data/inject.py \
    --staging_dir /home/PuMengYu/nnUNet_workspace/external_staging/ircad \
    [--staging_dir /home/PuMengYu/nnUNet_workspace/external_staging/chaos]

（--staging_dir 可重复指定多个目录）
"""

from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from pumengyu.tools.external_data._preprocess import preprocess_case

# ──────────────────────────────────────────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────────────────────────────────────────

DATASET_ROOT   = Path("/home/PuMengYu/nnUNet_workspace")
PREPROCESSED   = DATASET_ROOT / "preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres"
SPLITS_PATH    = DATASET_ROOT / "preprocessed/Dataset003_Liver/splits_final.json"
LOG_PATH       = DATASET_ROOT / "preprocessed/Dataset003_Liver/external_cases_log.json"


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def load_log() -> dict:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text())
    return {"injected_cases": []}


def save_log(log: dict) -> None:
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False))


def backup_splits() -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = SPLITS_PATH.with_name(f"splits_final.json.bak_{ts}")
    shutil.copy2(SPLITS_PATH, bak)
    print(f"  splits 已备份 → {bak.name}")
    return bak


def add_cases_to_splits(case_ids: list[str]) -> None:
    splits = json.loads(SPLITS_PATH.read_text())
    original_vals = [set(fold["val"]) for fold in splits]

    for fold in splits:
        existing = set(fold["train"]) | set(fold["val"])
        for cid in case_ids:
            if cid not in existing:
                fold["train"].append(cid)

    # 断言验证集未变动
    for i, fold in enumerate(splits):
        assert set(fold["val"]) == original_vals[i], f"fold_{i} val 集合被意外修改！"

    SPLITS_PATH.write_text(json.dumps(splits, indent=2, ensure_ascii=False))
    print(f"  splits 已更新：{len(case_ids)} 个 case 加入所有 fold 训练集")


def discover_cases(staging_dir: Path) -> list[tuple[Path, Path]]:
    """返回 [(ct_path, seg_path), ...] 列表。"""
    ct_files = sorted(staging_dir.glob("*_0000.nii.gz"))
    pairs = []
    for ct_path in ct_files:
        case_id  = ct_path.name.replace("_0000.nii.gz", "")
        seg_path = staging_dir / f"{case_id}.nii.gz"
        if not seg_path.exists():
            print(f"  [WARN] 缺少 seg 文件，跳过: {case_id}")
            continue
        pairs.append((ct_path, seg_path, case_id))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--staging_dir", action="append", required=True,
                   help="staging nii.gz 目录（可重复指定多个）")
    p.add_argument("--dry_run", action="store_true",
                   help="只打印计划，不实际执行")
    args = p.parse_args()

    # ── 收集所有 case ──────────────────────────────────────────────────────
    all_cases: list[tuple[Path, Path, str]] = []
    for d in args.staging_dir:
        staging = Path(d)
        cases   = discover_cases(staging)
        print(f"[{staging.name}] 找到 {len(cases)} 个 case")
        all_cases.extend(cases)

    if not all_cases:
        print("没有找到任何 case，退出。")
        return

    print(f"\n计划注入 {len(all_cases)} 个 case：")
    for ct, seg, cid in all_cases:
        b2nd_exists = (PREPROCESSED / f"{cid}.b2nd").exists()
        print(f"  {cid}  {'[已存在，跳过preprocessing]' if b2nd_exists else '[新 case]'}")

    if args.dry_run:
        print("\n[dry_run] 仅打印计划，未执行任何操作。")
        return

    # ── 加载已有 log，过滤重复 ────────────────────────────────────────────
    log = load_log()
    existing_ids = {e["case_id"] for e in log["injected_cases"]}
    new_cases    = [(ct, seg, cid) for ct, seg, cid in all_cases if cid not in existing_ids]

    if not new_cases:
        print("\n所有 case 已在 log 中，无需重复注入。")
        return

    print(f"\n新注入 {len(new_cases)} 个 case（已有 {len(existing_ids)} 个跳过）")

    # ── preprocessing → .b2nd + .pkl ────────────────────────────────────
    print("\n[1/3] preprocessing...")
    processed_ids = []
    for ct_path, seg_path, case_id in new_cases:
        if (PREPROCESSED / f"{case_id}.b2nd").exists():
            print(f"  [{case_id}] .b2nd 已存在，跳过 preprocessing")
        else:
            preprocess_case(ct_path, seg_path, case_id, PREPROCESSED, verbose=True)
        processed_ids.append(case_id)

    # ── 修改 splits_final.json ───────────────────────────────────────────
    print("\n[2/3] 修改 splits_final.json...")
    bak_path = backup_splits()
    add_cases_to_splits(processed_ids)

    # ── 写 log ───────────────────────────────────────────────────────────
    print("\n[3/3] 写入 log...")
    ts = datetime.now().isoformat()
    for ct_path, seg_path, case_id in new_cases:
        log["injected_cases"].append({
            "case_id":    case_id,
            "ct_src":     str(ct_path),
            "seg_src":    str(seg_path),
            "injected_at": ts,
            "splits_bak": str(bak_path),
        })
    save_log(log)
    print(f"  log 已写入: {LOG_PATH}")

    print(f"\n✓ 注入完成，共 {len(processed_ids)} 个 case。")
    print(f"  可用 eject.py 完整回退。")


if __name__ == "__main__":
    main()
