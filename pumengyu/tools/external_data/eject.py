"""
回退：从 Dataset003_Liver 移除所有外部注入的 case

操作：
  1. 读取 external_cases_log.json，获取所有注入的 case_id
  2. 删除 preprocessed/nnUNetPlans_3d_fullres/ 中的 .b2nd + .pkl 文件
  3. 恢复最近一次的 splits_final.json 备份
  4. 清空 external_cases_log.json

用法：
  python pumengyu/tools/external_data/eject.py [--dry_run]
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


DATASET_ROOT = Path("/home/PuMengYu/nnUNet_workspace")
PREPROCESSED = DATASET_ROOT / "preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres"
SPLITS_PATH  = DATASET_ROOT / "preprocessed/Dataset003_Liver/splits_final.json"
LOG_PATH     = DATASET_ROOT / "preprocessed/Dataset003_Liver/external_cases_log.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true", help="只打印计划，不实际删除")
    args = p.parse_args()

    if not LOG_PATH.exists():
        print("未找到 external_cases_log.json，无需回退。")
        return

    log      = json.loads(LOG_PATH.read_text())
    cases    = log.get("injected_cases", [])
    case_ids = [e["case_id"] for e in cases]

    if not case_ids:
        print("log 为空，无注入记录。")
        return

    print(f"准备回退 {len(case_ids)} 个 case：{case_ids}")

    # ── 找最近一次备份 ────────────────────────────────────────────────────
    bak_files = sorted(
        SPLITS_PATH.parent.glob("splits_final.json.bak_*"),
        reverse=True
    )
    if not bak_files:
        print("[WARN] 未找到 splits 备份文件，splits 不会恢复。")
        bak_to_restore = None
    else:
        bak_to_restore = bak_files[0]
        print(f"将恢复 splits 备份: {bak_to_restore.name}")

    # ── 删除 .b2nd + .pkl ────────────────────────────────────────────────
    print("\n[1/3] 删除 preprocessed 文件...")
    for case_id in case_ids:
        for suffix in [".b2nd", "_seg.b2nd", ".pkl"]:
            f = PREPROCESSED / f"{case_id}{suffix}"
            if f.exists():
                print(f"  删除: {f.name}")
                if not args.dry_run:
                    f.unlink()
            else:
                print(f"  [skip] 不存在: {f.name}")

    # ── 恢复 splits ───────────────────────────────────────────────────────
    print("\n[2/3] 恢复 splits_final.json...")
    if bak_to_restore:
        if not args.dry_run:
            import shutil
            shutil.copy2(bak_to_restore, SPLITS_PATH)
        print(f"  已恢复: {bak_to_restore.name} → splits_final.json")

        # 验证验证集完整性
        splits = json.loads(SPLITS_PATH.read_text() if not args.dry_run else bak_to_restore.read_text())
        for i, fold in enumerate(splits):
            for cid in case_ids:
                assert cid not in fold["val"], f"外部 case {cid} 出现在 fold_{i} val 中！"
        print("  验证集完整性检查通过")

    # ── 清空 log ──────────────────────────────────────────────────────────
    print("\n[3/3] 清空 log...")
    if not args.dry_run:
        LOG_PATH.write_text(json.dumps({"injected_cases": []}, indent=2))
        print(f"  log 已清空: {LOG_PATH.name}")

    if args.dry_run:
        print("\n[dry_run] 仅打印计划，未实际执行。")
    else:
        print(f"\n✓ 回退完成，{len(case_ids)} 个外部 case 已全部移除。")
        print("  备份文件保留（如需彻底清理请手动删除 splits_final.json.bak_*）")


if __name__ == "__main__":
    main()
