"""
阈值敏感性分析。

对给定的 CC 分类结果，扫描不同体积阈值 T，计算：
  - TP 保留率（有多少真实肿瘤 CC 被保留）
  - FP（无肿瘤 case）清除率
  - 净收益曲线

结果用于绘制"阈值敏感性曲线"，帮助判断体积阈值后处理是否可行。
"""
from __future__ import annotations
import numpy as np


def scan_thresholds(
    tp_cc: list[int],
    fp_notumor_cc: list[int],
    n_steps: int = 200,
) -> dict:
    """
    扫描阈值 T，计算各 T 下的 TP 保留率和 FP 清除率。

    参数
    ----
    tp_cc          : 有肿瘤 case 的真阳性 CC 体素数列表
    fp_notumor_cc  : 无肿瘤 case 的假阳性 CC 体素数列表

    返回
    ----
    dict with keys:
      thresholds   : array of T values
      tp_retain    : fraction of TP CC retained (size >= T)
      fp_remove    : fraction of FP CC removed  (size < T)
    """
    if not tp_cc and not fp_notumor_cc:
        return {"thresholds": [], "tp_retain": [], "fp_remove": []}

    all_sizes = tp_cc + fp_notumor_cc
    t_max = int(max(all_sizes)) + 1
    # 对数均匀采样，因为体积分布是重尾的
    thresholds = np.unique(np.concatenate([
        [0, 1],
        np.logspace(0, np.log10(t_max), n_steps).astype(int),
        [t_max],
    ]))

    tp_arr = np.array(tp_cc) if tp_cc else np.array([0])
    fp_arr = np.array(fp_notumor_cc) if fp_notumor_cc else np.array([0])

    tp_retain = []
    fp_remove = []
    for t in thresholds:
        tr = float((tp_arr >= t).mean()) if len(tp_cc) > 0 else 1.0
        fr = float((fp_arr < t).mean()) if len(fp_notumor_cc) > 0 else 0.0
        tp_retain.append(tr)
        fp_remove.append(fr)

    return {
        "thresholds": thresholds.tolist(),
        "tp_retain": tp_retain,
        "fp_remove": fp_remove,
    }
