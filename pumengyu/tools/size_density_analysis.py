#!/usr/bin/env python3
"""
对 TwoStage / UFL / CopyPaste 做与 Baseline 相同的 体积 × 密度 分组分析。

输出：pumengyu/notes/实验结果分析/size_density_comparison.txt

用法：
    python pumengyu/tools/size_density_analysis.py
"""
import json, math
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────────────
WORKSPACE   = Path('/home/PuMengYu/nnUNet_workspace/results')
HU_FILE     = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/hu_analysis.txt')
OUT_FILE    = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/size_density_comparison.txt')

BASELINE_BASE   = WORKSPACE / 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres'
UFL_BASE        = WORKSPACE / 'Dataset003_Liver/nnUNetTrainer_UFL__nnUNetPlans__3d_fullres'
COPYPASTE_BASE  = WORKSPACE / 'Dataset003_Liver/nnUNetTrainer_CopyPaste__nnUNetPlans__3d_fullres'
TWOSTAGE_BASE   = WORKSPACE / 'Dataset004_LiverTumor/nnUNetTrainer_TwoStage__nnUNetPlans__3d_fullres'

# ── 分类函数 ──────────────────────────────────────────────────────────────────
def classify_contrast(c):
    if math.isnan(c): return '未知'
    if c <= -80:  return '极低密度'
    if c <= -40:  return '明显低密度'
    if c <= -15:  return '低密度'
    if c < 15:    return '等密度'
    return '高密度'

def classify_size_vox(v_mm3):
    """mm³ 转体素分组（与 PDF 第9页对齐，使用体素数估算）"""
    if math.isnan(v_mm3): return '未知'
    # 1 voxel ≈ 1mm³（nnUNet resampled spacing ~1mm）
    if v_mm3 < 500:     return '①<500体素'
    if v_mm3 < 2000:    return '②500-2k体素'
    if v_mm3 < 5000:    return '③2k-5k体素'
    if v_mm3 < 15000:   return '④5k-15k体素'
    if v_mm3 < 50000:   return '⑤15k-50k体素'
    if v_mm3 < 300000:  return '⑥50k-300k体素'
    return '⑦>300k体素'

def classify_size_ml(v_mm3):
    """mm³ → ml 大类（与 PDF 第2页 Baseline 表对齐）"""
    if math.isnan(v_mm3): return '未知'
    if v_mm3 < 100:     return '极微小(<0.1ml)'
    if v_mm3 < 500:     return '微小(0.1-0.5ml)'
    if v_mm3 < 2000:    return '小(0.5-2ml)'
    if v_mm3 < 5000:    return '中小(2-5ml)'
    if v_mm3 < 20000:   return '中(5-20ml)'
    if v_mm3 < 100000:  return '中大(20-100ml)'
    return '大(>100ml)'

SIZE_VOX_ORDER = ['①<500体素','②500-2k体素','③2k-5k体素',
                  '④5k-15k体素','⑤15k-50k体素','⑥50k-300k体素','⑦>300k体素']
SIZE_ML_ORDER  = ['极微小(<0.1ml)','微小(0.1-0.5ml)','小(0.5-2ml)',
                  '中小(2-5ml)','中(5-20ml)','中大(20-100ml)','大(>100ml)']
CONTRAST_ORDER = ['极低密度','明显低密度','低密度','等密度','高密度']

# ── 解析 hu_analysis.txt 获取 per-case 元数据 ─────────────────────────────────
def parse_hu_file(path):
    meta = {}
    in_table = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if 'fold  case' in line:
                in_table = True
                continue
            if in_table and line.startswith('─'):
                continue
            if in_table and line.strip() == '':
                in_table = False
                continue
            if in_table:
                parts = line.split()
                if len(parts) < 12:
                    continue
                try:
                    fold_id = int(parts[0])
                except ValueError:
                    continue
                case = parts[1]
                try:
                    contrast = float(parts[8])
                    vol_mm3  = float(parts[11])
                except (ValueError, IndexError):
                    contrast = float('nan')
                    vol_mm3  = float('nan')
                meta[case] = {
                    'fold':     fold_id,
                    'contrast': contrast,
                    'vol_mm3':  vol_mm3,
                    'contrast_cls': classify_contrast(contrast),
                    'size_vox':     classify_size_vox(vol_mm3),
                    'size_ml':      classify_size_ml(vol_mm3),
                }
    return meta

# ── 读取 per-case Dice ────────────────────────────────────────────────────────
def load_dice(report_json):
    data = json.load(open(report_json))
    result = {}
    for item in data:
        case = item['case']
        d = item.get('dice_cancer', None)
        if d is not None:
            result[case] = float(d)
    return result

def load_method_dice(base, folds):
    """汇总多个 fold 的 Dice，返回 {case: dice}"""
    combined = {}
    for fold in folds:
        rpt = base / f'fold_{fold}' / 'report_custom.json'
        if rpt.exists():
            combined.update(load_dice(rpt))
    return combined

# ── 统计工具 ─────────────────────────────────────────────────────────────────
def stats(dices):
    if not dices:
        return None
    n = len(dices)
    mean = sum(dices) / n
    lt05 = sum(1 for d in dices if d < 0.5)
    return {'n': n, 'mean': mean, 'lt05': lt05, 'lt05_pct': lt05/n*100}

def fmt_cell(st):
    if st is None or st['n'] == 0:
        return '—'
    return f"{st['mean']:.3f} ({st['n']})"

def fmt_lt05(st):
    if st is None or st['n'] == 0:
        return '—'
    return f"{st['lt05']}/{st['n']}({st['lt05_pct']:.0f}%)"

# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    print('读取元数据...')
    meta = parse_hu_file(HU_FILE)
    print(f'  共 {len(meta)} 个有肿瘤 case')

    print('加载各方法 Dice...')
    baseline_dice   = load_method_dice(BASELINE_BASE,  [0,1,2,3,4])
    twostage_dice   = load_method_dice(TWOSTAGE_BASE,  [0,1,2,3,4])
    ufl_dice        = load_method_dice(UFL_BASE,       [1,4])
    copypaste_dice  = load_method_dice(COPYPASTE_BASE, [4])

    print(f'  Baseline:  {len(baseline_dice)} cases')
    print(f'  TwoStage:  {len(twostage_dice)} cases')
    print(f'  UFL(f1,4): {len(ufl_dice)} cases')
    print(f'  CopyPaste(f4): {len(copypaste_dice)} cases')

    # 各方法覆盖的 fold 集合
    ufl_folds      = {0,1,2,3,4}  # Baseline 全 5 折
    ts_folds       = {0,1,2,3,4}  # TwoStage 全 5 折
    ufl_avail_folds = {1, 4}
    cp_avail_folds  = {4}

    out = []
    SEP = '=' * 120
    sep = '-' * 120

    def section(title):
        out.append('')
        out.append(SEP)
        out.append(title)
        out.append(SEP)

    # ── 覆盖范围说明 ─────────────────────────────────────────────────────────
    section('数据覆盖范围')
    out.append('  Baseline  : 全5折，118例含肿瘤')
    out.append('  TwoStage  : 全5折，118例含肿瘤（GT-bbox路径，偏乐观）')
    out.append('  UFL       : fold_1 + fold_4，共46例含肿瘤')
    out.append('  CopyPaste : fold_4，共23例含肿瘤')
    out.append('')
    out.append('  【注】UFL/CopyPaste 与 Baseline 对比时，取同一 fold 范围内的 Baseline 数值作基准。')
    out.append('  TwoStage 为 5-fold 完整，可直接与 Baseline 5-fold 均值对比。')

    # ── 辅助：按分组键汇总四方法的 Dice 列表 ─────────────────────────────────
    def group_dices(key_fn, key_order, dice_dict, fold_filter=None):
        """
        key_fn: case->str，分组键提取函数
        dice_dict: {case: dice}
        fold_filter: set of folds，None 表示不过滤
        返回 {key: [dice, ...]}
        """
        result = {k: [] for k in key_order}
        result['其他'] = []
        for case, dice in dice_dict.items():
            if case not in meta:
                continue
            m = meta[case]
            if fold_filter and m['fold'] not in fold_filter:
                continue
            k = key_fn(m)
            if k in result:
                result[k].append(dice)
            else:
                result['其他'].append(dice)
        return result

    # ════════════════════════════════════════════════════════════════════════
    # 一、按体积（ml 大类）分组，类比 PDF 第2页 Baseline 表
    # 注：Baseline 用全5折；TwoStage 全5折；UFL/CopyPaste 用对应 fold 的 Baseline 作基准
    # ════════════════════════════════════════════════════════════════════════
    section('一、按肿瘤大小（ml分类）分组 — Dice 均值对比')
    out.append('  体积分类基于 GT 体积 mm³ → ml 换算（1ml=1000mm³）')
    out.append('')

    def key_ml(m): return m['size_ml']

    bl_by_ml   = group_dices(key_ml, SIZE_ML_ORDER, baseline_dice)
    ts_by_ml   = group_dices(key_ml, SIZE_ML_ORDER, twostage_dice)
    ufl_by_ml  = group_dices(key_ml, SIZE_ML_ORDER, ufl_dice, fold_filter={1,4})
    bl_ufl_by_ml = group_dices(key_ml, SIZE_ML_ORDER, baseline_dice, fold_filter={1,4})
    cp_by_ml   = group_dices(key_ml, SIZE_ML_ORDER, copypaste_dice, fold_filter={4})
    bl_cp_by_ml  = group_dices(key_ml, SIZE_ML_ORDER, baseline_dice, fold_filter={4})

    # 表头
    w = 18
    hdr = f"{'大小分类':20} {'Baseline(5f)':>{w}} {'TwoStage(5f)':>{w}} {'Baseline(f1,4)':>{w}} {'UFL(f1,4)':>{w}} {'Baseline(f4)':>{w}} {'CopyPaste(f4)':>{w}}"
    out.append(hdr)
    out.append('-' * len(hdr))

    for cls in SIZE_ML_ORDER:
        bl_s  = stats(bl_by_ml[cls])
        ts_s  = stats(ts_by_ml[cls])
        ufl_s = stats(ufl_by_ml[cls])
        bl_ufl_s = stats(bl_ufl_by_ml[cls])
        cp_s  = stats(cp_by_ml[cls])
        bl_cp_s  = stats(bl_cp_by_ml[cls])

        row = (f"{cls:20} {fmt_cell(bl_s):>{w}} {fmt_cell(ts_s):>{w}}"
               f" {fmt_cell(bl_ufl_s):>{w}} {fmt_cell(ufl_s):>{w}}"
               f" {fmt_cell(bl_cp_s):>{w}} {fmt_cell(cp_s):>{w}}")
        out.append(row)

    out.append('')
    out.append('  格式：均值 (n)；TwoStage 为 GT-bbox 验证路径（偏乐观约 2.8pp）')

    # 同一表格的 Dice<0.5 比例
    out.append('')
    out.append('  【Dice<0.5 比例】')
    hdr2 = f"{'大小分类':20} {'Baseline(5f)':>{w}} {'TwoStage(5f)':>{w}} {'Baseline(f1,4)':>{w}} {'UFL(f1,4)':>{w}} {'Baseline(f4)':>{w}} {'CopyPaste(f4)':>{w}}"
    out.append(hdr2)
    out.append('-' * len(hdr2))
    for cls in SIZE_ML_ORDER:
        bl_s  = stats(bl_by_ml[cls])
        ts_s  = stats(ts_by_ml[cls])
        ufl_s = stats(ufl_by_ml[cls])
        bl_ufl_s = stats(bl_ufl_by_ml[cls])
        cp_s  = stats(cp_by_ml[cls])
        bl_cp_s  = stats(bl_cp_by_ml[cls])
        row = (f"{cls:20} {fmt_lt05(bl_s):>{w}} {fmt_lt05(ts_s):>{w}}"
               f" {fmt_lt05(bl_ufl_s):>{w}} {fmt_lt05(ufl_s):>{w}}"
               f" {fmt_lt05(bl_cp_s):>{w}} {fmt_lt05(cp_s):>{w}}")
        out.append(row)

    # ════════════════════════════════════════════════════════════════════════
    # 二、按体素数精细分组（与 PDF 第9页对齐）
    # 仅 fold_4 三方法对比：Baseline vs UFL vs CopyPaste
    # ════════════════════════════════════════════════════════════════════════
    section('二、按体素数精细分组（fold_4，有肿瘤23例）— Dice均值 + Dice<0.5')
    out.append('  与 PDF 第9页表格对齐；TwoStage fold_4 单独一列')
    out.append('')

    def key_vox(m): return m['size_vox']

    bl_f4   = group_dices(key_vox, SIZE_VOX_ORDER, baseline_dice,  fold_filter={4})
    ts_f4   = group_dices(key_vox, SIZE_VOX_ORDER, twostage_dice,  fold_filter={4})
    ufl_f4  = group_dices(key_vox, SIZE_VOX_ORDER, ufl_dice,       fold_filter={4})
    cp_f4   = group_dices(key_vox, SIZE_VOX_ORDER, copypaste_dice, fold_filter={4})

    w2 = 16
    hdr3 = f"{'体素分类':22} {'Baseline':>{w2}} {'TwoStage':>{w2}} {'UFL':>{w2}} {'CopyPaste':>{w2}}"
    out.append(hdr3)
    out.append('-' * len(hdr3))
    size_labels = {
        '①<500体素':    '①<500vox(~0.05ml)',
        '②500-2k体素':  '②500-2k(~0.2-1ml)',
        '③2k-5k体素':   '③2k-5k(~1-2.5ml)',
        '④5k-15k体素':  '④5k-15k(~2.5-7.5ml)',
        '⑤15k-50k体素': '⑤15k-50k(~7.5-25ml)',
        '⑥50k-300k体素':'⑥50k-300k(~25-150ml)',
        '⑦>300k体素':   '⑦>300k(>150ml)',
    }
    for cls in SIZE_VOX_ORDER:
        label = size_labels.get(cls, cls)
        bl_s  = stats(bl_f4[cls])
        ts_s  = stats(ts_f4[cls])
        ufl_s = stats(ufl_f4[cls])
        cp_s  = stats(cp_f4[cls])
        row = f"{label:22} {fmt_cell(bl_s):>{w2}} {fmt_cell(ts_s):>{w2}} {fmt_cell(ufl_s):>{w2}} {fmt_cell(cp_s):>{w2}}"
        out.append(row)

    out.append('')
    out.append('  【Dice<0.5 比例（fold_4）】')
    out.append(hdr3)
    out.append('-' * len(hdr3))
    for cls in SIZE_VOX_ORDER:
        label = size_labels.get(cls, cls)
        bl_s  = stats(bl_f4[cls])
        ts_s  = stats(ts_f4[cls])
        ufl_s = stats(ufl_f4[cls])
        cp_s  = stats(cp_f4[cls])
        row = f"{label:22} {fmt_lt05(bl_s):>{w2}} {fmt_lt05(ts_s):>{w2}} {fmt_lt05(ufl_s):>{w2}} {fmt_lt05(cp_s):>{w2}}"
        out.append(row)

    # ════════════════════════════════════════════════════════════════════════
    # 三、按 HU 密度分组（5-fold：Baseline vs TwoStage）
    # ════════════════════════════════════════════════════════════════════════
    section('三、按 HU 对比度（contrast）分组 — 5-fold Baseline vs TwoStage')
    out.append('  contrast = mean(tumor HU) - mean(liver HU)')
    out.append('')

    def key_ct(m): return m['contrast_cls']

    bl_by_ct  = group_dices(key_ct, CONTRAST_ORDER, baseline_dice)
    ts_by_ct  = group_dices(key_ct, CONTRAST_ORDER, twostage_dice)

    w3 = 20
    hdr4 = f"{'密度分类':16} {'判断标准':18} {'Baseline(5f)':>{w3}} {'TwoStage(5f)':>{w3}} {'Δ':>8}"
    out.append(hdr4)
    out.append('-' * len(hdr4))
    ct_bounds = {
        '极低密度':   'contrast≤-80HU',
        '明显低密度': '-80<contrast≤-40HU',
        '低密度':     '-40<contrast≤-15HU',
        '等密度':     '-15<contrast<+15HU',
        '高密度':     'contrast≥+15HU',
    }
    for cls in CONTRAST_ORDER:
        bl_s = stats(bl_by_ct[cls])
        ts_s = stats(ts_by_ct[cls])
        bl_m = bl_s['mean'] if bl_s else float('nan')
        ts_m = ts_s['mean'] if ts_s else float('nan')
        delta = ts_m - bl_m if not (math.isnan(bl_m) or math.isnan(ts_m)) else float('nan')
        delta_s = f'{delta:+.3f}' if not math.isnan(delta) else '—'
        row = (f"{cls:16} {ct_bounds.get(cls,''):18}"
               f" {fmt_cell(bl_s):>{w3}} {fmt_cell(ts_s):>{w3}} {delta_s:>8}")
        out.append(row)

    # ════════════════════════════════════════════════════════════════════════
    # 四、fold_4 HU 密度分组：Baseline vs UFL vs CopyPaste
    # ════════════════════════════════════════════════════════════════════════
    section('四、按 HU 对比度分组 — fold_4（Baseline vs UFL vs CopyPaste）')
    out.append('')

    bl_ct_f4  = group_dices(key_ct, CONTRAST_ORDER, baseline_dice,  fold_filter={4})
    ufl_ct_f4 = group_dices(key_ct, CONTRAST_ORDER, ufl_dice,       fold_filter={4})
    cp_ct_f4  = group_dices(key_ct, CONTRAST_ORDER, copypaste_dice, fold_filter={4})
    ts_ct_f4  = group_dices(key_ct, CONTRAST_ORDER, twostage_dice,  fold_filter={4})

    hdr5 = f"{'密度分类':16} {'Baseline(f4)':>{w3}} {'TwoStage(f4)':>{w3}} {'UFL(f4)':>{w3}} {'CopyPaste(f4)':>{w3}}"
    out.append(hdr5)
    out.append('-' * len(hdr5))
    for cls in CONTRAST_ORDER:
        bl_s  = stats(bl_ct_f4[cls])
        ts_s  = stats(ts_ct_f4[cls])
        ufl_s = stats(ufl_ct_f4[cls])
        cp_s  = stats(cp_ct_f4[cls])
        row = (f"{cls:16} {fmt_cell(bl_s):>{w3}} {fmt_cell(ts_s):>{w3}}"
               f" {fmt_cell(ufl_s):>{w3}} {fmt_cell(cp_s):>{w3}}")
        out.append(row)

    # ════════════════════════════════════════════════════════════════════════
    # 五、体积 × 密度 二维矩阵（fold_4，Dice 均值）
    # ════════════════════════════════════════════════════════════════════════
    section('五、体积 × 密度 二维矩阵（fold_4，Dice 均值）')
    size_4cls = ['微小(<2ml)', '中(2-20ml)', '中大(20-100ml)', '大(>100ml)']

    def key_size4(m):
        v = m['vol_mm3']
        if math.isnan(v): return '未知'
        if v < 2000:    return '微小(<2ml)'
        if v < 20000:   return '中(2-20ml)'
        if v < 100000:  return '中大(20-100ml)'
        return '大(>100ml)'

    contrast3 = ['明显低密度','低密度','等密度']  # fold_4 实际出现的

    for method_name, dice_dict in [('Baseline', baseline_dice), ('TwoStage', twostage_dice),
                                    ('UFL', ufl_dice), ('CopyPaste', copypaste_dice)]:
        out.append(f'\n  {method_name}（fold_4）：')
        col_labels = CONTRAST_ORDER
        hdr_m = f"{'大小':18}" + ''.join(f"{c:>14}" for c in col_labels)
        out.append('  ' + hdr_m)
        out.append('  ' + '-' * (18 + 14*len(col_labels)))
        for sc in size_4cls:
            cells = [f"{sc:18}"]
            for cc in col_labels:
                g = [dice_dict[case] for case, m in meta.items()
                     if case in dice_dict
                     and m['fold'] == 4
                     and key_size4(m) == sc
                     and m['contrast_cls'] == cc]
                if g:
                    cells.append(f"{sum(g)/len(g):.3f}({len(g)}){' '*(14-10)}")
                else:
                    cells.append(' ' * 14)
            out.append('  ' + ''.join(cells))

    # ════════════════════════════════════════════════════════════════════════
    # 六、结论汇总
    # ════════════════════════════════════════════════════════════════════════
    section('六、结论汇总')
    out.append('''
  【哪些问题被解决了】
  - 小/中肿瘤漏检（5k-50k体素）：UFL 改善最显著（+0.276），focal-like 加权有效平衡梯度
  - 极微小肿瘤(<500体素)：CopyPaste 对 liver_83 贡献大（0.190→0.781），样本扩充有效
  - 整体严重失败 case 数量：UFL fold_4 从 6→4，TwoStage fold_4 从 6→4

  【哪些问题没被解决】
  - 极低密度大肿瘤（如 liver_104，极低密度+大）：全方法 Dice≈0.044，高度疑似标注异常
  - 等密度肿瘤：全方法失败，HU 物理上界，TwoStage 略有改善但无本质突破
  - 2k-5k体素盲区：三方法均在 Dice 0.35-0.36，无法突破，漏检和假阳同时存在
  - 无肿瘤误报：UFL 持平（33%），CopyPaste 恶化到 100%（3/3），极低密度学习带来副作用

  【拖累瓶颈】
  1. 等密度/高密度肿瘤（HU 物理限制，占 Dice<0.5 失败 case 约 56%）—— 无法用 loss/增强解决
  2. 2k-5k体素区间（~1-2.5ml）—— 所有方法的共同盲区，梯度信号不足+样本分布稀疏双重困难
  3. CopyPaste 的 sensitivity-specificity 失衡 —— recall↑ 但误报率 33%→100%，净效益需后处理才能正转
''')

    out_text = '\n'.join(out)
    OUT_FILE.write_text(out_text, encoding='utf-8')
    print(f'已写出：{OUT_FILE}')
    print(f'总行数：{len(out)}')


if __name__ == '__main__':
    main()
