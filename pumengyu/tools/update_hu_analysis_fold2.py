#!/usr/bin/env python3
"""
补全 fold_2 Dice 到 hu_analysis.txt，并重建所有统计 section。

用法:
    python pumengyu/tools/update_hu_analysis_fold2.py

效果:
    - fold_2 的 dice_t/dice_l N/A → 来自 summary.json 的实际值
    - 主表按 dice_t 升序重新排序（全 118 例）
    - 分组统计、相关分析、总结更新为全 118 例
    - 失败 case 深入分析：追加 fold_2 的等/高密度低 Dice case
    - 原文件备份为 hu_analysis.txt.bak
"""
import json, math, shutil
import numpy as np

WORKSPACE = '/home/PuMengYu/nnUNet_workspace'
HU_FILE = '/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/hu_analysis.txt'
RESULT_BASE = f'{WORKSPACE}/results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres'

SEP = '=' * 161


# ══════════════════════ 工具函数 ══════════════════════════════════════════════

def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float('nan')


def classify_contrast(c):
    if math.isnan(c): return '未知'
    if c <= -80:  return '极低密度'
    if c <= -40:  return '明显低密度'
    if c <= -15:  return '低密度'
    if c < 15:    return '等密度'
    return '高密度'


def classify_size(v):
    if math.isnan(v): return '未知'
    if v < 200:    return '微小 (<0.2ml)'
    if v < 2000:   return '小 (0.2-2ml)'
    if v < 20000:  return '中 (2-20ml)'
    return '大 (>20ml)'


def fmt_f(v, w=7, d=4, na='   N/A'):
    return na if math.isnan(v) else f'{v:{w}.{d}f}'


def pearson(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return float('nan')
    return float(np.corrcoef(xs, ys)[0, 1])


# ══════════════════════ 读取 Dice ═════════════════════════════════════════════

fold2_dice = {}
for fold in range(5):
    summary = json.load(open(f'{RESULT_BASE}/fold_{fold}/validation/summary.json'))
    for c in summary['metric_per_case']:
        name = c['reference_file'].split('/')[-1].replace('.nii.gz', '')
        dt = c['metrics']['2'].get('Dice', float('nan'))
        dl = c['metrics']['1'].get('Dice', float('nan'))
        fold2_dice[(fold, name)] = (dt, dl)


# ══════════════════════ 解析主表 ══════════════════════════════════════════════

with open(HU_FILE, 'r', encoding='utf-8') as f:
    raw = f.read()
lines = raw.splitlines()

def find_line(keyword, start=0):
    for i in range(start, len(lines)):
        if keyword in lines[i]:
            return i
    return -1

header_end   = find_line('fold  case')
sep_line     = find_line('─' * 10, header_end)
stats_start  = find_line('分组統計' if '分组統計' in raw else '分组统计', sep_line)
if stats_start == -1:
    stats_start = find_line('分组统计', sep_line)
detail_start = find_line('失败 case 深入分析', stats_start)
low_dice_start = find_line('低 Dice 但无明显', detail_start)
summary_start = find_line('总结', stats_start)

# 原来的详细分析块（等/高密度）
detail_block_lines = []
if detail_start != -1:
    end = low_dice_start if low_dice_start != -1 else summary_start
    detail_block_lines = lines[detail_start + 2 : end]  # +2 跳过 SEP 行和 section 标题

# 原来"低 Dice 但无明显"块
low_dice_lines = []
if low_dice_start != -1:
    end = summary_start if summary_start != -1 else len(lines)
    # 找到 ── 低 Dice 但无明显 HU... 那行
    low_dice_lines = lines[low_dice_start : end]

# 解析数据行
rows = []
for i in range(sep_line + 1, stats_start):
    line = lines[i].strip()
    if not line:
        continue
    parts = line.split(None, 16)
    if len(parts) < 16:
        continue
    try:
        fold_id = int(parts[0])
    except ValueError:
        continue
    rows.append({
        'fold':     fold_id,
        'case':     parts[1],
        'dice_t':   safe_float(parts[2]),
        'dice_l':   safe_float(parts[3]),
        'recall':   safe_float(parts[4]),
        'prec':     safe_float(parts[5]),
        'liverHU':  safe_float(parts[6]),
        'tumorHU':  safe_float(parts[7]),
        'contrast': safe_float(parts[8]),
        'cohens_d': safe_float(parts[9]),
        'overlap':  safe_float(parts[10]),
        'vol_mm3':  safe_float(parts[11]),
        'tl_pct':   safe_float(parts[12]),
        'fn_mm3':   safe_float(parts[13]),
        'fp_mm3':   safe_float(parts[14]),
        'fp_liv':   safe_float(parts[15]),
    })

print(f'解析到 {len(rows)} 行（含 fold_2 N/A 行）')

# 更新 fold_2 的 dice_t / dice_l
updated = 0
for r in rows:
    key = (r['fold'], r['case'])
    if key in fold2_dice:
        dt, dl = fold2_dice[key]
        if math.isnan(r['dice_t']) and not math.isnan(dt):
            r['dice_t'] = dt
            r['dice_l'] = dl
            updated += 1

print(f'更新了 {updated} 个 fold_2 case')

# 过滤：只保留 GT 有肿瘤的 case（dice_t 是有效数值，包括 0）
# dice_t=NaN 来自 GT=0 且 pred=0 的无肿瘤 case，排除
tumor_rows = [r for r in rows if not math.isnan(r['dice_t'])]
tumor_rows.sort(key=lambda r: r['dice_t'])
print(f'有效肿瘤 case 共 {len(tumor_rows)} 例')


# ══════════════════════ 格式化函数 ════════════════════════════════════════════

def fmt_row(r):
    warn = '⚠ ' if r['dice_t'] < 0.3 else '  '
    label = f"{warn}{classify_contrast(r['contrast'])} / {classify_size(r['vol_mm3'])}"
    dt = '   N/A' if math.isnan(r['dice_t']) else f'{r["dice_t"]:.4f}'
    dl = '   N/A' if math.isnan(r['dice_l']) else f'{r["dice_l"]:.4f}'
    re = '   N/A' if math.isnan(r['recall']) else f'{r["recall"]:.4f}'
    pr = '   N/A' if math.isnan(r['prec'])   else f'{r["prec"]:.4f}'
    return (
        f"{r['fold']:<5} {r['case']:<15} {dt:>6} {dl:>6} "
        f"{re:>6} {pr:>6} {r['liverHU']:>7.1f} {r['tumorHU']:>7.1f} "
        f"{r['contrast']:>8.1f} {r['cohens_d']:>8.3f} {r['overlap']:>7.3f} "
        f"{r['vol_mm3']:>10.0f} {r['tl_pct']:>6.2f} {r['fn_mm3']:>8.0f} "
        f"{r['fp_mm3']:>8.0f} {r['fp_liv']:>6.0f}  {label}"
    )


# ══════════════════════ 构建输出 ══════════════════════════════════════════════

out = []

# ── 头部（保留原文件前 6 行：各 fold case 统计）
for line in lines[:6]:
    out.append(line)
out.append('')

# ── 主表
out.append(SEP)
out.append('全 fold 验证集 — 按 tumor_dice 升序（全 5 fold，118 例含肿瘤）')
out.append(SEP)
out.append('fold  case            dice_t dice_l recall   prec liverHU tumorHU contrast Cohens_d overlap    vol_mm3  t/l_%   fn_mm3   fp_mm3 fp_liv 备注')
out.append('─' * 165)
for r in tumor_rows:
    out.append(fmt_row(r))
out.append('')

# ── 分组统计
out.append(SEP)
out.append('分组统计（全 5 fold，118 例含肿瘤）')
out.append(SEP)

contrast_order = ['极低密度', '明显低密度', '低密度', '等密度', '高密度']
size_order     = ['微小 (<0.2ml)', '小 (0.2-2ml)', '中 (2-20ml)', '大 (>20ml)']

# 按对比度
out.append('')
out.append('── 按 HU 对比度分类 ──')
for cls in contrast_order:
    g = [r for r in tumor_rows if classify_contrast(r['contrast']) == cls]
    if not g:
        continue
    dices  = [r['dice_t'] for r in g]
    vols   = [r['vol_mm3'] for r in g]
    lt05   = sum(1 for d in dices if d < 0.5)
    dices_s = sorted(dices)
    vols_s  = sorted(vols)
    median_d = dices_s[len(dices_s) // 2]
    median_v = vols_s[len(vols_s) // 2]
    out.append(f'  {cls} ({len(g)} cases):')
    out.append(f'    Dice 均值={sum(dices)/len(dices):.4f} 中位数={median_d:.4f} 范围=[{min(dices):.4f}, {max(dices):.4f}]')
    out.append(f'    体积中位数={median_v:.0f} mm³  范围=[{min(vols):.0f}, {max(vols):.0f}]')
    out.append(f'    Dice<0.5: {lt05}/{len(g)} ({lt05/len(g)*100:.1f}%)')
    out.append('')

# 按大小
out.append('── 按肿瘤大小分类 ──')
for cls in size_order:
    g = [r for r in tumor_rows if classify_size(r['vol_mm3']) == cls]
    if not g:
        continue
    dices    = [r['dice_t'] for r in g]
    contrasts = [r['contrast'] for r in g if not math.isnan(r['contrast'])]
    lt05     = sum(1 for d in dices if d < 0.5)
    dices_s  = sorted(dices)
    median_d = dices_s[len(dices_s) // 2]
    out.append(f'  {cls} ({len(g)} cases):')
    out.append(f'    Dice 均值={sum(dices)/len(dices):.4f} 中位数={median_d:.4f}')
    if contrasts:
        out.append(f'    Contrast 均值={sum(contrasts)/len(contrasts):.1f}')
    out.append(f'    Dice<0.5: {lt05}/{len(g)} ({lt05/len(g)*100:.1f}%)')
    out.append('')

# 联合分组
out.append('── 联合分组：对比度 × 大小 (Dice 均值) ──')
col_w = 18
header = ' ' * 16 + ''.join(f'{s:<{col_w}}' for s in size_order)
out.append(header)
out.append('─' * 78)
for cls_c in contrast_order:
    cells = [f'{cls_c:>14}']
    for cls_s in size_order:
        g = [r for r in tumor_rows
             if classify_contrast(r['contrast']) == cls_c
             and classify_size(r['vol_mm3']) == cls_s]
        if g:
            m = sum(r['dice_t'] for r in g) / len(g)
            cells.append(f'  {m:.3f}({len(g)}){" " * (col_w - 11)}')
        else:
            cells.append(' ' * col_w)
    out.append(''.join(cells))
out.append('')

# ── 相关分析
out.append(SEP)
out.append('相关分析')
out.append(SEP)
out.append('')
n = len(tumor_rows)
dice_v    = [r['dice_t']   for r in tumor_rows]
cont_v    = [r['contrast'] for r in tumor_rows]
cohd_v    = [r['cohens_d'] for r in tumor_rows]
over_v    = [r['overlap']  for r in tumor_rows]
vol_v     = [r['vol_mm3']  for r in tumor_rows]
tlp_v     = [r['tl_pct']   for r in tumor_rows]

features = [dice_v, cont_v, cohd_v, over_v, vol_v, tlp_v]
feat_names = ['Dice', 'Contrast', "Cohen's d", 'HistOverlap', '体积', '肿瘤/肝比']
corr = [[pearson(features[i], features[j]) for j in range(6)] for i in range(6)]

out.append(f'  Pearson 相关系数矩阵 (n={n}):')
header = ' ' * 14 + ''.join(f'{nm:>12}' for nm in feat_names)
out.append(header)
for i, nm in enumerate(feat_names):
    row_s = f'{nm:>14}' + ''.join(f'{corr[i][j]:>12.3f}' for j in range(6))
    out.append(row_s)
out.append('')

out.append('  与 Dice 的 Pearson 相关系数 (按绝对值排序):')
corr_with_dice = [(feat_names[j], corr[0][j]) for j in range(1, 6)]
for nm, r_val in sorted(corr_with_dice, key=lambda x: -abs(x[1])):
    out.append(f'  {nm:>16}: r = {r_val:+.4f}')
out.append('')

# ── 失败 case 深入分析（保留原文 + 追加 fold_2）
out.append(SEP)
out.append('失败 case 深入分析（Dice < 0.5 且等/高密度）')
out.append(SEP)
out.append('')

# 保留原文中 fold_0/1/3/4 的内容
for line in detail_block_lines:
    out.append(line)

# 追加 fold_2 等/高密度低 Dice case（无 ±std 数据，简化格式）
fold2_detail = [r for r in tumor_rows
                if r['fold'] == 2
                and r['dice_t'] < 0.5
                and classify_contrast(r['contrast']) in ('等密度', '高密度')]
if fold2_detail:
    out.append('── fold_2 新增（等/高密度，Dice < 0.5）──')
    for r in sorted(fold2_detail, key=lambda x: x['dice_t']):
        prec_s = 'nan' if math.isnan(r['prec']) else f'{r["prec"]:.3f}'
        out.append(f'  {r["case"]} (fold_2) — Dice={r["dice_t"]:.4f}, Rec={r["recall"]:.3f}, Prec={prec_s}')
        out.append(f'    HU: 肝={r["liverHU"]:.1f}（均值）, 肿瘤={r["tumorHU"]:.1f}')
        c_cls = classify_contrast(r['contrast'])
        out.append(f'    Contrast={r["contrast"]:+.1f}  ({c_cls}), Cohen\'s d={r["cohens_d"]:.3f}, 重叠度={r["overlap"]:.3f}')
        liver_vol = r['vol_mm3'] / r['tl_pct'] * 100 if r['tl_pct'] > 0 else float('nan')
        liver_s = f'{liver_vol:.0f} mm³' if not math.isnan(liver_vol) else 'N/A'
        out.append(f'    体积: 肿瘤={r["vol_mm3"]:.0f} mm³ ({classify_size(r["vol_mm3"])}), 肝脏≈{liver_s}, 占比={r["tl_pct"]:.2f}%')
        out.append(f'    漏检={r["fn_mm3"]:.0f} mm³, 假阳={r["fp_mm3"]:.0f} mm³ (其中在肝脏内={r["fp_liv"]:.0f} voxels)')
        # 推断可能原因
        reasons = []
        if r['overlap'] > 0.5:
            reasons.append(f'直方图重叠度高({r["overlap"]:.2f})')
        if r['cohens_d'] < 0.5:
            reasons.append(f'Cohen\'s d 小({r["cohens_d"]:.3f})')
        if r['fp_mm3'] > r['vol_mm3'] * 10:
            reasons.append('爆炸式假阳性（FP远超GT体积）')
        if r['recall'] > 0.5 and r['fp_mm3'] > r['vol_mm3']:
            reasons.append('召回率尚可但假阳性多，精度低')
        elif r['recall'] < 0.3:
            reasons.append('漏检远多于正确检测，召回率低')
        if reasons:
            out.append(f'    可能原因: {"; ".join(reasons)}')
        out.append('')

# ── 低 Dice 但无明显 HU 异常（保留原文 + 追加 fold_2）
if low_dice_lines:
    # 保留原文 "── 低 Dice 但无明显..." 行
    for line in low_dice_lines[:-1]:  # 去掉最后空行
        out.append(line)
    # 追加 fold_2 case
    fold2_low = [r for r in tumor_rows
                 if r['fold'] == 2
                 and r['dice_t'] < 0.5
                 and classify_contrast(r['contrast']) not in ('等密度', '高密度')]
    for r in sorted(fold2_low, key=lambda x: x['dice_t']):
        out.append(
            f'  {r["case"]} (fold_2) — Dice={r["dice_t"]:.4f}, '
            f'Contrast={r["contrast"]:.1f}, Volume={r["vol_mm3"]:.0f} mm³, '
            f"Cohen's d={r['cohens_d']:.3f}"
        )
    out.append('')

# ── 总结
valid_dices = [r['dice_t'] for r in tumor_rows]
dices_s     = sorted(valid_dices)
mean_d      = sum(valid_dices) / len(valid_dices)
median_d    = dices_s[len(dices_s) // 2]
lt05_all    = [d for d in valid_dices if d < 0.5]
hard        = [r for r in tumor_rows if r['dice_t'] < 0.5]
n_hard      = len(hard)
n_iso       = sum(1 for r in hard if abs(r['contrast']) < 15)
n_lowd      = sum(1 for r in hard if r['cohens_d'] < 0.5)
n_micro     = sum(1 for r in hard if r['vol_mm3'] < 200)
n_small     = sum(1 for r in hard if r['vol_mm3'] < 2000)
n_over      = sum(1 for r in hard if r['overlap'] > 0.5)

out.append(SEP)
out.append('总结')
out.append(SEP)
out.append(f'  共分析 {len(tumor_rows)} 个有肿瘤 case（全 5 fold）')
out.append(f'  平均 Dice = {mean_d:.4f}, 中位数 = {median_d:.4f}')
out.append(f'  Dice < 0.5: {len(lt05_all)} ({len(lt05_all)/len(tumor_rows)*100:.1f}%)')
out.append('')
out.append(f'  失败因素分析 (Dice < 0.5 的 case):')
out.append(f'    等密度/高密度 (|contrast|<15 或 >15): {n_iso}/{n_hard} ({n_iso/n_hard*100:.0f}%)')
out.append(f'    Cohen\'s d < 0.5: {n_lowd}/{n_hard} ({n_lowd/n_hard*100:.0f}%)')
out.append(f'    肿瘤微小 (<200 mm³): {n_micro}/{n_hard} ({n_micro/n_hard*100:.0f}%)')
out.append(f'    肿瘤小 (<2000 mm³): {n_small}/{n_hard} ({n_small/n_hard*100:.0f}%)')
out.append(f'    直方图重叠 > 0.5: {n_over}/{n_hard} ({n_over/n_hard*100:.0f}%)')
out.append('')
out.append('  结论:')
out.append('    - 与 Dice 最相关的单一指标: HistOverlap')
out.append('    - 存在多因素叠加效应: 小体积 + 低对比度 = 极低 Dice')
out.append('    - 大部分失败 case 可以用 HU 特征解释')
out.append('')


# ══════════════════════ 写出文件 ══════════════════════════════════════════════

shutil.copy(HU_FILE, HU_FILE + '.bak')
print(f'已备份: {HU_FILE}.bak')

with open(HU_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
    f.write('\n')

print(f'已写出: {HU_FILE}')
print(f'总行数: {len(out)}')
print(f'验证: 主表共 {len(tumor_rows)} 行，均值 Dice={mean_d:.4f}')
