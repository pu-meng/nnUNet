"""Read existing source-only experiments; write tables without running inference."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import ast
import csv
import hashlib
import json
import re
import subprocess

import torch

REPO = Path(__file__).resolve().parents[3]
WS = Path('/home/PuMengYu/nnUNet_workspace')
BASE = WS / 'results_v2'
STATS = REPO / 'pumengyu/notes/paper/statistics'
NOTES = REPO / 'pumengyu/notes/md/葛老师的指导记录'
REPAIRED = {
    'MedNeXt_MHA_MoE': WS / 'results_v2_baseline_retrain_20260902',
    'MedNeXt_MLA_MoE': Path('/home/PuMengYu/8T/nnUNet_result/results_v2_moe_retrain_after_fix_20260907'),
    'MedNeXt_MLA_MoE_SizeOV4': Path('/home/PuMengYu/8T/nnUNet_result/results_v2_moe_sizeov4_retrain_after_fix_20260910'),
}
DOMAINS = ['LiTS', 'IRCADb', 'HCC']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint(fold, trainer):
    p = fold / 'checkpoint_best.pth'
    out = dict(path=str(p), resolved_path=str(p.resolve()), status='未验证', dataset='Dataset003_Liver', trainer=trainer, fold=0)
    if not p.is_file():
        out['reason'] = 'best 文件缺失'
        return out
    try:
        d = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
        epoch = int(d.get('current_epoch', -1))
        init = d.get('init_args', {})
        out.update(epoch=epoch, mtime=datetime.fromtimestamp(p.stat().st_mtime).isoformat(), size=p.stat().st_size,
                   recorded_trainer=d.get('trainer_name'), recorded_fold=init.get('fold'),
                   recorded_dataset=init.get('plans', {}).get('dataset_name'))
        del d
        final = fold / 'checkpoint_final.pth'
        if final.is_file():
            d = torch.load(final, map_location='cpu', weights_only=False, mmap=True)
            out['final_epoch'] = int(d.get('current_epoch', -1))
            out['final_mtime'] = datetime.fromtimestamp(final.stat().st_mtime).isoformat()
            del d
        logs = sorted(fold.glob('training_log*.txt'))
        # Preserve evidence locations; do not infer training success from directory names.
        out['logs'] = [str(x) for x in logs]
        texts = '\n'.join(x.read_text(errors='replace') for x in logs)
        out['epoch_log_seen'] = bool(re.search(rf'Epoch\s+{epoch-1}(?!\d)', texts))
        if epoch < 1 or (out.get('final_epoch', epoch) < epoch):
            out.update(status='无效', reason='best epoch 与 final 矛盾')
        elif final.exists() and p.stat().st_mtime > final.stat().st_mtime and epoch < out['final_epoch']:
            out.update(status='可疑', reason='best 写入晚于 final 且 epoch 更早，需人工来源复核')
        elif out['recorded_trainer'] != trainer or out['recorded_dataset'] != 'Dataset003_Liver' or out['recorded_fold'] != 0:
            out['reason'] = 'trainer/dataset/fold 不一致；历史改名需补迁移证据'
        elif out['epoch_log_seen'] and final.exists():
            out.update(status='基础检查通过', reason='trainer/dataset/fold、epoch、mtime、训练日志相符；预测来源对应关系另查')
        else:
            out['reason'] = '训练日志或 final 证据不足'
    except Exception as exc:
        out['reason'] = f'{type(exc).__name__}: {exc}'
    return out


def aggregate(cases):
    pos = [c for c in cases if c['metrics']['2']['TP'] + c['metrics']['2']['FN'] > 0]
    neg = [c for c in cases if c not in pos]
    def mean(values):
        return sum(values) / len(values) if values else None
    liver = mean([c['metrics']['1']['Dice'] for c in cases])
    tumor = mean([c['metrics']['2']['Dice'] for c in pos])
    def rate(c, kind):
        t = c['metrics']['2']; tp, fp, fn = [t[k] for k in ('TP', 'FP', 'FN')]
        den = {'recall': tp+fn, 'precision': tp+fp, 'iou': tp+fp+fn}[kind]
        return tp / den if den else 0.0
    return dict(liver=liver, tumor=tumor, overall=(liver+tumor)/2 if tumor is not None else None,
                recall=mean([rate(c, 'recall') for c in pos]), precision=mean([rate(c, 'precision') for c in pos]),
                iou=mean([rate(c, 'iou') for c in pos]), n_positive=len(pos), n_no_tumor=len(neg),
                fp_no_tumor=sum(c['metrics']['2']['TP']+c['metrics']['2']['FP'] > 0 for c in neg),
                fp_no_tumor_rate=mean([float(c['metrics']['2']['TP']+c['metrics']['2']['FP'] > 0) for c in neg]))


def audit_domain(run, domain, expected, cp):
    root, method, fold = run['root'], run['method'], run['fold']
    if domain == 'LiTS':
        directory, pred, report = fold, fold/'test_prediction', fold/'test_report_custom.txt'
    else:
        candidates = [root/'IRCADb/source_only'/method] if domain == 'IRCADb' else [
            root/'Dataset013_HCCReferencedCT/source_only'/method, root/'ExternalVal_HCCReferencedCT'/method]
        existing = [p for p in candidates if (p/'report_custom.txt').is_file() or (p/'predictions/summary.json').is_file()]
        directory = existing[0] if existing else candidates[0]
        pred, report = directory/'predictions', directory/'report_custom.txt'
    summary = pred/'summary.json'
    actual = {p.name.removesuffix('.nii.gz') for p in pred.glob('*.nii.gz')}
    pngs = list((directory/'test_viz').rglob('*.png'))
    row = dict(run_id=run['id'], method=method, lineage=run['lineage'], domain=domain,
               expected=len(expected), prediction_count=len(expected & actual), extra_predictions=sorted(actual-expected),
               missing_predictions=sorted(expected-actual), summary=str(summary), summary_exists=summary.is_file(),
               report=str(report), report_exists=report.is_file() and report.stat().st_size > 0,
               png_count=len(pngs), checkpoint_status=cp['status'], binding='未验证', errors=[])
    if summary.is_file():
        try:
            data = read(summary)['metric_per_case']
            keyed = {Path(c['reference_file']).name.removesuffix('.nii.gz'): c for c in data}
            row['summary_missing'] = sorted(expected-set(keyed))
            row['summary_extra'] = sorted(set(keyed)-expected)
            row['summary_sha256'] = digest(summary)
            if len(keyed) != len(data):
                raise ValueError('summary 有重复病例')
            if row['summary_missing']:
                raise ValueError('summary 缺固定测试病例，禁止部分病例均值冒充全测试结果')
            metrics = aggregate([keyed[c] for c in sorted(expected)])
            row.update(metrics, metric_source='summary 固定病例重算')
        except Exception as exc:
            row['errors'].append(str(exc))
    elif row['report_exists']:
        row['errors'].append('summary 缺失；仅保留历史报告数值，病例与数值未独立验证')
        text = report.read_text()
        liver = re.search(r'Liver\s+Dice:\s*mean=([\d.]+)', text)
        section = text.split('Tumor 综合指标', 1)[-1].split('nnUNet foreground_mean')[0]
        tumor = re.search(r'Dice\s*:\s*mean=([\d.]+)', section)
        if liver and tumor and 'Tumor 综合指标' in text:
            l, t = float(liver[1]), float(tumor[1])
            row.update(liver=l, tumor=t, overall=(l+t)/2, metric_source='仅历史报告')
    if row['report_exists']:
        text = report.read_text()
        row['report_sha256'] = digest(report)
        if row.get('metric_source') == 'summary 固定病例重算':
            section = text.split('Tumor 综合指标', 1)[-1].split('nnUNet foreground_mean')[0]
            matches = [re.search(r'Liver\s+Dice:\s*mean=([\d.]+)', text), re.search(r'Dice\s*:\s*mean=([\d.]+)', section)]
            row['report_metrics_match'] = all(m and abs(float(m[1])-row[k]) <= 0.00011 for m,k in zip(matches,['liver','tumor']))
            if not row['report_metrics_match']: row['errors'].append('报告与固定病例 summary 指标不一致')
    prov = directory/'evaluation_provenance.json'
    if prov.is_file() and Path(cp['path']).is_file():
        try:
            p = read(prov)
            if p.get('checkpoint_sha256'):
                if 'sha256' not in cp: cp['sha256'] = digest(Path(cp['path']))
                row['binding'] = '哈希一致' if p['checkpoint_sha256'] == cp['sha256'] and p.get('checkpoint') == 'checkpoint_best.pth' and p.get('checkpoint_epoch') == cp.get('epoch') else '不一致'
            row['provenance'] = str(prov)
        except Exception as exc: row['errors'].append(str(exc))
    if domain == 'LiTS' and (fold/'inference_usage.json').is_file():
        entries = read(fold/'inference_usage.json').get('entries', [])
        entries = [x for x in entries if x.get('scope') == 'internal_test']
        if entries and entries[-1].get('checkpoint_epoch') == cp.get('epoch'):
            row['binding'] = '运行记录 epoch 一致（无哈希）'
    if cp['status'] in ('可疑', '无效') or row['binding'] == '不一致':
        row['withheld_metrics'] = {k:row.pop(k) for k in ['liver','tumor','overall','recall','precision','iou','fp_no_tumor_rate'] if k in row}
        row['errors'].append('checkpoint 来源异常，正式指标隐藏')
    row['artifact_complete'] = (row['prediction_count'] == row['expected'] and row['summary_exists'] and row['report_exists'] and row['png_count'] > 0 and not row.get('summary_missing') and 'tumor' in row and row.get('report_metrics_match', False))
    if cp['status'] in ('可疑','无效') or row['binding'] == '不一致' or (row['summary_exists'] and 'tumor' not in row):
        row['status'] = '失败/待核查'
    elif row['artifact_complete'] and cp['status'] == '基础检查通过':
        row['status'] = '完整'
    elif not any([row['prediction_count'],row['summary_exists'],row['report_exists'],row['png_count']]):
        row['status'] = '未开始'
    else: row['status'] = '部分完成'
    if not pngs: row['errors'].append('PNG=0；未验证是阈值无切片还是生成失败/未生成')
    return row


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def write_notes(runs, rows, cps, process_lines):
    lookup = {(r['id'],d):next(x for x in rows if x['run_id']==r['id'] and x['domain']==d) for r in runs for d in DOMAINS}
    def display_status(x):
        if x['checkpoint_status'] in ('可疑', '无效') or x['binding'] == '不一致' or (x['summary_exists'] and 'tumor' not in x):
            return '不通过'
        metrics_ready = (
            x['summary_exists'] and x['report_exists'] and x['png_count'] > 0
            and not x.get('summary_missing') and 'tumor' in x
            and x.get('report_metrics_match', False)
            and x['checkpoint_status'] == '基础检查通过'
        )
        return '通过' if metrics_ready else '不完整'

    display_counts = Counter(display_status(x) for x in rows)
    incomplete = [x for x in rows if display_status(x) == '不完整']
    for mednext, filename, title in [(False,'实验指标_全方法.md','全方法三域实验指标'),(True,'实验指标_MedNeXt系列.md','MedNeXt 系列三域实验指标')]:
        chosen = [r for r in runs if not mednext or 'MedNeXt' in r['method']]
        scope = (
            '范围：Dataset003_Liver/fold 0 的 MedNeXt source-only 对照与三项修复后 MoE 重训。修复前 MoE、HCC 适配、联合训练和 Task02 不进入本表。'
            if mednext else
            '范围：Dataset003_Liver/fold 0 的 source-only 对照方法与三项修复后 MoE 重训。修复前 MoE、SizeOV3、HCC 适配、联合训练和 Task02 不进入本表。'
        )
        lines = ['# '+title, '', scope, '',
                 f'LiTS 为自定义划分中的固定 26 例；IRCADb 为 20 例；HCC-TACE-Seg 为固定 21 例。Liver Dice 对全部病例取均值，Tumor Dice 只对真实有肿瘤的病例取均值，Overall=(Liver+Tumor)/2。排名按本表 {len(chosen)} 个方法在对应数据域的 Overall 从高到低计算。', '']
        if not mednext:
            lines += [f'当前纳入 {len(chosen)} 个方法、{len(rows)} 个方法×数据域组合：通过 {display_counts["通过"]}，不完整 {display_counts["不完整"]}，不通过 {display_counts["不通过"]}。旧 LiTS 预测仅缺 NIfTI 时不影响指标使用，按通过处理。', '']
            if incomplete:
                lines += ['当前不完整：' + '、'.join(f"{x['method']} / {x['domain']}" for x in incomplete) + '。', '']
        lines += ['## 指标表', '']
        groups=['当前对照','修复后']
        domain_ranks = {}
        for domain in DOMAINS:
            rankable = [r for r in chosen if lookup[r['id'],domain].get('overall') is not None]
            rankable.sort(key=lambda r: (-lookup[r['id'],domain]['overall'], r['method']))
            ranks = {r['id']: index for index, r in enumerate(rankable, 1)}
            domain_ranks[domain] = ranks
            table_header = ['| 方法 | Liver Dice | Tumor Dice | Overall | 排名 |', '|---|---:|---:|---:|---:|']
            lines += ['### '+domain, ''] + table_header
            for group in groups:
                for r in chosen:
                    if r['lineage']!=group: continue
                    x=lookup[r['id'],domain]
                    rank = f"{ranks[r['id']]}/{len(rankable)}" if r['id'] in ranks else '—'
                    method_label = '**Baseline**' if r['method'] == 'Baseline' else r['method']
                    lines.append(f"| {method_label} | {fmt(x.get('liver'))} | {fmt(x.get('tumor'))} | {fmt(x.get('overall'))} | {rank} |")
            lines += ['']
        composite = []
        for r in chosen:
            if not all(r['id'] in domain_ranks[d] for d in DOMAINS):
                continue
            average_rank = sum(domain_ranks[d][r['id']] for d in DOMAINS) / len(DOMAINS)
            mean_overall = sum(lookup[r['id'],d]['overall'] for d in DOMAINS) / len(DOMAINS)
            composite.append((average_rank, -mean_overall, r['method'], r, mean_overall))
        composite.sort(key=lambda item: item[:3])
        lines += ['## 三域综合排名', '',
                  '综合排名按 LiTS、IRCADb、HCC 三个 Overall 排名的平均值从小到大排列；平均排名相同时，三域 Overall 均值更高者优先。', '',
                  '| 综合排名 | 方法 | LiTS 排名 | IRCADb 排名 | HCC 排名 | 平均排名 | 三域 Overall 均值 |',
                  '|---:|---|---:|---:|---:|---:|---:|']
        for composite_rank, (average_rank, _, _, r, mean_overall) in enumerate(composite, 1):
            method_label = '**Baseline**' if r['method'] == 'Baseline' else r['method']
            lines.append(
                f"| {composite_rank}/{len(composite)} | {method_label} | "
                f"{domain_ranks['LiTS'][r['id']]}/{len(domain_ranks['LiTS'])} | "
                f"{domain_ranks['IRCADb'][r['id']]}/{len(domain_ranks['IRCADb'])} | "
                f"{domain_ranks['HCC'][r['id']]}/{len(domain_ranks['HCC'])} | "
                f"{average_rank:.2f} | {mean_overall:.4f} |"
            )
        lines += ['']
        if mednext:
            (NOTES/filename).write_text('\n'.join(lines))
            continue
        lines += ['## 检查结果','', '“通过”表示指标、summary、txt、PNG 和 checkpoint 基础检查可用；旧 LiTS 预测仅缺 NIfTI 时仍按通过处理。', '',
                  '| 方法 | LiTS | IRCADb | HCC |', '|---|---|---|---|']
        for r in chosen:
            cells=[]
            for d in DOMAINS:
                x=lookup[r['id'],d]
                cells.append(display_status(x))
            method_label = '**Baseline**' if r['method'] == 'Baseline' else r['method']
            lines.append(f"| {method_label} | "+' | '.join(cells)+' |')
        lines += ['', '## 模型与预测检查','', '旧流程没有保存额外来源记录时不判失败；只有 checkpoint 本身异常或已有记录明确冲突才判为不通过。','',
                  '| 方法 | checkpoint 检查 | 预测关系 |','|---|---|---|']
        for r in chosen:
            cp=cps[r['id']]
            checkpoint_ok = cp['status'] == '基础检查通过'
            prediction_ok = checkpoint_ok and all(lookup[r['id'],d]['binding'] != '不一致' for d in DOMAINS)
            method_label = '**Baseline**' if r['method'] == 'Baseline' else r['method']
            lines.append(f"| {method_label} | {'通过' if checkpoint_ok else '不通过'} | {'通过' if prediction_ok else '不通过'} |")
        lines += ['', '[详细指标 CSV](../../paper/statistics/guidance_result_metrics.csv) · [审计 JSON](../../paper/statistics/guidance_result_audit.json)', '']
        (NOTES/filename).write_text('\n'.join(lines))


def main():
    aliases={}
    for node in ast.parse((REPO/'pumengyu/ext_val/06_batch_hcc_ext_val.py').read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALIASES' for t in node.targets):
            aliases={v:k for k,v in ast.literal_eval(node.value).items()}
    excluded_changed_training = [p.name for p in (BASE/'IRCADb/source_only').iterdir() if any(x in p.name for x in ['HCCAdapter','HCCRefOnly','MSDHCC'])]
    excluded_old_moe = [p.name for p in (BASE/'IRCADb/source_only').iterdir() if p.is_dir() and 'MoE' in p.name]
    excluded = set(excluded_changed_training + excluded_old_moe + ['SizeOV3'])
    methods=sorted(p.name for p in (BASE/'IRCADb/source_only').iterdir() if p.is_dir() and p.name not in excluded)
    runs=[]
    for method in methods:
        trainer=aliases.get(method,'nnUNetTrainer_'+method)
        runs.append(dict(method=method,trainer=trainer,root=BASE,lineage='当前对照'))
    for method,root in REPAIRED.items():
        runs.append(dict(method=method,trainer='nnUNetTrainer_'+method,root=root,lineage='修复后'))
    for i,r in enumerate(runs,1):
        r['id']=f'R{i:02d}';r['fold']=r['root']/'Dataset003_Liver'/(r['trainer']+'__nnUNetPlans__3d_fullres')/'fold_0'
    expected = {
        'LiTS':set(read(WS/'preprocessed/Dataset003_Liver/split_info_712.json')['test']['cases']),
        'HCC':set(read(WS/'preprocessed/Dataset013_HCCReferencedCT/split_info_701020_stratified_v2.json')['test']['cases']),
        'IRCADb':{f'ircadb_{i:03}' for i in range(1,21)},
    }
    rows=[];cps={}
    for r in runs:
        cp=checkpoint(r['fold'],r['trainer']);cps[r['id']]=cp
        for d in DOMAINS:
            try: rows.append(audit_domain(r,d,expected[d],cp))
            except Exception as e:
                rows.append(dict(run_id=r['id'], method=r['method'], lineage=r['lineage'], domain=d,
                                 expected=len(expected[d]), prediction_count=0,
                                 extra_predictions=[], missing_predictions=sorted(expected[d]),
                                 summary='', summary_exists=False, report='', report_exists=False,
                                 png_count=0, checkpoint_status=cp['status'], binding='未验证',
                                 artifact_complete=False, status='失败/待核查',
                                 errors=[f'审计异常，产物数量未验证：{type(e).__name__}: {e}']))
        print(r['id'],r['method'],cp['status'],flush=True)
    process_lines=[s for s in subprocess.check_output(['ps','-eo','pid,comm,args'],text=True).splitlines() if re.search(r'(nnUNetv2_train|python.*(?:run_task02|evaluate_task02|nnunetv2_predict))',s) and 'build_guidance_result_tables' not in s]
    STATS.mkdir(exist_ok=True)
    payload=dict(generated_at=datetime.now().isoformat(),scope='source-only architecture inventory',
                 excluded_changed_training_data=sorted(excluded_changed_training),
                 excluded_old_moe_count=len(excluded_old_moe),
                 expected_cases={k:sorted(v) for k,v in expected.items()},runs=runs,checkpoints=cps,domains=rows,process_snapshot=process_lines)
    (STATS/'guidance_result_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
    fields=['run_id','method','lineage','domain','liver','tumor','overall','iou','recall','precision','n_positive','n_no_tumor','fp_no_tumor','fp_no_tumor_rate','prediction_count','expected','summary_exists','report_exists','png_count','checkpoint_status','binding','status','metric_source','summary','report']
    with (STATS/'guidance_result_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    write_notes(runs,rows,cps,process_lines)
    print('TOTAL',len(runs),Counter(x['status'] for x in rows),flush=True)


if __name__=='__main__':
    main()
