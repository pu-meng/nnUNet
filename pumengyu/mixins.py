"""
OOP=Object-Oriented Programming,面向对象编程
核心概念三个:封装,把数据和方法打包到一个类里;
继承,子类复用父类的代码
多态,同一个方法名,不同类有不同的行为;
通用 Trainer Mixin — 供各子 Trainer 按需继承，不依赖具体网络结构。

使用方式：
    class nnUNetTrainer_MyMethod(SomeMixin, AutoReportMixin, nnUNetTrainer):
        ...

Mixin 列表
----------
SizeStratifiedOversampleMixin  ← 当前推荐
    按肿瘤大小自动分层重复：极小/小/极大/无肿瘤各设重复次数，边界从训练集百分位自动算，
    无需外部 JSON 或硬编码阈值，适配任意数据集。

StratifiedOversampleMixin
    分层过采样：无肿瘤 case 固定重复，肿瘤 case 按 wasserstein 难度映射重复次数。
    依赖外部 separability_metrics.json，已被 SizeStratifiedOversampleMixin 替代。

SmallTumorOversampleMixin
    训练集中小肿瘤 case 在 identifiers 层面重复采样（硬编码阈值 6000 locs）。
    已被 SizeStratifiedOversampleMixin 替代，保留供参考。

CopyPasteMixin
    训练时从小肿瘤库随机抽取 ROI，粘贴到其他 case 的肝脏区域，
    直接增加极小肿瘤的出现频率，解决 LiTS 中极小肿瘤严重失败问题。

UnifiedFocalLossMixin
    在 nnUNet 默认 CE+Dice loss 之上叠加 AsymmetricUnifiedFocalLoss，
    自动平衡极小前景与大量背景体素的梯度贡献，无需手动调 class weight。
    依赖 compound-loss-pytorch-main/unified_focal_loss_pytorch.py（官方库）。

BboxJitterMixin
    训练时对图像边界随机置零，模拟 Stage1 预测框与 GT 框的偏差，
    弥合两阶段流水线的 train-test distribution gap。

AutoReportMixin
    验证结束后自动调用 run_auto_report，生成 report_custom.txt。
"""

from __future__ import annotations
import importlib.util as _ilu
import os as _os#_os是os模块的别名
from os.path import join

from pathlib import Path
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from pumengyu.tools.analyasis.auto_report import run_auto_report

import gc
import ctypes
import numpy as np
import torch
import blosc2
from scipy.ndimage import label as cc_label
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

# ------------------------------------------------------------------ #
# UnifiedFocalLoss 辅助：懒加载官方库，避免启动时强依赖               #
# ------------------------------------------------------------------ #
_UFL_FILE = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    '..', 'compound-loss-pytorch-main', 'unified_focal_loss_pytorch.py'
)
_AUFL_CLS = None#写在文件的最外层,不在任何函数或类里面,这个是模块级别的变量


def _load_aufl():
    """懒加载 AsymmetricUnifiedFocalLoss，结果全局缓存。"""
    global _AUFL_CLS #global声明全局变量,这样在函数内部就可以修改全局变量的值了
    if _AUFL_CLS is None:
        #第一步.spec_from_file_location()找到文件,创建"说明书"",告诉python有一个模块,名字叫做'unified_focal_loss_pytorch'

        spec = _ilu.spec_from_file_location('unified_focal_loss_pytorch', _UFL_FILE)
        #第二步:创建空模块对象,
        mod  = _ilu.module_from_spec(spec)#type:ignore
        #第三步:执行文件,填充内容
        spec.loader.exec_module(mod)      # type: ignore[union-attr]
        _AUFL_CLS = mod.AsymmetricUnifiedFocalLoss
    return _AUFL_CLS


class _UFLWrapper(torch.nn.Module):
    """
    将 AsymmetricUnifiedFocalLoss（二值肿瘤）叠加到 nnUNet 默认 loss 上。
    这个类是包装器,将两个loss合并为一个,让nnUNet使用
    """

    def __init__(self, base_loss, ufl_fn, tumor_cls_idx: int, ufl_lambda: float):
        """
        ufl_fn是AsymmetricUnifiedFocalLoss类的一个实例,负责计算UFL
        tumor_cls_idx是肿瘤的类别索引,LiTS种,肝脏=1,肿瘤=2,所以tumor_cls_idx=2
        ufl_lambda是UFL的权重,用于平衡UFL和CE+Dice的loss
        """
        super().__init__()
        self.base_loss  = base_loss
        self.ufl_fn     = ufl_fn
        self.tumor_idx  = tumor_cls_idx
        self.ufl_lambda = ufl_lambda

    def forward(self, net_output, target):
        """
        nnUNet默认开启深监督,
        net_output是网络输出,是一个list,包含多个输出,
        =[
        全分辨率logits,#(B,C,Z,Y,X)
        1/4分辨率logits,#(B,C,Z/4,Y/4,X/4)
        ]
        target是真实标签,是一个list,包含多个标签,对应各个分辨率的标签
        nnunet计算loss,这个的unet的decoder不是很多层吗,每层的输出都会记录,给与权重,总的loss=加权求和,
        低分辨率层权重小,全分辨率层权重大
        """
        base = self.base_loss(net_output, target)

        # 只在全分辨率（deep-supervision 第0层）上计算 UFL
        logits = net_output[0] if isinstance(net_output, list) else net_output  # (B,C,Z,Y,X)
        tgt    = target[0]    if isinstance(target,     list) else target        # (B,1,Z,Y,X)

        # 转 float32，防止 autocast 下精度问题
        probs  = torch.softmax(logits.float(), dim=1)                           # (B,C,Z,Y,X)
        p_tumor    = probs[:, self.tumor_idx]                                       # (B,Z,Y,X)
        y_pred = torch.stack([1.0 - p_tumor, p_tumor], dim=1)                            # (B,2,Z,Y,X)
#默认规则,注释写的是等号左边的变量的维度
#tgt[:,0]维度是(B,Z,Y,X),值是0/1/2/.../C-1整数
#.long()将值转为整数,.float()是把True/False转为1.0/0.0
        tm     = (tgt[:, 0].long() == self.tumor_idx).float()                  # (B,Z,Y,X)
        y_true = torch.stack([1.0 - tm, tm], dim=1)                            # (B,2,Z,Y,X)
#list []里面装两个元素是完全合法
        ufl = self.ufl_fn(y_pred, y_true)
        return base + self.ufl_lambda * ufl

from batchgenerators.utilities.file_and_folder_operations import load_pickle


class DifficultyOversampleMixin:
    """
    按 difficulty.json 中的内在难度分数对训练 case 重复采样。

    原理：
      SmallTumorOversampleMixin 只重复"小肿瘤 case"，忽略了极大肿瘤（长尾）、
      无肿瘤 case（baseline 误报多）等同样困难的样例。
      本 Mixin 以难度分数直接映射重复次数，覆盖所有困难来源，无需手动定义阈值。

      难度分数 → 重复次数（线性映射）：
        score == min  → DOS_MIN_REPEAT（最少 1 次）
        score == max  → DOS_MAX_REPEAT（最多 N 次）

    子类可覆盖：
        DIFFICULTY_JSON   难度权重文件路径
        DOS_MAX_REPEAT    最难 case 重复上限（默认 5）
        DOS_MIN_REPEAT    最易 case 最少出现次数（默认 1）
    """

    DIFFICULTY_JSON: str = '/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/difficulty.json'
    DOS_MAX_REPEAT: int = 5
    DOS_MIN_REPEAT: int = 1

    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()  # type: ignore
        self._expand_by_difficulty(dataset_tr)
        return dataset_tr, dataset_val

    def _expand_by_difficulty(self, dataset_tr):
        import json
        with open(self.DIFFICULTY_JSON, encoding='utf-8') as f:
            diff = json.load(f)

        keys = list(dataset_tr.identifiers)
        vals = list(diff.values())
        med = float(np.median(vals)) if vals else 1.0

        weights = np.array([diff.get(k, med) for k in keys], dtype=np.float64)
        w_min, w_max = weights.min(), weights.max()

        if w_max > w_min:
            repeats = (
                (weights - w_min) / (w_max - w_min)
                * (self.DOS_MAX_REPEAT - self.DOS_MIN_REPEAT)
                + self.DOS_MIN_REPEAT
            ).round().astype(int)
        else:
            repeats = np.ones(len(keys), dtype=int)

        extra = []
        for key, r in zip(keys, repeats):
            extra.extend([key] * (r - 1))  # 已有1次，再加(r-1)次

        n_before = len(dataset_tr.identifiers)
        dataset_tr.identifiers.extend(extra)
        self.print_to_log_file(  # type: ignore
            f"[DifficultyOversample] {n_before} → {len(dataset_tr.identifiers)} identifiers, "
            f"repeat [{repeats.min()}, {repeats.max()}], "
            f"difficulty fallback(median)={med:.4f}"
        )


class StratifiedOversampleMixin:
    """
    分层过采样：根据 case 类型分别制定重复策略，替代 SmallTumorOversampleMixin 和
    DifficultyOversampleMixin。

    三类 case 的处理方式：
      1. 无肿瘤 case（不在 separability_metrics.json 中）
             → 固定重复 SOS_NO_TUMOR_REPEAT 次
             → 目的：降低无肿瘤 case 被训练忽略导致的误报
      2. 肿瘤 case（在 JSON 中）
             → 按 wasserstein 距离（越低 = 肿瘤肝脏越难区分 = 越难 = 重复越多）
               线性映射到 [SOS_MIN_REPEAT, SOS_MAX_REPEAT]
             → 自然覆盖极小肿瘤（低 wasserstein）和正常大肿瘤（高 wasserstein）

    难度方向（wasserstein）：
        w 低 → 肿瘤与肝脏 HU 分布重叠大 → 难分割 → 重复次数多（SOS_MAX_REPEAT）
        w 高 → 肿瘤与肝脏 HU 分布差异大 → 易分割 → 重复次数少（SOS_MIN_REPEAT）

    子类可覆盖：
        SOS_METRICS_JSON      separability_metrics.json 路径
        SOS_METRIC            使用哪个指标（'wasserstein' 或 'auc'）
        SOS_NO_TUMOR_REPEAT   无肿瘤 case 的固定重复次数（默认 4）
        SOS_MIN_REPEAT        肿瘤 case 最少重复次数（默认 1）
        SOS_MAX_REPEAT        肿瘤 case 最多重复次数（默认 5）
        SOS_METRIC_INVERT     指标越高越难时设 True（默认 False，即低值=难）
    """

    SOS_METRICS_JSON: str  = '/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/separability_metrics.json'
    SOS_METRIC:       str  = 'wasserstein'
    SOS_NO_TUMOR_REPEAT: int = 4
    SOS_MIN_REPEAT:   int  = 1
    SOS_MAX_REPEAT:   int  = 5
    SOS_METRIC_INVERT: bool = False  # wasserstein/AUC 都是高值=易，False 表示低值=难

    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()  # type: ignore
        self._expand_stratified(dataset_tr)
        return dataset_tr, dataset_val

    def _expand_stratified(self, dataset_tr):
        import json
        with open(self.SOS_METRICS_JSON, encoding='utf-8') as f:
            metrics = json.load(f)  # {case_id: {metric: value, ...}, ...}

        keys = list(dataset_tr.identifiers)

        # 提取肿瘤 case 的指标值（no-tumor case 不在 JSON 中）
        tumor_keys = [k for k in keys if k in metrics]
        tumor_vals = np.array(
            [metrics[k].get(self.SOS_METRIC, None) for k in tumor_keys],
            dtype=np.float64,
        )
        # 过滤掉 None（某些 case 指标缺失）
        valid_mask = np.array([v is not None for v in tumor_vals])
        if valid_mask.any():
            tumor_vals_clean = tumor_vals[valid_mask].astype(np.float64)
            w_min, w_max = tumor_vals_clean.min(), tumor_vals_clean.max()
        else:
            w_min = w_max = 0.0

        def _metric_to_repeat(val) -> int:
            """将指标值映射到重复次数（低 wasserstein → 多重复）。"""
            if val is None or w_max <= w_min:
                return self.SOS_MIN_REPEAT
            # 归一化到 [0, 1]，0=最难，1=最易
            norm = (val - w_min) / (w_max - w_min)
            if self.SOS_METRIC_INVERT:
                norm = 1.0 - norm  # 高值=难时翻转
            # 难度越高（norm 越小）→ 重复越多
            difficulty = 1.0 - norm
            r = self.SOS_MIN_REPEAT + difficulty * (self.SOS_MAX_REPEAT - self.SOS_MIN_REPEAT)
            return int(round(r))

        # 计算每个 case 的重复次数
        extra = []
        n_no_tumor, n_tumor = 0, 0
        repeat_counts = []
        for key in keys:
            if key not in metrics:
                # 无肿瘤 case：固定重复
                r = self.SOS_NO_TUMOR_REPEAT
                n_no_tumor += 1
            else:
                val = metrics[key].get(self.SOS_METRIC, None)
                r = _metric_to_repeat(val)
                n_tumor += 1
            repeat_counts.append(r)
            extra.extend([key] * (r - 1))  # 已有 1 次，再加 (r-1) 次

        n_before = len(dataset_tr.identifiers)
        dataset_tr.identifiers.extend(extra)

        repeat_arr = np.array(repeat_counts)
        self.print_to_log_file(  # type: ignore
            f"[StratifiedOversample] metric={self.SOS_METRIC}, "
            f"no_tumor={n_no_tumor}×{self.SOS_NO_TUMOR_REPEAT}, "
            f"tumor={n_tumor} repeat[{repeat_arr[repeat_arr > 1].min() if (repeat_arr > 1).any() else 1}"
            f", {repeat_arr.max()}], "
            f"identifiers {n_before} → {len(dataset_tr.identifiers)}"
        )


class SmallTumorOversampleMixin:
    """
    在 get_tr_and_val_datasets() 阶段扫描 class_locations，
    将小肿瘤 case 在 identifiers 中重复 SMALL_TUMOR_REPEAT 次。

    class_locations 长度说明：
      - nnUNet 上限约 10000；实测小肿瘤(<5k voxel)通常 < 6000
      - 0 = 无肿瘤 case（不做重复，避免放大误报）

    子类可覆盖类变量调整力度：
        SMALL_TUMOR_THRESH_LOCS = 4000
        SMALL_TUMOR_REPEAT      = 2
    nnUNet的每轮是固定迭代次数(默认250个batch)
    每个batch:
    1.从identifiers中随机抽取一个case
    2.从该case中随机抽取一个patch
    3.将两个patch拼接成一个batch,[2,C,Z,Y,X ]送进网络,nnunet默认batch=2
    一个batch=一次独立的forward+backward+update
    identifiers是所有case的名字列表
    """

    SMALL_TUMOR_THRESH_LOCS: int = 6000 #小于这个值的case会被认定为小肿瘤case
    SMALL_TUMOR_REPEAT:      int = 3#小肿瘤case在identifiers中重复的次数

    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()#type:ignore
        #dataset_tr是dataset对象,有identifiers属性,是所有case的名字列表
        self._expand_small_tumor_indices(dataset_tr)#把小肿瘤case的名字重复加入进去
        return dataset_tr, dataset_val

    def _expand_small_tumor_indices(self, dataset_tr):
        tumor_cls = self.label_manager.foreground_labels[-1]#type:ignore
        folder    = self.preprocessed_dataset_folder#type:ignore
        extra, n_small = [], 0

        for key in list(dataset_tr.identifiers):
            props  = load_pickle(join(folder, key + '.pkl'))
            #拼成完整的路径 /nnUNet_workspace/preprocessed/Dataset003/liver_0001.pkl
            #props有"class_locations",里面有肝脏体素的位置列表,肿瘤体素的位置列表
            n_locs = len(props.get('class_locations', {}).get(tumor_cls, []))
            #n_locs是肿瘤体素的位置的记录数,表示到底有多少个肿瘤
            if 0 < n_locs < self.SMALL_TUMOR_THRESH_LOCS:
                extra.extend([key] * (self.SMALL_TUMOR_REPEAT - 1))
                n_small += 1

        n_before = len(dataset_tr.identifiers)
        dataset_tr.identifiers.extend(extra)
        self.print_to_log_file(  #type:ignore
            f"[小肿瘤重复] 阈值={self.SMALL_TUMOR_THRESH_LOCS} locs, "
            f"重复次数={self.SMALL_TUMOR_REPEAT}x, "
            f"small={n_small}/{n_before}, "
            f"identifiers {n_before} → {len(dataset_tr.identifiers)}"
        )


class SizeStratifiedOversampleMixin:
    """
    按肿瘤大小自动分层重复过采样。不依赖任何外部 JSON，边界从当前 fold 训练集
    的大小分布中自动计算（百分位数），适配任意数据集。

    分层策略（按 class_locations 长度衡量体素数）：
        无肿瘤（n=0）  → SSO_NO_TUMOR_REPEAT 次
        极小（< P_TINY） → SSO_TINY_REPEAT 次
        小（P_TINY ~ P_SMALL）→ SSO_SMALL_REPEAT 次
        中/大（中间段） → 1 次（不额外重复）
        极大（> P_HUGE） → SSO_HUGE_REPEAT 次

    百分位数从 **有肿瘤** case 的大小分布中计算，保证每个过采样类别始终有 case。
    极大：最大的 (1-SSO_HUGE_PCTL/100) 比例；极小：最小的 SSO_TINY_PCTL/100 比例。

    可覆盖的类变量：
        SSO_TINY_PCTL        极小上界百分位（默认 20 → 最小的 20% 为极小）
        SSO_SMALL_PCTL       小上界百分位（默认 45 → P20~P45 为小）
        SSO_HUGE_PCTL        极大下界百分位（默认 90 → 最大的 10% 为极大）
        SSO_NO_TUMOR_REPEAT  无肿瘤重复次数（默认 3）
        SSO_TINY_REPEAT      极小重复次数（默认 3）
        SSO_SMALL_REPEAT     小重复次数（默认 2）
        SSO_HUGE_REPEAT      极大重复次数（默认 3）
    """

    SSO_TINY_PCTL:  float = 20.0
    SSO_SMALL_PCTL: float = 45.0
    SSO_HUGE_PCTL:  float = 90.0

    SSO_NO_TUMOR_REPEAT: int = 3
    SSO_TINY_REPEAT:     int = 3
    SSO_SMALL_REPEAT:    int = 2
    SSO_HUGE_REPEAT:     int = 3

    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()  # type: ignore
        self._expand_size_stratified(dataset_tr)
        return dataset_tr, dataset_val

    def _expand_size_stratified(self, dataset_tr):
        tumor_cls = self.label_manager.foreground_labels[-1]  # type: ignore
        folder    = self.preprocessed_dataset_folder           # type: ignore
        keys      = list(dataset_tr.identifiers)

        # 读取每个 case 的肿瘤体素数
        sizes = {}
        for key in keys:
            props  = load_pickle(join(folder, key + '.pkl'))
            n_locs = len(props.get('class_locations', {}).get(tumor_cls, []))
            sizes[key] = n_locs

        # 仅用有肿瘤 case 的大小来计算百分位边界
        tumor_sizes = np.array([sizes[k] for k in keys if sizes[k] > 0])

        if len(tumor_sizes) == 0:
            self.print_to_log_file("[SizeStratifiedOversample] 无肿瘤 case，跳过")  # type: ignore
            return

        p_tiny  = float(np.percentile(tumor_sizes, self.SSO_TINY_PCTL))
        p_small = float(np.percentile(tumor_sizes, self.SSO_SMALL_PCTL))
        p_huge  = float(np.percentile(tumor_sizes, self.SSO_HUGE_PCTL))

        # 避免边界坍塌（极端分布下相邻百分位相等）
        p_small = max(p_small, p_tiny + 1)
        p_huge  = max(p_huge,  p_small + 1)

        extra  = []
        counts = {'no_tumor': 0, 'tiny': 0, 'small': 0, 'mid': 0, 'huge': 0}

        for key in keys:
            n = sizes[key]
            if n == 0:
                r = self.SSO_NO_TUMOR_REPEAT
                counts['no_tumor'] += 1
            elif n <= p_tiny:
                r = self.SSO_TINY_REPEAT
                counts['tiny'] += 1
            elif n <= p_small:
                r = self.SSO_SMALL_REPEAT
                counts['small'] += 1
            elif n > p_huge:
                r = self.SSO_HUGE_REPEAT
                counts['huge'] += 1
            else:
                r = 1
                counts['mid'] += 1
            extra.extend([key] * (r - 1))

        n_before = len(dataset_tr.identifiers)
        dataset_tr.identifiers.extend(extra)

        self.print_to_log_file(  # type: ignore
            f"[SizeStratifiedOversample] 自动边界（训练集 {len(tumor_sizes)} 个有肿瘤 case）:\n"
            f"  极小 ≤{p_tiny:.0f} locs (P{self.SSO_TINY_PCTL}), "
            f"小 ≤{p_small:.0f} locs (P{self.SSO_SMALL_PCTL}), "
            f"极大 >{p_huge:.0f} locs (P{self.SSO_HUGE_PCTL})\n"
            f"  无肿瘤={counts['no_tumor']}×{self.SSO_NO_TUMOR_REPEAT}, "
            f"极小={counts['tiny']}×{self.SSO_TINY_REPEAT}, "
            f"小={counts['small']}×{self.SSO_SMALL_REPEAT}, "
            f"中/大={counts['mid']}×1, "
            f"极大={counts['huge']}×{self.SSO_HUGE_REPEAT}\n"
            f"  identifiers {n_before} → {len(dataset_tr.identifiers)}"
        )


class CopyPasteMixin:
    """
    Copy-Paste 小肿瘤增强。

    on_train_start 时扫描训练集，将所有小肿瘤（class_locations 数量 ≤ CP_MAX_LOCS）
    的肿瘤 ROI 提取到内存中建库。
    train_step 对每个 batch sample 以 CP_PROB 概率随机从库中抽取一个 ROI，
    粘贴到当前 patch 的肝脏区域（seg > 0 且非肿瘤类），同步更新 CT 和 seg。

    粘贴策略：
      - 只粘贴肿瘤 mask 内的体素（不粘贴 bounding box 边缘，减少边界伪影）
      - 粘贴位置随机选取 patch 内有效前景体素为中心
      - 对深监督各分辨率层，只更新全分辨率 target[0]；极小肿瘤在低分辨率
        几乎不可见，不一致性可接受

    子类可覆盖：
        CP_PROB     = 0.5    每个 batch sample 被粘贴的概率
        CP_MAX_LOCS = 5000   class_locations 条目数上限（超过视为"非小肿瘤"）
        CP_MARGIN   = 3      提取 ROI 时肿瘤 bbox 四周扩展的 voxel 数
    """

    CP_PROB:        float = 0.5
    CP_MAX_LOCS:    int   = 5000
    CP_MARGIN:      int   = 3
    CP_NUM_DA_PROC: int   = 4   # 限制 DA worker 数，4 workers×3.5GB=14GB vs 默认12×3.5GB=42GB

    def get_dataloaders(self):
        import os
#下面是临时改变环境变量的惯用法
        prev = os.environ.get('nnUNet_n_proc_DA')#记住原值,没设过就为None
        os.environ['nnUNet_n_proc_DA'] = str(self.CP_NUM_DA_PROC)#设计为self.CP_NUM_DA_PROC
        try:
            #我们临时设计的这个'nnUNet_n_proc_DA'环境变量,只用于这一个函数.get_dataloaders()
            result = super().get_dataloaders()  #type:ignore
        #finally意思是不管try里面有没有异常,都会执行finally
        finally:
            if prev is None:
                #如果原本就没有'nnUNet_n_proc_DA'这个环境变量,就删除当作我们没设计这个
                del os.environ['nnUNet_n_proc_DA']
            else:
                #原本有值就立马放回去
                os.environ['nnUNet_n_proc_DA'] = prev
        return result

    # ------------------------------------------------------------------ #
    # 库构建                                                               #
    # ------------------------------------------------------------------ #
    def on_train_start(self):
        super().on_train_start()#type:ignore #先跑父类的正常开始训练流程
        #_cp_library
        self._cp_library: list = []#建一个空的list当作库 
        self._build_cp_library()#把库填满

    def _build_cp_library(self):
    
        tumor_cls = self.label_manager.foreground_labels[-1]#type:ignore
        folder    = self.preprocessed_dataset_folder#type:ignore
        tr_keys, _ = self.do_split()#type:ignore #tr_keys是训练名单,_是验证名单

        self._no_tumor_keys: set = set()#存储的是无肿瘤的case名字
        n_loaded = 0
        from tqdm import tqdm
        #tqdm的desc参数是进度条前面的描述,unit是进度条的单位,tr_keys是所有case的名字列表
        pbar = tqdm(tr_keys, desc='[CopyPaste] 建库', unit='case')
        #pbar是一个迭代器,每次迭代返回一个case名字
        for key in pbar:
            pbar.set_postfix(ROIs=n_loaded)#更新进度条末尾的附加信息,
            props  = load_pickle(join(folder, key + '.pkl'))
            n_locs = len(props.get('class_locations', {}).get(tumor_cls, []))
            if n_locs == 0:
                self._no_tumor_keys.add(key)
                continue
#blosc2是高性能的压缩裤,专门为科学计算设计
#.npy是numpy原生,无压缩;.nii.gz是医学图像标准压缩慢;
#.b2nd是blosc2压缩,压缩比高,解压速度快,读写极快,支持切片懒加载
#import blosc2
#.nii.gz必须把整个文件解压到内存,才能访问任意的位置
#blosc2内部把数据分成很多小块,存储,访问某个区域时只解压几块

            # 先读 seg 判断是否有小肿瘤连通域，有才读 CT，避免为大肿瘤 case 加载整个 CT
            seg_arr = blosc2.open(join(folder, key + '_seg.b2nd'), mode='r')[0]   #type:ignore
            tmask = (seg_arr == tumor_cls)
            if not tmask.any():
                continue
            labeled, n_cc = cc_label(tmask)  #type:ignore
            Z, Y, X = seg_arr.shape  #type:ignore
            m = self.CP_MARGIN
#下面七行是挑出小肿瘤,如果一个没有,就彻底清空内存,跳到下一个case
            small_ccs = [i for i in range(1, n_cc + 1)
                         if (labeled == i).sum() <= self.CP_MAX_LOCS]#这个是挑选小的连通域
            if not small_ccs:#如果small_ccs为空,就进入这个,清空内存跳过
                del seg_arr, tmask, labeled #这个是删除让他们的引用计数归零
                gc.collect()#gc是python的垃圾回收机制,gc.collect()是强制立即回收内存
                #平时python靠着引用技术自动回收,但有时候引用计数不会归零,需要手动回收
                ctypes.CDLL('libc.so.6').malloc_trim(0)#这行解决一个隐蔽问题
                #也就是python把内存释放了,但是进程没有,
                #malloc_trim(0)是glibc的一个C函数,作用是"把空闲内存尽量还给操作系统",但是python不能直接调用C函数,所以用ctypes
                #ctypes是python的一个库,可以调用C函数,调用  C函数的桥,一个外部函数接口,FFI
                #.CDLL('libc.so.6')是加载C标准库的这个共享库,CDLL=按照C调用约定加载.so
                #这个效果是真正让进程的内存占用降下来,而不只是Python内部"标记为可用"
                continue

            ct_arr = blosc2.open(join(folder, key + '.b2nd'), mode='r')[:]  #type:ignore

            for cc_id in small_ccs:
                cc_mask = (labeled == cc_id)
                coords = np.where(cc_mask)
                z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
                y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
                x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
                z0m, z1m = max(0, z0-m), min(Z, z1+m)
                y0m, y1m = max(0, y0-m), min(Y, y1+m)
                x0m, x1m = max(0, x0-m), min(X, x1+m)

                ct_roi    = ct_arr[:, z0m:z1m, y0m:y1m, x0m:x1m].astype(np.float32) #type:ignore
                # 必须 .copy()：切片是视图，torch.from_numpy 会共享内存锁住整卷 cc_mask(~82MB)
                tmask_roi = cc_mask[z0m:z1m, y0m:y1m, x0m:x1m].copy()#就是这一行决定了为什么copypaste内存会爆炸
                #加一个.copy()就解决了内存爆炸的问题
#不加入.copy(),我们就是类似a=b[1:23]这种,取切片,这种不是复制,而是共享内存,所以a,b指向同一块内存,如果a修改了,b也会跟着修改
#只要视图a还活着,b就不会被回收,所以内存占用会一直增加,直到a,b都死掉,才会被回收
#加入.copy()就是复制,不是共享内存,所以a,b指向不同的内存,修改a不会影响b,
#这样a和b都可以独立取消释放
#不然因为tmask_roi被加入_cp_library,如果不加.copy(),tmask_roi会一直活着,导致内存占用一直增加
#如果加了保存的只有每一个循环的tmask_roi,

                self._cp_library.append({
                    'ct':    torch.from_numpy(ct_roi),
                    'tmask': torch.from_numpy(tmask_roi),
                    'src':   key,   # 来源 case；难度加权子类据此查权重，均匀采样忽略此字段
                })
                n_loaded += 1

            # 每处理完一个 case 立即释放大数组，并强制 C allocator 归还内存给 OS
            del ct_arr, seg_arr, tmask, labeled
            gc.collect()
            ctypes.CDLL('libc.so.6').malloc_trim(0)

        self.print_to_log_file( #type:ignore
            f"[CopyPaste] library built: {n_loaded} ROIs "
            f"(CP_MAX_LOCS={self.CP_MAX_LOCS}, CP_MARGIN={self.CP_MARGIN}, "
            f"no_tumor_keys={len(self._no_tumor_keys)} 跳过粘贴)"
        )

    # ------------------------------------------------------------------ #
    # 粘贴逻辑                                                             #
    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict) -> dict:
        if self._cp_library:#self._cp_library是建好的小肿瘤库
            batch = self._apply_copy_paste(batch)#这个是实际的粘贴操作
        return super().train_step(batch) #type:ignore

    def _sample_roi_index(self) -> int:
        """选一个 ROI 的下标。默认均匀随机；难度加权子类覆盖此方法。"""
        return int(np.random.randint(len(self._cp_library)))

    def _apply_copy_paste(self, batch: dict) -> dict:
        data      = batch['data']    # (B,C,PZ,PY,PX) CPU float32
        target    = batch['target']  # list[(B,1,...)] 或 (B,1,PZ,PY,PX) int16
        tumor_cls = self.label_manager.foreground_labels[-1]#type:ignore

        B, C, PZ, PY, PX = data.shape
        seg_full = target[0] if isinstance(target, list) else target  # (B,1,PZ,PY,PX)
        # batch['keys'] 可能是 numpy 数组，转 list 避免数组真值歧义
        batch_keys = batch.get('keys', [])
        batch_keys = list(batch_keys) if batch_keys is not None else []

        for b in range(B):
            # 无肿瘤 case 跳过粘贴，防止污染训练分布导致推理误报
            if b < len(batch_keys) and batch_keys[b] in self._no_tumor_keys:#无肿瘤case跳过粘贴
                continue               
            if np.random.random() > self.CP_PROB:#概率门
                continue

            item  = self._cp_library[self._sample_roi_index()]
            ct_r  = item['ct']    # (C,dz,dy,dx)#小肿瘤的CT值
            tm_r  = item['tmask'] # (dz,dy,dx) bool  #小肿瘤的连通域掩码
            _, dz, dy, dx = ct_r.shape

            if dz > PZ or dy > PY or dx > PX:
                continue
#选位置粘贴
            # 有效粘贴区域：前景且非肿瘤（对 Dataset003 即肝脏体素）
            #seg_full是整个batch的分割标签,(B,1,PZ,PY,PX) int16 tensor
            seg_b = seg_full[b, 0]           # (PZ,PY,PX) int16 tensor
            #seg_b是(PZ,PY,PX)的分割标签,值是0/1/2的整数数组
            valid = (seg_b > 0) & (seg_b != tumor_cls)#必须是前景且不是肿瘤=肝脏才能粘贴
            if not valid.any():#如果没有肝脏,不沾
                continue

            # 以随机有效体素为中心计算粘贴起点
            max_z, max_y, max_x = PZ-dz, PY-dy, PX-dx
            if max_z < 0 or max_y < 0 or max_x < 0:
                continue

            vc      = torch.nonzero(valid, as_tuple=False)  # (N,3)#vc是所有前景且非肿瘤的体素坐标
            #torch.randint的第一个参数len(vc)是生成的随机整数范围是[0,len(vc))
            #第二个参数(1,)是生成的tensor的形状
            #item()是把只有一个元素的tensor变成普通的int
            pick    = vc[torch.randint(len(vc), (1,)).item()] #type:ignore #挑选一个可以的肝脏体素做中心
            pz = int((pick[0] - dz//2).clamp(0, max_z))
            py = int((pick[1] - dy//2).clamp(0, max_y))
            px = int((pick[2] - dx//2).clamp(0, max_x))
#pick 是随机选出的肝脏体素坐标[z,y,x]作为粘贴的目标中心点
            # 粘贴 CT（仅肿瘤 mask 体素）
            #data:(B,C,PZ,PY,PX) float32 tensor,CT图像的HU值,
            #ct_r:(C,dz,dy,dx) float32 tensor,小肿瘤的CT值
            #ct_r是库里整个肿瘤ROI的CT值
            patch_ct = data[b, :, pz:pz+dz, py:py+dy, px:px+dx].clone()  # (C,dz,dy,dx)
            patch_ct[:, tm_r] = ct_r[:, tm_r]
            #tm_r是肿瘤掩码,shape:(dz,dy,dx) bool tensor,表示小肿瘤ROI内的体素
            #.clone()是独立拷贝
            data[b, :, pz:pz+dz, py:py+dy, px:px+dx] = patch_ct

            # 粘贴 seg label
            patch_seg = seg_full[b, 0, pz:pz+dz, py:py+dy, px:px+dx].clone()
            patch_seg[tm_r] = tumor_cls
            seg_full[b, 0, pz:pz+dz, py:py+dy, px:px+dx] = patch_seg

        if isinstance(target, list):
            target[0] = seg_full
        else:
            batch['target'] = seg_full
        batch['data'] = data
        return batch


class DifficultyCopyPasteMixin(CopyPasteMixin):
    """
    在 CopyPasteMixin 基础上，把"均匀随机抽 ROI"改成"按内在难度加权抽 ROI"。

    难度来自 compute_difficulty.py 产出的 difficulty.json（{case: weight}，
    纯内在 overlap 排名、零泄露）。与 CopyPasteMixin 的**唯一区别 = 抽哪个 ROI**；
    建库、粘贴、无肿瘤 case 跳过等逻辑全部复用父类。

    无泄露说明：库仍只来自本折 tr_keys（do_split），验证折 ROI 不进库；
    difficulty.json 虽含全部 case，但没进库的权重用不上。

    子类可覆盖：
        DIFFICULTY_JSON   难度权重文件路径
    """

    DIFFICULTY_JSON: str = '/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/difficulty.json'

    def on_train_start(self):
        super().on_train_start()          # 先按父类逻辑建库（每个 ROI 已带 'src'）
        self._setup_difficulty_weights()

    def _setup_difficulty_weights(self):
        import json
        if not self._cp_library:
            self._cp_weights = np.array([])
            self.print_to_log_file("[DifficultyCopyPaste] 库为空，退回均匀采样")  # type:ignore
            return
        with open(self.DIFFICULTY_JSON, encoding='utf-8') as f:
            diff = json.load(f)#diff 是难度表的json,json.load()是把JSON文件解析成Python字典

        med = float(np.median(list(diff.values()))) if diff else 1.0

        w = np.array([diff.get(it['src'], med) for it in self._cp_library], dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones_like(w)#造一个和w形状,dtype都一样,但是全部填1.0的新数组
        self._cp_weights = w / w.sum()
        n_fb = sum(1 for it in self._cp_library if it['src'] not in diff)
        self.print_to_log_file(  # type:ignore
            f"[DifficultyCopyPaste] 难度加权就绪：{len(self._cp_library)} ROIs, "
            f"权重文件={self.DIFFICULTY_JSON}, 缺失用中位数兜底={n_fb}"
        )

    def _sample_roi_index(self) -> int:
        if self._cp_weights.size == 0:
            return super()._sample_roi_index()
        return int(np.random.choice(len(self._cp_library), p=self._cp_weights))


class UnifiedFocalLossMixin:
    """
    在 nnUNet 默认 CE+Dice loss 基础上叠加 AsymmetricUnifiedFocalLoss（肿瘤类二值化）。

    原理：AsymFTL 惩罚 FN（漏检），AsymFL 抑制背景 easy-example 梯度，
    两者联合自动平衡极小肿瘤体素与大量背景体素的梯度贡献。

    子类可覆盖：
        UFL_LAMBDA: float = 0.5   UFL 项相对默认 loss 的整体权重
        UFL_DELTA:  float = 0.5   Tversky delta（0.5=对称；>0.5 → 漏检惩罚 > 误报惩罚）
        UFL_GAMMA:  float = 0.2   focal 参数（越大越聚焦难样本）
    总的loss=CE+Dice+lambda*AUFL

    这个类没有父类,但是经常是组合使用,类似
    class MyTrainer(nnUNetTrainerV2,UnifiedFocalLossMixin):
    MRO方法解析顺序
    
    """

    UFL_LAMBDA: float = 0.5
    UFL_DELTA:  float = 0.5  # 0.6→0.5：对称惩罚，消除对 FN 的系统性偏置，降低无肿瘤 case 误报
    UFL_GAMMA:  float = 0.2

    def _build_loss(self):
        #super()沿着MRO找到nnUNetTrainer._build_loss(),返回nnUNet的默认的CE+Dice loss对象,存进base

        base      = super()._build_loss()#type:ignore
        AUFL      = _load_aufl()
        #weight=0.5是内部AsymFTL和AsymFL的各自占50%
        ufl_fn    = AUFL(weight=0.5, delta=self.UFL_DELTA, gamma=self.UFL_GAMMA)
        tumor_cls = self.label_manager.foreground_labels[-1]#type:ignore
        self.print_to_log_file(#type:ignore
            f"[UnifiedFocalLoss] AUFL added: lambda={self.UFL_LAMBDA}, "
            f"delta={self.UFL_DELTA}, gamma={self.UFL_GAMMA}, tumor_cls={tumor_cls}"
        )
        return _UFLWrapper(base, ufl_fn, tumor_cls, self.UFL_LAMBDA)


class _NoTumorFPWrapper(torch.nn.Module):
    """
    无肿瘤误报专项惩罚 loss 包装器。

    对 batch 中 GT 完全无肿瘤的 sample，在基础 loss 之上额外叠加
    "预测肿瘤概率均值" 作为惩罚项，促使模型在无肿瘤 patch 上不乱预测肿瘤。
    有肿瘤 sample 的 loss 路径不受任何影响。
    """

    def __init__(self, base_loss, tumor_cls_idx: int, penalty_lambda: float):
        super().__init__()
        self.base_loss      = base_loss
        self.tumor_idx      = tumor_cls_idx
        self.penalty_lambda = penalty_lambda

    def forward(self, net_output, target):
        base = self.base_loss(net_output, target)

        logits  = net_output[0] if isinstance(net_output, list) else net_output  # (B,C,Z,Y,X)
        tgt     = target[0]    if isinstance(target,     list) else target        # (B,1,Z,Y,X)

        probs   = torch.softmax(logits.float(), dim=1)   # (B,C,Z,Y,X)
        p_tumor = probs[:, self.tumor_idx]                # (B,Z,Y,X)

        # 逐 sample 判断 GT 是否含肿瘤
        has_tumor    = (tgt[:, 0].long() == self.tumor_idx).any(dim=(1, 2, 3))  # (B,)
        no_tumor_idx = (~has_tumor).nonzero(as_tuple=True)[0]

        if no_tumor_idx.numel() == 0:
            return base

        penalty = p_tumor[no_tumor_idx].mean()
        return base + self.penalty_lambda * penalty


class NoTumorFPPenaltyMixin:
    """
    无肿瘤专项误报惩罚 Mixin。

    当 batch 中某 sample 的 GT 无任何肿瘤体素时，
    额外惩罚预测出肿瘤的概率（p_tumor 均值作为 penalty term）。
    有肿瘤 sample 的 loss 不受影响，精确针对误报场景。

    可覆盖的类变量：
        NTFP_LAMBDA  惩罚权重（默认 1.0）
    """

    NTFP_LAMBDA: float = 1.0

    def _build_loss(self):
        base      = super()._build_loss()  # type: ignore
        tumor_cls = self.label_manager.foreground_labels[-1]  # type: ignore
        self.print_to_log_file(  # type: ignore
            f"[NoTumorFPPenalty] 无肿瘤误报惩罚已启用: lambda={self.NTFP_LAMBDA}, tumor_cls={tumor_cls}"
        )
        return _NoTumorFPWrapper(base, tumor_cls, self.NTFP_LAMBDA)


class BboxJitterMixin:
    """
    Stage-aware Crop Jitter：弥合两阶段流水线 train-test distribution gap。

    问题根源：
      Dataset004 训练时用 GT 肝脏框裁剪（边界完美），
      推理时用 Stage1 预测框裁剪（边界有几 mm 偏差）。
      Stage2 模型从未见过"偏移的框"，E2E Dice 因此低于 GT-crop 验证值。

    解决方案：
      训练的每个 batch，以概率 JITTER_P 对图像的随机 1-3 个面
      置零宽度 ∈ [1, JITTER_MAX_MM/spacing] 个体素的边界条带，
      模拟 Stage1 预测框与 GT 框之间的偏差。
      Stage2 因此学会对不完美的裁剪边界保持鲁棒。

    类变量（子类可覆盖）：
        JITTER_MAX_MM: float = 10.0   最大扰动宽度（mm）
        JITTER_P:      float = 0.5    每个 sample 被扰动的概率
    """

    JITTER_MAX_MM: float = 10.0
    JITTER_P:      float = 0.5

    def train_step(self, batch: dict) -> dict:
        batch['data'] = self._apply_bbox_jitter(batch['data'])
        return super().train_step(batch)#type:ignore

    def _apply_bbox_jitter(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: (B, C, Z, Y, X) float32 CPU Tensor
        随机对 1-3 个面置零边界条带，模拟 Stage1 crop 偏差。
        """
        spacing = self.configuration_manager.spacing  #type:ignore # [sp_z, sp_y, sp_x] mm/voxel
        max_vox = [max(1, int(np.ceil(self.JITTER_MAX_MM / s))) for s in spacing]
#np.ceil()是向上取整,无论小数多小,只要有小数就进一位
        data = data.clone()
        B = data.shape[0]
        for b in range(B):
            if np.random.random() >= self.JITTER_P:
                continue
#n_faces是随机决定这次扰动几个面,范围是[1,4)
            n_faces = np.random.randint(1, 4)
#replace=False不放回抽样,选出的编号不会重复,
#faces是随机选出的编号,范围是[0,6),形状是(n_faces,)
            faces   = np.random.choice(6, size=n_faces, replace=False)
            for face in faces:
                ax   = int(face) // 2          # 0=Z, 1=Y, 2=X
#0是起始端(上/前/左),1是末端(下/后/右)
                side = int(face) % 2           # 0=start, 1=end
                ax_size = data.shape[ax + 2]   # shape: (B, C, Z, Y, X)
                n_drop  = np.random.randint(1, max_vox[ax] + 1)
                n_drop  = min(n_drop, ax_size)
                if ax == 0:
                    if side == 0:
                        data[b, :, :n_drop, :, :]          = 0.0
                    else:
                        data[b, :, ax_size - n_drop:, :, :] = 0.0
                elif ax == 1:
                    if side == 0:
                        data[b, :, :, :n_drop, :]          = 0.0
                    else:
                        data[b, :, :, ax_size - n_drop:, :] = 0.0
                else:
                    if side == 0:
                        data[b, :, :, :, :n_drop]          = 0.0
                    else:
                        data[b, :, :, :, ax_size - n_drop:] = 0.0
        return data


class AutoReportMixin:
    """
    验证结束后自动生成 report_custom.txt / report_custom.json。
    依赖环境变量 nnUNet_preprocessed / nnUNet_raw。
    """

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)#type:ignore
        try:
          
            dataset_name = self.plans_manager.dataset_name#type:ignore
            fold_dir = Path(self.output_folder)#type:ignore
            gt_dir   = Path(nnUNet_preprocessed) / dataset_name / "gt_segmentations"
            img_dir  = Path(nnUNet_raw)           / dataset_name / "imagesTr"

            self.print_to_log_file(f"[AutoReport] 生成报告: {fold_dir.name}")#type:ignore
            run_auto_report(fold_dir, gt_dir, img_dir, min_tumor_size=0)
            self.print_to_log_file("[AutoReport] 报告生成完成")#type:ignore
        except Exception as e:
            self.print_to_log_file(f"[AutoReport] 失败: {e}")#type:ignore
