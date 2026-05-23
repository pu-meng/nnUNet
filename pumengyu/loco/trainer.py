"""
nnUNetTrainer_LoCo — BCE / ICE 对比学习辅助 loss

来源思路：Low-Contrast-Enhanced Contrastive Learning (LoCo, arXiv:2412.02314)
适配：全监督 3D CT 肝脏肿瘤分割（Dataset004）

ICE（Inter-class Contrast Enhancement，类间对比增强）
    找特征向量与自身类原型相似度最低的体素（模型最混淆的体素），
    用对比 loss 把它们往类原型拉近。

BCE（Boundary Contrast Enhancement，边界对比增强）
    找边界体素（3×3×3邻域内存在不同类），从中选与类原型相似度最低的，
    用对比 loss 强化边界区域的特征区分度。

实验变体（逐步消融）：
    nnUNetTrainer_BCE      — 只加 BCE
    nnUNetTrainer_ICE      — 只加 ICE
    nnUNetTrainer_BCE_ICE  — BCE + ICE

内存设计：
    不对整个 feature map 展开，而是先采样 MAX_VOXELS 个体素坐标，
    再用 gather 取特征，通过 MLP projector 投影到低维对比空间。
    MAX_VOXELS=20000 时显存增加约 200MB。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from pumengyu.mixins import SmallTumorOversampleMixin, AutoReportMixin


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_mask(gt: torch.Tensor) -> torch.Tensor:
    """
    3D 边界检测：3×3×3 邻域内存在不同类的体素 = 边界体素。
    gt : [B, 1, D, H, W] int
    返回 [B, D, H, W] bool
    """
    gt_f = gt.float()
    mx = F.max_pool3d(gt_f, kernel_size=3, stride=1, padding=1)
    mn = -F.max_pool3d(-gt_f, kernel_size=3, stride=1, padding=1)
    return (mx != mn).squeeze(1)


def _class_prototypes(
    z_s: torch.Tensor,   # [N_s, D]  已 detach
    gt_s: torch.Tensor,  # [N_s]
) -> dict[int, torch.Tensor]:
    """
    按类别计算特征均值（类原型）。
    原型 detach，不参与梯度计算，避免 prototype collapse。
    """
    protos = {}
    for cls in gt_s.unique():
        c = cls.item()
        if c < 0:
            continue
        mask = gt_s == cls
        if mask.sum() == 0:
            continue
        protos[c] = z_s[mask].mean(0)   # z_s 已 detach，此处无梯度
    return protos


# ─────────────────────────────────────────────────────────────────────────────
# LCC 对比 loss
# ─────────────────────────────────────────────────────────────────────────────

class LCCLoss(nn.Module):
    """
    监督对比 loss：困难体素向自身类原型靠近，远离其他类原型。
    实现为 softmax cross-entropy over prototype similarities（InfoNCE 简化版）。
    """

    def __init__(self, temp: float = 0.1):
        super().__init__()
        self.temp = temp

    def forward(
        self,
        z_hard: torch.Tensor,                  # [N, D]  带梯度
        gt_hard: torch.Tensor,                  # [N]     int
        prototypes: dict[int, torch.Tensor],    # {cls: [D]}  已 detach
    ) -> torch.Tensor:
        if len(prototypes) < 2 or len(z_hard) == 0:
            return z_hard.sum() * 0.0

        classes   = sorted(prototypes.keys())
        proto_mat = torch.stack([prototypes[c] for c in classes])   # [K, D]
        cls2idx   = {c: i for i, c in enumerate(classes)}

        sim    = torch.mm(z_hard, proto_mat.T) / self.temp          # [N, K]
        target = torch.tensor(
            [cls2idx.get(g.item(), -1) for g in gt_hard],# type: ignore
            device=z_hard.device, dtype=torch.long,
        )
        valid = target >= 0
        if valid.sum() == 0:
            return z_hard.sum() * 0.0

        return F.cross_entropy(sim[valid], target[valid])


# ─────────────────────────────────────────────────────────────────────────────
# Base Trainer
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainer_LoCoBase(SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    在 nnUNetTrainer 基础上添加 BCE / ICE 对比辅助 loss。
    子类只需设置 USE_BCE / USE_ICE 标志，无需重写其他方法。
    """

    USE_BCE: bool = False
    USE_ICE: bool = False

    PROJ_DIM:      int   = 64      # projector 输出维度
    TOP_K_PCT:     float = 0.25    # 每类中选取困难体素的比例
    LAMBDA_LCC:    float = 0.1     # LCC loss 权重
    TEMP:          float = 0.1     # 对比 loss 温度参数
    WARMUP_EPOCHS: int   = 50      # λ 线性预热轮数
    MAX_VOXELS:    int   = 20_000  # 每次采样的体素数上限（显存保护）

    # ── 初始化 ───────────────────────────────────────────────────────────────

    def initialize(self):
        super().initialize()

        # 在最高分辨率 decoder stage 上注册 forward hook，捕获特征图
        self._decoder_features: torch.Tensor | None = None
        net = getattr(self.network, 'module', self.network)  # 兼容 DDP
        net.decoder.stages[-1].register_forward_hook( #type: ignore
            lambda m, i, o: setattr(self, '_decoder_features', o)
        )

        # 一次 dummy forward 以获取 feat_dim
        patch = self.configuration_manager.patch_size
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.num_input_channels, *patch, device=self.device)# type: ignore
            self.network(dummy)# type: ignore
        feat_dim = self._decoder_features.shape[1]# type: ignore
        self._decoder_features = None

        # MLP projector（作用于采样体素特征向量，非整张 feature map）
        self._projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, self.PROJ_DIM, bias=False),
        ).to(self.device)
        self._lcc = LCCLoss(self.TEMP)

        # 将 projector 参数加入已有 optimizer（随主网络 LR schedule 更新）
        self.optimizer.add_param_group({# type: ignore
            'params':       list(self._projector.parameters()),
            'lr':           self.optimizer.param_groups[0]['lr'],# type: ignore
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 3e-5),# type: ignore
        })
        self.print_to_log_file(
            f"[LoCo] feat_dim={feat_dim}, PROJ_DIM={self.PROJ_DIM}, "
            f"BCE={self.USE_BCE}, ICE={self.USE_ICE}, "
            f"lambda={self.LAMBDA_LCC}, warmup={self.WARMUP_EPOCHS}"
        )

    # ── 采样 + 投影 ──────────────────────────────────────────────────────────

    def _sample_and_project(
        self,
        feat: torch.Tensor,   # [B, C, D, H, W]  decoder 输出特征（带梯度）
        gt:   torch.Tensor,   # [B, 1, D, H, W]  GT 标签
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        随机采样 MAX_VOXELS 个有效体素，gather 其特征后投影。
        返回：
          z_s   [N_s, PROJ_DIM]  投影后特征（带梯度，用于 loss 计算）
          gt_s  [N_s]            对应 GT 类别
          bnd_s [N_s] bool | None   是否为边界体素（USE_BCE=True 时返回）
        """
        B, C, D, H, W = feat.shape
        gt_flat = gt.reshape(-1)

        bnd_flat = None
        if self.USE_BCE:
            bnd_flat = _boundary_mask(gt).reshape(-1)   # [B*D*H*W]

        # 有效体素（排除 ignore_label=-1）
        valid = (gt_flat >= 0).nonzero(as_tuple=True)[0]
        if len(valid) > self.MAX_VOXELS:
            perm  = torch.randperm(len(valid), device=feat.device)[:self.MAX_VOXELS]
            valid = valid[perm]

        gt_s  = gt_flat[valid]
        bnd_s = bnd_flat[valid] if bnd_flat is not None else None

        # flat index → (b, d, h, w)
        b_i   = valid // (D * H * W)
        rem   = valid %  (D * H * W)
        d_i   = rem   // (H * W)
        rem   = rem   %  (H * W)
        h_i   = rem   // W
        w_i   = rem   %  W

        feat_s = feat[b_i, :, d_i, h_i, w_i].float()   # [N_s, C]  带梯度
        z_s    = F.normalize(self._projector(feat_s), dim=-1)  # [N_s, PROJ_DIM]

        return z_s, gt_s, bnd_s

    # ── ICE：类间困难体素选取 ─────────────────────────────────────────────────

    def _ice_indices(
        self,
        z_s:    torch.Tensor,
        gt_s:   torch.Tensor,
        protos: dict,
    ) -> torch.Tensor:
        """
        每类中相似度最低的 TOP_K_PCT 体素（与类原型最远 = 最困难）。
        相似度用 detach 的特征计算，避免影响 prototype 稳定性。
        """
        sim = torch.zeros(len(z_s), device=z_s.device)
        for cls, proto in protos.items():
            m = gt_s == cls
            if m.sum():
                sim[m] = (z_s[m].detach() * proto).sum(-1)

        hard = []
        for cls in protos:
            m = (gt_s == cls).nonzero(as_tuple=True)[0]
            if len(m):
                k = max(1, int(len(m) * self.TOP_K_PCT))
                _, r = sim[m].topk(k, largest=False)
                hard.append(m[r])
        if not hard:
            return torch.empty(0, dtype=torch.long, device=z_s.device)
        return torch.cat(hard)

    # ── BCE：边界困难体素选取 ─────────────────────────────────────────────────

    def _bce_indices(
        self,
        z_s:    torch.Tensor,
        gt_s:   torch.Tensor,
        bnd_s:  torch.Tensor,   # [N_s] bool
        protos: dict,
    ) -> torch.Tensor:
        """
        边界体素中相似度最低的 TOP_K_PCT（与类原型最远的边界体素）。
        """
        b_idx = bnd_s.nonzero(as_tuple=True)[0]
        if len(b_idx) == 0:
            return torch.empty(0, dtype=torch.long, device=z_s.device)

        sim = torch.zeros(len(b_idx), device=z_s.device)
        for cls, proto in protos.items():
            m = gt_s[b_idx] == cls
            if m.sum():
                sim[m] = (z_s[b_idx][m].detach() * proto).sum(-1)

        k = max(1, int(len(b_idx) * self.TOP_K_PCT))
        _, r = sim.topk(k, largest=False)
        return b_idx[r]

    # ── train_step ───────────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        data   = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)# type: ignore
        ctx = (autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())

        lam    = min(self.current_epoch / max(self.WARMUP_EPOCHS, 1), 1.0) * self.LAMBDA_LCC
        do_lcc = lam > 0 and (self.USE_BCE or self.USE_ICE)

        with ctx:
            output = self.network(data)# type: ignore
            l_seg  = self.loss(output, target)# type: ignore

            l_lcc = torch.tensor(0., device=self.device)
            if do_lcc:
                feat  = self._decoder_features   # 由 hook 捕获，带梯度
                gt_hr = target[0] if isinstance(target, list) else target

                z_s, gt_s, bnd_s = self._sample_and_project(feat, gt_hr)# type: ignore
                protos = _class_prototypes(z_s.detach(), gt_s)

                if self.USE_ICE:
                    ice_idx = self._ice_indices(z_s, gt_s, protos)
                    if len(ice_idx):
                        l_lcc = l_lcc + self._lcc(z_s[ice_idx], gt_s[ice_idx], protos)

                if self.USE_BCE and bnd_s is not None:
                    bce_idx = self._bce_indices(z_s, gt_s, bnd_s, protos)
                    if len(bce_idx):
                        l_lcc = l_lcc + self._lcc(z_s[bce_idx], gt_s[bce_idx], protos)

            l = l_seg + lam * l_lcc

        all_params = (list(self.network.parameters())# type: ignore
                      + list(self._projector.parameters()))
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)# type: ignore
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.grad_scaler.step(self.optimizer)# type: ignore
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.optimizer.step()# type: ignore

        return {
            'loss':  l.detach().cpu().numpy(),
            'l_seg': l_seg.detach().cpu().numpy(),
            'l_lcc': l_lcc.detach().cpu().numpy(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 实验变体（逐步消融）
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainer_BCE(nnUNetTrainer_LoCoBase):
    """实验 Step 1：只加 BCE（边界对比增强）"""
    USE_BCE = True
    USE_ICE = False


class nnUNetTrainer_ICE(nnUNetTrainer_LoCoBase):
    """实验 Step 2：只加 ICE（类间对比增强）"""
    USE_BCE = False
    USE_ICE = True


class nnUNetTrainer_BCE_ICE(nnUNetTrainer_LoCoBase):
    """实验 Step 3：BCE + ICE 合用"""
    USE_BCE = True
    USE_ICE = True
