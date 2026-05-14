"""
nnUNetTrainer_BoundaryLoss,这个损失计算
nnUNetTrainer_BoundaryLoss
Adds boundary-aware surface distance loss (BATseg, arXiv:2412.06507) to
the standard nnUNet training loop.  Drop-in replacement for nnUNetTrainer.

Architecture
  Standard nnUNet backbone + parallel distance-field prediction head
  tapped from the highest-resolution decoder features.

Loss
  L = L_seg + L_boundary
  L_seg      = CE + Dice  (unchanged from nnUNetTrainer)
  L_boundary = mean(|e|² · |e|)  where e = pred_dist - gt_dist
  (cubic penalty: large errors near boundaries dominate; tiny far-field
  errors vanish automatically — no hard truncation needed in the loss.)

Distance field GT
  Computed online from the *augmented* segmentation mask inside each
  DataLoader worker process.  This is correct: spatial augmentation
  (rotation, scaling, elastic) changes geometry, so the distance field
  must be derived after the transform, not before.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import autocast
#autocast是Pytorch得自动混合精度(AMP),训练时自动把部分计算从float32降到float16,加速训练,减少显存,精度损失极小

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _trainer_module
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader, crop_and_pad_nd
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
#get_network_from_plans是按照plans文件里面得配置自动构建网络结构(层数,通道数等),nnUNet得核心自动化功能之一

from nnunetv2.utilities.helpers import dummy_context

from pumengyu.boundary.dist_field import compute_batch_distance_field


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class _BoundaryDataLoader(nnUNetDataLoader):
    """
    Extends nnUNetDataLoader to compute a surface distance field for each
    training sample after augmentation.
    Class-level _dist_params must be populated before worker processes fork.
    """
    _dist_params: dict = {'num_classes': None, 'spacing': None}
#self.变量,得规则是先找__init__定义的self.=,再找类变量,类里面直接写的变量,类似这个_dist_params
#再次找父类变量,还找不到就报AttributeError
    def generate_train_batch(self) -> dict:
        selected_keys = self.get_indices()
        #.get_indices()是父类得实例方法
        from threadpoolctl import threadpool_limits
        #threadpoolctl是控制底层线程数得库,

        data_all = seg_all = dist_all = None
        num_classes = self._dist_params['num_classes']
        spacing     = self._dist_params['spacing']

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                #limits=1限制底层库只用1个线程,user_api=None是对所有底层库生效
                for j, i in enumerate(selected_keys):
#selected_keys=["liver_001","liver_047","liver_023","liver_089"]
#i是"liver_001"这些
                    force_fg = self.get_do_oversample(j)
                    #self.get_do_oversample()是判断这个样本是否需要强制包含前景
#nnUNet得策略是batch得前几个样本随机采样,后几个样本强制包含前景,防止全是背景得patch主导训练
#
                    data, seg, seg_prev, properties = self._data.load_case(i)
                #data是CT图像,seg是分割标签,seg_prev上一个阶段得预测,没有则None,properties是元数据字典,包含spacing,等

                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape, force_fg, properties['class_locations'])
#bbox_lbs这里的lbs是下界,ubs是上界
#self.get_bbox()是返回这个patch在完整图像得裁剪范围                    
                    bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]

                    data_c = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    #0是超出边界时候填充得值,
                    seg_c  = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1,
                                        cast_cropped_to=np.int16)).to(torch.int16)
                    if seg_prev is not None:
            #seg_prev是上一个阶段得预测,如果不是None就一起裁剪,并拼接到seg_c得后面,作为额外得监督信号
                        seg_prev_c = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1,
                                            cast_cropped_to=np.int16)).to(torch.int16)
                        seg_c = torch.cat((seg_c, seg_prev_c[None]), dim=0)
#None是numpy/PyTorch得索引技巧,在该位置插入
                    if self.patch_size_was_2d:
                        data_c = data_c[:, 0]
                        seg_c  = seg_c[:, 0]
#self.patch_size_was_2d=true表示这个数据集本身是2D的,但是nnUNet内部统一用3D格式处理,所以数据被临时拓展成[C,1,H,W]
#[:,0]第0维全取,第1维取索引0,去掉这个维度
                    if self.transforms is not None:
                        out       = self.transforms(image=data_c, segmentation=seg_c)
                        data_s    = out['image']
                        seg_s     = out['segmentation']
                    else:
                        data_s, seg_s = data_c, seg_c

                    seg_for_edt = seg_s[0] if isinstance(seg_s, list) else seg_s
                    dist_s = compute_batch_distance_field(
                        seg_for_edt.unsqueeze(0),
                        num_classes=num_classes,
                        spacing=spacing,
                    )[0]  # (K, H, W, D)

                    if data_all is None:
                        data_all = torch.empty(
                            (self.batch_size, *data_s.shape), dtype=torch.float32)
                    data_all[j] = data_s

                    if isinstance(seg_s, list):
                        if seg_all is None:
                            seg_all = [
                                torch.empty((self.batch_size, *s.shape), dtype=s.dtype)
                                for s in seg_s]
                        for idx, s in enumerate(seg_s):
                            seg_all[idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty(
                                (self.batch_size, *seg_s.shape), dtype=seg_s.dtype)
                        seg_all[j] = seg_s

                    if dist_all is None:
                        dist_all = torch.empty(
                            (self.batch_size, *dist_s.shape), dtype=torch.float32)
                    dist_all[j] = dist_s

        return {'data': data_all, 'target': seg_all,
                'dist_field': dist_all, 'keys': selected_keys}


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class _DistHeadNet(nn.Module):
    """
    Wraps a standard nnUNet backbone and adds a parallel distance-field head.
    train → (seg_out, dist_out)
    eval  → seg_out only  (inference-compatible with standard nnUNet pipeline)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

        decoder   = backbone.decoder
        in_ch     = decoder.encoder.output_channels[0]
        #in_ch是decoder最后一层输出的通道数,也就是_dec_features的通道数
        conv_op   = decoder.encoder.conv_op
        #conv_op是卷积操作的类型是, 2D还是3D卷积
        num_cls   = decoder.num_classes

        self.dist_head = conv_op(in_ch, num_cls, 1, 1, 0, bias=True)
        nn.init.zeros_(self.dist_head.weight)
        nn.init.zeros_(self.dist_head.bias)

        self._dec_features: torch.Tensor | None = None
        decoder.stages[-1].register_forward_hook(self._hook)
#hook的核心是:不改原有的代码,在某个事件触发时自动插入自己的逻辑
#.register_forward_hook()是在forward函数执行到decoder.stages[-1]这个模块时,自动调用self._hook()这个函数
#a.register_forward_hook(self._hook)是PyTorch的hook机制
#意思是等a的forward执行完,自动触发括号里面的函数
    @property
    def decoder(self):
        return self.backbone.decoder

    def _hook(self, module, inp, out):
        self._dec_features = out

    def forward(self, x: torch.Tensor):
        seg_out = self.backbone(x)
        if self.training:
            return seg_out, self.dist_head(self._dec_features)
        return seg_out


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L_boundary = mean(|e|² · |e|) = mean(e² · |e|)  where e = pred − gt."""
    pred = pred.float()
    gt   = gt.to(pred.device, dtype=pred.dtype)
    err  = pred - gt
    return (err.pow(2) * err.abs()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainer_BoundaryLoss(nnUNetTrainer):

    def initialize(self):
        super().initialize()
        self._spacing = tuple(self.configuration_manager.spacing)
#self._spacing的前面的_表示这个是私有变量,其他开发者不要碰
#self.__spacing的前面是双下划线,表示这个变量是类的私有变量,不能被子类访问,只能在这个类里面用
#__表示强制,阻止外部访问
    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        backbone = get_network_from_plans(
            configuration_manager.network_arch_class_name,#网络类名,如"PlainConvUNet"
            configuration_manager.network_arch_init_kwargs,#网络构造参数,如层数,通道数
            configuration_manager.network_arch_init_kwargs_req_import,#构造参数需要额外import 的类别
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return _DistHeadNet(backbone)

    def get_dataloaders(self):
        _BoundaryDataLoader._dist_params = {
            'num_classes': self.label_manager.num_segmentation_heads,
            'spacing':     self._spacing,
        }
        _orig = _trainer_module.nnUNetDataLoader
        _trainer_module.nnUNetDataLoader = _BoundaryDataLoader
        try:
            loaders = super().get_dataloaders()
        finally:
            _trainer_module.nnUNetDataLoader = _orig
        return loaders

    def train_step(self, batch: dict) -> dict:
        data   = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        gt_dist = batch['dist_field']

        self.optimizer.zero_grad(set_to_none=True)  # type: ignore
        ctx = (autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
        with ctx:
            seg_out, dist_out = self.network(data)          # type: ignore
            l_seg = self.loss(seg_out, target)              # type: ignore
            l_ba  = _boundary_loss(dist_out, gt_dist)
            # 前 50 epoch 线性 warm-up: 0 → lambda_bd_max，之后固定
            lambda_bd = min(self.current_epoch / 50, 1.0) * 0.1
            l     = l_seg + lambda_bd * l_ba

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()#是把loss放大一个倍数,再反向传播,float16的数值范围小,梯度很小时候会变成0,放大后再算梯度可以避免这个问题

            self.grad_scaler.unscale_(self.optimizer) #反向传播完成后,把梯度除回原本的大小,还原成真实梯度值      # type: ignore
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12) #把所有的参数的梯度范说限制在12以内 # type: ignore
            self.grad_scaler.step(self.optimizer)           # type: ignore
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # type: ignore
            self.optimizer.step()                           # type: ignore

        return {'loss': l.detach().cpu().numpy(),
                'l_seg': l_seg.detach().cpu().numpy(),
                'l_ba': (lambda_bd * l_ba).detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        #这个是验证集上的单个batch推理,不更新参数,只计算指标
        return super().validation_step(batch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)
        self._run_report()

    def _run_report(self):
        try:
            from pathlib import Path
            from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
            from pumengyu.tools.analyasis.auto_report import run_auto_report

            dataset_name = self.plans_manager.dataset_name
            fold_dir = Path(self.output_folder)
            gt_dir   = Path(nnUNet_preprocessed) / dataset_name / "gt_segmentations"
            img_dir  = Path(nnUNet_raw)           / dataset_name / "imagesTr"

            self.print_to_log_file(f"[report] 生成报告: {fold_dir.name}")
            run_auto_report(fold_dir, gt_dir, img_dir)
            self.print_to_log_file("[report] 报告生成完成")
        except Exception as e:
            self.print_to_log_file(f"[report] 失败: {e}")
