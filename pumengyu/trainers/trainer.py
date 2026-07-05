import pydoc
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from pumengyu.mixins import (
    CopyPasteMixin, DifficultyCopyPasteMixin,
    SmallTumorOversampleMixin,
    SizeStratifiedOversampleMixin,
    UnifiedFocalLossMixin, TverskyLossMixin, AutoReportMixin, AutoInternalTestMixin,
    NoTumorFPPenaltyMixin, TopKNoTumorFPPenaltyMixin,
    ExternalNoTumorMixin,
    TumorOnlyTrainMixin,
    NoMirrorMixin,
    Stage2FPSupMixin,
)
from pumengyu.architectures.umamba import UMambaBot3D
from pumengyu.architectures.mla_unetr import MLAUNetBot3D, MLAUNetDWBot3D, MLAUNetIBBot3D, IBConvUNet3D, DWSepUNet3D, MLAUNetDWSepResBot3D
from pumengyu.architectures.mednext import build_mednext_large, build_mednext_large_mla
from pumengyu.architectures.deep_plain_res_gn import (
    build_deep_plain_res_gn,
    build_deep_res_gn_mla,
    build_deep_dwib_res_gn,
    build_deep_dwib_med_config,
)
from pumengyu.architectures.swinunetr import build_swinunetr
from pumengyu.architectures.nnformer import build_nnformer


class nnUNetTrainer_Baseline(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """原版 nnUNetTrainer + 验证报告 + 训练结束自动跑内部测试集（26 cases）。"""
    pass


class nnUNetTrainer_UFL(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """
    仅叠加 AsymmetricUnifiedFocalLoss，不做 CopyPaste / 过采样。

    用途：单独评估 UFL 对 Dataset003_Liver 的贡献，
    与 nnUNetTrainer_CopyPaste（无UFL）形成 2×2 消融对照。

    结果目录：nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_UFL_v2(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """结果目录：nnUNetTrainer_UFL_v2__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_UFL_delta06(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """
    UFL_DELTA=0.6，偏向惩罚漏检（FN），对应原 nnUNetTrainer_UFL 训练时的实际超参。
    用于与 UFL_v2（delta=0.5）做显式对照消融。
    结果目录：nnUNetTrainer_UFL_delta06__nnUNetPlans__3d_fullres/
    """
    UFL_DELTA = 0.6


# ------------------------------------------------------------------ #
# 以下为已放弃方向，仅保留以维持历史结果目录可复现                    #
# ------------------------------------------------------------------ #

class nnUNetTrainer_CopyPaste(CopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """已放弃。结果目录：nnUNetTrainer_CopyPaste__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_CopyPaste_v2(CopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """已放弃。结果目录：nnUNetTrainer_CopyPaste_v2__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_CopyPaste_Diff(
    DifficultyCopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """已放弃。结果目录：nnUNetTrainer_CopyPaste_Diff__nnUNetPlans__3d_fullres/"""


# ------------------------------------------------------------------ #
# 当前主线                                                            #
# ------------------------------------------------------------------ #

class nnUNetTrainer_Ext25(AutoInternalTestMixin, ExternalNoTumorMixin, AutoReportMixin, nnUNetTrainer):
    """
    Baseline + 外部无肿瘤 25 case（IRCADb×5 + CHAOS CT×20），其余与 nnUNetTrainer 完全一致。

    消融目标：隔离外部数据的独立贡献，与 SizeOversampleV2_Ext25 形成 2×2 对照：
        Baseline            vs  Ext25               → 外部数据的独立效果
        SizeOversampleV2    vs  SizeOversampleV2_Ext25 → 外部数据在 V2 上的增量

    结果目录：nnUNetTrainer_Ext25__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_SizeOversample(AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    大小分层重复过采样（无 CopyPaste）。

    边界从当前 fold 训练集的肿瘤大小分布中自动计算（百分位数），无硬编码阈值：
        极小（≤ P20） → 3× 重复
        小（P20~P45） → 2× 重复
        中/大（中间段）→ 不额外重复
        极大（> P90） → 3× 重复
        无肿瘤 case   → 3× 重复（抑制误报）

    消融目标：单独评估"纯重复过采样（不粘贴）"对极小/极大肿瘤和无肿瘤误报的贡献。

    结果目录：nnUNetTrainer_SizeOversample__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_SizeOversample_UFL(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer
):
    """
    大小分层重复过采样 + UFL（delta=0.6）。

    SizeOversample 负责数据层面的频率均衡，UFL 负责 loss 层面对漏检的惩罚。

    结果目录：nnUNetTrainer_SizeOversample_UFL__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_SizeOversampleV2(AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    大小分层重复过采样 V2（加强版倍数）。

    对比 V1（极小×3, 小×2, 无肿瘤×3），V2 大幅提升小肿瘤和无肿瘤曝光频率：
        极小（≤P20）  → 6× 重复
        小（P20~P45） → 5× 重复
        中/大（中间段）→ 1×（不变）
        极大（>P90）  → 3×（不变）
        无肿瘤 case   → 6× 重复

    消融目标：单独评估过采样倍数提升对小肿瘤召回和无肿瘤误报的影响（不加 loss 惩罚）。
    结果目录：nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV3(AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    大小分层重复过采样 V3（全尺寸均衡版）。

    在 V2 基础上新增中等肿瘤过采样，同时进一步提升极小/小的曝光频率；
    极大和无肿瘤保持与 V2 一致，不引入额外变量：
        极小（≤P20）  → 8× 重复
        小（P20~P45） → 6× 重复
        中（P45~P90） → 3× 重复（V2 为 1×，新增）
        极大（>P90）  → 3×（与 V2 相同）
        无肿瘤 case   → 6×（与 V2 相同）

    消融目标：验证"全尺寸均衡过采样"能否在保持 V2 Precision 增益的同时恢复小肿瘤召回率。
    结果目录：nnUNetTrainer_SizeOversampleV3__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 8
    SSO_SMALL_REPEAT:    int = 6
    SSO_MID_REPEAT:      int = 3
    SSO_HUGE_REPEAT:     int = 3
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV3_NoMirror(
    AutoInternalTestMixin, NoMirrorMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    SizeOversampleV3 + 关闭镜像增强。

    消融目标：验证"全尺寸均衡过采样 + 禁用镜像"叠加是否进一步降低无肿瘤误报率。
    SizeOversampleV3 过采样倍数完全一致，唯一变量是 NoMirrorMixin。

    结果目录：nnUNetTrainer_SizeOversampleV3_NoMirror__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 8
    SSO_SMALL_REPEAT:    int = 6
    SSO_MID_REPEAT:      int = 3
    SSO_HUGE_REPEAT:     int = 3
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV2_NTFP(
    AutoInternalTestMixin, NoTumorFPPenaltyMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    大小分层重复过采样 V2 + 无肿瘤专项误报惩罚（NoTumorFPPenalty）。

    数据层：极小×6, 小×5, 无肿瘤×6（增加困难 case 曝光频率）。
    Loss 层：GT 无肿瘤的 patch 中，额外惩罚预测出肿瘤的概率均值（lambda=1.0）。

    两路协同：oversample 让模型更多见无肿瘤 case，NTFP 直接在 loss 上惩罚误报；
    有肿瘤 sample 的 loss 路径不受影响，不牺牲小肿瘤召回。

    结果目录：nnUNetTrainer_SizeOversampleV2_NTFP__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_FPSafe(AutoInternalTestMixin, TopKNoTumorFPPenaltyMixin, AutoReportMixin, nnUNetTrainer):
    """
    FP-Safe baseline：原版 nnUNet + 无肿瘤 Top-K 误报惩罚。

    目标：验证 loss 层面的 false-positive-aware 约束能否降低无肿瘤 case 误报，
    同时尽量不牺牲有肿瘤 case 的召回。

    结果目录：nnUNetTrainer_FPSafe__nnUNetPlans__3d_fullres/
    """
    TKN_TUMOR_FP_LAMBDA: float = 1.0
    TKN_TOPK_PERCENT: float = 0.01


class nnUNetTrainer_SizeOV4_FPSafe(
    AutoInternalTestMixin, TopKNoTumorFPPenaltyMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    SizeOV4 + FP-Safe：均匀全量 2× 过采样 + 无肿瘤 Top-K 误报惩罚。

    数据层：所有 size/no-tumor 分层均 2×，对齐当前 SizeOV4 系列。
    Loss 层：只在 GT 无肿瘤 sample 上惩罚最高置信 tumor 概率。

    结果目录：nnUNetTrainer_SizeOV4_FPSafe__nnUNetPlans__3d_fullres/
    """
    TKN_TUMOR_FP_LAMBDA: float = 1.0
    TKN_TOPK_PERCENT: float = 0.01
    SSO_TINY_REPEAT:     int = 2
    SSO_SMALL_REPEAT:    int = 2
    SSO_MID_REPEAT:      int = 2
    SSO_HUGE_REPEAT:     int = 2
    SSO_NO_TUMOR_REPEAT: int = 2


class nnUNetTrainer_SizeOversampleV2_Ext25(
    AutoInternalTestMixin, ExternalNoTumorMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    SizeOversampleV2 + 外部无肿瘤 25 case（IRCADb×5 + CHAOS CT×20）。

    ExternalNoTumorMixin 在运行时向 tr_keys 追加外部 case，
    splits_final.json 始终保持干净，不需要手动修改文件。

    对照实验：与 SizeOversampleV2 完全相同的过采样倍数，
    唯一变量是训练集多了 25 个外部无肿瘤 case。

    结果目录：nnUNetTrainer_SizeOversampleV2_Ext25__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV2_NTFP_Ext25(
    AutoInternalTestMixin, ExternalNoTumorMixin, NoTumorFPPenaltyMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    SizeOversampleV2_NTFP + 外部无肿瘤 25 case（IRCADb×5 + CHAOS CT×20）。

    结果目录：nnUNetTrainer_SizeOversampleV2_NTFP_Ext25__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV2_Tversky(
    AutoInternalTestMixin, TverskyLossMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    大小分层重复过采样 V2 + AsymmetricFocalTverskyLoss（delta=0.7, gamma=0.75）。

    数据层：极小×6, 小×5, 无肿瘤×6（与 SizeOversampleV2 完全一致）。
    Loss 层：在 CE+Dice 基础上叠加 AsymmetricFocalTverskyLoss（compound-loss 库），
             FN 权重 0.7，FP 权重 0.3，focal 调制前景 easy 样本，专项提升小肿瘤召回率。

    消融目标：隔离"换损失函数"对召回的贡献，与 SizeOversampleV2 形成单一变量对照。

    结果目录：nnUNetTrainer_SizeOversampleV2_Tversky__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


# ------------------------------------------------------------------ #
# 数据增强消融                                                         #
# ------------------------------------------------------------------ #

class nnUNetTrainer_NoMirror(AutoInternalTestMixin, NoMirrorMixin, AutoReportMixin, nnUNetTrainer):
    """
    关闭所有镜像增强的消融实验。

    动机：肝脏是右侧不对称器官，nnUNet 默认的左右/前后/上下镜像
    会产生"肝脏在左侧"等解剖上不存在的假图像，可能引入训练噪音。

    消融目标：与 Baseline 单一变量对照，观察去除镜像是否
    改善无肿瘤 case 误报率和小肿瘤 Dice。

    结果目录：nnUNetTrainer_NoMirror__nnUNetPlans__3d_fullres/
    """


# ------------------------------------------------------------------ #
# 两阶段 FP 抑制                                                      #
# ------------------------------------------------------------------ #

class Tr_Stage2_FPSup(AutoInternalTestMixin, Stage2FPSupMixin, AutoReportMixin, nnUNetTrainer):
    """
    两阶段 FP 抑制 — Stage2 v1（基线版，Loss=Dice+CE，无专项 FP 惩罚）。
    结果与 Baseline 持平，网络未能利用 Stage1 概率图通道。
    对照实验保留，不再训练。

    输入：3 通道（CT + Stage1 肿瘤概率图 + Stage1 二值图）
    Loss：Dice+CE（与 nnUNetTrainer 一致）
    采样：无肿瘤 case 过采样 3×

    前置条件：Stage1 已对全部 imagesTr 推理并保存概率图（--save_probabilities），
    文件位于 S2_STAGE1_SOFTMAX_DIR（首次训练时自动 resample 并缓存到 s2feat/）。

    训练命令：
        RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 \\
        nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup

    结果目录：Tr_Stage2_FPSup__nnUNetPlans__3d_fullres/
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400               # Stage1热启动，400 epoch ≈ 21h（利用长时间窗口）
        self.num_iterations_per_epoch = 150


class Tr_Stage1_TumorOnly(AutoInternalTestMixin, TumorOnlyTrainMixin, AutoReportMixin, nnUNetTrainer):
    """
    两阶段 FP 抑制 — Stage1。

    训练集只保留有肿瘤 case（118/131），无肿瘤 case 从训练集剔除。
    目的：故意让模型从未见过"应全黑"的输入，推理时对无肿瘤 case 大量误报，
    产生 FP 概率图（--save_probabilities）供 Stage2 作为第 2/3 输入通道。

    推理命令（全 131 case，保存概率图）：
      CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \\
        -i .../imagesTr \\
        -o .../Tr_Stage1_TumorOnly.../fold_0/stage1_softmax \\
        -d Dataset003_Liver -c 3d_fullres -tr Tr_Stage1_TumorOnly -f 0 \\
        --save_probabilities

    结果目录：Tr_Stage1_TumorOnly__nnUNetPlans__3d_fullres/
    """


class Tr_Stage2_FPSup_v2(AutoInternalTestMixin, NoTumorFPPenaltyMixin, Stage2FPSupMixin, AutoReportMixin, nnUNetTrainer):
    """
    两阶段 FP 抑制 — Stage2 v2（修复版）。

    v1 失败原因：Dice+CE 对 Stage1 附加通道无梯度信号，网络退化为 Stage1 行为。

    修复方案：加入 NoTumorFPPenaltyMixin，
    对 GT 无肿瘤 patch 额外惩罚 mean(p_tumor)，
    提供明确梯度信号让网络学会利用 Stage1 概率图通道压制 FP。

    输入：3 通道（CT + Stage1 概率图 + Stage1 二值图）
    Loss：Dice+CE + λ·mean(p_tumor)  [仅 GT 无肿瘤 patch]
    采样：无肿瘤 case 过采样 3×

    训练命令：
        RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 \\
        nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup_v2

    结果目录：Tr_Stage2_FPSup_v2__nnUNetPlans__3d_fullres/
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400
        self.num_iterations_per_epoch = 150


# ------------------------------------------------------------------ #
# U-Mamba 系列                                                        #
# ------------------------------------------------------------------ #

def _build_umamba_bot(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    **mamba_kwargs,
) -> UMambaBot3D:
    """
    从 nnUNet plans 读取网络结构参数，构建 UMambaBot3D。
    与 get_network_from_plans 完全等价，仅把网络类换成 UMambaBot3D。
    """
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    return UMambaBot3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **arch_kwargs,
        **mamba_kwargs,
    )


class nnUNetTrainer_UMamba(AutoReportMixin, nnUNetTrainer):
    """
    U-Mamba Bot：在 nnUNet 标准 PlainConvUNet 的瓶颈层插入 Mamba 块。

    其余（encoder、decoder、loss、data augmentation）与 nnUNetTrainer 完全一致，
    唯一变量 = 瓶颈处的 Mamba 全局上下文建模。

    消融目标：评估 Mamba 瓶颈对小肿瘤长程依赖建模的贡献（对照 Baseline 和 SizeOversample）。

    依赖（需先安装）：
        pip install mamba-ssm causal-conv1d

    结果目录：nnUNetTrainer_UMamba__nnUNetPlans__3d_fullres/
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_umamba_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
        )


class nnUNetTrainer_UMamba_SizeOversample(SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    U-Mamba Bot + 大小分层重复过采样。

    组合：Mamba 瓶颈（全局上下文）+ 数据层过采样（极小/极大/无肿瘤频率均衡）。

    结果目录：nnUNetTrainer_UMamba_SizeOversample__nnUNetPlans__3d_fullres/
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_umamba_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
        )


# ------------------------------------------------------------------ #
# MLA-UNet 系列                                                       #
# ------------------------------------------------------------------ #

def _build_mla_unet_bot(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    **mla_kwargs,
) -> MLAUNetBot3D:
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    return MLAUNetBot3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **arch_kwargs,
        **mla_kwargs,
    )


class nnUNetTrainer_MLAUNet(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    MLA-UNet Bot：在 nnUNet 标准 PlainConvUNet 的瓶颈层插入 Multi-head Latent Attention 块。

    动机：DeepSeek-V2 的 MLA 通过低秩 KV 压缩（d_c = d_model // 4）在保持 full attention
    全局感受野的同时将 KV 显存从 O(N·d) 降至 O(N·d_c)，允许在 3D 医学图像分割中
    使用更小 patch（8³ 代替 16³）做细粒度分割，或用更大 batch size。

    其余（encoder、decoder、loss、data augmentation）与 nnUNetTrainer 完全一致，
    唯一变量 = 瓶颈处的 MLA 全局上下文建模。

    超参（如需调整请子类化并覆盖 MLA_* 类变量）：
        MLA_NUM_HEADS         = 8
        MLA_NUM_BLOCKS        = 2
        MLA_COMPRESSION_RATIO = 4   （d_c = d_bot // 4）
        MLA_MLP_RATIO         = 4

    结果目录：nnUNetTrainer_MLAUNet__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    MLA_USE_MOE:           bool = False  # 旧 checkpoint 用简单 MLP，与 mla_bot.blocks.*.mlp.* 命名对齐

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
            mla_use_moe=cls.MLA_USE_MOE,
        )


class nnUNetTrainer_MLAUNet_1500(nnUNetTrainer_MLAUNet):
    """MLAUNet 从头训练 1500 epoch，与 MLAUNet@1000 做收敛对比。"""
    def run_training(self):
        self.num_epochs = 1500
        return super().run_training()


class nnUNetTrainer_MLAUNet_MoE(nnUNetTrainer_MLAUNet):
    """MLA-UNet Bot + MoE-FFN（DeepSeek V3 风格）。

    与 nnUNetTrainer_MLAUNet 唯一区别：FFN 替换为 MoE-FFN
        - 1 shared expert + 4 路由专家，top_k=2
        - loss-free bias 均衡（expert_bias 非 Parameter，不进优化器）
        - 激活容量约等于原 FFN（3 个半宽专家）

    结果目录：nnUNetTrainer_MLAUNet_MoE__nnUNetPlans__3d_fullres/
    """
    MLA_USE_MOE: bool = True


class nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（含 MoE-FFN）+ 大小分层重复过采样 V2 倍数。

    对照实验：验证 SizeOversampleV2(+4.4pp) 与 MLAUNet_MoE(+4.1pp) 的增益能否叠加。
    两个提升机制正交：数据层（过采样频率均衡）vs 模型层（全局 attention + MoE）。

    倍数与 SizeOversampleV2 完全一致（极小×6，小×5，无肿瘤×6），
    网络架构与 nnUNetTrainer_MLAUNet_MoE 完全一致。

    结果目录：nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 6
    SSO_SMALL_REPEAT:      int = 5
    SSO_NO_TUMOR_REPEAT:   int = 6

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLAUNet_MoE_SizeOversampleV3(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（含 MoE-FFN）+ 大小分层重复过采样 V3 倍数。

    倍数与 SizeOversampleV3 完全一致（极小×8，小×6，中×3，极大×3，无肿瘤×6），
    网络架构与 nnUNetTrainer_MLAUNet_MoE 完全一致。

    结果目录：nnUNetTrainer_MLAUNet_MoE_SizeOversampleV3__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 8
    SSO_SMALL_REPEAT:      int = 6
    SSO_MID_REPEAT:        int = 3
    SSO_HUGE_REPEAT:       int = 3
    SSO_NO_TUMOR_REPEAT:   int = 6

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（含 MoE-FFN）+ 均匀全量 2× 过采样。

    假设：MoE 本身已具备对不同大小肿瘤的内部专门化路由，
    无需按大小差异化过采样；统一将全部 case 重复 2× 保持自然分布，
    让每个 expert 在更多样本上充分训练，避免 V2/V3 的分布扭曲。

        全部 case（极小/小/中/大/无肿瘤）→ 2× 重复

    对照：nnUNetTrainer_MLAUNet_MoE（无过采样）vs 本实验（均匀 2×）
    结果目录：nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（含 MoE-FFN）+ 均匀全量 3× 过采样（V5）。

    与 V4 相同策略——全部 case 均匀过采样，维持自然分布；
    但倍数从 2× 提升到 3×，让模型在更多样本上充分训练。
    基于 V4 的综合优异表现（Overall 0.8330, Tumor Dice 0.7146）的进一步探索，
    验证"更多数据 + MoE"的边际收益。

        全部 case（极小/小/中/大/无肿瘤）→ 3× 重复

    对照：nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4（均匀 2×）vs 本实验（均匀 3×）
    结果目录：nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 3
    SSO_SMALL_REPEAT:      int = 3
    SSO_MID_REPEAT:        int = 3
    SSO_HUGE_REPEAT:       int = 3
    SSO_NO_TUMOR_REPEAT:   int = 3

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLAUNet_MoE_SizeOversampleV6(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（含 MoE-FFN）+ 均匀全量 4× 过采样（V6）。

    延续 V4（2×）→ V5（3×）→ V6（4×）的过采样梯度实验，
    验证 4× 过采样是否能进一步提升外部泛化能力。
    外部验证：V5（3×）外部 Overall 0.8025 > V4（2×）0.7871，
    探索曲线拐点：3× 已经是最优还是 4× 更好？

        全部 case（极小/小/中/大/无肿瘤）→ 4× 重复

    对照：MoE_SizeOV4（2×）、MoE_SizeOV5（3×）vs 本实验（4×）
    结果目录：nnUNetTrainer_MLAUNet_MoE_SizeOversampleV6__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 4
    SSO_SMALL_REPEAT:      int = 4
    SSO_MID_REPEAT:        int = 4
    SSO_HUGE_REPEAT:       int = 4
    SSO_NO_TUMOR_REPEAT:   int = 4

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


# ------------------------------------------------------------------ #
# MLA-UNet DW（Encoder 换 Depthwise Separable Conv）                  #
# ------------------------------------------------------------------ #

def _build_mla_unet_dw_bot(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    dw_kernel_size: int = 5,
    **mla_kwargs,
) -> MLAUNetDWBot3D:
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    return MLAUNetDWBot3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        dw_kernel_size=dw_kernel_size,
        **arch_kwargs,
        **mla_kwargs,
    )


class nnUNetTrainer_MLAUNet_MoE_DW7_SizeOversampleV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（MoE-FFN）+ 全网络 5×5×5 DW Sep Conv + 均匀全量 2× 过采样（内部最优配置）。

    名字保留 DW7 标记历史，实际 kernel=5（7×7×7 在 3D 上 cuDNN 无优化实现，慢 10×）。

    基于 nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4（内部 Overall 0.8330，最优）：
        Encoder + Decoder 所有 3×3×3 Conv3d → 5×5×5 DWSepConv3d
        k=5 DWSep 参数量 ≈ k=3 标准卷积的 14%（C=128），感受野 5³=125 体素
        MLA Bottleneck 保持标准卷积不变

    过采样：全部 case 均匀 2× 重复（与 SizeOV4 一致）。

    结果目录：nnUNetTrainer_MLAUNet_MoE_DW7_SizeOversampleV4__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    DW_KERNEL_SIZE:        int = 5
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_dw_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            dw_kernel_size=cls.DW_KERNEL_SIZE,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLAUNet_MoE_DW7_SizeOversampleV5(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（MoE-FFN）+ 全网络 5×5×5 DW Sep Conv + 均匀全量 3× 过采样（外部最优配置）。

    名字保留 DW7 标记历史，实际 kernel=5（7×7×7 在 3D 上 cuDNN 无优化实现，慢 10×）。

    基于 nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5（外部 IRCADb Overall 0.8025，最优）：
        Encoder + Decoder 所有 3×3×3 Conv3d → 5×5×5 DWSepConv3d
        MLA Bottleneck 保持标准卷积不变

    过采样：全部 case 均匀 3× 重复（与 SizeOV5 一致）。

    结果目录：nnUNetTrainer_MLAUNet_MoE_DW7_SizeOversampleV5__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    DW_KERNEL_SIZE:        int = 5
    SSO_TINY_REPEAT:       int = 3
    SSO_SMALL_REPEAT:      int = 3
    SSO_MID_REPEAT:        int = 3
    SSO_HUGE_REPEAT:       int = 3
    SSO_NO_TUMOR_REPEAT:   int = 3

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_dw_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            dw_kernel_size=cls.DW_KERNEL_SIZE,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


def _build_mla_unet_ib_bot(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    ib_kernel_size: int = 7,
    exp_r: int = 4,
    channels_per_group: int = 1,
    use_checkpoint: bool = False,
    **mla_kwargs,
) -> MLAUNetIBBot3D:
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    return MLAUNetIBBot3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        ib_kernel_size=ib_kernel_size,
        exp_r=exp_r,
        channels_per_group=channels_per_group,
        use_checkpoint=use_checkpoint,
        **arch_kwargs,
        **mla_kwargs,
    )


def _build_mla_unet_dwsep_res_bot(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    enc_conv_per_stage: int = 3,
    dw_kernel_size: int = 3,
    **mla_kwargs,
) -> MLAUNetDWSepResBot3D:
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

    # 增加 encoder 深度：n_conv_per_stage 统一设为 enc_conv_per_stage
    orig = arch_kwargs.get('n_conv_per_stage')
    if orig is not None:
        arch_kwargs['n_conv_per_stage'] = [enc_conv_per_stage] * len(orig)

    return MLAUNetDWSepResBot3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        dw_kernel_size=dw_kernel_size,
        **arch_kwargs,
        **mla_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DWSepRes4 + MoE + SizeOV4
# ──────────────────────────────────────────────────────────────────────────────

class nnUNetTrainer_DWSepRes4_MoE_SizeOV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    DWSep(k=3)+残差 encoder + MLA-MoE 瓶颈 + plain conv decoder + 均匀 2× 过采样。

    架构改动（对比 MLAUNet_MoE_SizeOV4）：
      - Encoder stride=1 块 → DWSepResBlock3D
            DW(k=3, groups=C) → InstanceNorm3d → GELU → PW(1×1) → + residual
      - n_conv_per_stage: 2 → 4（每 encoder stage 多 2 层，以深度补偿 k=3 小感受野）
        DWSepRes 每块参数极少，n=4 总参 23.84M，比原 baseline(30.75M) 少 23%
      - Decoder 保持 plain conv 不变（encoder/decoder 不对称，集中容量在 encoder）
      - MLA Bottleneck（MoE-FFN）不变

    过采样策略（与 MLAUNet_MoE_SizeOV4 相同）：
      全部 case 均匀 2× 重复，维持自然分布。

    消融意义：
      验证"DWSep 参数效率 + 残差 + encoder 加深(4层)"能否超过
      纯 IBConvBlock3D(k=7) 方案（IB7_V4 Overall=0.8192）及 MoE_SizeOV4(0.8330)。
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    ENC_CONV_PER_STAGE:    int = 4
    DW_KERNEL_SIZE:        int = 3
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_dwsep_res_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            enc_conv_per_stage=cls.ENC_CONV_PER_STAGE,
            dw_kernel_size=cls.DW_KERNEL_SIZE,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


def _build_ib_unet(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    ib_kernel_size: int = 7,
    exp_r: int = 4,
    channels_per_group: int = 1,
    use_checkpoint: bool = False,
) -> IBConvUNet3D:
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    return IBConvUNet3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        ib_kernel_size=ib_kernel_size,
        exp_r=exp_r,
        channels_per_group=channels_per_group,
        use_checkpoint=use_checkpoint,
        **arch_kwargs,
    )


def _build_dwsep_unet(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    dw_kernel_size: int = 7,
) -> DWSepUNet3D:
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    return DWSepUNet3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        dw_kernel_size=dw_kernel_size,
        **arch_kwargs,
    )


class nnUNetTrainer_DWSep7(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    PlainConvUNet + DWSepConv3d(k=7)，无 MLA，无 MoE，无 SizeOversample。

    消融目标：单独验证"大核感受野"的贡献，不引入倒置瓶颈结构。
    替换所有 nn.Conv3d（含 stride=2）为 DW(7×7×7)+PW(1×1×1)，1×1×1 跳过。

    与 IBConv7 的区别：
      DWSep7  = DW(k) → PW           （无 Norm/激活/残差，极简）
      IBConv7 = DW(k) → Norm → PW_expand → GELU → PW_compress → +residual

    结果目录：nnUNetTrainer_DWSep7__nnUNetPlans__3d_fullres/
    """

    DW_KERNEL_SIZE: int = 7

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_dwsep_unet(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            dw_kernel_size=cls.DW_KERNEL_SIZE,
        )


class nnUNetTrainer_IBConv7(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    PlainConvUNet + IBConvBlock3D(k=7, exp_r=4, depthwise)，无 MLA，无 MoE，无 SizeOversample。

    消融目标：单独验证大核倒置瓶颈卷积（InvertedBottleneck k=7）对肝脏肿瘤分割的贡献，
    排除 MLA 全局注意力和 MoE 路由的干扰。

    架构：PlainConvUNet 全网 stride=1 卷积块 → IBConvBlock3D
        DW(7×7×7, groups=C) → InstanceNorm3d → PW_expand(×4) → GELU → PW_compress → + residual
    stride=2 下采样保持标准 3×3×3 不变。

    参数量估计：与 MedNeXt-L 相近（DW 大核 + 倒置瓶颈结构相同）。
    训练：nnUNet 默认 1000 epoch，无额外修改。
    结果目录：nnUNetTrainer_IBConv7__nnUNetPlans__3d_fullres/
    """

    IB_KERNEL_SIZE:        int = 7
    IB_EXP_R:              int = 4
    IB_CHANNELS_PER_GROUP: int = 1   # 纯 depthwise，与 MedNeXt 一致
    IB_USE_CHECKPOINT:     bool = False

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_ib_unet(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            ib_kernel_size=cls.IB_KERNEL_SIZE,
            exp_r=cls.IB_EXP_R,
            channels_per_group=cls.IB_CHANNELS_PER_GROUP,
            use_checkpoint=cls.IB_USE_CHECKPOINT,
        )


class nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（MoE-FFN）+ 倒置瓶颈 7×7×7 大核 + 均匀 2× 过采样（内部最优）。

    架构（MLAUNetIBBot3D）：
        stride=1 ConvDropoutNormReLU → IBConvBlock3D(k=7, exp_r=4)
            DW(7, stride=1, groups=C) → InstanceNorm → PW_expand → GELU → PW_compress → +residual
        stride=2 下采样卷积保持标准 3×3×3（避免大核步长 cuDNN 无优化问题）
        MLA Bottleneck 不变

    与 DW7（DWSepConv3d）的区别：大核不碰 stride=2；有 expand/compress 非线性；有 residual 连接。

    过采样（SizeOV4）：全 case 均匀 2×，对齐内部最优 nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4。
    结果目录：nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    IB_KERNEL_SIZE:        int = 7
    IB_EXP_R:              int = 4
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_ib_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            ib_kernel_size=cls.IB_KERNEL_SIZE,
            exp_r=cls.IB_EXP_R,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV5(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA-UNet Bot（MoE-FFN）+ 倒置瓶颈 7×7×7 大核 + 均匀 3× 过采样（外部最优）。

    架构同 IB7_V4（MLAUNetIBBot3D, k=7）。
    过采样改为 3×，对齐外部 IRCADb 最优 nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5（Overall 0.8025）。
    结果目录：nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV5__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    IB_KERNEL_SIZE:        int = 7
    IB_EXP_R:              int = 4
    SSO_TINY_REPEAT:       int = 3
    SSO_SMALL_REPEAT:      int = 3
    SSO_MID_REPEAT:        int = 3
    SSO_HUGE_REPEAT:       int = 3
    SSO_NO_TUMOR_REPEAT:   int = 3

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_ib_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            ib_kernel_size=cls.IB_KERNEL_SIZE,
            exp_r=cls.IB_EXP_R,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLA_GK7_V4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA = MLAUNet MoE 基础架构
    GK7 = Grouped Conv k=7（channels_per_group=16，M=16，Tensor Core 可用）
    V4  = SizeOversampleV4（均匀 2×）

    动机：IB7 纯 depthwise M=1 → Tensor Core 闲置（84% 时间在 CUDA Core）。
    改用 groups=C/16（每组 16 通道），M=16 满足 Tensor Core 最低要求，k=7 感受野不变。
    结果目录：nnUNetTrainer_MLA_GK7_V4__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    IB_KERNEL_SIZE:        int = 7
    IB_EXP_R:              int = 4
    IB_CHANNELS_PER_GROUP: int = 16
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_ib_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            ib_kernel_size=cls.IB_KERNEL_SIZE,
            exp_r=cls.IB_EXP_R,
            channels_per_group=cls.IB_CHANNELS_PER_GROUP,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLA_GK5_V4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MLA = MLAUNet MoE 基础架构
    GK5 = Grouped Conv k=5（channels_per_group=16，M=16，Tensor Core 可用）
    V4  = SizeOversampleV4（均匀 2×）
    + Gradient Checkpointing（激活不保留，反向时重算，显存 ~50% 减少）

    vs GK7：k=5³=125 < k=7³=343，DW 计算量 2.7×减少，感受野从 7→5。
    结果目录：nnUNetTrainer_MLA_GK5_V4__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    IB_KERNEL_SIZE:        int = 5
    IB_EXP_R:              int = 4
    IB_CHANNELS_PER_GROUP: int = 16
    IB_USE_CHECKPOINT:     bool = True
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_ib_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            ib_kernel_size=cls.IB_KERNEL_SIZE,
            exp_r=cls.IB_EXP_R,
            channels_per_group=cls.IB_CHANNELS_PER_GROUP,
            use_checkpoint=cls.IB_USE_CHECKPOINT,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_MLAUNet_deeper(nnUNetTrainer_MLAUNet):
    """MLA-UNet Bot，瓶颈加深到 4 层 MLA block（默认 2 层）。"""
    MLA_NUM_BLOCKS = 4


class nnUNetTrainer_MLAUNet_SizeOversample(AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    MLA-UNet Bot（含 MoE-FFN）+ 大小分层重复过采样（V2 倍数）。

    组合：MLA 瓶颈（低秩 KV 全局 attention + MoE-FFN）+ 数据层过采样（小肿瘤/无肿瘤频率均衡）。
    验证 SizeOversampleV2(+4.4pp) 和 MLAUNet_MoE(+4.1pp) 的增益能否叠加。

    注：commit 4aec13e 后 MLATransformerBlock 已将 MoE 硬编码，此 trainer 自动使用 MoE。

    结果目录：nnUNetTrainer_MLAUNet_SizeOversample__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 6
    SSO_SMALL_REPEAT:      int = 5
    SSO_NO_TUMOR_REPEAT:   int = 6

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_mla_unet_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


# ------------------------------------------------------------------ #
# MedNeXt-L                                                          #
# ------------------------------------------------------------------ #

class nnUNetTrainer_MedNeXt(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    MedNeXt-L（Large）对比实验。

    架构：ConvNeXt 风格的 3D UNet，固定配置 MedNeXt-L：
        n_channels=32, kernel_size=3, exp_r=[3,4,8,8,8,8,8,4,3]
        block_counts=[3,4,8,8,8,8,8,4,3], do_res=True, do_res_up_down=True

    训练策略：与 nnUNetTrainer_Baseline 完全一致（CE+Dice loss, mirror TTA）。
    唯一变量 = 网络架构。

    结果目录：nnUNetTrainer_MedNeXt__nnUNetPlans__3d_fullres/
    """

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_mednext_large(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        # MedNeXt 用 do_ds 而非 decoder.deep_supervision
        from torch._dynamo import OptimizedModule
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod  # 解包到内层真实模型，否则 torch.compile 追踪内层 do_ds 不变
        mod.do_ds = enabled


class nnUNetTrainer_MedNeXt_FPSafe(TopKNoTumorFPPenaltyMixin, nnUNetTrainer_MedNeXt):
    """
    MedNeXt-L + FP-Safe：强 DW+IB backbone 上验证无肿瘤 Top-K 误报惩罚。

    目标：观察 MedNeXt 的同域高分是否可以在降低无肿瘤误报、改善外部泛化的同时保留。

    结果目录：nnUNetTrainer_MedNeXt_FPSafe__nnUNetPlans__3d_fullres/
    """
    TKN_TUMOR_FP_LAMBDA: float = 1.0
    TKN_TOPK_PERCENT: float = 0.01


class nnUNetTrainer_MedNeXt_SizeOV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MedNeXt-L + 均匀全量 2× 过采样（SizeOV4）。

    消融目标：验证 SizeOV4 策略是否在 MedNeXt 骨干上同样有效。
    对照 nnUNetTrainer_MedNeXt（0.8402）单一变量 = 过采样。

    结果目录：nnUNetTrainer_MedNeXt_SizeOV4__nnUNetPlans__3d_fullres/
    """

    SSO_TINY_REPEAT:     int = 2
    SSO_SMALL_REPEAT:    int = 2
    SSO_MID_REPEAT:      int = 2
    SSO_HUGE_REPEAT:     int = 2
    SSO_NO_TUMOR_REPEAT: int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_mednext_large(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        from torch._dynamo import OptimizedModule
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.do_ds = enabled

    def _do_i_compile(self):
        return False

    def _do_i_compile(self):
        return False

    def _do_i_compile(self):
        return False


class nnUNetTrainer_MedNeXt_MLA(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    MedNeXt-L + MLA Bottleneck。

    架构（MedNeXtMLABot）：
        Encoder/Decoder：MedNeXt-L（k=3 IB 卷积 + 残差，block_counts=[3,4,8,8,8,8,8,4,3]）
        Bottleneck：MedNeXt bottleneck（8 IB blocks, 512 ch）→ MLABottleneck3D（2 层 MLA）

    动机：MedNeXt IB 卷积提供强局部特征，MLA 在最低分辨率特征图建立全局依赖，两者互补。
    对照 nnUNetTrainer_MedNeXt（0.8402）单一变量 = MLA bottleneck。

    结果目录：nnUNetTrainer_MedNeXt_MLA__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_mednext_large_mla(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        from torch._dynamo import OptimizedModule
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.do_ds = enabled


class nnUNetTrainer_MedNeXt_MLA_FPSafe(TopKNoTumorFPPenaltyMixin, nnUNetTrainer_MedNeXt_MLA):
    """
    MedNeXt-L + MLA Bottleneck + FP-Safe。

    当前主线候选：在外部验证最强的 MedNeXt_MLA 基础上，加入无肿瘤 Top-K
    tumor probability 惩罚，验证是否能进一步降低外部无肿瘤误报，同时保留
    MedNeXt_MLA 的跨域泛化优势。

    对照：
      - nnUNetTrainer_MedNeXt
      - nnUNetTrainer_MedNeXt_FPSafe
      - nnUNetTrainer_MedNeXt_MLA

    结果目录：nnUNetTrainer_MedNeXt_MLA_FPSafe__nnUNetPlans__3d_fullres/
    """
    TKN_TUMOR_FP_LAMBDA: float = 1.0
    TKN_TOPK_PERCENT: float = 0.01


class nnUNetTrainer_MedNeXt_MLA_SizeOV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    MedNeXt-L + MLA Bottleneck + 均匀全量 2× 过采样（SizeOV4）。

    对照实验链（单一变量逐步叠加）：
        MedNeXt（0.8402）
        → + SizeOV4          → nnUNetTrainer_MedNeXt_SizeOV4
        → + MLA              → nnUNetTrainer_MedNeXt_MLA
        → + MLA + SizeOV4   → 本实验（完整组合）

    结果目录：nnUNetTrainer_MedNeXt_MLA_SizeOV4__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_mednext_large_mla(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        from torch._dynamo import OptimizedModule
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.do_ds = enabled


# ------------------------------------------------------------------ #
# SwinUNETR                                                          #
# ------------------------------------------------------------------ #

class nnUNetTrainer_SwinUNETR(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    SwinUNETR-B 对比实验（MONAI 官方实现）。

    架构：Swin Transformer encoder + CNN decoder，固定配置 SwinUNETR-B：
        feature_size=48, depths=(2,2,2,2), num_heads=(3,6,12,24), window_size=7

    注意：MONAI SwinUNETR 不含 deep supervision，此 trainer 关闭 DS（enable_deep_supervision=False）。
    训练策略与 Baseline 完全一致，唯一变量 = 网络架构。

    结果目录：nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_fullres/
    """

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        patch_size = tuple(configuration_manager.patch_size)
        return build_swinunetr(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            patch_size=patch_size,
        )

    def initialize(self):
        self.enable_deep_supervision = False
        super().initialize()

    def set_deep_supervision_enabled(self, enabled: bool):
        pass  # SwinUNETR 无 DS 机制，no-op

    def _do_i_compile(self):
        return False  # torch.compile + use_checkpoint 已知冲突

    def _get_deep_supervision_scales(self):
        return None


# ------------------------------------------------------------------ #
# nnFormer                                                           #
# ------------------------------------------------------------------ #

class nnUNetTrainer_nnFormer(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    nnFormer 对比实验（mednextv1 包中附带的 nnFormer_tumor 实现）。

    架构：3D Swin Transformer UNet（Encoder + Decoder 均为 Transformer），固定配置：
        embedding_dim=96, depths=[2,2,2,2], num_heads=[3,6,12,24]
        patch_size=[4,4,4], window_size=[4,4,8,4]

    注意：该实现的 DS 已注释掉，始终返回单 tensor，此 trainer 关闭 DS。
    训练策略与 Baseline 完全一致，唯一变量 = 网络架构。

    显存优化：
        - 梯度检查点：对所有 SwinTransformerBlock 启用，节省 ~40% 激活显存
        - bf16 autocast：覆盖基类默认的 fp16，exponent 位数更多，无需 GradScaler

    实测（RTX 4090 D, bs=2, 128³, bf16）：训练峰值 ~3 GB，无显存压力。

    结果目录：nnUNetTrainer_nnFormer__nnUNetPlans__3d_fullres/
    """

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        patch_size = tuple(configuration_manager.patch_size)
        return build_nnformer(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            patch_size=patch_size,
            use_gradient_checkpointing=True,
        )

    def initialize(self):
        self.enable_deep_supervision = False
        super().initialize()
        # bf16 exponent 范围充足，不需要 loss scaling
        self.grad_scaler = None

    def set_deep_supervision_enabled(self, enabled: bool):
        pass  # nnFormer 无 DS 机制，no-op

    def _do_i_compile(self):
        return False  # Transformer 架构与 torch.compile 兼容性差

    def _get_deep_supervision_scales(self):
        return None

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        # bf16 比 fp16 指数位多（8 vs 5），不溢出，不需要 GradScaler
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}


# ------------------------------------------------------------------ #
# DeepPlainResGN：消融实验 —— 深层 Plain Conv + 残差 + GroupNorm     #
# ------------------------------------------------------------------ #

class nnUNetTrainer_DeepPlainResGN(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    消融实验：证明 MedNeXt 的性能提升来自"更深 + 残差 + GroupNorm"，
    而非 depthwise separable conv（倒置瓶颈）。

    架构（DeepPlainResGN）：
        plain 3×3 Conv3d + BasicBlockD 残差 + GroupNorm(8)
        9 个处理位置（5 enc + 4 dec），参数量 ~61.1M ≈ MedNeXt-L（61.8M）
        encoder block_counts=[3,4,4,4,4]，decoder=[4,4,4,3]
        features=[32,64,128,256,384]，strides 匹配 MedNeXt（底部 8³）

    对照链：
        Baseline（6 stages, InstanceNorm, 无残差）→ 0.7941
        ↓ +深度 +残差 +GroupNorm（本实验）         → ?
        MedNeXt（+DW sep conv）                   → 0.8402

    结果目录：nnUNetTrainer_DeepPlainResGN__nnUNetPlans__3d_fullres/
    """

    USE_CHECKPOINT: bool = True

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_deep_plain_res_gn(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            use_checkpoint=cls.USE_CHECKPOINT,
        )


class nnUNetTrainer_DeepPlainResGN_SizeOV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    DeepPlainResGN + 均匀全量 2× 过采样（SizeOV4）。

    完整消融对齐：
        MedNeXt_SizeOV4（DW sep + 残差 + GN + SizeOV4） → 0.8431
        DeepPlainResGN_SizeOV4（Plain + 残差 + GN + SizeOV4）→ ?

    若结果接近 MedNeXt_SizeOV4，说明：
        1. DW sep conv 对性能影响有限
        2. 真正关键因素 = 深度 + 残差 + GroupNorm + 过采样策略

    结果目录：nnUNetTrainer_DeepPlainResGN_SizeOV4__nnUNetPlans__3d_fullres/
    """

    USE_CHECKPOINT: bool = True
    SSO_TINY_REPEAT:     int = 2
    SSO_SMALL_REPEAT:    int = 2
    SSO_MID_REPEAT:      int = 2
    SSO_HUGE_REPEAT:     int = 2
    SSO_NO_TUMOR_REPEAT: int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_deep_plain_res_gn(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            use_checkpoint=cls.USE_CHECKPOINT,
        )



# ------------------------------------------------------------------ #
# DeepResGN_MLA：你真正的贡献架构                                    #
# 强底座（深层残差GN）+ MLA 全局注意力瓶颈                           #
# ------------------------------------------------------------------ #

class nnUNetTrainer_DeepResGN_MLA(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    核心架构：DeepResGN 骨干 + MLA Bottleneck。

    之前 MLAUNet_MoE_SizeOV4（0.8330）< MedNeXt_SizeOV4（0.8431）的原因：
        旧 MLAUNet 底座 = 默认 nnUNet（6 stages, [2,2,2,2,2,2], InstanceNorm, 无残差）
        MLA 插在弱底座上，MLA 的全局建模能力无法发挥。

    本架构修正：
        底座换成 DeepResGN（5 stages, [3,4,4,4,4], GroupNorm, 残差，61M≈MedNeXt）
        MLA 插在强底座的 bottleneck（384ch, 8³），在有足够局部特征的基础上建模全局依赖。

    消融链（单一变量）：
        DeepPlainResGN（强底座，无MLA）  → ?
        DeepResGN_MLA （强底座，有MLA）  → ?（期望更高）
        MedNeXt（DW sep，无MLA）         → 0.8402

    结果目录：nnUNetTrainer_DeepResGN_MLA__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_deep_res_gn_mla(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            use_checkpoint=True,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_DeepResGN_MLA_SizeOV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    DeepResGN_MLA + SizeOV4：完整贡献架构。

    完整对比：
        MedNeXt_SizeOV4（DW sep，无MLA，有SizeOV4）          → 0.8431
        DeepResGN_MLA_SizeOV4（Plain残差GN，有MLA，有SizeOV4）→ ?

    结果目录：nnUNetTrainer_DeepResGN_MLA_SizeOV4__nnUNetPlans__3d_fullres/
    """

    MLA_NUM_HEADS:         int = 8
    MLA_NUM_BLOCKS:        int = 2
    MLA_COMPRESSION_RATIO: int = 4
    MLA_MLP_RATIO:         int = 4
    SSO_TINY_REPEAT:       int = 2
    SSO_SMALL_REPEAT:      int = 2
    SSO_MID_REPEAT:        int = 2
    SSO_HUGE_REPEAT:       int = 2
    SSO_NO_TUMOR_REPEAT:   int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_deep_res_gn_mla(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            use_checkpoint=True,
            mla_num_heads=cls.MLA_NUM_HEADS,
            mla_num_blocks=cls.MLA_NUM_BLOCKS,
            mla_compression_ratio=cls.MLA_COMPRESSION_RATIO,
            mla_mlp_ratio=cls.MLA_MLP_RATIO,
        )


class nnUNetTrainer_DeepDWIBResGN(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    消融实验：证明 DW+IB conv 是可迁移的有效成分，而非 MedNeXt 精心调参的结果。

    架构（DeepDWIBResGN）：
        Encoder：DWIBBlock（DW k3 + IB r=8 + GN + 残差），features=[32,64,128,256,512]
                 block counts=[3,4,8,8,8]，参数量 ~58M
        Decoder：标准 plain-conv UNetDecoder，n_conv=[4,4,4,3]

    对照链：
        DeepPlainResGN（plain 3×3, GN, 残差）→ 0.7966   ← 唯一变量 = block 类型
        DeepDWIBResGN （DW+IB,    GN, 残差）→ ?
        MedNeXt-L（专有架构, DW+IB）         → 0.8402

    若 DeepDWIBResGN ≈ MedNeXt → DW+IB 是可迁移方法，不依赖 MedNeXt 作者的架构设计。

    训练策略：与 nnUNetTrainer_Baseline 完全一致（CE+Dice loss, mirror TTA）。

    结果目录：nnUNetTrainer_DeepDWIBResGN__nnUNetPlans__3d_fullres/
    """

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_deep_dwib_res_gn(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            use_checkpoint=True,
        )


class nnUNetTrainer_DeepDWIBResGN_SizeOV4(
    AutoInternalTestMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    DeepDWIBResGN + 均匀全量 2× 过采样（SizeOV4）。

    消融对齐：
        MedNeXt_SizeOV4      （DW+IB, 专有架构, SizeOV4）→ 0.8431
        DeepDWIBResGN_SizeOV4（DW+IB, 通用架构, SizeOV4）→ ?

    对照 nnUNetTrainer_DeepDWIBResGN，单一变量 = 过采样。

    结果目录：nnUNetTrainer_DeepDWIBResGN_SizeOV4__nnUNetPlans__3d_fullres/
    """

    SSO_TINY_REPEAT:     int = 2
    SSO_SMALL_REPEAT:    int = 2
    SSO_MID_REPEAT:      int = 2
    SSO_HUGE_REPEAT:     int = 2
    SSO_NO_TUMOR_REPEAT: int = 2

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_deep_dwib_res_gn(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            use_checkpoint=True,
        )


class nnUNetTrainer_DeepDWIBMedConfig(AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer):
    """
    显式 MedNeXt-L 配置对齐版的独立 DW+IB 复现。

    目标：
        检验只靠公开可见的结构配置，是否能独立复现 MedNeXt 的高性能。

    显式对齐：
        channels     = 32,64,128,256,512,256,128,64,32
        block_counts = [3,4,8,8,8,8,8,4,3]
        exp_r        = [3,4,8,8,8,8,8,4,3]
        skip fusion  = add
        encoder/decoder 都使用 DW+IB residual block

    刻意不对齐的代码级细节：
        不复用官方 MedNeXt 类；
        up/down block、padding、normalization group 数、conv bias 等保持自写实现。

    对照：
        nnUNetTrainer_MedNeXt        官方/移植 MedNeXt-L
        nnUNetTrainer_DeepDWIBResGN  只对齐 DW+IB 思想，未对齐 decoder exp_r/block_counts

    结果目录：nnUNetTrainer_DeepDWIBMedConfig__nnUNetPlans__3d_fullres/
    """

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_deep_dwib_med_config(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
            use_checkpoint=True,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        from torch._dynamo import OptimizedModule
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.do_ds = enabled

    def _do_i_compile(self):
        return False
