## 目录

[toc]

**选修课课程作业,要求做一个多模态相关的工作,我选择这个,是因为不想单纯敷衍,想完成任务的同时,帮助自己的事业**

---

## 项目结构说明

### 顶层目录
- `BiomedCLIP/` — BiomedCLIP预训练模型权重和tokenizer（748MB，已下载）
- `dataset/` — VQA-Med 2019数据集（train/val/test三个split）
- `medical_vqa/` — 核心代码目录
- `read.md` — 本文件，项目说明

### dataset/
- `train/All_QA_Pairs_train.txt` — 训练集全部QA对，12791条，格式: imgname|question|answer
- `train/QAPairsByCategory/C1_Modality_train.txt` — 按类别分的训练QA：影像模态类
- `train/QAPairsByCategory/C2_Plane_train.txt` — 按类别分的训练QA：成像平面类
- `train/QAPairsByCategory/C3_Organ_train.txt` — 按类别分的训练QA：器官类
- `train/QAPairsByCategory/C4_Abnormality_train.txt` — 按类别分的训练QA：异常/病变类
- `train/Train_images/` — 训练集医学图像（.jpg）
- `val/All_QA_Pairs_val.txt` — 验证集全部QA对，1999条
- `val/QAPairsByCategory/` — 按类别分的验证QA
- `val/Val_images/` — 验证集医学图像
- `test/VQAMed2019_Test_Questions.txt` — 测试集问题（无答案）
- `test/VQAMed2019_Test_Questions_w_Ref_Answers.txt` — 测试集问题+参考答案
- `test/VQAMed2019_Test_Images/` — 测试集医学图像

### medical_vqa/
- `configs/default.yaml` — 所有训练超参数配置（数据路径、模型参数、训练参数、输出路径）
- `data/dataset.py` — 数据加载：load_qa_files读取QA文件，build_answer_vocab构建答案词表，VQAMedDataset是PyTorch Dataset类
- `data/__init__.py` — 包初始化
- `models/vqa_model.py` — 模型定义：CrossAttentionFusion（图文跨模态注意力融合）+ MLPClassifier + BiomedVQA主模型
- `models/__init__.py` — 包初始化
- `train.py` — 主训练脚本：CE Loss + CLIP对比损失，AMP混合精度，cosine LR调度，保存最优checkpoint
- `ablation.py` — 消融实验：E1 Zero-shot CLIP / E2 Concat+Linear / E3 Element-wise multiply
- `evaluate.py` — 评估脚本：加载checkpoint，输出Overall Accuracy，对比BAN 2019基线(58.3%)
- `infer.py` — 单样本推理脚本
- `requirements.txt` — 依赖包列表
- `scripts/download_data.sh` — 数据下载脚本
- `scripts/verify_env.py` — 环境验证脚本

---

## 当前进度
- [x] 模型架构完整实现
- [x] 数据加载完整实现
- [x] 训练/消融/评估脚本完整实现
- [x] BiomedCLIP权重已下载
- [x] 数据集已就绪
- [ ] 还未跑过任何训练（experiments/目录为空）

## 运行方式
```bash
cd /home/PuMengYu/MED_VQA/medical_vqa

# 先验证环境（E1 zero-shot，无需训练）
# CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python ablation.py --config configs/default.yaml --exp E1

# 主模型训练（E4 Cross-Attention）
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/default.yaml --exp_name e4_crossattn

# 消融对比

CUDA_VISIBLE_DEVICES=0 python ablation.py --config configs/default.yaml --exp E2


CUDA_VISIBLE_DEVICES=0 python ablation.py --config configs/default.yaml --exp E3

# 评估
CUDA_VISIBLE_DEVICES=0 python evaluate.py --ckpt /home/PuMengYu/MED_VQA/experiments/medical_vqa/checkpoints/e4_crossattn/best.pth --split val


```


## 阅读顺序
1. 先看配置和文档
  - scripts/执行.md — 你已经看过，整体实验设计                                    
  - configs/default.yaml — 所有超参数、路径配置  

2. 看数据（了解输入）
  
  - data/dataset.py — 数据怎么加载、预处理，图像和问题怎么组织             
                         
3. 看模型（核心）   
- models/vqa_model.py — 最重要，4种融合方式（E1~E4）的模型定义都在这               
              
4.看训练和评估（了解流程） 

- train.py — E4 主模型的训练循环                                      
- ablation.py — E1/E2/E3 消融实验的入口
- evaluate.py — 评估指标怎么算  
   
5.最后看推理      
- infer.py — 单张图片推理                                      
       

6.一句话总结：从 configs/default.yaml → data/dataset.py → models/vqa_model.py → train.py 这条线读下来，能理解 80%
  的项目。要不要我帮你逐个解析这些文件？                                                                                      
