# Task03_Liver 数据准备流程

## 环境变量（已写入 ~/.bashrc）
```bash
export nnUNet_raw="/home/PuMengYu/nnUNet_workspace/raw"
export nnUNet_preprocessed="/home/PuMengYu/nnUNet_workspace/preprocessed"
export nnUNet_results="/home/PuMengYu/nnUNet_workspace/results"
```
/home/PuMengYu/anaconda3/envs/medseg/bin/pip install -e /home/PuMengYu/nnUNet  


/home/PuMengYu/anaconda3/envs/medseg/bin/python -c "import nnunetv2; print(nnunetv2.__version__)" 
   
## 1. 解压原始数据
```bash
tar -xf /home/PuMengYu/nnUNet_workspace/raw/Task03_Liver.tar -C /home/PuMengYu/nnUNet_workspace/raw/
```

## 2. MSD 格式转换为 nnUNet v2 格式
```bash

nnUNetv2_convert_MSD_dataset -i /home/PuMengYu/nnUNet_workspace/raw/Task03_Liver -overwrite_id 3
-i 的是输入路径,指向原始MSD格式的数据集文件夹(Task03_Liver)
-overwrite_id 3是指定转换后的数据集编号,生成Dataset003_Liver文件夹;不加的话自动从Task名称取编号

nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
-d 3是指定要处理的数据集编号,对应于Dataset003_Liver文件夹
--verify_dataset_integrity是可选参数,预处理前检查数据集的完整性,文件是否缺失,label值是否合法,不加也能跑,但是不做检查

```
输出目录：`$nnUNet_raw/Dataset003_Liver/`

## 3. 规划 + 预处理
```bash
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity -np 1 1 1 

```

## 4. 训练（5折交叉验证）
```bash
# fold 0~4 分别跑，或只跑 fold 0 快速验证
#3是数据集编号,对应Dataset003_Liver文件夹,3d_fullers是训练配置,2d按照切片训练,速度快精度低,
#3d_lowers,用于大图像的粗分割,3d_fullres是全分辨率训练,精度最高,论文首选
#第二个0是fold编号,

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres 1,2

```
```bash

for fold in 1 2 ;do
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres $fold
done

```

## 5. 推理
```bash
nnUNetv2_predict \
  -i /path/to/imagesTs \
  -o /path/to/output \
  -d 3 \
  -c 3d_fullres \
  -f 0
```

## 数据结构
```
nnUNet_workspace/raw/
└── Dataset003_Liver/
    ├── dataset.json
    ├── imagesTr/        # liver_0_0000.nii.gz ...  (131 cases)
    └── labelsTr/        # liver_0.nii.gz ...
```
