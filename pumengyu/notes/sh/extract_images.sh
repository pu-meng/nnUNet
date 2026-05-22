#!/bin/bash
set -e
#bash脚本默认任何命令失败都会继续执行,set -e让脚本遇到错误立即退出

TAR=/home/PuMengYu/8T/Task03_Liver.tar
OUT=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr
TMP=/tmp/task03_extract
#-p是父目录不存在则一并创建,目录已存在则不报错
#/tmp是Linux得系统得临时目录,系统重启后自动清空
mkdir -p "$OUT"
rm -rf "$TMP" && mkdir -p "$TMP"
#命令A&&命令B,如果A成功则执行B,如果A失败则不执行B,并且整个脚本因为set -e而退出

echo "=== 解压 imagesTr（只解压图像目录）==="
tar -xf "$TAR" -C "$TMP" Task03_Liver/imagesTr/
#tar -C这里的-C指定输出目录,Task03_Liver/imagesTr/是tar包内的目录,用来筛选,只选择这个解压
echo "=== 重命名：liver_X.nii.gz → liver_X_0000.nii.gz ==="
COUNT=0
#""外面得*是通配符
for f in "$TMP/Task03_Liver/imagesTr/liver_"*.nii.gz; do
    [ -f "$f" ] || continue
    #[]是bash得条件判断,-f "$f"是判断$f是否是普通文件
    #命令A||命令B,如果A成功则不执行B,如果A失败则执行B
    #continue是跳过本次循环,继续下一次循环
    base=$(basename "$f" .nii.gz)
    #basename是对路径取最后的一层文件名,加上.nii,gz是为了去掉这个后缀,得到liver_X

    cp "$f" "$OUT/${base}_0000.nii.gz"
    COUNT=$((COUNT + 1))
done
#$(COUNT+1)这个是命令替换,执行托号内得命令,执行一个叫做COUNT+1的命令,
#$(())这个是算术运算,计算托号内得数学表达式,这里是把COUNT加1

rm -rf "$TMP"
echo "完成：$COUNT 个图像文件 → $OUT"
