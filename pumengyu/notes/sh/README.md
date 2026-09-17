# Shell 脚本

只保留仍有用途或仍被代码引用的脚本。

- `kill_orphans.sh`：检查当前用户的 medseg GPU 孤儿进程；默认只预览。它也是 `/home/PuMengYu/kill_orphans.sh` 的实际目标。
- `rerun_best_reports_gpu1.sh`：重建仍存在的 `results_v2_best` 独立报告目录。
- `01_run_inference.sh`：旧 IRCADb 五折推理入口；`pumengyu/ext_val` 的旧评估提示仍引用它，只有恢复对应历史 checkpoint 后才可用。

外部无肿瘤病例导入的唯一入口已移到：

`pumengyu/tools/external_data/run_external_import.sh`

Task02 训练与评估不使用本目录，命令见 `pumengyu/notes/md/02_实验档案/Task02_domain_shift.md`。
