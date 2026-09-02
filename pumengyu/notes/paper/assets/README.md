# 病例图原始材料

本目录保存病例图脚本使用的切片级 PNG 输入。它们是证据材料，不是当前正文直接引用的成品；哈希核查未发现目录内重复文件，因此不按“当前六图是否引用”删除。

- 文件名记录病例号与切片号，例如 `liver_33_z50_full.png`。
- `draw_failure_case_figures.py` 当前直接使用的 LiTS 输入包括 `liver_41_z45`、`liver_30_z152`、`liver_33_z49/z50` 和 `liver_13_z327/z328/z334`；其余文件保留为失败病例复核证据池。
- 正式图输出在 [`../figures/`](../figures/)；来源记录在 [`../statistics/`](../statistics/)。
- 不应仅凭单张 PNG 推断 checkpoint、数据 split 或模型身份，使用前需结合 provenance 文件。
- 不在本目录保存正式图、PPT 副本、绘图脚本缓存或 Markdown 导出物。
