# 实验多模态软故障数据与基准

主入口为 data/structured_features_3355.csv：3355条、五类、32个明确输入特征。2224条低BER子集独立提供。Normal的WSS名义带宽为75 GHz，70/65/60/55 GHz为带宽收窄；中心偏移单独分类。低BER按有效记录的双偏振平均BER<0.024筛选，不等于真实FEC译码结果。

data_dictionary定义标签、配置、特征、数组和QoT；data提供派生数据；splits保存每个实验的训练/测试成员；predictions和metrics保存结果；src包含模型、核验和绘图；scripts提供入口；figures保存PDF；reproduction保存复现输出。包内不含原始示波器文件或私有DSP代码。

figures/main保存正文实际使用的图，figures/supplementary保存单独的扩展图。[扩展结果与星座图集](docs/Extended_Results.pdf) 汇集分类与迁移结果、逐工况BER，以及按全部299个物理配置索引的代表星座图；目录和配置索引可跳转至相应页面。对应数值结果及绘图代码位于上述各目录。

采集日期分组压力测试同时改变日期与入纤功率：990条组中989条为5 dBm、1条功率未记录；2365条组中1214条为1 dBm、1151条为3 dBm。因此该测试不能隔离日期效应，也不代表独立采集会话间的迁移。固定成员和预测保留在acquisition_day_proxy目录。

安装 requirements.txt 后，在仓库根目录依次运行 scripts/quick_start.py、scripts/run_baselines.py、scripts/run_targeted_analysis.py 和 scripts/plot_results.py。可用参数见英文README。重新运行的结果进入 reproduction/reruns，不覆盖已报告预测。特征输入仅使用 feature_columns.json 中的32列，不要把工况或BER当成模型输入。完整工况留出采用主资源中具备五类支持的1867条，其他实验也以各自splits为准。数组2048位置是同次采集内均匀选点，不是连续恢复符号。
