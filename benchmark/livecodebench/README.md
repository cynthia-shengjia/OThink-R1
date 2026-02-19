# LiveCodeBench Benchmark for OThink-R1

将 [LiveCodeBench](https://github.com/cynthia-shengjia/LiveCodeBench) 集成到 OThink-R1 项目中，
支持标准评测和 DEER early-exit 适配。

## 目录结构
benchmark/livecodebench/
├── LiveCodeBench/ # LiveCodeBench 源码 (git clone)
├── run_lcb.sh # 标准评测入口
├── run_deer_lcb.sh # DEER 适配运行脚本
├── deer_lcb.py # DEER × LiveCodeBench 适配代码
├── cp_router_lcb_stub.py # CP-Router 不适用标记
├── download_data.sh # 数据集下载脚本
└── README.md # 本文件
