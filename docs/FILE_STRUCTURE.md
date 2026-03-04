# 文件结构说明（lcf_clean）

本文档简要说明 `lcf_clean/` 的目录结构和各部分用途，方便快速浏览与讲解。

```text
lcf_clean/
├── README.md                # 工程总体介绍与快速上手
├── requirements.txt         # Python 依赖列表
├── main.py                  # 统一入口，可路由到不同实验脚本
├── evaluate.py              # 评估与可视化入口（模板）
├── __init__.py              # 使 lcf_clean 成为一个 Python 包
├── .gitignore               # Git 忽略规则
├── lcf/
│   ├── __init__.py          # LCF 核心包入口
│   ├── data/                # 各类数据集的 dataloader 与预处理
│   ├── models/              # Latent Causal Flow 主模型实现
│   ├── modules/             # 子模块（编码器、速度网络、GMM 先验等）
│   ├── utils/               # 通用工具与评估函数
│   └── scripts/
│       └── experiments/     # 实验脚本（可直接运行）
│           ├── test_improved_encoder.py
│           ├── test_traffic.py
│           ├── test_harmonic_vp.py
│           └── catsg_benchmark.py
└── docs/
    ├── FILE_STRUCTURE.md              # 当前文件：结构说明
    ├── MODEL_ARCHITECTURE.md          # 模型架构与训练配置
    ├── Latent_Causal_Flow_Technical_Report.md  # 技术报告（可从原项目复制过来）
    ├── EXPERIMENT_RESULTS.md          # 实验结果与对标（可选）
    └── LCF_architecture.pdf           # 架构图 PDF（README 中会链接到它）
```

如果你只想给别人快速介绍项目，建议配合：
- `README.md`
- `docs/MODEL_ARCHITECTURE.md`
- `docs/LCF_architecture.pdf`

