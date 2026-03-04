Latent Causal Flow (LCF) - Clean Version
========================================

这是一个为展示与复现整理过的 **精简版 LCF 工程结构**，只保留核心代码、关键实验脚本和文档，更方便你或合作的同学快速上手。

完整技术细节与背景请参考原始项目与技术报告，这里只保留最常用的入口与说明。

## 目录结构

```text
lcf_clean/
├── README.md
├── requirements.txt
├── main.py
├── evaluate.py
├── __init__.py
├── .gitignore
├── lcf/
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   ├── modules/
│   ├── utils/
│   └── scripts/
│       └── experiments/
│           ├── test_improved_encoder.py
│           ├── test_traffic.py
│           ├── test_harmonic_vp.py
│           └── catsg_benchmark.py
└── docs/
    ├── FILE_STRUCTURE.md
    ├── MODEL_ARCHITECTURE.md
    ├── Latent_Causal_Flow_Technical_Report.md
    ├── EXPERIMENT_RESULTS.md
    └── LCF_architecture.pdf
```

架构图 PDF 请放在 `docs/LCF_architecture.pdf`，并用下面这个链接即可直接点击查看：

- [点击查看 LCF 模型架构（PDF）](docs/LCF_architecture.pdf)

## 快速开始

1. 安装依赖（先根据自己环境修改 `requirements.txt`，然后执行）：

```bash
pip install -r requirements.txt
```

2. 运行典型实验脚本（示例）：

```bash
python -m lcf.scripts.experiments.test_improved_encoder --epochs 50
python -m lcf.scripts.experiments.test_traffic --epochs 10
python -m lcf.scripts.experiments.test_harmonic_vp --epochs 80
python -m lcf.scripts.experiments.catsg_benchmark --dataset harmonic_vp
```

具体参数说明请直接看对应脚本的开头注释。

## 主入口说明

- `main.py`：你可以在这里封装一个统一的训练入口（例如根据命令行参数选择数据集与实验脚本）。
- `evaluate.py`：集中放评估与可视化相关的入口（加载已训练模型、画图、导出指标等）。

目前这两个文件只是简单模板，你可以按自己习惯补充逻辑。

