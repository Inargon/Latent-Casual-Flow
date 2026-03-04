# 模型架构概览（LCF Clean）

本文件只给出一个简要框架，方便与你的代码和技术报告对齐。详细数学推导建议直接看
`Latent_Causal_Flow_Technical_Report.md`。

## 1. 整体结构

- **环境编码器 `EnvironmentEncoderV2`**
  - 输入：时间序列 \(x\)、条件变量 \(c\)
  - 输出：环境表示的均值与方差 \((\mu, \log \sigma^2)\) 以及采样的 \(e\)
- **速度网络 `VelocityNetwork`**
  - 输入：当前状态 \(x_t\)、时间 \(t\)、条件 \(c\)、环境 \(e\)
  - 输出：速度场 \(v_\theta(x_t, t, c, e)\)
- **训练目标**
  - Flow Matching 损失：拟合 \(x_1 - x_0\)
  - KL / 方差 / 协方差等正则项
  - 可选：对比学习、GMM 先验等

## 2. 典型训练脚本

- `lcf.scripts.experiments.test_improved_encoder`
  - 重点：改进环境编码器的多样性与 GMM 先验
- `lcf.scripts.experiments.test_traffic`
  - 真实 Traffic 数据上的性能与生成质量
- `lcf.scripts.experiments.test_harmonic_vp`
  - 合成 Harmonic-VP 数据上的可辨识性与生成质量
- `lcf.scripts.experiments.catsg_benchmark`
  - 在 CaTSG 数据集上的统一对标（MMD / MDD / J-FTSD 等）

详细说明建议直接在上面这几个脚本的头部注释中阅读，你已经写得很清楚了；这里不重复展开。

