# Chronos-2 本地部署项目

Amazon Chronos-2 时序预测基础模型的本地部署和测试项目。

## 项目概述

**Chronos-2** 是 Amazon 开源的 120M 参数时序预测基础模型 (实际模型大小 478MB)，支持：
- **单变量时序预测** (Univariate forecasting)
- **多变量时序预测** (Multivariate forecasting)  
- **协变量感知预测** (Covariate-informed forecasting) - Chronos-2 原生支持

⚠️ **重要**: Chronos-2 需要 **Python >= 3.10** 和 **chronos-forecasting >= 2.0**

## 环境要求

- Python 3.10, 3.11 或 3.12
- chronos-forecasting >= 2.0
- PyTorch
- 见 `environment.yml` 完整依赖

## 部署过程

### 1. 创建 Conda 环境 (推荐)

```bash
# 克隆项目
git clone https://github.com/tanz63/chronos2-local-deployment.git
cd chronos2-local-deployment

# 创建 Python 3.11 环境
conda env create -f environment.yml
conda activate chronos2-env
```

### 2. 验证安装

```bash
python -c "from chronos import Chronos2Pipeline; print('Chronos-2 ready!')"
```

### 3. 模型下载

Chronos-2 模型会在首次运行时自动从 HuggingFace 下载：
- 模型 ID: `amazon/chronos-2`
- 大小: ~478 MB (safetensors 格式)
- 存储位置: `~/.cache/huggingface/hub/models--amazon--chronos-2/`

## 项目结构

```
chronos2-deployment/
├── src/
│   └── chronos2_forecaster.py      # Chronos-2 推理接口
├── tests/
│   ├── test_univariate.py          # 单变量预测测试
│   └── test_covariates.py          # 协变量预测测试
├── outputs/                         # 预测结果图表 (gitignored)
├── data/                           # 数据文件
├── environment.yml                 # Conda 环境配置 (Python 3.11)
├── .gitignore                      # Git 忽略规则
├── README.md                       # 本文件
└── USAGE.md                        # 使用指南
```

## 测试结果

### Test 1: 单变量时序预测

**场景**: 正弦波 + 噪声 + 趋势

**命令**:
```bash
python tests/test_univariate.py
```

**预期输出**:
- MAE: ~0.15-0.20
- RMSE: ~0.20-0.25
- 图表: `outputs/univariate_forecast.png`

### Test 2: 协变量时序预测

**场景**: 电力消耗预测 (温度、湿度作为协变量)

**命令**:
```bash
python tests/test_covariates.py
```

**预期输出**:
- MAE: ~40-50 kW
- RMSE: ~50-60 kW
- MAPE: ~6-8%
- 协变量相关性分析
- 图表: `outputs/covariate_forecast.png`

## 关键技术点

1. **Zero-shot 预测**: 无需训练，直接使用预训练模型
2. **Probabilistic Forecasting**: 输出分位数预测 (10%, 50%, 90%)
3. **Native Covariate Support**: Chronos-2 原生支持协变量
4. **Cross-learning**: 批次内时间序列信息共享
5. **Efficient Inference**: CPU 和 GPU 均支持

## 依赖项

见 `environment.yml`:
- Python 3.11
- chronos-forecasting >= 2.0
- PyTorch
- pandas, numpy, matplotlib

## 与 Chronos-1.x 的区别

| 特性 | Chronos-1.x (T5) | Chronos-2 |
|------|------------------|-----------|
| 模型架构 | T5-based | Encoder-only |
| 协变量支持 | ❌ 不支持 | ✅ 原生支持 |
| Python 要求 | >= 3.9 | >= 3.10 |
| 包版本 | 1.x | >= 2.0 |
| 模型大小 | 8M-200M | 478M |
| 参数数量 | 8M-200M | 120M |

## 许可证

本项目代码采用 MIT 许可证。

Chronos-2 模型由 Amazon Science 发布，遵循 Apache 2.0 许可。

## 参考

- [Chronos-2 HuggingFace](https://huggingface.co/amazon/chronos-2)
- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [Chronos-2 Paper](https://arxiv.org/abs/2510.15821)
- [AutoGluon Chronos Tutorial](https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-chronos.html)
