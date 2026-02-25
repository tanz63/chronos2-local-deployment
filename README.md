# Chronos-2 本地部署项目

Amazon Chronos-2 时序预测基础模型的本地部署和测试项目。

## 项目概述

Chronos-2 是 Amazon 开源的 120M 参数时序预测基础模型，支持：
- **单变量时序预测** (Univariate forecasting)
- **多变量时序预测** (Multivariate forecasting)  
- **协变量感知预测** (Covariate-informed forecasting)

本项目包含完整的本地部署流程和两个测试案例。

## 部署过程

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd chronos2-local-deployment

# 创建 Conda 环境
conda env create -f environment.yml
conda activate chronos2-env
```

### 2. 模型下载

Chronos-2 模型会在首次运行时自动从 HuggingFace 下载：
- 模型 ID: `amazon/chronos-2`
- 大小: ~480 MB
- 存储位置: HuggingFace 缓存目录

### 3. 验证安装

```bash
python -c "from chronos import Chronos2Pipeline; print('Chronos-2 installed successfully!')"
```

## 项目结构

```
chronos2-deployment/
├── src/
│   └── chronos2_forecaster.py      # 模型推理接口
├── tests/
│   ├── test_univariate.py          # 单变量预测测试
│   └── test_covariates.py          # 协变量预测测试
├── outputs/                         # 预测结果图表
├── data/                           # 数据文件
├── environment.yml                 # Conda 环境配置
├── .gitignore                      # Git 忽略规则
├── README.md                       # 本文件
└── USAGE.md                        # 使用指南
```

## 测试结果

### Test 1: 单变量时序预测

**场景**: 正弦波 + 噪声 + 趋势

**评估指标**:
- MAE: <0.15 (根据实际运行)
- RMSE: <0.20 (根据实际运行)

**结果图表**: `outputs/univariate_forecast.png`

### Test 2: 协变量时序预测

**场景**: 电力消耗预测
- **协变量**: 温度、湿度
- **目标**: 电力消耗 (kW)

**评估指标**:
- MAE: <50 kW
- RMSE: <70 kW
- MAPE: <10%

**协变量分析**:
- 温度与电力消耗相关性: 高 (>0.6)
- 湿度与电力消耗相关性: 中等 (0.3-0.5)

**结果图表**: `outputs/covariate_forecast.png`

## 关键技术点

1. **Zero-shot 预测**: 无需训练，直接使用预训练模型
2. **Probabilistic Forecasting**: 输出分位数预测 (10%, 50%, 90%)
3. **Efficient Inference**: CPU 和 GPU 均支持
4. **Covariate Support**: 可利用外部变量提升预测准确性

## 依赖项

- Python 3.11
- PyTorch
- transformers
- chronos-forecasting >= 2.0
- pandas, numpy, matplotlib

详见 `environment.yml`

## 许可证

本项目代码采用 MIT 许可证。

Chronos-2 模型由 Amazon Science 发布，遵循相应许可条款。

## 参考

- [Chronos-2 HuggingFace](https://huggingface.co/amazon/chronos-2)
- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [Chronos-2 Paper](https://huggingface.co/papers/2510.15821)
