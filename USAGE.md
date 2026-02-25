# Chronos-2 使用指南

## 快速开始

### 1. 基础预测

```python
import numpy as np
from src.chronos2_forecaster import Chronos2Forecaster

# 初始化预测器
forecaster = Chronos2Forecaster(model_id="amazon/chronos-2")

# 准备历史数据 (numpy array, pandas Series, 或 list)
history = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1

# 生成预测
forecast = forecaster.predict(
    context=history,
    prediction_length=24,  # 预测未来24步
    quantile_levels=[0.1, 0.5, 0.9]  # 10%, 50%, 90% 分位数
)

# 获取均值预测
mean_prediction = forecast['mean']

# 获取置信区间
lower_bound = forecast['q10']
upper_bound = forecast['q90']
```

### 2. 带协变量的预测

```python
import pandas as pd

# 准备历史数据 (包含目标变量和协变量)
context_df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=168, freq='H'),
    'id': 'series_1',
    'target': [/* 历史目标值 */],
    'temperature': [/* 历史温度 */],
    'humidity': [/* 历史湿度 */]
})

# 准备未来协变量
future_df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-08', periods=24, freq='H'),
    'id': 'series_1',
    'temperature': [/* 未来温度 */],
    'humidity': [/* 未来湿度 */]
})

# 生成预测
forecast_df = forecaster.predict_with_covariates(
    context_df=context_df,
    future_df=future_df,
    target_col='target',
    id_col='id',
    timestamp_col='timestamp',
    prediction_length=24
)
```

## 模型配置

### 选择模型大小

Chronos 系列提供多个模型大小：

| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| `amazon/chronos-t5-tiny` | 8M | 快速原型、资源受限 |
| `amazon/chronos-t5-mini` | 20M | 平衡速度和精度 |
| `amazon/chronos-t5-small` | 46M | 推荐用于大多数任务 |
| `amazon/chronos-t5-base` | 200M | 最高精度 |
| `amazon/chronos-2` | 120M | 最新版本，支持协变量 |

### 设备选择

```python
# 自动选择 (推荐)
forecaster = Chronos2Forecaster()

# 强制使用 CPU
forecaster = Chronos2Forecaster(device="cpu")

# 强制使用 GPU
forecaster = Chronos2Forecaster(device="cuda")
```

## 高级用法

### 批处理预测

```python
# 多个时间序列同时预测
multiple_series = [series1, series2, series3]
forecasts = []

for series in multiple_series:
    forecast = forecaster.predict(series, prediction_length=24)
    forecasts.append(forecast)
```

### 自定义分位数

```python
# 获取更多分位数
forecast = forecaster.predict(
    context=history,
    prediction_length=24,
    quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95]
)
```

### 调整采样数

```python
# 增加采样数以获得更稳定的预测
forecast = forecaster.predict(
    context=history,
    prediction_length=24,
    num_samples=500  # 默认 100
)
```

## 性能优化

### CPU 推理优化

```python
import torch

# 设置线程数
torch.set_num_threads(4)

# 初始化预测器
forecaster = Chronos2Forecaster(device="cpu")
```

### 内存优化

对于长序列，可以分段预测：

```python
def predict_long_sequence(history, prediction_length, chunk_size=1000):
    """分段预测长序列"""
    results = []
    for i in range(0, prediction_length, chunk_size):
        chunk_len = min(chunk_size, prediction_length - i)
        forecast = forecaster.predict(
            context=history if i == 0 else np.concatenate([history, results]),
            prediction_length=chunk_len
        )
        results.extend(forecast['mean'])
    return np.array(results)
```

## 故障排除

### 模型下载失败

```bash
# 手动下载模型
huggingface-cli download amazon/chronos-2
```

### 内存不足

- 使用更小的模型 (`chronos-t5-mini`)
- 减少 `num_samples`
- 使用 CPU 而非 GPU

### CUDA 错误

```python
# 强制使用 CPU
forecaster = Chronos2Forecaster(device="cpu")
```

## 最佳实践

1. **数据预处理**: 确保输入数据没有缺失值
2. **数据长度**: 提供足够长的历史数据 (至少 50-100 个时间点)
3. **异常值处理**: 在输入前移除或平滑异常值
4. **协变量选择**: 选择真正影响目标变量的协变量
5. **分位数解释**: 
   - q10: 悲观预测 (10% 概率低于此值)
   - q50: 中位数预测 (最可能值)
   - q90: 乐观预测 (90% 概率低于此值)

## 示例脚本

```bash
# 运行单变量测试
python tests/test_univariate.py

# 运行协变量测试
python tests/test_covariates.py

# 运行所有测试
python -m pytest tests/ -v --cov=src --cov-report=html
```
