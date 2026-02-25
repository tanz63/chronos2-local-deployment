#!/usr/bin/env python3
"""
协变量时序预测测试
模拟电力消耗场景：温度、湿度作为协变量影响电力消耗
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from chronos2_forecaster import Chronos2Forecaster


def generate_power_consumption_data(n_days=60, freq='H'):
    """
    生成电力消耗模拟数据
    
    场景：电力消耗受以下因素影响：
    - 温度：高温和低温都会增加电力消耗（空调/暖气）
    - 湿度：高湿度会增加体感温度，间接影响电力消耗
    - 时间模式：工作日vs周末，白天vs夜晚
    - 基础负荷：随时间缓慢增长
    """
    np.random.seed(42)
    
    # 生成时间序列
    timestamps = pd.date_range(start='2024-01-01', periods=n_days*24, freq=freq)
    n_points = len(timestamps)
    
    # 1. 温度：正弦波模拟季节变化 + 日变化 + 噪声
    # 季节变化（年度周期简化）
    seasonal_temp = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, n_points))
    # 日变化
    daily_temp = 5 * np.sin(np.linspace(0, 2*np.pi*24*n_days, n_points))
    temp_noise = np.random.randn(n_points) * 2
    temperature = seasonal_temp + daily_temp + temp_noise
    
    # 2. 湿度：与温度负相关（通常温度高时湿度低）+ 基础值
    base_humidity = 60
    humidity = base_humidity - 0.5 * (temperature - 20) + np.random.randn(n_points) * 5
    humidity = np.clip(humidity, 30, 90)
    
    # 3. 电力消耗（目标变量）- 基于协变量计算
    # 基础负荷
    base_load = 500
    
    # 温度影响：极端温度（<10度或>28度）增加电力消耗
    temp_effect = np.where(
        temperature < 10, (10 - temperature) * 15,
        np.where(temperature > 28, (temperature - 28) * 20, 0)
    )
    
    # 湿度影响：高湿度增加体感温度
    humidity_effect = np.where(humidity > 70, (humidity - 70) * 3, 0)
    
    # 时间模式：工作日白天用电多
    is_weekday = pd.Series(timestamps).dt.dayofweek < 5
    hour = pd.Series(timestamps).dt.hour
    is_daytime = (hour >= 8) & (hour <= 18)
    time_effect = np.where(is_weekday & is_daytime, 100, 0)
    
    # 增长趋势
    trend = np.linspace(0, 50, n_points)
    
    # 噪声
    noise = np.random.randn(n_points) * 20
    
    # 总电力消耗
    power_consumption = base_load + temp_effect + humidity_effect + time_effect + trend + noise
    power_consumption = np.clip(power_consumption, 300, 1500)
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'id': 'power_station_1',
        'target': power_consumption,
        'temperature': temperature,
        'humidity': humidity
    })
    
    return df


def plot_covariate_forecast(history_df, forecast_df, context_len, title="Covariate-Informed Forecast"):
    """绘制协变量预测结果"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 准备数据
    full_len = context_len + len(forecast_df)
    x_full = np.arange(full_len)
    x_context = np.arange(context_len)
    x_forecast = np.arange(context_len, full_len)
    
    # 1. 电力消耗（主图）
    ax = axes[0]
    ax.plot(x_context, history_df['target'], 'b-', label='History', linewidth=2)
    ax.plot(x_forecast, forecast_df['target_q50'], 'r-', label='Forecast (q50)', linewidth=2)
    
    if 'target_q10' in forecast_df.columns and 'target_q90' in forecast_df.columns:
        ax.fill_between(
            x_forecast, 
            forecast_df['target_q10'], 
            forecast_df['target_q90'], 
            alpha=0.3, color='red', label='Confidence Interval'
        )
    
    ax.axvline(x=context_len-1, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Power Consumption (kW)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 温度（协变量）
    ax = axes[1]
    ax.plot(x_full, pd.concat([history_df['temperature'], forecast_df['temperature']]), 
            'orange', linewidth=1.5, label='Temperature')
    ax.axvline(x=context_len-1, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 湿度（协变量）
    ax = axes[2]
    ax.plot(x_full, pd.concat([history_df['humidity'], forecast_df['humidity']]), 
            'green', linewidth=1.5, label='Humidity')
    ax.axvline(x=context_len-1, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Humidity (%)')
    ax.set_xlabel('Time Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def test_covariate_forecast():
    """测试协变量时序预测"""
    print("=" * 60)
    print("Test 2: Covariate-Informed Time Series Forecasting")
    print("Scenario: Power consumption with temperature and humidity")
    print("=" * 60)
    
    # 初始化预测器
    print("\n1. Initializing Chronos-2 forecaster...")
    forecaster = Chronos2Forecaster()
    
    # 生成模拟数据
    print("\n2. Generating synthetic power consumption data...")
    print("   - Temperature affects power (AC/heating)")
    print("   - Humidity affects comfort level")
    print("   - Time patterns (weekday/daytime)")
    
    full_df = generate_power_consumption_data(n_days=60)
    
    # 分割数据
    context_length = 24 * 50  # 50天的历史数据
    prediction_length = 24 * 7  # 预测7天
    
    context_df = full_df[:context_length].copy()
    # 对于测试，我们需要未来的协变量值
    future_with_target = full_df[context_length:context_length + prediction_length].copy()
    future_df = future_with_target[['timestamp', 'id', 'temperature', 'humidity']].copy()
    
    print(f"   Context length: {len(context_df)} hours ({len(context_df)//24} days)")
    print(f"   Prediction length: {len(future_df)} hours ({len(future_df)//24} days)")
    
    # 生成预测
    print("\n3. Generating forecast with covariates...")
    forecast_df = forecaster.predict_with_covariates(
        context_df=context_df,
        future_df=future_df,
        target_col='target',
        id_col='id',
        timestamp_col='timestamp',
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    # 评估（对比实际值）
    actual_values = future_with_target['target'].values
    predicted_values = forecast_df['target_q50'].values
    
    mae = np.mean(np.abs(predicted_values - actual_values))
    rmse = np.sqrt(np.mean((predicted_values - actual_values)**2))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    
    print(f"\n4. Evaluation Metrics:")
    print(f"   MAE: {mae:.2f} kW")
    print(f"   RMSE: {rmse:.2f} kW")
    print(f"   MAPE: {mape:.2f}%")
    
    # 分析协变量的预测价值
    print("\n5. Covariate Analysis:")
    print("   - Temperature range:", 
          f"{future_df['temperature'].min():.1f}°C to {future_df['temperature'].max():.1f}°C")
    print("   - Humidity range:", 
          f"{future_df['humidity'].min():.1f}% to {future_df['humidity'].max():.1f}%")
    
    # 相关性分析
    temp_corr = np.corrcoef(future_df['temperature'], actual_values)[0, 1]
    hum_corr = np.corrcoef(future_df['humidity'], actual_values)[0, 1]
    print(f"   - Temperature correlation with power: {temp_corr:.3f}")
    print(f"   - Humidity correlation with power: {hum_corr:.3f}")
    
    # 绘制图表
    print("\n6. Generating visualization...")
    fig = plot_covariate_forecast(
        context_df, 
        forecast_df, 
        len(context_df),
        title="Chronos-2 Covariate Forecast: Power Consumption"
    )
    
    # 在图中添加实际值对比
    ax = fig.axes[0]
    x_forecast = np.arange(len(context_df), len(context_df) + len(actual_values))
    ax.plot(x_forecast, actual_values, 'g--', label='Actual', linewidth=2, alpha=0.7)
    ax.legend()
    
    # 保存图表
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'covariate_forecast.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Chart saved to: {output_path}")
    
    plt.close(fig)
    
    print("\n✅ Covariate forecast test completed!")
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'chart_path': output_path,
        'covariate_correlations': {
            'temperature': temp_corr,
            'humidity': hum_corr
        }
    }


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'outputs'), exist_ok=True)
    
    results = test_covariate_forecast()
    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"  MAE: {results['mae']:.2f} kW")
    print(f"  RMSE: {results['rmse']:.2f} kW")
    print(f"  MAPE: {results['mape']:.2f}%")
    print(f"  Chart: {results['chart_path']}")
    print("=" * 60)
