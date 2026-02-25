#!/usr/bin/env python3
"""
单变量时序预测测试
生成正弦波+噪声数据，使用 Chronos-2 进行预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chronos2_forecaster import Chronos2Forecaster


def generate_sine_data(n_points=200, noise_level=0.1, trend=0.0):
    """生成正弦波+噪声+趋势的模拟时序数据"""
    t = np.linspace(0, 8*np.pi, n_points)
    
    # 基础正弦波
    sine = np.sin(t)
    
    # 添加噪声
    noise = np.random.randn(n_points) * noise_level
    
    # 添加趋势
    trend_component = np.linspace(0, trend, n_points)
    
    # 组合
    data = sine + noise + trend_component
    
    return data, t


def plot_univariate_forecast(history, forecast, quantiles, title="Univariate Time Series Forecast"):
    """绘制单变量预测结果"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 历史数据
    hist_len = len(history)
    x_hist = np.arange(hist_len)
    ax.plot(x_hist, history, 'b-', label='History', linewidth=2)
    
    # 预测数据
    x_pred = np.arange(hist_len, hist_len + len(forecast['mean']))
    
    # 均值预测
    ax.plot(x_pred, forecast['mean'], 'r-', label='Forecast (mean)', linewidth=2)
    
    # 置信区间
    q_levels = forecast['quantile_levels']
    if len(q_levels) >= 3:
        lower = forecast.get(f'q{int(q_levels[0]*100)}', forecast['mean'])
        upper = forecast.get(f'q{int(q_levels[-1]*100)}', forecast['mean'])
        ax.fill_between(x_pred, lower, upper, alpha=0.3, color='red', label='Confidence Interval')
    
    # 垂直线分隔历史和预测
    ax.axvline(x=hist_len-1, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def test_univariate_forecast():
    """测试单变量时序预测"""
    print("=" * 60)
    print("Test 1: Univariate Time Series Forecasting")
    print("=" * 60)
    
    # 初始化预测器
    print("\n1. Initializing Chronos-2 forecaster...")
    forecaster = Chronos2Forecaster()
    
    # 生成模拟数据
    print("\n2. Generating synthetic sine wave data...")
    np.random.seed(42)
    data, time_axis = generate_sine_data(n_points=200, noise_level=0.15, trend=0.5)
    
    # 分割历史数据和测试数据
    context_length = 150
    context = data[:context_length]
    actual_future = data[context_length:]
    prediction_length = len(actual_future)
    
    print(f"   Context length: {context_length}")
    print(f"   Prediction length: {prediction_length}")
    
    # 生成预测
    print("\n3. Generating forecast...")
    forecast = forecaster.predict(
        context=context,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        num_samples=100
    )
    
    # 计算误差
    mae = np.mean(np.abs(forecast['mean'] - actual_future))
    rmse = np.sqrt(np.mean((forecast['mean'] - actual_future)**2))
    
    print(f"\n4. Evaluation Metrics:")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    # 绘制图表
    print("\n5. Generating visualization...")
    history_for_plot = data[:context_length]
    fig = plot_univariate_forecast(
        history_for_plot, 
        forecast, 
        forecast['quantile_levels'],
        title="Chronos-2 Univariate Forecast: Sine Wave + Noise + Trend"
    )
    
    # 添加实际值进行对比
    ax = fig.axes[0]
    x_pred = np.arange(context_length, context_length + prediction_length)
    ax.plot(x_pred, actual_future, 'g--', label='Actual', linewidth=2, alpha=0.7)
    ax.legend()
    
    # 保存图表
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'univariate_forecast.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Chart saved to: {output_path}")
    
    plt.close(fig)
    
    print("\n✅ Univariate forecast test completed!")
    return {
        'mae': mae,
        'rmse': rmse,
        'forecast': forecast,
        'chart_path': output_path
    }


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'outputs'), exist_ok=True)
    
    results = test_univariate_forecast()
    print("\n" + "=" * 60)
    print(f"Final Results:")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  Chart: {results['chart_path']}")
    print("=" * 60)
