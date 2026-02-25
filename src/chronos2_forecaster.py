#!/usr/bin/env python3
"""
Chronos-2 模型部署和推理接口
使用 chronos-forecasting >= 2.0
"""

import torch
import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
from typing import Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Chronos2Forecaster:
    """Chronos-2 时序预测器"""
    
    def __init__(self, model_id: str = "amazon/chronos-2", device: Optional[str] = None):
        """
        初始化 Chronos-2 预测器
        
        Args:
            model_id: HuggingFace 模型 ID (默认 amazon/chronos-2)
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        logger.info(f"Loading Chronos-2 model: {self.model_id}")
        logger.info(f"Using device: {self.device}")
        
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.device
        )
        logger.info("Chronos-2 model loaded successfully!")
    
    def predict(
        self,
        context: Union[np.ndarray, pd.Series, List[float]],
        prediction_length: int = 24,
        quantile_levels: Optional[List[float]] = None,
        num_samples: int = 100
    ) -> dict:
        """
        单变量时序预测
        
        Args:
            context: 历史时序数据
            prediction_length: 预测步数
            quantile_levels: 分位数水平 (如 [0.1, 0.5, 0.9])
            num_samples: 采样次数
            
        Returns:
            预测结果字典
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]
        
        # 转换为 tensor
        if isinstance(context, pd.Series):
            context = context.values
        if isinstance(context, list):
            context = np.array(context)
        
        context_tensor = torch.tensor(context, dtype=torch.float32)
        
        logger.info(f"Predicting {prediction_length} steps ahead...")
        
        # 生成预测
        forecast = self.pipeline.predict(
            context=[context_tensor],
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            num_samples=num_samples
        )
        
        # 提取分位数预测
        results = {
            'model': self.model_id,
            'quantile_levels': quantile_levels,
            'prediction_length': prediction_length,
        }
        
        for i, q in enumerate(quantile_levels):
            results[f'q{int(q*100)}'] = forecast[0, :, i].numpy()
        
        # 使用 q50 作为均值
        results['mean'] = results.get('q50', forecast[0, :, 1].numpy())
        
        return results
    
    def predict_with_covariates(
        self,
        context_df: pd.DataFrame,
        future_df: pd.DataFrame,
        target_col: str = "target",
        id_col: str = "id",
        timestamp_col: str = "timestamp",
        prediction_length: int = 24,
        quantile_levels: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        带协变量的时序预测 (Chronos-2 原生支持)
        
        Args:
            context_df: 历史数据 (包含 target 和 covariates)
            future_df: 未来协变量数据
            target_col: 目标列名
            id_col: ID 列名
            timestamp_col: 时间戳列名
            prediction_length: 预测步数
            quantile_levels: 分位数水平
            
        Returns:
            预测结果 DataFrame
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]
        
        logger.info(f"Predicting with covariates: {prediction_length} steps")
        
        forecast = self.pipeline.predict_df(
            context_df=context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column=id_col,
            timestamp_column=timestamp_col,
            target=target_col
        )
        
        return forecast


if __name__ == "__main__":
    # 简单测试
    print("Testing Chronos2Forecaster with Chronos-2...")
    forecaster = Chronos2Forecaster()
    
    # 生成测试数据
    test_data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
    
    # 预测
    results = forecaster.predict(test_data, prediction_length=24)
    print(f"Prediction shape: {results['mean'].shape}")
    print(f"Quantiles: {results['quantile_levels']}")
    print("Chronos-2 test passed!")
