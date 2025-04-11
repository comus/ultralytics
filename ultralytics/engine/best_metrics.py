import json
from pathlib import Path
from datetime import datetime

class BestMetricsLogger:
    """用于记录最佳模型指标的类"""
    
    def __init__(self, save_dir):
        """
        初始化 BestMetricsLogger
        
        参数:
            save_dir (str | Path): 保存目录路径
        """
        self.save_dir = Path(save_dir)
        self.metrics_file = self.save_dir / "best_metrics.json"
        
    def save_metrics(self, epoch, metrics, fitness):
        """
        保存最佳模型的指标
        
        参数:
            epoch (int): 最佳模型的轮次
            metrics (dict): 模型的评估指标
            fitness (float): 适应度分数
        """
        data = {
            "best_epoch": epoch + 1,
            "fitness": float(fitness),
            "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
            "date": datetime.now().isoformat()
        }
        
        self.metrics_file.write_text(json.dumps(data, indent=4)) 
