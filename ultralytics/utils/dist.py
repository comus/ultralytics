# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import socket
import sys
import tempfile
import uuid
from pathlib import Path

import torch

from . import USER_CONFIG_DIR

# Constants
TORCH_1_9 = int(torch.__version__.split(".")[0]) == 1 and int(torch.__version__.split(".")[1]) >= 9


def find_free_network_port() -> int:
    """
    Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """
    Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs.
    The file contains the necessary configuration to initialize the trainer in a distributed environment.

    Args:
        trainer (object): The trainer object containing training configuration and arguments.
                         Must have args attribute and be a class instance.

    Returns:
        (str): Path to the generated temporary DDP file.
    """
    # 創建臨時腳本文件
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    temp_file_path = USER_CONFIG_DIR / "DDP" / f"_temp_{uuid.uuid4().hex}.py"
    
    # 獲取當前目錄路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 獲取項目根目錄
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 寫入腳本內容
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(f"""
# Ultralytics Multi-GPU training temp file (自動生成的DDP訓練腳本)
import os
import sys

# 添加項目根目錄到Python路徑，確保能夠導入模組
current_dir = {repr(current_dir)}
project_root = {repr(project_root)}
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 設置PYTHONPATH環境變量，確保子進程也能夠找到模組
os.environ["PYTHONPATH"] = f"{{project_root}}:{{os.environ.get('PYTHONPATH', '')}}"

# 打印當前進程的路徑以及環境信息（用於調試）
print(f"Python路徑: {{sys.path}}")
print(f"當前工作目錄: {{os.getcwd()}}")
print(f"PYTHONPATH: {{os.environ.get('PYTHONPATH', '未設置')}}")

# 傳遞的訓練參數
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    # 從ultralytics導入所需模組
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.utils import DEFAULT_CFG
    
    # 初始化訓練器
    trainer = PoseTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    
    # 顯式設置模型路徑
    trainer.args.model = "{getattr(trainer.hub_session, 'model_url', trainer.args.model)}"
    
    # 注意：這裡不直接調用train()，而是調用_do_train()以避免重複的DDP初始化
    # 因為train()方法會再次調用DDP進程，從而導致錯誤
    trainer._setup_train(world_size=int(os.environ.get('WORLD_SIZE', 1)))
    trainer._do_train(world_size=int(os.environ.get('WORLD_SIZE', 1)))
""")
    
    return str(temp_file_path)


def generate_ddp_command(world_size, trainer):
    """
    Generate command for distributed training.

    Args:
        world_size (int): Number of processes to spawn for distributed training.
        trainer (object): The trainer object containing configuration for distributed training.

    Returns:
        cmd (List[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    # 獲取當前路徑和PYTHONPATH設置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 設置環境變量
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 輸出更詳細的分佈式訓練日誌
    
    # 打印環境信息
    print(f"設置環境變量 PYTHONPATH={env['PYTHONPATH']}")
    
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
        
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    
    # 在命令中添加環境變量
    return cmd, file, env


def ddp_cleanup(trainer, file):
    """
    Delete temporary file if created during distributed data parallel (DDP) training.

    Args:
        trainer (object): The trainer object used for distributed training.
        file (str): Path to the file that might need to be deleted.
    """
    try:
        if os.path.exists(file):
            os.remove(file)
            print(f"已刪除臨時DDP文件: {file}")
    except Exception as e:
        print(f"刪除文件時出錯: {e}")
