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
import torch
import datetime

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

# 首先初始化分布式進程組
def init_distributed():
    # 獲取環境變量
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"初始化分布式進程: rank={{rank}}, local_rank={{local_rank}}, world_size={{world_size}}")
    
    # 設置設備
    device = torch.device(f"cuda:{{local_rank}}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # 初始化進程組
    if torch.distributed.is_available():
        torch.distributed.init_process_group(
            backend="nccl" if torch.distributed.is_nccl_available() else "gloo",
            init_method="env://",
            timeout=datetime.timedelta(seconds=10800),  # 3小時超時
            world_size=world_size,
            rank=rank,
        )
        print(f"進程組初始化成功: {{torch.distributed.get_rank()}}/{{torch.distributed.get_world_size()}}")
        torch.distributed.barrier()  # 同步所有進程
    else:
        print("警告: PyTorch分布式不可用")
    
    return device, local_rank, rank, world_size

# 傳遞的訓練參數
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    # 初始化分布式環境
    device, local_rank, rank, world_size = init_distributed()
    
    # 從ultralytics導入所需模組
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.utils import DEFAULT_CFG
    import torch.distributed as dist
    from ultralytics.utils import LOGGER
    
    # 初始化訓練器
    trainer = PoseTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    
    # 設置設備
    trainer.device = device
    trainer.args.device = device
    
    # 顯式設置模型路徑
    trainer.args.model = "{getattr(trainer.hub_session, 'model_url', trainer.args.model)}"
    
    # 直接調用_do_train而不是_setup_train
    # _setup_train會再次調用dist.init_process_group導致錯誤
    if rank == 0:
        LOGGER.info(f"開始在 {{world_size}} 個 GPU 上訓練")
        
    # 手動設置訓練需要的屬性
    trainer.setup_model()
    trainer.model = trainer.model.to(device)
    trainer.set_model_attributes()
    
    # 初始化數據加載器
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    
    # 獲取數據集
    data = check_det_dataset(trainer.args.data)
    trainer.data = data
    
    # 初始化批次大小
    batch_size = trainer.args.batch // world_size
    
    # 構建數據加載器
    trainset, _ = data["train"], data.get("val") or data.get("test")
    train_loader = build_dataloader(
        trainset, 
        batch_size=batch_size,
        rank=rank,
        mode="train",
        workers=trainer.args.workers
    )
    trainer.train_loader = train_loader
    
    # 只在主節點初始化驗證器
    if rank in [0, -1]:
        testset = data.get("val") or data.get("test")
        test_loader = build_dataloader(
            testset, 
            batch_size=batch_size * 2,
            rank=-1,
            mode="val",
            workers=trainer.args.workers
        )
        trainer.test_loader = test_loader
        
        # 初始化驗證器
        trainer.validator = trainer.get_validator()
    
    # 所有進程同步一下
    if dist.is_initialized():
        dist.barrier()
        
    # 執行訓練
    trainer._do_train(world_size)
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
