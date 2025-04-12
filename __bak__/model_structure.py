# model_structure.py - 打印模型结构
from ultralytics import YOLO
import torch

# 加载模型
student = YOLO("yolo11n-pose.pt")
teacher = YOLO("yolo11n-pose.pt")

print("=" * 50)
print("学生模型结构:")
print("=" * 50)

# 打印学生模型结构
student_modules = {}
for name, module in student.model.named_modules():
    if isinstance(name, str) and name.startswith("model.") and isinstance(module, torch.nn.Conv2d):
        layer_parts = name.split(".")
        if len(layer_parts) > 1:
            layer_id = layer_parts[1]
            student_modules[layer_id] = {
                "name": name,
                "type": type(module).__name__,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels
            }

# 按层ID排序
for layer_id in sorted(student_modules.keys(), key=int):
    info = student_modules[layer_id]
    print(f"层 {layer_id}: {info['name']} - {info['type']} - 输入通道: {info['in_channels']}, 输出通道: {info['out_channels']}")

print("\n" + "=" * 50)
print("教师模型结构:")
print("=" * 50)

# 打印教师模型结构
teacher_modules = {}
for name, module in teacher.model.named_modules():
    if isinstance(name, str) and name.startswith("model.") and isinstance(module, torch.nn.Conv2d):
        layer_parts = name.split(".")
        if len(layer_parts) > 1:
            layer_id = layer_parts[1]
            teacher_modules[layer_id] = {
                "name": name,
                "type": type(module).__name__,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels
            }

# 按层ID排序
for layer_id in sorted(teacher_modules.keys(), key=int):
    info = teacher_modules[layer_id]
    print(f"层 {layer_id}: {info['name']} - {info['type']} - 输入通道: {info['in_channels']}, 输出通道: {info['out_channels']}")

print("\n" + "=" * 50)
print("共同层分析:")
print("=" * 50)

# 找出共同的层
common_layers = set(student_modules.keys()) & set(teacher_modules.keys())
print(f"共同层ID: {sorted(common_layers, key=int)}")

# 分析通道差异
print("\n通道差异分析:")
for layer_id in sorted(common_layers, key=int):
    s_info = student_modules[layer_id]
    t_info = teacher_modules[layer_id]
    print(f"层 {layer_id}: 学生通道 {s_info['out_channels']} vs 教师通道 {t_info['out_channels']}")

# 建议蒸馏配置
print("\n建议的蒸馏配置:")
for layer_id in ["15", "19", "22"]:  # 常见蒸馏层
    if layer_id in common_layers:
        s_info = student_modules[layer_id]
        t_info = teacher_modules[layer_id]
        print(f"层 {layer_id}: 学生 {s_info['name']} ({s_info['out_channels']}通道) -> 教师 {t_info['name']} ({t_info['out_channels']}通道)")
        