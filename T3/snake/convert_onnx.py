import torch
import torch.nn as nn
import argparse
import os

from model import SnakeNet

def convert_to_onnx(weight_path: str):
    # 加载模型
    board_size = 5 if "5" in weight_path.split('/')[-1] else 8
    model = SnakeNet(board_size, num_actions=4)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 生成一个虚拟输入
    dummy_input = torch.randn(1, 3, board_size, board_size)
    
    # 生成 ONNX 路径
    onnx_path = os.path.splitext(weight_path)[0] + ".onnx"
    
    # 导出模型
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX 模型已保存至: {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="输入的 .pth 权重文件路径")
    args = parser.parse_args()
    
    convert_to_onnx(args.path)
