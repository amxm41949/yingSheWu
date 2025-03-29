import torch
import numpy as np

# 加载 state_dict
path = "./snake/checkpoints/dqn_model_death_5_26000.pth"
state_dict = torch.load(path)

board_size = "5" if "5" in path else "8"

# 准备生成的 TypeScript 文件内容
ts_code = ''

for key, value in state_dict.items():
    # 将 key 中的 '.' 替换为 '_'
    ts_key = key.replace('.', '_')
    
    # 获取权重的 NumPy 数组并转换为 f32 数组
    weight_array = value.cpu().detach().numpy().astype(np.float32)
    
    # 根据权重数组的维度来确定如何表示
    if weight_array.ndim == 1:
        # 1D 数组：f32[]
        ts_value = f'export const {ts_key}_{board_size}: f32[] = [{", ".join(map(str, weight_array.flatten()))}];\n'
    elif weight_array.ndim == 2:
        # 2D 数组：f32[][]
        ts_value = 'export const ' + ts_key+"_"+board_size + ': f32[][] = [\n'
        ts_value += ',\n'.join(['    ' + str(row) for row in weight_array.tolist()])
        ts_value += '\n];\n'
    elif weight_array.ndim == 3:
        # 3D 数组：f32[][][]
        ts_value = 'export const ' + ts_key +"_"+board_size+ ': f32[][][] = [\n'
        ts_value += ',\n'.join(['    ' + str(layer) for layer in weight_array.tolist()])
        ts_value += '\n];\n'
    else:
        # 其他维度（一般会是 4D 或更高）
        ts_value = 'export const ' + ts_key+"_"+board_size + ': f32[][][][] = [\n'
        ts_value += ',\n'.join(['    ' + str(layer) for layer in weight_array.tolist()])
        ts_value += '\n];\n'

    # 将生成的内容加入 TypeScript 代码
    ts_code += ts_value

# 将生成的代码写入到一个 TypeScript 文件
with open(f"snakenet_{board_size}_params.ts", "w") as f:
    f.write(ts_code)

print(f"TypeScript 文件 snakenet_{board_size}_params.ts 已生成。")