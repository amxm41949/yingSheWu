
// 定义 Tensor 数据结构
export type Tensor = f32[][][]; // 3D Tensor: [C, H, W]
export type Weights = f32[][];  // Weights for Linear

let zero:f32 = 0.0;

// ReLU 激活函数
function relu(tensor: Tensor): Tensor {
    return tensor.map((channel: f32[][]) => 
        channel.map((row: f32[]) => 
            row.map((value: f32) => (value > zero ? value : zero))
        )
    );
}

function relu_f32arr(tensor: f32[]): f32[] {
    return tensor.map((value: f32) => (value > zero ? value : zero));
}

function conv2d(input: Tensor, weight: Tensor[], bias: f32[]): Tensor {
    const outChannels: i32 = weight.length;  // 输出通道数
    const inChannels: i32 = weight[0].length;  // 输入通道数
    const H: i32 = input[0].length;  // 输入的高度
    const W: i32 = input[0][0].length;  // 输入的宽度
    const kernelSize: i32 = weight[0][0].length;  // 卷积核的大小
    const pad: i32 = 1;  // 填充大小
    

    // 创建输出数组，使用 Float32Array 保证所有数据都是 32 位浮点数
    const output: Tensor = new Array<f32[][]>(outChannels);

    // 使用 for 循环初始化每个输出通道
    for (let c = 0; c < outChannels; c++) {
        output[c] = new Array<f32[]>(H);
    
        // 为每个通道初始化高度 H
        for (let h = 0; h < H; h++) {
            output[c][h] = new Array<f32>(W).fill(0.0);  // 初始化宽度 W，并填充为 0.0
        }
    }

    // 卷积运算
    for (let oc = 0; oc < outChannels; oc++) {  // 遍历输出通道
        for (let i = 0; i < H; i++) {  // 遍历输入高度
            for (let j = 0; j < W; j++) {  // 遍历输入宽度
                let sum: f32 = bias[oc];  // 初始化当前输出的值为对应的偏置
                for (let ic = 0; ic < inChannels; ic++) {  // 遍历输入通道
                    for (let ki = 0; ki < kernelSize; ki++) {  // 遍历卷积核的高度
                        for (let kj = 0; kj < kernelSize; kj++) {  // 遍历卷积核的宽度
                            let ni: i32 = i + ki - pad;  // 计算对应的输入坐标
                            let nj: i32 = j + kj - pad;  // 计算对应的输入坐标
                            if (ni >= 0 && ni < H && nj >= 0 && nj < W) {  // 确保坐标在范围内
                                sum += input[ic][ni][nj] * weight[oc][ic][ki][kj];  // 计算卷积和
                            }
                        }
                    }
                }
                output[oc][i][j] = sum;  // 将计算结果赋值到输出
            }
        }
    }

    return output;  // 返回卷积后的输出
}

// Flatten 3D Tensor to 1D
function flatten(tensor: Tensor): f32[] {
    // 使用 reduce 来展平三维数组
    return tensor.reduce((acc: f32[], channel:f32[][]) => {
        // 对每个通道 (二维数组) 进行展平
        return acc.concat(channel.reduce((acc2: f32[], row:f32[]) => {
            // 对每一行进行展平
            return acc2.concat(row);
        }, []));
    }, []);  // 初始值为一个空数组
}

// 全连接层
function linear(input: f32[], weight: Weights, bias: f32[]): f32[] {
    let output: f32[] = new Array(weight.length);  // Initialize an output array with the same length as `weight`

    for (let i = 0; i < weight.length; i++) {
        let sum: f32 = bias[i];  // Start with the bias for this row
        for (let j = 0; j < weight[i].length; j++) {
            sum += weight[i][j] * input[j];  // Sum up the weighted inputs
        }
        output[i] = sum;  // Store the result for this row in the output array
    }

    return output;
}

// SnakeNet 推理类
export class SnakeNet {
    conv1_weight: f32[][][][]; // 4D tensor for conv1 weights
    conv2_weight: f32[][][][]; // 4D tensor for conv2 weights
    conv1_bias: f32[];         // 1D tensor for conv1 bias
    conv2_bias: f32[];         // 1D tensor for conv2 bias
    fc1_weight: f32[][];       // 2D tensor for fc1 weights
    fc1_bias: f32[];           // 1D tensor for fc1 bias
    fc2_weight: f32[][];       // 2D tensor for fc2 weights
    fc2_bias: f32[];           // 1D tensor for fc2 bias
    boardSize: i32;

    constructor(
        conv1_weight: f32[][][][], 
        conv2_weight: f32[][][][], 
        conv1_bias: f32[], 
        conv2_bias: f32[], 
        fc1_weight: f32[][], 
        fc1_bias: f32[], 
        fc2_weight: f32[][], 
        fc2_bias: f32[], 
        boardSize: i32
    ) {
        this.conv1_weight = conv1_weight;
        this.conv2_weight = conv2_weight;
        this.conv1_bias = conv1_bias;
        this.conv2_bias = conv2_bias;
        this.fc1_weight = fc1_weight;
        this.fc1_bias = fc1_bias;
        this.fc2_weight = fc2_weight;
        this.fc2_bias = fc2_bias;
        this.boardSize = boardSize;
    }

    forward(input: Tensor): f32[] {
        let x = conv2d(input, this.conv1_weight, this.conv1_bias);
        x = relu(x);
        x = conv2d(x, this.conv2_weight, this.conv2_bias);
        x = relu(x);
        let flattened = flatten(x);
        let fc1_out = relu_f32arr(linear(flattened, this.fc1_weight, this.fc1_bias));
        let fc2_out = linear(fc1_out, this.fc2_weight, this.fc2_bias);
        return fc2_out;
    }
}


export function convertState(
    board_size: i32,
    snake: Array<i32>,
    otherSnakes: Array<i32>,
    foods: Array<i32>
): Tensor {
    // 初始化通道
    let ch_snake: Array<Array<f32>> = [];
    let ch_others: Array<Array<f32>> = [];
    let ch_foods: Array<Array<f32>> = [];

    // 创建零填充的 board_size x board_size 矩阵
    for (let i = 0; i < board_size; i++) {
        ch_snake.push(new Array<f32>(board_size).fill(0));
        ch_others.push(new Array<f32>(board_size).fill(0));
        ch_foods.push(new Array<f32>(board_size).fill(0));
    }

    // 解析 snake 坐标，确保在合法范围
    let snake_pt = reshapeAndClamp(snake, board_size);
    let others_pt = reshapeAndClamp(otherSnakes, board_size);
    let foods_pt = reshapeAndClamp(foods, board_size);

    // 处理 snake 通道
    ch_snake[snake_pt[0][0]][snake_pt[0][1]] = 1; // 头部
    ch_snake[snake_pt[1][0]][snake_pt[1][1]] = -1; // 身体

    for (let i = 0; i < others_pt.length; i++) {
        let x = others_pt[i][0], y = others_pt[i][1];
        if ((i % 4) == 0) {
            ch_others[x][y] = 1; // 敌人头部
        } else{
            ch_others[x][y] = -1; // 敌人身体
        }
    }

    // 处理 foods 通道
    for (let i = 0; i < foods_pt.length; i++) {
        let x = foods_pt[i][0], y = foods_pt[i][1];
        ch_foods[x][y] = 1; // 食物点
    }

    return [ch_snake, ch_others, ch_foods]; // 返回 3 通道
}

function reshapeAndClamp(data: Array<i32>, board_size: i32): Array<Array<i32>> {
    let reshaped: Array<Array<i32>> = [];
    for (let i = 0; i < data.length; i += 2) {
        let coord = new Array<i32>();

        let x = data[i] - 1;
        let y = board_size - data[i + 1];
        coord.push(y);
        coord.push(x);

        // let x = data[i] - 1; // debug
        // let y = data[i + 1] - 1;
        // coord.push(x);
        // coord.push(y);

        reshaped.push(coord);
    }
    return reshaped;
}