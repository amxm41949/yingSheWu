// The entry file of your WebAssembly module.

import { convertState, SnakeNet, Tensor } from "./snake_net";
import { conv1_weight_5, conv2_weight_5, conv1_bias_5, conv2_bias_5, fc1_weight_5, fc1_bias_5, fc2_weight_5, fc2_bias_5 } from "./snakenet_5_params";
import { conv1_weight_8, conv2_weight_8, conv1_bias_8, conv2_bias_8, fc1_weight_8, fc1_bias_8, fc2_weight_8, fc2_bias_8 } from "./snakenet_8_params";


export function greedy_snake_move(snake: Array<i32>, food: Array<i32>): i32 {  // T1
    const headX: i32 = snake[0];
    const headY: i32 = snake[1];
    const foodX: i32 = food[0];
    const foodY: i32 = food[1];

    const directions: Array<Array<i32>> = [
        [0, 0, 1], // 上
        [1, -1, 0], // 左
        [2, 0, -1],  // 下
        [3, 1, 0]   // 右
    ];

    for (let i = 0; i < directions.length; i++) {
        for (let j = i + 1; j < directions.length; j++) {
            const distI = abs(headX + directions[i][1] - foodX) + abs(headY + directions[i][2] - foodY);
            const distJ = abs(headX + directions[j][1] - foodX) + abs(headY + directions[j][2] - foodY);
            if (distI > distJ) {
                const temp = directions[i];
                directions[i] = directions[j];
                directions[j] = temp;
            }
        }
    }

    for (let i = 0; i < directions.length; i++) {
        const dir: i32 = directions[i][0];
        const newX: i32 = headX + directions[i][1];
        const newY: i32 = headY + directions[i][2];

        if (newX < 1 || newX > 8 || newY < 1 || newY > 8) continue;

        let futureSnake: Array<i32> = new Array<i32>(snake.length);
        futureSnake[0] = newX;
        futureSnake[1] = newY;
        for (let j = 2; j < snake.length; j++) {
            futureSnake[j] = snake[j - 2];
        }

        let isColliding: bool = false;
        for (let j = 2; j < futureSnake.length; j += 2) {
            if (newX == futureSnake[j] && newY == futureSnake[j + 1]) {
                isColliding = true;
                break;
            }
        }

        if (!isColliding) {
            return dir;
        }
    }

    return 0;
}

export function greedySnakeMoveBarriers(n: i32, snake: Array<i32>, food: Array<i32>, barriers: Array<i32>): i32 { // T2

    let directions: Array<Array<i32>> = [
        [0, 1],   // up
        [-1, 0],  // left
        [0, -1],  // down
        [1, 0]    // right
    ];

    let grid: Array<Array<boolean>> = new Array<Array<boolean>>(n);
    for (let i: i32 = 0; i < n; i++) {
        grid[i] = new Array<boolean>(n).fill(true);
    }

    for (let i: i32 = 0; i < barriers.length; i += 2) {
        const x: i32 = barriers[i];
        const y: i32 = barriers[i + 1];
        if (x >= 1 && x <= n && y >= 1 && y <= n) {
            grid[x - 1][y - 1] = false;
        }
    }

    let queue: Array<Array<i32>> = [];
    let visited: Set<string> = new Set();

    for (let i: i32 = 0; i < 4; i++) {
        let nx: i32 = snake[0] + directions[i][0];
        let ny: i32 = snake[1] + directions[i][1];

        if (nx >= 1 && nx <= n && ny >= 1 && ny <= n && grid[nx - 1][ny - 1] && !(nx == snake[2] && ny == snake[3])) {
            queue.push([nx, ny, 1, i, snake[0], snake[1]]);
            visited.add(`${nx},${ny},${snake[0]},${snake[1]}`);
        }
    }

    // bfs
    while (queue.length > 0) {
        let item = queue.shift();
        let x: i32 = item[0];
        let y: i32 = item[1];
        let step: i32 = item[2];
        let firstMove: i32 = item[3];
        let oldSecondX: i32 = item[4];
        let oldSecondY: i32 = item[5];

        if (x === food[0] && y === food[1]) return firstMove;

        for (let i: i32 = 0; i < 4; i++) {
            let nx: i32 = x + directions[i][0];
            let ny: i32 = y + directions[i][1];

            if (nx >= 1 && nx <= n && ny >= 1 && ny <= n && grid[nx - 1][ny - 1] && !(nx == oldSecondX && ny == oldSecondY)) {
                let newSecondX = x;
                let newSecondY = y;
                if (!visited.has(`${nx},${ny},${newSecondX},${newSecondY}`)) {
                    visited.add(`${nx},${ny},${newSecondX},${newSecondY}`);
                    queue.push([nx, ny, step + 1, firstMove, newSecondX, newSecondY]);
                }
            }

        }
    }
    return -1;
}

export function greedy_policy_wrapper(n: i32, snake: Array<i32>, otherSnakeCount: i32, otherSnakes: Array<i32>, food_num: i32, foods: Array<i32>, remainingRounds: i32): i32 {
    let head_x = snake[0];
    let head_y = snake[1];

    let min_dist = Number.MAX_VALUE;
    let closest_food = [0, 0];

    for (let i = 0; i < food_num; i++) {
        let food_x = foods[i * 2];
        let food_y = foods[i * 2 + 1];

        let dist = Math.abs(head_x - food_x) + Math.abs(head_y - food_y); // 曼哈顿距离

        if (dist < min_dist) {
            min_dist = dist;
            closest_food = [food_x, food_y];
        }
    }

    let snakeBarriers: Array<i32> = [];
    for (let i = 0; i < otherSnakeCount; i++) {
        for (let j = 0; j < 3; j++) { // 取前三节
            let index = i * 8 + j * 2;
            snakeBarriers.push(otherSnakes[index]);
            snakeBarriers.push(otherSnakes[index + 1]);
        }
    }

    let t2_policy: i32 = greedySnakeMoveBarriers(n, snake, closest_food, snakeBarriers);

    if (t2_policy == -1) {
        return greedy_snake_move(snake, closest_food);
    }

    return t2_policy;
}



export function greedy_snake_step(n: i32, snake: Array<i32>, otherSnakeCount: i32, otherSnakes: Array<i32>, food_num: i32, foods: Array<i32>, remainingRounds: i32): i32 {
    // let greedy_policy_output: i32 = greedy_policy_wrapper(n, snake, otherSnakeCount, otherSnakes, food_num, foods);

    let inputTensor:Tensor = convertState(n, snake, otherSnakes, foods);
    // printTensor(inputTensor);

    let conv1_weight = n == 5 ? conv1_weight_5 : conv1_weight_8;
    let conv2_weight = n == 5 ? conv2_weight_5 : conv2_weight_8;
    let conv1_bias = n == 5 ? conv1_bias_5 : conv1_bias_8;
    let conv2_bias = n == 5 ? conv2_bias_5 : conv2_bias_8;
    let fc1_weight = n == 5 ? fc1_weight_5 : fc1_weight_8;
    let fc1_bias = n == 5 ? fc1_bias_5 : fc1_bias_8;
    let fc2_weight = n == 5 ? fc2_weight_5 : fc2_weight_8;
    let fc2_bias = n == 5 ? fc2_bias_5 : fc2_bias_8;

    const snakeNet = new SnakeNet(
        conv1_weight, 
        conv2_weight, 
        conv1_bias, 
        conv2_bias, 
        fc1_weight, 
        fc1_bias, 
        fc2_weight, 
        fc2_bias, 
        n
    );
    const output = snakeNet.forward(inputTensor);
    
    // console.log(output.toString())

    let mx:f32 = -Infinity;
    let out = 0;
    for(let i:i32=0;i<4;i++){
        if (output[i] > mx){
            out = i;
            mx = output[i];
        }
    }
    return out;
}

function printTensor(tensor: Tensor): void {
    for (let c = 0; c < tensor.length; c++) {
        console.log(`Channel ${c}:`);
        for (let i = 0; i < tensor[c].length; i++) {
            console.log(tensor[c][i].join("\t")); // 使用 \t 让列对齐
        }
        console.log(""); // 每个通道之间加个空行
    }
}
