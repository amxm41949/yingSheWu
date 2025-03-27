import assert from "assert";

// Choose proper "import" depending on your PL.
import { greedySnakeMoveBarriers } from "../build/release.js";
// import { greedy_snake_move_barriers as greedySnakeMoveBarriers } from "./t2_rust/pkg/t2_rust.js"
// [Write your own "import" for other PLs.]

function greedy_snake_barriers_checker(initial_snake, food_num, foods, barriers, access) {
    if (initial_snake.length !== 8) throw "Invalid snake length";

    let current_snake = [...initial_snake];
    let current_foods = [...foods];
    const barriers_list = [];
    for (let i = 0; i < barriers.length; i += 2) {
        const x = barriers[i];
        const y = barriers[i + 1];
        if (x !== -1 && y !== -1) {
            barriers_list.push({ x, y });
        }
    }
    let turn = 1;

    while (turn <= 200) {
        const direction = greedySnakeMoveBarriers(current_snake, current_foods, barriers);

        if (access === 0) {
            if (direction !== -1) {
                return -5;
            } else {
                return 1;
            }
        }

        // invalid direction
        if (direction < 0 || direction > 3) return -4;

        let new_snake = [
            current_snake[0] + (direction == 3) - (direction == 1),
            current_snake[1] + (direction == 0) - (direction == 2),
            current_snake[0],
            current_snake[1],
            current_snake[2],
            current_snake[3],
            current_snake[4],
            current_snake[5],
        ];


        // out of range
        if (new_snake[0] < 1 || new_snake[0] > 8 || new_snake[1] < 1 || new_snake[1] > 8) return -1;

        // hit a barrier
        if (barriers_list.some(ob => ob.x === new_snake[0] && ob.y === new_snake[1])) return -2;

        // eat food
        let ate_index = -1;
        for (let i = 0; i < current_foods.length; i += 2) {
            if (new_snake[0] === current_foods[i] && new_snake[1] === current_foods[i + 1]) {
                ate_index = i;
                break;
            }
        }

        // console.log("turn " + turn + " :" + direction);
        if (ate_index !== -1) {
            current_foods.splice(ate_index, 2);
            food_num -= 1;
        }

        if (food_num === 0) {
            console.log("Total turn: " + turn);
            return turn;
        }

        current_snake = new_snake;
        turn++;
    }

    // timeout
    return -3;
}

function dfs(x, y, food, grid, directions, visited, lastX, lastY) {
    if (x === food[0] && y === food[1]) return true;
    if (x < 1 || x > 8 || y < 1 || y > 8 || !grid[x - 1][y - 1]) return false;

    for (let i = 0; i < directions.length; i++) {
        const dx = directions[i][0];
        const dy = directions[i][1];
        let nx = x + dx;
        let ny = y + dy;
        if (nx == lastX && ny == lastY) {
            continue;
        }
        if (!visited.has(`${nx},${ny},${x},${y}`)) {
            visited.set(`${nx},${ny},${x},${y}`, true);
            if (dfs(nx, ny, food, grid, directions, visited, x, y)) return true;
        }
    }



    return false;
}

function canReachFood(snake, food, barriers) {
    const directions = [
        [0, 1],   // 上
        [-1, 0],  // 左
        [0, -1],  // 下
        [1, 0]    // 右
    ];

    const grid = Array.from({ length: 8 }, () => Array(8).fill(true));

    for (let i = 0; i < barriers.length; i += 2) {
        const x = barriers[i];
        const y = barriers[i + 1];
        if (x >= 1 && x <= 8 && y >= 1 && y <= 8) {
            grid[x - 1][y - 1] = false;
        }
    }

    let visited = new Map();

    return dfs(snake[0], snake[1], food, grid, directions, visited, snake[2], snake[3]);
}

function generateRandomSnakeAndFoodAndBarrier() {
    const snake = [];
    const food = [];
    const barriers = [];

    const headX = Math.floor(Math.random() * 8) + 1;
    const headY = Math.floor(Math.random() * 8) + 1;

    snake.push(headX, headY);

    const directions = [
        [0, 1], [0, -1], [-1, 0], [1, 0]
    ];

    let curX = headX, curY = headY;
    for (let i = 0; i < 3; i++) {
        while (true) {
            let dir = directions[Math.floor(Math.random() * 4)];
            let nx = curX - dir[0];
            let ny = curY - dir[1];

            if (nx < 1 || nx > 8 || ny < 1 || ny > 8) {
                continue;
            }

            let isColliding = false;
            for (let j = 0; j < snake.length; j += 2) {
                if (snake[j] === nx && snake[j + 1] === ny) {
                    isColliding = true;
                    break;
                }
            }
            if (isColliding) {
                continue;
            }

            curX = nx;
            curY = ny;
            snake.push(curX, curY);
            break;
        }
    }

    while (true) {
        const foodX = Math.floor(Math.random() * 8) + 1;
        const foodY = Math.floor(Math.random() * 8) + 1;

        let isValid = true;
        for (let i = 0; i < snake.length; i += 2) {
            if (snake[i] === foodX && snake[i + 1] === foodY) {
                isValid = false;
                break;
            }
        }

        if (isValid) {
            food.push(foodX, foodY);
            break;
        }
    }

    while (barriers.length < 24) {
        let barrierX = Math.floor(Math.random() * 8) + 1;
        let barrierY = Math.floor(Math.random() * 8) + 1;

        let isOccupied = false;
        for (let i = 0; i < snake.length; i += 2) {
            if (snake[i] === barrierX && snake[i + 1] === barrierY) {
                isOccupied = true;
                break;
            }
        }
        if (!isOccupied && !(barrierX === food[0] && barrierY === food[1])) {
            barriers.push(barrierX, barrierY);
        }
    }


    const access = canReachFood(snake, food, barriers) ? 1 : 0;

    return [snake, food, barriers, access];
}

let num_tests = 2000
for (let i = 0; i < num_tests; i++) {
    const [snake, food, barrier, access] = generateRandomSnakeAndFoodAndBarrier();
    let res = greedy_snake_barriers_checker(snake, 1, food, barrier, access);
    if (res <= 0) {
        console.log([snake, food, barrier, access], res);
        drawTestCase(snake, food, barrier);
        break
    }

}
console.log("🎉 You have passed", num_tests, "tests provided.");

function drawTestCase(snake, food, barriers) {
    // 创建 8×8 空白矩阵
    let grid = Array.from({ length: 8 }, () => Array(8).fill('.'));

    // 标记蛇的位置
    for (let i = 0; i < snake.length; i += 2) {
        let x = snake[i] - 1;  // 由于索引从 1 开始，这里转换为 0 索引
        let y = snake[i + 1] - 1;
        grid[x][y] = i == 0 ? 'S' : 's';
    }

    // 标记食物位置
    let foodX = food[0] - 1;
    let foodY = food[1] - 1;
    grid[foodX][foodY] = 'F';

    // 标记障碍物位置
    for (let i = 0; i < barriers.length; i += 2) {
        let bx = barriers[i] - 1;
        let by = barriers[i + 1] - 1;
        grid[bx][by] = '#';
    }

    // 打印矩阵
    console.log(grid.map(row => row.join(' ')).join('\n'));
}
