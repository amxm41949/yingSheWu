import assert from "assert";
import { greedy_snake_move } from "../build/release.js";
// import { greedy_snake_move } from "./t1_rust/pkg/t1_rust.js";
// [Write your own "import" for other PLs.]

function greedy_snake_fn_checker (snake, food) {
    let now_snake = [
        snake[0], snake[1], snake[2], snake[3], snake[4], snake[5], snake[6], snake[7]
    ];
    let turn = 1;
    while (true) {
        let result = greedy_snake_move(now_snake, food);
        let new_snake = [
            now_snake[0] + (result == 3) - (result == 1),
            now_snake[1] + (result == 0) - (result == 2),
            now_snake[0],
            now_snake[1],
            now_snake[2],
            now_snake[3],
            now_snake[4],
            now_snake[5],
        ];
        if (new_snake[0] < 1 || new_snake[0] > 8 || new_snake[1] < 1 || new_snake[1] > 8) {
            return -1;
        }
        if (new_snake[0] == new_snake[4] && new_snake[1] == new_snake[5]) {
            return -2;
        }
        if (new_snake[0] == food[0] && new_snake[1] == food[1]) {
            console.log("Total turn: " + turn);
            return turn;
        }
        now_snake = [
            new_snake[0], new_snake[1], new_snake[2], new_snake[3], new_snake[4], new_snake[5], new_snake[6], new_snake[7]
        ];
        if (turn > 200) {
            return -3;
        }
        turn += 1;
    }
}

function generateRandomSnakeAndFood() {
    const snake = [];
    const food = [];

    const headX = Math.floor(Math.random() * 7) + 1;
    const headY = Math.floor(Math.random() * 7) + 1;

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

    return [snake, food];
}

let num_tests = 2000
for (let i = 0; i < num_tests; i++) {
    const [snake, food] = generateRandomSnakeAndFood();
    // console.log(`Test ${i + 1}:`, { snake, food });
    assert.strictEqual(greedy_snake_fn_checker(snake, food, greedy_snake_move) >= 0, true);
}
console.log("🎉 You have passed",num_tests,"tests provided.");