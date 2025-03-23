// The entry file of your WebAssembly module.

export function add(a: i32, b: i32): i32 {
  return a + b;
}

export function greedy_snake_move(snake: Array<i32>, food: Array<i32>): i32 {
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
