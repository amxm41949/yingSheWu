export function greedySnakeMoveBarriers(snake: Array<i32>, food: Array<i32>, barriers: Array<i32>): i32 {

  let directions: Array<Array<i32>> = [
    [0, 1],   // up
    [-1, 0],  // left
    [0, -1],  // down
    [1, 0]    // right
  ];

  let grid: Array<Array<boolean>> = new Array<Array<boolean>>(8);
  for (let i: i32 = 0; i < 8; i++) {
    grid[i] = new Array<boolean>(8).fill(true);
  }

  for (let i: i32 = 0; i < barriers.length; i += 2) {
    const x: i32 = barriers[i];
    const y: i32 = barriers[i + 1];
    if (x >= 1 && x <= 8 && y >= 1 && y <= 8) {
      grid[x - 1][y - 1] = false;
    }
  }

  let queue: Array<Array<i32>> = [];
  let visited: Set<string> = new Set();

  for (let i: i32 = 0; i < 4; i++) {
    let nx: i32 = snake[0] + directions[i][0];
    let ny: i32 = snake[1] + directions[i][1];

    if (nx >= 1 && nx <= 8 && ny >= 1 && ny <= 8 && grid[nx - 1][ny - 1] && !(nx == snake[2] && ny == snake[3])) {
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

      if (nx >= 1 && nx <= 8 && ny >= 1 && ny <= 8 && grid[nx - 1][ny - 1] && !(nx == oldSecondX && ny == oldSecondY)) {
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
