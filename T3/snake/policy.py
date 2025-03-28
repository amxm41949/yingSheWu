from collections import deque

def greedy_snake_move_barriers(n, directions, snake, food, barriers):
    
    grid = [[True] * n for _ in range(n)]
    
    for i in range(0, len(barriers), 2):
        x, y = barriers[i], barriers[i + 1]
        if 1 <= x <= n and 1 <= y <= n:
            grid[x - 1][y - 1] = False
    
    queue = deque()
    visited = set()
    
    for i in range(4):
        nx, ny = snake[0] + directions[i][0], snake[1] + directions[i][1]
        
        if (1 <= nx <= n and 1 <= ny <= n and grid[nx - 1][ny - 1] and 
            not (nx == snake[2] and ny == snake[3])):
            queue.append((nx, ny, 1, i, snake[0], snake[1]))
            visited.add((nx, ny, snake[0], snake[1]))
    
    cnt = 0
    while queue:
        cnt +=1
        if cnt > 5000:
            breakpoint()
            cnt=0
        x, y, step, first_move, old_second_x, old_second_y = queue.popleft()
        
        if (x, y) == (food[0], food[1]):
            return first_move
        
        for i in range(4):
            nx, ny = x + directions[i][0], y + directions[i][1]
            
            if (1 <= nx <= n and 1 <= ny <= n and grid[nx - 1][ny - 1] and 
                not (nx == old_second_x and ny == old_second_y)):
                new_second_x, new_second_y = x, y
                if (nx, ny, new_second_x, new_second_y) not in visited:
                    visited.add((nx, ny, new_second_x, new_second_y))
                    queue.append((nx, ny, step + 1, first_move, new_second_x, new_second_y))
                    
    # all blocked
    return -1

def greedy_policy_wrapper(board_size, snake, food, others):
    directions = [
        (-1, 0),   # up
        (0, -1),  # left
        (1, 0),  # down
        (0, 1)    # right
    ]
    return greedy_snake_move_barriers(board_size, directions,
                                      (snake.reshape(-1).clone().numpy()+1).tolist(),
                                      (food.reshape(-1).clone().numpy()+1).tolist(),
                                      (others[:, :-1].reshape(-1).clone().numpy()+1).tolist())