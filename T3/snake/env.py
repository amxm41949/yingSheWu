import gym
import numpy as np
import torch
from gym import spaces
import torch.nn.functional as F

from policy import greedy_policy_wrapper

class SnakeEnv(gym.Env):
    def __init__(self, board_size=5, num_others_init=1, num_foods=5):
        super(SnakeEnv, self).__init__()

        self.board_size = board_size
        self.num_others_init = num_others_init
        self.num_others = num_others_init
        self.num_foods = num_foods
        self.action_space = spaces.Discrete(4)  # 4个动作（上左下右）
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, board_size, board_size), dtype=np.float32)
        self.snake_dead = False

        self.reset()

    def reset(self):
        """初始化游戏状态"""
        self.done = False
        self.snake_dead = False
        snakes = self.place_snakes(self.num_others_init + 1) 
        self.snake = snakes[0]

        self.others = snakes[1:]
        
        # 生成不与蛇重叠的食物
        self.foods = torch.zeros([0,2])
        self.foods = self.gen_food()

        return self.get_state()
    
    def get_snakes(self):
        return torch.concat([self.snake.unsqueeze(0), self.others])
    
    def gen_food(self):
        foods = self.foods.clone()  # 复制已有食物
        snakes = self.get_snakes().view(-1, 2)  # 展开所有蛇的位置
        while foods.shape[0] < self.num_foods:
            new_food = torch.randint(0, self.board_size, (1, 2))
            # 检查是否与蛇或已有食物重叠
            if not ((new_food == snakes).all(dim=1).any() or (new_food == foods).all(dim=1).any()):
                foods = torch.cat([foods, new_food], dim=0)
        return foods.int()
        

    def gen_snake(self, l, r, u, d, device="cpu"):
        # 生成起始点
        x = torch.randint(l, r + 1, (1,), device=device).item()
        y = torch.randint(u, d + 1, (1,), device=device).item()

        # 方向：上、左、下、右
        directions = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]], device=device)

        snake = torch.zeros((4, 2), dtype=torch.int32, device=device)
        snake[0] = torch.tensor([x, y], dtype=torch.int32, device=device)

        for i in range(1, 4):
            directions = directions[torch.randperm(4, device=device)]  # 打乱方向
            for dx, dy in directions:
                new_x = snake[i-1, 0] + dx
                new_y = snake[i-1, 1] + dy
                if l <= new_x <= r and u <= new_y <= d and not any((snake[:i] == torch.tensor([new_x, new_y], device=device)).all(dim=1)):
                    snake[i] = torch.tensor([new_x, new_y], dtype=torch.int32, device=device)
                    break

        return snake
        
    def place_snakes(self, total):
        board_half = self.board_size // 2
        snakes = []
        if total == 2:
            
            if torch.randint(0,2, (1,),).item()> 0.5: # up and down
                snakes.append(self.gen_snake(0, self.board_size-1, 0, board_half-1))
                snakes.append(self.gen_snake(0, self.board_size-1, board_half, self.board_size-1))
            else: # left and right
                snakes.append(self.gen_snake(0, board_half-1, 0, self.board_size-1))
                snakes.append(self.gen_snake(board_half, self.board_size-1, 0, self.board_size-1))

        elif total == 4: # four corners
            snakes.append(self.gen_snake(0, board_half-1, 0, board_half-1))
            snakes.append(self.gen_snake(0, board_half-1, board_half, self.board_size-1))
            snakes.append(self.gen_snake(board_half, self.board_size-1, 0, board_half-1))
            snakes.append(self.gen_snake(board_half, self.board_size-1, board_half, self.board_size-1))

        snakes = torch.stack(snakes)[torch.randperm(total)]
                
        return snakes


    def step(self, action):
        """执行一步动作，包括 AI 和其他蛇"""
        if self.done:
            return self.get_state(), torch.tensor(0.0), True, {}

        old_snakes = self.get_snakes()
        old_snake = self.snake.clone()
        # 计算 AI 蛇的下一步位置
        self.snake = torch.concat([self.move(self.snake[0], action).unsqueeze(0), self.snake[:-1]])
        # 计算其他蛇的动作（使用贪心策略）
        for i in range(len(self.others)):
            best_move = self.greedy_move(old_snakes, i+1)
            self.others[i] = torch.concat([self.move(self.others[i][0], best_move).unsqueeze(0), self.others[i][:-1]])

        # 检查游戏是否结束（碰壁或碰自己）
        snakes = self.get_snakes()
        dead = []
        for i in range(self.num_others+1):
            if self.check_collision(snakes[i][0], snakes[i][1:]):  # 自己撞自己
                dead.append(i)
            
            for j in range(self.num_others+1):
                if i != j and self.check_collision(snakes[i][0], snakes[j]):
                    dead.append(i)

        if 0 in dead:
            reward = torch.tensor(-10.0)
            self.done = True
            self.snake_dead = True
            return self.get_state(), reward, self.done, {}
            
        if len(dead) > 0:
            dead_mask = torch.zeros([self.num_others]).bool()
            dead_mask[np.array(dead)-1]=1

            self.others = self.others[~dead_mask]
            self.num_others = self.others.shape[0]

        ate = False

        for i in range(self.num_others+1):
            for j, food in enumerate(self.foods):
                    if (snakes[i][0] == food).all():
                        self.foods = torch.concat([self.foods[:j], self.foods[j+1:]])
                        if i == 0:
                            ate = True
        
        if self.foods.shape[0] < self.num_foods:
            self.foods = self.gen_food()
        
        if ate:
            reward = torch.tensor(10.0)
        else:
            fx, fy = self.get_nearest_food(old_snake[0])
            old_dis = abs(old_snake[0][0] - fx) + abs(old_snake[0][1] - fy)
            new_dis = abs(self.snake[0][0] - fx) + abs(self.snake[0][1] - fy)
            reward = (old_dis - new_dis).float()

        reward += torch.tensor(0.1)

        return self.get_state(), reward, self.done, {}

    def move(self, pos, action):
        delta = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]])  # 上、左、下、右
        return pos + delta[action]

    def get_nearest_food(self, pos):
        nearest_food_idx = torch.argmin(torch.sum((self.foods - pos) ** 2,dim=-1))
        return self.foods[nearest_food_idx]
    
    def greedy_move(self, snakes, idx):
        """其他蛇采用贪心策略：朝食物移动"""
        snake = snakes[idx]
        others = torch.concat([snakes[:idx], snakes[idx+1:]])
        food = self.get_nearest_food(snake[0])
        direction = greedy_policy_wrapper(self.board_size, snake, food, others)
        if direction != -1:
            return direction
        
        x, y = snake[0]
        fx, fy = food
        if abs(fx - x) < abs(fy - y):
            return 3 if y < fy else 1
        else:
            return 2 if x < fx else 0

    def check_collision(self, pos, others):
        """检查是否撞墙或撞到其他蛇"""
        if not (0 <= pos[0] < 8 and 0 <= pos[1] < 8):
            return True
        return (pos == others).all(dim=1).any()
    
    def convert_state(self, board_size, snake, others, foods):
        ch_snake = torch.zeros([board_size, board_size])
        ch_others = torch.zeros([board_size, board_size])
        ch_foods = torch.zeros([board_size, board_size])

        snake_pt = snake.reshape(-1, 2).clamp(0, self.board_size-1)
        others_pt = others.reshape(-1,2).clamp(0, self.board_size-1)
        foods_pt = foods.reshape(-1,2).clamp(0, self.board_size-1)

        ch_snake[snake_pt[0, 0], snake_pt[0, 1]] = 1
        ch_snake[snake_pt[1:2, 0], snake_pt[1:2, 1]] = -1
        ch_snake[snake_pt[2:, 0], snake_pt[2:, 1]] = 0.5

        head_mask = torch.zeros([others_pt.shape[0]]).bool()
        head_mask[::4] = 1
        ch_others[others_pt[head_mask, 0], others_pt[head_mask, 1]] = 1
        ch_others[others_pt[~head_mask, 0], others_pt[~head_mask, 1]] = -1

        ch_foods[foods_pt[:, 0], foods_pt[:, 1]] = 1

        return torch.stack([ch_snake, ch_others, ch_foods])

    def get_state(self):
        """返回游戏状态：蛇、食物、其他蛇"""
        state = self.convert_state(self.board_size, self.snake, self.others, self.foods)
        return state

    def render(self):
        """可视化游戏状态，蛇头显示为 'S' / 'O'，蛇身显示为 's' / 'o'，食物显示为 'F'"""
        board = np.full((self.board_size, self.board_size), " . ")

        # 画食物
        for fx, fy in self.foods:
            board[fx, fy] = " F "

        # 画AI蛇
        if not self.snake_dead:
            board[self.snake[0, 0], self.snake[0, 1]] = " S "
            for i, (x, y) in enumerate(self.snake[1:]):
                board[x, y] = f" {i+1} "

        # 画其他蛇
        for other in self.others:
            board[other[0, 0], other[0, 1]] = " O "
            for x, y in other[1:]:
                board[x, y] = " o "

        print("\n".join(["".join(row) for row in board]) + "\n")


def main():
    env = SnakeEnv(5,1)
    env.render()
    while True:
        direction = input()
        if 'w' in direction:
            direction = 0
        elif 'a' in direction:
            direction = 1
        elif 's' in direction:
            direction = 2
        elif 'd' in direction:
            direction = 3
        state, reward, done, _ = env.step(direction)
        
        env.render()
        print("reward:", reward)
        if done:
            break

if __name__ == "__main__":
    main()

