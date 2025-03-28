from tkinter.tix import Tree
import gym
import numpy as np
import random
import torch
from gym import spaces
import torch.nn.functional as F

class SnakeEnv(gym.Env):
    def __init__(self, board_size=5, num_others=1, num_foods=5):
        super(SnakeEnv, self).__init__()

        self.board_size = board_size
        self.num_others = num_others
        self.num_foods = num_foods
        self.action_space = spaces.Discrete(4)  # 4个动作（左下右上）
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, board_size, board_size), dtype=np.float32)

        self.reset()

    def reset(self):
        """初始化游戏状态"""
        self.done = False
        snakes = self.place_snakes(self.num_others + 1) 
        self.snake = snakes[0]

        self.others = snakes[1:]
        
        # 生成不与蛇重叠的食物
        self.foods = torch.zeros([0,2])
        self.foods = self.gen_food()

        return self.get_state()
    
    def gen_food(self):
        foods = self.foods.numpy().tolist()
        snakes = torch.concat([self.snake.unsqueeze(0),self.others])
        while len(foods) < self.num_foods:
            fx, fy = torch.randint(0, self.board_size, (1,)).item(), torch.randint(0, self.board_size, (1,)).item()
            if not any([[fx, fy] == coord for coord in  snakes.view(-1,2).numpy().tolist()]) \
                and not any([[fx, fy] == coord for coord in foods]):
                foods.append([fx, fy])

        return torch.tensor(foods)
        

    def gen_snake(self, l, r, u, d, device="cpu"):
        # 生成起始点
        x = torch.randint(l, r + 1, (1,), device=device).item()
        y = torch.randint(u, d + 1, (1,), device=device).item()

        # 可能的方向：右、下、左、上
        directions = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], device=device)

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

        elif total == 4:
            snakes.append(self.gen_snake(0, board_half-1, 0, board_half-1))
            snakes.append(self.gen_snake(0, board_half-1, board_half, self.board_size-1))
            snakes.append(self.gen_snake(board_half, self.board_size-1, 0, board_half-1))
            snakes.append(self.gen_snake(board_half, self.board_size-1, board_half, self.board_size-1))

        snakes = torch.stack(snakes)[torch.randperm(total)]
                
        return snakes


    def step(self, action):
        """执行一步动作，包括 AI 和其他蛇"""
        if self.done:
            return self.get_state(), 0, True, {}

        # 计算 AI 蛇的下一步位置
        self.snake = torch.concat([torch.tensor(self.move(self.snake[0], action)).unsqueeze(0), self.snake[:-1]])
        # 计算其他蛇的动作（使用贪心策略）
        for i in range(len(self.others)):
            best_move = self.greedy_move(self.others[i])
            self.others[i] = torch.concat([torch.tensor(self.move(self.others[i][0], best_move)).unsqueeze(0), self.others[i][:-1]])

        # 检查游戏是否结束（碰壁或碰自己）
        snakes = torch.concat([self.snake.unsqueeze(0),self.others])
        dead = []
        for i in range(self.num_others+1):
            for j in range(self.num_others+1):
                if i == j and self.check_collision(snakes[i][0], snakes[j][1:]):
                    dead.append(i)
                if i != j and self.check_collision(snakes[i][0], snakes[j]):
                    dead.append(i)

        if 0 in dead:
            reward = -10
            self.done = True
            dead.remove(0)
            

        dead_mask = torch.zeros([self.num_others])
        dead_mask[np.array(dead)-1]=1

        self.others = self.others[~dead_mask]
        self.num_others = self.others.shape[0]

        for i in range(self.num_others+1):
            pass
        if self.snake[0] == self.foods:
            reward = 10
            self.foods = [torch.randint(0, self.board_size, (1,)).item(), torch.randint(0, self.board_size, (1,)).item()]
        else:
            reward = -0.1
            self.snake.pop()

        

        

        return self.get_state(), reward, self.done, {}

    def move(self, pos, action):
        """根据动作计算新位置"""
        x, y = pos
        if action == 0: y -= 1  # 左
        if action == 1: x += 1  # 下
        if action == 2: y += 1  # 右
        if action == 3: x -= 1  # 上
        return [x , y]

    def greedy_move(self, pos):
        """其他蛇采用贪心策略：朝食物移动"""
        x, y = pos
        food_idx = torch.argmin(torch.sum((self.foods - pos) **2,dim=-1))
        fx, fy = self.foods[food_idx]
        if abs(fx - x) < abs(fy - y):
            return 2 if y < fy else 0
        else:
            return 1 if x < fx else 3

    def check_collision(self, pos, others):
        """检查是否撞墙或撞到其他蛇"""
        return (pos == others).all(dim=1).any()
    
    def convert_state(self, board_size, snake, others, foods):
        ch_snake = torch.zeros([board_size, board_size])
        ch_others = torch.zeros([board_size, board_size])
        ch_foods = torch.zeros([board_size, board_size])

        snake_pt = snake.reshape(-1, 2)
        others_pt = others.reshape(-1,2)
        foods_pt = foods.reshape(-1,2)

        ch_snake[snake_pt[:, 0], snake_pt[:, 1]] = 1
        ch_others[others_pt[:, 0], others_pt[:, 1]] = 1
        ch_foods[foods_pt[:, 0], foods_pt[:, 1]] = 1

        return torch.stack([ch_snake, ch_others, ch_foods])

    def get_state(self):
        """返回游戏状态：蛇、食物、其他蛇"""
        state = self.convert_state(self.board_size, self.snake, self.others, self.foods)
        return state

    def render(self):
        """可视化游戏状态（简化版）"""
        board = np.full((self.board_size, self.board_size), " . ")
        for x, y in self.snake:
            board[x, y] = " S "
        for x, y in self.others:
            board[x, y] = " O "
        fx, fy = self.foods
        board[fx, fy] = " F "
        print("\n".join(["".join(row) for row in board]) + "\n")


env = SnakeEnv(8,3)
breakpoint()
