import torch

from env import SnakeEnv
from model import SnakeNet


def main():
    board_size = 8
    num_others = 1 if board_size == 5 else 3
    num_foods = 5 if board_size == 5 else 10
    max_round = 50 if board_size == 5 else 100
    
    model = SnakeNet(board_size, 4)
    state_dict = torch.load(f'./snake/checkpoints/dqn_model_{board_size}_sota.pth')
    model.load_state_dict(state_dict)
    model.eval()

    env = SnakeEnv(board_size, num_others, num_foods, max_round)
    env.render()
    reward_sum = 0
    with torch.no_grad():
        while env.steps < max_round:
            
            q_values = model(env.get_state().unsqueeze(0))
            direction = q_values.argmax().item()
            # breakpoint()
            state, reward, done, _ = env.step(direction)
            reward_sum += reward
            env.render()
            print("reward:", reward, " step:", env.steps)
            if done :
                break

    print("reward_sum:",reward_sum)

if __name__ == "__main__":
    main()
