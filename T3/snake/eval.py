import torch

from env import SnakeEnv
from model import SnakeNet


def main():
    board_size = 8
    num_others = 1 if board_size == 5 else 3
    num_foods = 5 if board_size == 5 else 10
    max_round = 50 if board_size == 5 else 100
    
    model = SnakeNet(board_size, 4)
    state_dict = torch.load(f'./snake/sotas/dqn_model_{board_size}_sota.pth')
    model.load_state_dict(state_dict)
    model.eval()

    env = SnakeEnv(board_size, num_others, num_foods, max_round)
   
    # state =  env.convert_state(board_size,
    #                   torch.tensor([2, 5, 2, 6, 2, 7, 3, 7])  ,
    #                   torch.tensor([6, 3, 5, 3, 4, 3, 4, 2, 6, 5, 6, 4, 5, 4, 4, 4, 3, 0, 3, 1, 3, 2, 2, 2]) ,
    #                   torch.tensor([1, 7, 1, 3, 0, 3, 4, 1, 7, 4, 2, 1, 2, 3, 5, 7, 4, 5, 5, 6])
    #                   )

    # print(state)
    # print(model(state.unsqueeze(0)))

    env.render()
    breakpoint()
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
