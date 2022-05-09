# Sai Kiran Maddela, Vivek Ramanathan
# CS394R - Reinforcement Learning Final Project

from ple import PLE
import numpy as np
import random
import pickle as pkl
from ple.games.flappybird import FlappyBird
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

device = torch.device('cpu')
learning_rate = 1e-5

# Utility functions to save and load model data
def save_model(model, itera):
    from torch import save
    from os import path
    if isinstance(model, FeedForwardNet):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), './dqn_models/ffn_scale' + str(itera) + '.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = FeedForwardNet(2)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'dqn_models/ffn_scale70000.th'), map_location='cpu'))
    return r

# FFN Network Architecture
class FeedForwardNet(torch.nn.Module):
    def __init__(self, state_dims):
        # Initialization
        super().__init__()

        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(3, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 1000),
                torch.nn.ReLU(),
                torch.nn.Linear(1000, state_dims))
    
    def forward(self, states):
        output = self.classifier(states)       
        return output

# Value Function Parameterized 
class ValueFunction():
    def __init__(self,
                 state_dims):
        # Initialize model
        self.model = FeedForwardNet(state_dims)
        self.model = self.model.double()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.loss_f = torch.nn.MSELoss()
        self.to_tens = transforms.ToTensor()
        

    def __call__(self,s):
        self.model.eval()
        inp = s
        inp = inp.to(device)
        inp = inp.double()

        output = self.model(inp)
        
        return output

    def update(self,y_j,s_j, a_j):
        self.model.train()
        self.optimizer.zero_grad()

        # Process inp to model and loss function
        inp = s_j
        inp = inp.to(device)
        inp = inp.double()
        a_j = a_j.long()
        target = y_j
        target.to(device)
        target = target.double()

        self.optimizer.zero_grad()

        output = self.model(inp)
        target.to(device)
        loss = self.loss_f(output[:, a_j], target.to(device))
        loss.backward()
        self.optimizer.step()
 
        return None

# Use deep Q Learning with target network
Q = ValueFunction(2)
Q_target = ValueFunction(2)

Q.model = Q.model.double()
Q_target.model = Q_target.model.double()

# Experience replay and other constants
update_target = 1000
b_s = 16
experience_replay_size = 5000
experience_replay = torch.zeros(5000, 9)

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()

epsilon_decay = 0.999995
epsilon = 1
gamma = 0.9

possible_actions = [0, 119]
max_score = 0

# Passing Pipe Reward = +5
# Surviving in Frame = +0.5
# Lose Game = -1000.0
game.adjustRewards({'positive' : 5.0, 'loss' : -1000.0, 'tick' : 0.5})

# State space constants
max_vert = 300
min_vert = -150
max_horiz = 400
min_horiz = 0
max_vel = 20
min_vel = -20

to_tens = transforms.ToTensor()
episode = 1
curr_score = 0
it = -1

while True:
    it += 1
    if p.game_over():
        print("Episode: ", episode, "Max Score: ", max_score, "Curr Score: ", curr_score, "epsilon: ", epsilon)
        p.reset_game()
        epsilon = max(epsilon * epsilon_decay, 0)
        if episode % 10000 == 0:
            save_model(Q.model, episode)
        curr_score = 0
        episode += 1
    
    max_score = max(game.pipes_passed, max_score)
    curr_score = max(game.pipes_passed, curr_score)
    
    # Get current state values
    state = game.getGameState()
    vertical_dist = int(state['next_pipe_bottom_y'] - state['player_y'])
    horizontal_dist = int(state['next_pipe_dist_to_player'])
    velocity = int(state['player_vel'])

    vertical_dist = (vertical_dist - min_vert) / (max_vert - min_vert)
    horizontal_dist = (horizontal_dist - min_horiz) / (max_horiz - min_horiz)
    velocity = (velocity - min_vel) / (max_vel - min_vel)

    # Following e-greedy approach
    prob = random.uniform(0, 1)
    if prob <= epsilon or it < experience_replay_size:
        a_t = random.randint(0, 1)
    else:
        # Pass input into Value Function network
        inp = torch.Tensor([vertical_dist, horizontal_dist, velocity])
        value = Q(inp[None, :])
        if value[0, 0] == value[0, 1]:
            a_t = random.randint(0, 1)
        else:
            a_t = value.argmax()
        
    action = possible_actions[a_t]
    r_t = p.act(action)

    next_state = game.getGameState()
    next_state_vertical_dist = int(next_state['next_pipe_bottom_y'] - next_state['player_y'])
    next_horizontal_dist = int(next_state['next_pipe_dist_to_player'])
    next_velocity = int(next_state['player_vel'])

    next_state_vertical_dist = (next_state_vertical_dist - min_vert) / (max_vert - min_vert)
    next_horizontal_dist = (next_horizontal_dist - min_horiz) / (max_horiz - min_horiz)
    next_velocity = (next_velocity - min_vel) / (max_vel - min_vel)

    s_t = (vertical_dist, horizontal_dist, velocity)
    s_t_1 = (next_state_vertical_dist, next_horizontal_dist, next_velocity)

    # Save current state, next state, action, reward, and game condition into experience replay
    if it < experience_replay_size:
        experience_replay[it, :] = torch.Tensor([*s_t, a_t, r_t, *s_t_1, p.game_over()])
    else:
        experience_replay[it%experience_replay_size, :] = torch.Tensor([*s_t, a_t, r_t, *s_t_1, p.game_over()])
    
    if it >= experience_replay_size:
        # Create batch of state values, find target values, and update network
        idxs = torch.randperm(experience_replay_size)[:b_s]
        batch = experience_replay[idxs]
        game_status = batch[:, 8]
        rewards = batch[:, 4]
        labels = torch.zeros(b_s, )
        actions = torch.zeros(b_s, )

        rewards = rewards.double()
        labels = labels.double()

        mask = game_status == 1
        labels[mask] = rewards[mask]
        mask = game_status != 1
        next_states = batch[:, 5:8]
        a_j, _ = torch.max(Q_target(next_states), dim=1)
        labels[mask] = rewards[mask] + gamma * a_j[mask]
        curr_batch = batch[:, 0:3]
        actions = batch[:, 3]

        Q.update(labels, curr_batch, actions)
        update_target -= 1
        
        # Sync target network every 1000 updates
        if update_target == 0:
            update_target = 1000
            Q_target.model.load_state_dict(Q.model.state_dict())
            print("UPDATE Network")
