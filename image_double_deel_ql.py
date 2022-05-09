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
    if isinstance(model, CNN):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn' + str(itera) + '.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = CNN(2)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), './cnn570000.th'), map_location='cpu'))
    return r

# CNN Architecture
class CNN(torch.nn.Module):
    def __init__(self, state_dims):
        # Initialization
        super().__init__()
        self.network = torch.nn.Sequential(
                transforms.Resize((120,120)),                
                torch.nn.Conv2d(4, 16, 5, stride = 2, padding = 2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, stride = 2, padding = 1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
            )
            
        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(28800, state_dims))
    
    def forward(self, obs):
        obs = torch.transpose(obs, -2, -1)

        output = self.network(obs)       

        # Flatten CNN output and pass to classifier
        output = output.view(output.size()[0], -1)
        result = self.classifier(output)
        return result

class ValueFunction():
    def __init__(self,
                 state_dims):

        # Define model and other training variables
        self.model = CNN(state_dims)
        self.model = self.model.double()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.loss_f = torch.nn.MSELoss()
        self.to_tens = transforms.ToTensor()
        

    def __call__(self,s):
        self.model.eval()
        input = s
        input = input.to(device)
        input = input.double()

        output = self.model(input)
        
        return output

    def update(self,y_j,s_j, a_j):
        self.model.train()
        self.optimizer.zero_grad()

        # Process input to model and loss function
        input = s_j
        input = input.to(device)
        input = input.double()
        a_j = a_j.long()
        target = y_j
        target.to(device)
        target = target.double()

        self.optimizer.zero_grad()

        output = self.model(input)
        target.to(device)
        loss = self.loss_f(output[:, a_j], target.to(device))
        loss.backward()
        self.optimizer.step()
        
        return None

# Use deep Q Learning with target network
Q = ValueFunction(2)
Q_target = ValueFunction(2)

update_target = 1000

experience_replay_size = 5000
experience_replay = []

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()

# Algorithm constants
epsilon_decay = 0.99995
epsilon = 0.5
num_frames_prev = 4
gamma = 0.9

possible_actions = [0, 119]
max_score = 0
it = -1

# Passing Pipe Reward = +5
# Surviving in Frame = +0.5
# Lose Game = -1000.0
game.adjustRewards({'positive' : 5.0, 'loss' : -1000.0, 'tick' : 0.5})

to_tens = transforms.ToTensor()
while True:
    it += 1
    if p.game_over():
        print("Iteration: ", it, "Max Score: ", max_score, "epsilon: ", epsilon)
        p.reset_game()
        epsilon = max(epsilon * epsilon_decay, 0)

    
    if it % 10000 == 0:
        save_model(Q.model, it)

    max_score = max(game.pipes_passed, max_score)

    # Use Grayscale output
    s_t = p.getScreenGrayscale()

    prob = random.uniform(0, 1)
    if prob <= epsilon or len(experience_replay) < experience_replay_size:
        a_t = random.randint(0, 1)
    else:
        # Create batch of the current frame plus the last three frames
        batch = torch.zeros((1, 4, 288, 512))
        batch[0, 0] = to_tens(experience_replay[-3][0]).squeeze()
        batch[0, 1] = to_tens(experience_replay[-2][0]).squeeze()
        batch[0, 2] = to_tens(experience_replay[-1][0]).squeeze()
        batch[0, 3] = to_tens(s_t).squeeze()

        value = Q(batch)
        if value[0, 0] == value[0, 1]:
            a_t = random.randint(0, 1)
        else:
            a_t = value.argmax()

    action = possible_actions[a_t]
    r_t = p.act(action)

    s_t_1 = p.getScreenGrayscale()

    # Add necessary information to experience replay
    if len(experience_replay) < experience_replay_size:
        experience_replay.append((s_t, a_t, r_t, s_t_1, p.game_over()))
    else:
        experience_replay[it%experience_replay_size] = (s_t, a_t, r_t, s_t_1, p.game_over())

    if len(experience_replay) == experience_replay_size:
        k = 0

        labels = torch.zeros((1,))
        actions = torch.zeros((1, ))
        exp_idx = random.randint(0, len(experience_replay) - 4)
        
        curr_batch = torch.zeros((1, num_frames_prev, 288, 512))
        next_batch = torch.zeros((1, num_frames_prev, 288, 512))

        # Create batch of size 1 that is a random starting point frame and 3 more frames after that 
        while k < num_frames_prev:
            s, a, r, s_prime, done = experience_replay[exp_idx]
            curr_batch[0, k, :, :] = to_tens(s).squeeze()
            next_batch[0, k, :, :] = to_tens(s_prime).squeeze()
            k += 1
            exp_idx += 1
        
        actions = torch.as_tensor(a)

        if done:
            y_j = r
        else:
            a_j = max(Q_target(next_batch).squeeze())
            a_j = a_j.detach()
            y_j = r + a_j * gamma

        # Perform update to Q network
        labels = torch.as_tensor(y_j)
        Q.update(labels, curr_batch, actions)
        update_target -= 1

        # Sync target network every 1000 updates
        if update_target == 0:
            update_target = 1000
            Q_target.model.load_state_dict(Q.model.state_dict())
            print("UPDATE Network")


