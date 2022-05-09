# Sai Kiran Maddela, Vivek Ramanathan
# CS394R - Reinforcement Learning Final Project

from ple import PLE
import numpy as np
import random
import pickle as pkl
from ple.games.flappybird import FlappyBird
import torch
import torchvision.transforms as transforms

# CNN Architecture
class CNN(torch.nn.Module):
    def __init__(self, state_dims):
        super().__init__()
        self.network = torch.nn.Sequential(
                transforms.Resize((128,128)),
                torch.nn.Conv2d(1, 32, 15, stride = 2, padding = 2),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride = 1, padding = 1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 128, 3, stride = 1, padding = 1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(4, stride=2)
            )
            
        self.to_tens = transforms.ToTensor()
        self.classifier = torch.nn.Linear(128, state_dims)
    
    def forward(self, obs):
        obs = self.to_tens(obs)[None, :, :, :]
        obs[:, 0] = (obs[:, 0] - 0.5) / 0.5
        obs = obs.double()
        output = self.network(obs)
        output = output.mean([2, 3])
        result = self.classifier(output)
        return result

class ValueFunction():
    def __init__(self,
                 state_dims):
        
        # Initialize model and training details
        self.model = CNN(state_dims)
        self.model = self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001, betas=(0.9, 0.999))
        self.loss_f = torch.nn.MSELoss()
        self.to_tens = transforms.ToTensor()
                

    def __call__(self,s):
        self.model.eval()
        input = s

        # Return value function for state
        output = self.model(input)
        
        return output

    def update(self,y_j,s_j, a_j):
        self.model.train()
        self.optimizer.zero_grad()

        # Process input to model and loss function
        input = s_j
        target = torch.as_tensor(y_j)
        target = target.double()

        output = self.model(input)
        
        # Calculate loss and update weights
        loss = self.loss_f(target, output[:, a_j])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return None

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()
Q = ValueFunction(2)

experience_replay = []

# Algorithm constants
epsilon = 1
min_epsilon = 0
max_epsilon = 1
decay = 0.01
actions = [0, 119]
gamma = 0.95
num_minibatch = 16

iter = 0
episode = 0
score = 0
while True:
    print("Max Score so far: ", score)
    while not p.game_over():
        score = max(score, game.pipes_passed)

        # Get grayscale image
        s_t = p.getScreenGrayscale()
        prob = random.uniform(0, 1)
        if prob <= epsilon:
            a_t = random.randint(0, 1)
        else:
            a_t = Q(s_t).argmax()
        
        action = actions[a_t]
        r_t = p.act(action)
        s_t_1 = p.getScreenGrayscale()

        # Add to experience replay
        if len(experience_replay) <= 5000: 
            experience_replay.append((s_t, a_t, r_t, s_t_1, p.game_over()))
        else:
            # Evict older experiences with new ones!
            experience_replay[iter%5000] = (s_t, a_t, r_t, s_t_1, p.game_over())

        iter += 1
        k = num_minibatch
        if (iter % 4 == 0):
            # Perform num_minibatch updates to network
            while (k > 0):
                # Sample from experience replay
                minibatch = experience_replay[random.randint(0, len(experience_replay) - 1)]
                if minibatch[-1]:
                    y_j = minibatch[2]
                else:
                    a_j = max(Q(minibatch[-2]).squeeze())
                    y_j = minibatch[2] + a_j * gamma
                
                Q.update(y_j, minibatch[0], minibatch[1])
                k -= 1

    episode += 1
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    p.reset_game()


