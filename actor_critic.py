from ple import PLE
from ple.games.flappybird import FlappyBird

from torchvision.utils import save_image

import random

import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np

batch_size = 16

def log_loss(action_pi):
    return torch.neg(torch.log(action_pi))

class my_NN(torch.nn.Module):
    def __init__(self, policyp):
        super().__init__()

        # pi(s)
        self.fc1 = nn.Linear(in_features=3, out_features=64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=16, out_features=32)
        self.relu3 = nn.ReLU()
        if policyp:
            self.fc4 = nn.Linear(in_features=32, out_features=2)
        else:
            self.fc4 = nn.Linear(in_features=32, out_features=1)

        self.policyp = policyp
        if policyp:
            self.softmax = nn.Softmax(dim=1) # change back to 1 if batching!!!

    def forward(self,x):

        # input -> big
        x = self.fc1(x)
        x = self.relu1(x)
        # big -> small
        x = self.fc2(x)
        x = self.relu2(x)
        # small -> big
        x = self.fc3(x)
        x = self.relu3(x)
        # big -> output
        x = self.fc4(x)
        if self.policyp:
            x = self.softmax(x)

        return x


class Value():
    def __init__(self):

        self.model = my_NN(False)  # no softmax output
        self.model = self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6, \
                                          betas=(0.9, 0.999))
        self.loss_func = torch.nn.MSELoss()

    def __call__(self,s):
        self.model.eval()
        # input = s[None,:,:,:]
        out = self.model(s)

        # return value of max action, as a float
        return out

    # def update(self,Vs,Vsprime,R,gamma):
    def update(self,VbatchS,VbatchSprime,batchR,batchGamma):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.loss_func(batchR+(batchGamma*VbatchSprime),VbatchS)
        loss.backward()
        self.optimizer.step()

# produces softmax probability for action 0 and 1
class Policy():
    def __init__ (self):

        self.model = my_NN(True)
        self.model = self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6, \
                                          betas=(0.9, 0.999))
        self.loss_func = log_loss # torch.nn.NLLLoss()

    def __call__(self,s):
        self.model.eval()

        with torch.no_grad():

            # out is a tensor of probabilities
            out = self.model(s)

            prob_of_zero = out[0][0].item() # is this bad?

            if random.uniform(0,1) < prob_of_zero:
                # do nothing
                return 0
            else:
                # go up
                return 1

    # def update(self,delta,gamma_t,batch,a):
    def update(self,batchS,batchR,batchGamma,VbatchSprime,VbatchS,batchA_index,gamma):

        self.model.eval()

        # out is a tensor of probabilities for each state in minibatch
        out = self.model(batchS)

        # delta = R + (gamma * Vsprime.item()) - Vs.item()
        deltas = batchR + (gamma * VbatchSprime) - VbatchS

        # index out based on indices in batchA_index
        out = out[range(batch_size),batchA_index]
        # raise Exception()
    
        loss = deltas*batchGamma*self.loss_func(out)

        self.model.train()
        self.optimizer.zero_grad()

        # loss = delta*gamma_t*self.loss_func(prediction)
        loss.mean().backward()
        self.optimizer.step()

def update_batch(batch, new_val):
    for i in range(0, batch_size-1):
        batch[i] = batch[i+1]
    batch[batch_size-1] = new_val

def state_features(state):
    max_vert = 300
    max_horiz = 400
    max_vel = 20
    v_dist = int(state['next_pipe_bottom_y'] - state['player_y'])
    h_dist = int(state['next_pipe_dist_to_player'])
    vel = int(state['player_vel'])

    return torch.from_numpy(np.array([v_dist/max_vert, h_dist/max_horiz, vel/max_vel]))

def init_batch(batch, init_val):
    for i in range(0, batch_size):
        batch[i] = init_val

actions = [0,119]
gamma = 0.99

# game setup
game = FlappyBird()
# https://shalabhsingh.github.io/Deep-RL-Flappy-Bird/
game.adjustRewards({'positive' : 1.5, 'loss' : -1.0, 'tick' : 0.1})
p = PLE(game, fps=30, display_screen=True)
p.init()

pi = Policy()
V = Value()
max_score = 0

# algorithm
episode = 0
while True:
    print("e",episode, ", max_score=", max_score)
    batchS = np.zeros((batch_size, 3))
    batchS = torch.from_numpy(batchS)
    
    batchR = np.zeros((batch_size,1))
    batchR = torch.from_numpy(batchR)

    batchGamma = np.ones((batch_size,1))
    batchGamma = torch.from_numpy(batchGamma)

    batchA_index = [0]*batch_size

    gamma_t = 1
    state = game.getGameState()
    S = state_features(state)
    # for first batch, fill with first state
    init_batch(batchS, S)

    itr = 0

    while not p.game_over():

        update_batch(batchGamma, gamma_t)
        update_batch(batchS, S)
        A_index = pi(S[None,:])
        update_batch(batchA_index, A_index)
        A = actions[A_index]
        R = p.act(A)
        update_batch(batchR, R)
        max_score = max(game.pipes_passed, max_score)
        stateprime = game.getGameState()
        Sprime = state_features(stateprime)

        batchSprime = batchS.detach()
        update_batch(batchSprime, Sprime)

        VbatchS = V(batchS)
        VbatchSprime = V(batchSprime)

        # if itr != 0 and itr % batch_size == 0:
        if itr >= batch_size:
            # print("UPDATING")
            # print("result=",A_index)
            V.update(VbatchS,VbatchSprime,batchR,gamma)
            pi.update(batchS.detach(), batchR.detach(),batchGamma.detach() \
                  ,VbatchSprime.detach(),VbatchS.detach(),batchA_index,gamma)
        # delta = R + (gamma * Vsprime.item()) - Vs.item()

        S = Sprime.detach()
        gamma_t *= gamma
        itr += 1
    # change the learning rate every "epoch" (episode)
    # pi.lr_scheduler.step()
    # V.lr_scheduler.step()

    p.reset_game()
    episode += 1



