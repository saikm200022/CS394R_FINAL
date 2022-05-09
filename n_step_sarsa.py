
# Sai Kiran Maddela and Vivek Ramanathan

# Reinforcement Learning - CS 394R

from ple.games.flappybird import FlappyBird
from ple import PLE

import random

import numpy as np

import pickle as pkl
import sys
import csv

max_vert = 300
min_vert = -150
max_horiz = 400
min_horiz = 0
max_vel = 20
min_vel = -20

# policy function
def pi(Q,state_features,eps):
    # returns action INDEX!
    s1,s2,s3 = state_features
    if random.uniform(0,1) < eps:
        action = random.randint(0, 1)
    else:
        if Q[s1,s2,s3][0] > Q[s1,s2,s3][1]:
            action = 0
        elif Q[s1,s2,s3][0] == Q[s1,s2,s3][1]:
            action = random.randint(0, 1)
        else:
            action = 1
    return action

# state extraction function
def state_features(state):
    vertical_dist = int(state['next_pipe_bottom_y'] - state['player_y'])
    horizontal_dist = int(state['next_pipe_dist_to_player'])
    velocity = int(state['player_vel'])

    v_idx = vertical_dist - min_vert #s1
    h_idx = horizontal_dist          #s2
    vel_idx = velocity - min_vel     #s3

    return (v_idx, h_idx, vel_idx)

# main algorithm

# relevant algorithm constants
alpha = 0.1
epsilon = 0.00
gamma = 0.90
n = 2

# game setup
game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()
game.adjustRewards({'positive' : 5.0, 'loss' : -1000, 'tick' : 0.5})
action_map = [0,119]

num_episodes = 100000
episode = 0
pipes_passed = [0]*num_episodes

Q = np.random.rand(max_vert - min_vert, max_horiz - min_horiz, max_vel - min_vel, 2)
max_score = 0

# pseudocode from Sutton and Barto
while episode < num_episodes:
    # print("e", episode, ", max_score", max_score)
    states = []
    rewards = []
    actions = []

    state = game.getGameState()
    S = state_features(state)
    A = pi(Q,S,epsilon) # A is an index into action_map

    T = 9e18 # close to max int
    t = 0
    tau = 0
    # NOTE: rewards will be one behind!
    states.append(S)
    # we store the action indices
    actions.append(A)

    # main loop for episode
    while tau != T - 1:
        pipes_passed[episode] = game.pipes_passed
        if t < T:
            # we ONLY need the action map when actually flapping/not flapping
            R = p.act(action_map[A])
            max_score = max(game.pipes_passed, max_score)
            state_sp = game.getGameState()
            Sp = state_features(state_sp)
            states.append(Sp)
            rewards.append(R)

            if p.game_over():
                T = t+1
            else:
                Atp1 = pi(Q,Sp,epsilon)
                actions.append(Atp1)

        tau = t-n+1
        if tau >= 0:
            G = 0
            i = tau+1
            while i <= min(tau+n,T):
                G += (gamma ** (i-tau-1))*rewards[i-1]  # i-1 b/c rewards behind
                i += 1

            if tau+n < T:
                # S_{tau+n} = S_{t+1} = Sprime
                sp1,sp2,sp3 = Sp
                G += (gamma ** n)*Q[sp1,sp2,sp3][Atp1]
            stau1,stau2,stau3 = states[tau]
            atau = actions[tau]
            Q[stau1,stau2,stau3][atau] += alpha*(G - Q[stau1,stau2,stau3][atau])

        A = Atp1
        t += 1

    # Provide update to terminal with percentage done and max score
    if episode % 100 == 0:
        sys.stdout.write("\r%d%%" % int((episode/num_episodes)*100))
        sys.stdout.write(", max_score=%d" % max_score)
        sys.stdout.flush()
    episode += 1
    p.reset_game()

csv_file = 'csvs/sarsa_' + str(n) + '_step_learning.csv'
pkl_file = 'pkls/sarsa_' + str(n) + '_step.pkl'

# save learning progress for graphing later
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['episode','pipes_passed'])
    for i, pipes in enumerate(pipes_passed):
        writer.writerow([i,pipes])

# save Q as serialized pkl file
with open(pkl_file,'wb') as f:
    pkl.dump(Q, f)
