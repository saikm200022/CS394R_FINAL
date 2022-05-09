# Sai Kiran Maddela, Vivek Ramanathan
# CS394R - Reinforcement Learning Final Project

from ple import PLE
import numpy as np
import random
import pickle as pkl
from ple.games.flappybird import FlappyBird

# Behavior Policy Act Method - epsilon greedy policy
def b_soft_policy(s0, s1, s2, Q, epsilon = 0.10):
    actions = [0, 119]
    prob = random.uniform(0, 1)
    if prob <= epsilon:
        action = random.randint(0, 1)
    else:
        if Q[s0, s1, s2][0] > Q[s0, s1, s2][1]:
            action = 0
        elif Q[s0, s1, s2][0] < Q[s0, s1, s2][1]:
            action = 1
        else:
            action = random.randint(0, 1)
    
    return actions[action], action

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()

# State Space Constants
max_vert = 300
min_vert = -150
max_horiz = 400
min_horiz = 0
max_vel = 20
min_vel = -20

iteration = 0
epsilon = 0.0

# Initialize Relevant Tables
Q = np.random.rand(max_vert - min_vert, max_horiz - min_horiz, max_vel - min_vel, 2)
C = np.zeros((max_vert - min_vert, max_horiz - min_horiz, max_vel - min_vel, 2))

# Reward signals
game.adjustRewards({'positive' : 5.0, 'loss' : -1000, 'tick' : 0.5})
gamma = 0.90
max_score = -1
i = 1

mean = []
max_s = []
scores = []

# Follows pseudocode provided in Sutton Textbook
for episode in range(100000):
    S = []
    A = []
    R = [None]
    game_score = 0

    # Loop through entire episode and store state, action, and rewards
    while not p.game_over():
        state = game.getGameState()
        vertical_dist = int(state['next_pipe_bottom_y'] - state['player_y'])
        horizontal_dist = int(state['next_pipe_dist_to_player'])
        velocity = int(state['player_vel'])

        v_idx = vertical_dist - min_vert
        h_idx = horizontal_dist
        vel_idx = velocity - min_vel

        action, idx = b_soft_policy(v_idx, h_idx, vel_idx, Q, epsilon=0.0) # Hardcoded here

        reward = p.act(action)

        game_score = max(game_score, game.pipes_passed)
        max_score = max(max_score, game.pipes_passed)

        S.append((v_idx, h_idx, vel_idx))
        A.append(idx)
        R.append(reward)
        

    print("Game", i,"Score: ", game_score, " Max Score: ", max_score)
    scores.append(game_score)
    mean.append(sum(scores) / len(scores))
    max_s.append(max_score)

    i += 1
    G = 0
    W = 1

    for t in range(len(S) - 1, -1, -1):
        G = gamma * G + R[t + 1]
        s0, s1, s2 = S[t]
        a_t = A[t]
        C[s0, s1, s2, a_t] += W
        copy = list(Q[s0, s1, s2])
        Q[s0, s1, s2, a_t] = Q[s0, s1, s2, a_t] + W / C[s0, s1, s2, a_t] * (G - Q[s0, s1, s2, a_t])

        other = 0
        if a_t == 0:
            other = 1

        # If taken action agrees with target greedy policy, continue updates, else break out
        if Q[s0, s1, s2, a_t] >= Q[s0, s1, s2, other]:
            W = W * 1 / (1 - epsilon + epsilon / 2)
        
        else:
            break
    
    p.reset_game()

    # Code to save files
    if episode % 10000 == 0:
        with open('mc_Q_0.pkl','wb') as f: pkl.dump(Q, f)
        with open('mc_C_0.pkl','wb') as f: pkl.dump(C, f)
        with open('mc_mean_0.pkl','wb') as f: pkl.dump(mean, f)
        with open('mc_max_0.pkl','wb') as f: pkl.dump(max_s, f)
