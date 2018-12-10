#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import sys

from helpers import *
from model import *
from generate import generate

def save(decoder, filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '_shortened.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.model)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        decoder.cuda()

    # necessary to unroll args
    filename = args.model
    del args.model
    num_episodes = 10000
    gamma = 0.01
    eps = np.finfo(np.float32).eps.item()

    # keep track
    lengths = []
    for i in range(num_episodes):
        sent, log_probs, policy_rewards = generate(decoder, predict_len=10000)

        R = 0
        rewards = []
        for r in policy_rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        policy_loss = []
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.tensor(policy_loss, requires_grad=True)
        # policy_loss = torch.tensor([t.item() for t in policy_loss], requires_grad=True)

        optimizer.zero_grad()
        policy_loss = torch.sum(policy_loss)
        policy_loss.backward()
        optimizer.step()

        lengths.append(len(sent))

        if i % 100 == 0:
            print("Episode %d completed, Avg length: %.02f" %  (i, sum(lengths[-100:])/100))

    save(decoder, filename)
    plt.plot(lengths)