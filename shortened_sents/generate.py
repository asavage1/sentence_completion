#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch
import sys
import torch
import os
import argparse
from torch.distributions import Multinomial, Categorical, Exponential, Binomial

from helpers import *
from model import *

TERMINAL_STATES = [".", "?", "!"]

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    log_probs = []
    rewards = []

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        # print("Output")
        # print(output)
        # print("View")
        # print(output.data.view(-1))
        # print("Temperature")
        # print(output.data.view(-1).div(temperature))
        # print("Exp")
        # print(output.data.view(-1).div(temperature).exp())
        # print("Experiment")
        # print(output.div(temperature))
        # Sample from the network as a multinomial distribution
        output_dist = output.div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # sys.exit()
        # print(output)
        # print(output_dist)
        # predicted_prob = torch.log(output_dist[top_i])
        # log_probs.append(predicted_prob)

        m = Multinomial(100, output_dist)
        action = m.sample()
        log_probs.append(m.log_prob(action))

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

        # the episode is over if sentence ends
        if predicted_char in TERMINAL_STATES:
            rewards.append(100)
            break
        else:
            rewards.append(0)

    return predicted, log_probs, rewards

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename

    sent, _, _ = generate(decoder, **vars(args))
    print(sent)

