import sys
import torch
import os
import argparse
from torch.distributions import Multinomial
from torch.distributions import Categorical
import pickle
import spacy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

with open ('outfile', 'rb') as fp:
    vocab = pickle.load(fp)
print("Loading spacy vectors...")
nlp = spacy.load('en_core_web_md')
print("Done loading spacy vectors\n")
word2vec = lambda word: nlp.vocab[word].vector
parse_sent = lambda sent: np.sum(list(map(word2vec, sent)), axis=0)

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
         # TODO: input=300, hidden=?, output=len(vocab)
        self.affine1 = nn.Linear(input_size, 128) 
        self.affine2 = nn.Linear(128, output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return word2vec(vocab[action.item()]), action
    
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    args = argparser.parse_args()

    decoder = Policy(300, len(vocab))
    decoder.load_state_dict(torch.load(args.filename))

    prime_str = "Another day of finals"
    sent_len = 10
    for i in range(sent_len):
        state = parse_sent(prime_str)
        _, action_idx = select_action(state, decoder)
        prime_str += " " + vocab[action_idx]

    print(prime_str)

if __name__ == '__main__':
    main()
    