import argparse
import gym
import numpy as np
import unidecode
import re
import random
import math
from itertools import count
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import spacy

print("Loading spacy vectors...")
nlp = spacy.load('en_core_web_md')
print("Done loading spacy vectors\n")

word2vec = lambda word: nlp.vocab[word].vector


def get_vocab(filename):
    """ Returns a list of *unique* words in the file 
        and the number of unique words in the file.
    """
    file = unidecode.unidecode(open(filename).read())
    file = re.findall(r"[\w']+", file)

    unique_words = list(set(file))

    return unique_words, len(unique_words)


def get_qa_pairs(filename):
    """ Returns the a list of (q,a) tuples
    """
    qa_pairs = []
    with open(filename, 'r') as f:
        qa_pairs = [line.rstrip() for line in f.readlines()]

    qa_pairs = list(zip(qa_pairs, qa_pairs[1:]))
    qa_pairs = [qa_pairs[i] for i in range(0, len(qa_pairs), 2)]

    return qa_pairs


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('filename', type=str)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--threshold', type=int, default=1, help='cutoff dist')
args = parser.parse_args()


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


vocab, size = get_vocab(args.filename)
policy = Policy(300, size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
parse_sent = lambda sent: np.sum(list(map(word2vec, sent)), axis=0)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return word2vec(vocab[action.item()]), action


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)

    # reinforce with baseline
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()

    # backprop
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def step(state, action, expected_action):
    """ new_state is the sum of the action and the state word vectors
    """
    new_state = np.sum([state, action], axis=0)

    # Inverse distance as a reward
    # distance = np.linalg.norm(np.subtract(new_state, expected_action))
    # reward = 1 / distance

    # Inverse angle as a distance
    angle = math.acos(np.dot(new_state, expected_action) / (np.linalg.norm(new_state) * np.linalg.norm(expected_action)))
    reward = math.cos(angle)  # could just use dot product of unit vectors, but it isn't as obvious what's happening

    return new_state, reward


def parse_qa(qa_pairs):
    """ Converts each (q,a) pair into word vector and returns
        the (q,a) pair as word vectors: the individual word vectors in the 
        pair are summed.
        Returns a list of the converted (q,a) word vectors
    """
    trainings = []

    for q, a in qa_pairs:
        # curr_q = q
        # for a_word in a:
        #     trainings.append((parse_sent(curr_q), word2vec(a_word)))
        #     curr_q += a_word
        trainings.append((parse_sent(q), parse_sent(a)))
    return trainings


def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(policy, save_filename)
    print('Saved as %s' % save_filename)


def main():
    qa = get_qa_pairs(args.filename)

    # build the training set
    training_set = parse_qa(qa)

    print("Number of total episodes: ", len(training_set))
    for i in range(len(training_set)):
        # TODO: Why wouldn't we do this?
        # generate a random q,a pair
        rand_index = i  # random.randint(0, len(training_set) - 1)
        random_q, random_a = training_set[rand_index]

        # TODO: Should we be telling it when to stop? We can give it a large reward if it stops
        # TODO: when it is supposed to, but stopping it at the actual length seems a little too hardcoded
        ending_length = len(qa[rand_index][1])  # length of answer

        state = np.array(random_q)

        # Don't infinite loop while learning
        for t in range(100):
            action_vec, action_idx = select_action(state)

            state, reward = step(state, action_vec, random_a)
            policy.rewards.append(reward)

            # TODO: may just want to add to reward
            if t == ending_length:
                break

        finish_episode()
        if i % args.log_interval == 0:
            print('Episode {}\t'.format(i))
            save()

    # test time
    prime_str = "Another day of finals"
    sent_len = 10
    for i in range(sent_len):
        init_state = parse_sent(prime_str)
        _, action_idx = select_action(init_state)
        prime_str += " " + vocab[action_idx]

    print(prime_str)


if __name__ == '__main__':
    main()
