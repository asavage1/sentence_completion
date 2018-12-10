import argparse
import gym
import numpy as np
import unidecode
import re
import random
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import spacy
nlp = spacy.load('en_core_web_md')
word2vec = lambda word: nlp.vocab[word].vector

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    file = re.findall(r"[\w']+", file)
    return file, len(set(file))

def read_file2(filename):
    qa_pairs = []
    with open(filename, 'r') as f:
        qa_pairs = [line.rstrip() for line in f.readlines()]

    qa_pairs = list(zip(qa_pairs, qa_pairs[1:]))
    qa_pairs = [qa_pairs[i] for i in range(0, len(qa_pairs), 2)]

    # training_pairs = []
    # for q, a in qa_pairs:
    #     partial_answer = []
    #     for ans_word in a.split():
    #         training_pairs.append((' '.join([q] + partial_answer), ans_word))
    #         partial_answer.append(ans_word)

    return qa_pairs


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('filename', type=str)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--threshold', type=int, default=1, help='cutoff dist')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)  # TODO: input=300, hidden=?, output=len(vocab)
        self.affine2 = nn.Linear(128, output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


vocab, size = read_file(args.filename)
policy = Policy(300, size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return word2vec(vocab[action.item()])


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def step(state, action, expected_action, done):
    new_state = np.sum([state, action], axis=0)
    dist = np.linalg.norm(np.subtract(new_state, expected_action))

    return new_state, 1 / dist, done


def parse_qa(qa_pairs):
    trainings = []
    for q, a in qa_pairs:
        parse_sent = lambda sent: np.sum(list(map(word2vec, sent)), axis=0)
        trainings.append((parse_sent(q), parse_sent(a)))
    return trainings


def main():
    vocab, vocab_len = read_file(args.filename)
    qa = read_file2(args.filename)[:10]

    training_set = parse_qa(qa)

    running_reward = 10
    for i_episode in count(1):
        rand_index = random.randint(0, len(training_set) - 1)
        random_q, random_a = training_set[rand_index]
        ending_length = len(qa[rand_index][1])

        state = np.array(random_q)  # TODO: state should be 300-dimensional word vector (numpy.ndarray)
        # state = env.reset()
        for t in range(100):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done = step(state, action, random_a, t == ending_length)  # TODO: reward:float done:bool
            # state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
