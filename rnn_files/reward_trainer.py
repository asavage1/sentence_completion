""" Train with reward function
""" 
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from helpers import *
from model import *
from reward import compute_reward
from torch.autograd import Variable

TERMINAL_STATES = [".", "?", "!"]

def reward_train(decoder, lengths):
    """ ok this is going to look very similar to our generate function
        generate sentence, compute reward, backpropagate
    """ 
    eps = np.finfo(np.float32).eps.item()
    sent, log_probs, policy_rewards = generate(decoder, 'Wh', 100, cuda=args.cuda) 
    lengths.append(len(sent))

    R = 0
    policy_loss = []
    rewards = []
    for r in policy_rewards[::-1]:
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards, requires_grad=True)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
    return 1
    
def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '_rewarded.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

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
        # this gives us the new output vector from our current policy
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        log_prob = math.log(output_dist[top_i])
        log_probs.append(log_prob)

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]

        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

        # the episode is over if sentence ends
        if predicted_char in TERMINAL_STATES:
            rewards.append(10000000000)
            break
        else:
            rewards.append(-2000)

    return predicted, log_probs, rewards

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--chunk_len', type=int, default=20)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--learning_rate', type=float, default=0.01)
args = argparser.parse_args()


decoder = torch.load(args.filename)
optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

start = time.time()
lengths = []
loss_total = 0

print("Training for %d epochs..." % args.n_epochs)
for epoch in tqdm(range(1, args.n_epochs + 1)):
    loss = reward_train(decoder, lengths)
    loss_total += loss

    if epoch % args.print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss_total/epoch))
        sent, _ , _= generate(decoder, 'Wh', 100, cuda=args.cuda)
        print(sent, '\n')

plt.plot(lengths)
plt.show()

print("Saving...")
save()

