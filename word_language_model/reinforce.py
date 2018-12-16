import argparse
import sys
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import data
import numpy as np
import torch.optim as optim
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

parser = argparse.ArgumentParser()

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

device = torch.device("cuda" if args.cuda else "cpu")
TERMINAL_STATES = ['!', '.', '?']

# load model 
model = torch.load('model.pt', map_location={'cuda:0': 'cpu'})
# put params in continuous chunk of memory for faster forward pass
model.rnn.flatten_parameters() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Loading vocabulary...")
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
print("Finished loading vocabulary")

################### TODO: Make these commandline args ###############
num_episodes = 5
num_epochs = 5
gamma = 0.01
#####################################################################

# Sentiment Classifier
sentiment = SentimentIntensityAnalyzer()

# keep track
avg_lengths = []
lengths = []

def compute_reward(action):
    """ Return reward, done
            reward (int)
            done   (bool)
    """
    # return long_sentences(action)
    return short_sentences(action)
    # return sentiment_classifier(action)
    
######################################################################
####################### REWARD FUNCTIONS #############################
######################################################################
def short_sentences(action):
    if action in TERMINAL_STATES:
        return 0.1, True
    else:
        return 0, False

def long_sentences(action):
    if action in TERMINAL_STATES:
        return 0, True
    else:
        return 0.1, False

def sentiment_classifier(action):
    score = sentiment.polarity_scores(action)
    reward = score['compound']

    if action in TERMINAL_STATES:
        done = True
    else:
        done = False

    return reward, done

######################################################################
######################################################################
######################################################################
def select_action(state):
    output, hid = model(state, hidden)
    word_weights = output.squeeze().div(args.temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    word = corpus.dictionary.idx2word[word_idx]
    log_prob = torch.log(word_weights[word_idx] / torch.sum(word_weights))
    return word, log_prob, word_idx

# backpropagation is done at the end of every epoch
for j in range(num_epochs):
    loss_bank = [] # stores the losses of all episodes of an epoch
    state = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    
    # generate multiple episodes to reduce variance
    for i in range(num_episodes):
        done = False
        policy_rewards = []
        action_probs = []
        sent = ''
        model.train() # this allows params to be trained

        # keep selecting actions until the episode ends
        while not done:
            # action prob is a log probability
            action, action_prob, word_idx = select_action(state)
            policy_reward, done = compute_reward(action)
            policy_rewards.append(policy_reward)
            action_probs.append(action_prob)
            sent += action + ' '
            state = torch.tensor(np.array([[word_idx.item()]]))

        # leave the eligibility trace
        R = 0
        rewards = []
        for r in policy_rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

        # subtract mean as a baseline for stability
        rewards = (rewards - rewards.mean()) 

        # - log_prob * reward to make loss inverse of reward.
        policy_loss = []
        for log_prob, reward in zip(action_probs, rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.sum(torch.stack(policy_loss))

        loss_bank.append(policy_loss) 

        # len(action_probs) = num words in episode
        lengths.append(len(action_probs)) 
        avg_lengths.append(sum(lengths)/len(lengths))

        # log at the end of the epoch
        if i == num_episodes - 1:
            print("Epoch %d, %s" % (j, sent))
            print("Average lengths: %2.2f" % (avg_lengths[-1]) )

    # backprop all of the episodes at end of epoch
    for policy_loss in loss_bank:
        model.zero_grad()
        policy_loss.backward()
        # crucial: this prevents the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

plt.plot(avg_lengths)
plt.title("Reward : 0.1 if action is end of sentence, 0.0 otherwise")
plt.xlabel("Number of episodes")
plt.ylabel("Average length of sentences (in words)")
plt.savefig('lengths.jpg')
