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
import enchant

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
parser.add_argument('--gamma', type=float, default=0.01,
                    help='gamma higher will increase responsilbility of previous actions')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--num_episodes', type=int, default=5,
                    help='num episodes - the number of episodes to train on before weights update')
parser.add_argument('--num_iter', type=int, default=3,
                    help='num_iter - number of times the episode batches are run')
parser.add_argument('--validity', action='store_true',
                    help="Runs the experiment where the reward is for valid words in a sentence.")
parser.add_argument('--short_sent', action='store_true',
                    help="Runs the experiment where the the reward is for shorter sentences.")
parser.add_argument('--long_sent', action='store_true',
                    help="Runs the experiment where the the reward is for shorter sentences.")
parser.add_argument('--sentiment', action='store_true',
                    help="Runs the experiment where the the reward is for positive sentences.")
args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

device = torch.device("cuda" if args.cuda else "cpu")

# load model 
model = torch.load(args.checkpoint, map_location={'cuda:0': 'cpu'})
# put params in continuous chunk of memory for faster forward pass
model.rnn.flatten_parameters() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Loading vocabulary...")
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
print("Finished loading vocabulary")

# Sentiment Classifier
sentiment = SentimentIntensityAnalyzer()

# valid word checker
validity = enchant.Dict("en_US")

TERMINAL_STATES = ['!', '.', '?']

# keep track
avg_lengths = []
lengths = []
validity_scores = []
sentiment_scores = []

def save(iteration):
    # delete previous checkpoints

    if args.validity:
        save_filename = "validity_" + str(iteration) + '.pt'
    elif args.sentiment:
        save_filename = "sentiment_" + str(iteration) + '.pt'
    elif args.short_sent:
        save_filename = "short_sent_" + str(iteration) + '.pt'
    elif args.long_sent:
        save_filename = "long_sent_" + str(iteration) + '.pt'
    else:
        save_filename = "weighted_reward_" + str(iteration) + '.pt'

    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
    
######################################################################
####################### REWARD FUNCTIONS #############################
######################################################################

def compute_reward(action, sent):
    """ Return reward, done
            reward (int)
            done   (bool)

        Args:
            action (word)
            sent    Sentence generated so far 
    """
    if args.long_sent:
        return long_sentences(action)
    if args.short_sent:
        return short_sentences(action)
    if args.validity:
        return validity_checker(action)
    if args.sentiment:
        return sentiment_classifier(action, sent)

    return weighted_reward(action, sent)

def short_sentences(action):
    if action in TERMINAL_STATES:
        return 0.1, True
    else:
        return 0.0, False

def long_sentences(action):
    if action in TERMINAL_STATES:
        return 0.0, True
    else:
        return 0.1, False

def sentiment_classifier(action, sent):
    score = sentiment.polarity_scores(action)
    reward = score['compound']
    if action in TERMINAL_STATES:
        done = True
    else:
        # reward = 0.0
        done = False
    return reward, done

def validity_checker(action):
    if (validity.check(action)):
        reward = 0.1
    else:
        reward = -0.1

    # always return done=False, we let the main program
    # decide how long the episode is going to be
    return reward, False

def weighted_reward(action, sent):
    sent_score, done = sentiment_classifier(action, sent)
    validity_score, _ = validity_checker(action)
    repetion_score, _ = avoid_repetition(action, sent)
    # if we don't have a shortness metric, then the agent 
    # starts lengthening the sentence to reduce repetition
    shortness_score, _ = short_sentences(action)

    sent_weight = 0.0
    validity_weight = 0.4
    repetition_weight = 0.5
    shortness_weight = 0.1

    reward = sent_weight * sent_score \
           + validity_weight * validity_score \
           + repetition_weight * repetion_score \
           + shortness_weight * shortness_score
    return reward, done

def avoid_repetition(action, sent):
    sent = sent + action
    words = sent.split(' ')

    # the higher the score, the worse the repetition
    reward = len(set(words)) / len(words)

    if action in TERMINAL_STATES:
        done = True 
    else:
        done = False   
        
    return  reward, done

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
for j in range(args.num_iter):
    loss_bank = [] # stores the losses of all episodes of an epoch
    state = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    
    # generate multiple episodes to reduce variance
    for i in range(args.num_episodes):
        num_actions = 0
        done = False
        policy_rewards = []
        action_probs = []
        sent = ''
        model.train() # this allows params to be trained

        # keep selecting until the episode ends or 100 actions
        while not done and num_actions < 100:
            num_actions += 1
            # action prob is a log probability
            action, action_prob, word_idx = select_action(state)
            policy_reward, done = compute_reward(action, sent)
            policy_rewards.append(policy_reward)
            action_probs.append(action_prob)
            sent += action + ' '
            state = torch.tensor(np.array([[word_idx.item()]]))

        # average of score in epoch
        if args.validity:
            validity_scores.append(sum(policy_rewards)/len(policy_rewards))
        elif args.sentiment:
            sentiment_scores.append(sum(policy_rewards)/len(policy_rewards))

        # leave the eligibility trace
        R = 0
        rewards = []
        for r in policy_rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

        # subtract mean as a baseline for stability
        rewards = (rewards - rewards.mean()) 

        # - log_prob * reward to make loss inverse of reward.
        policy_loss = []
        for log_prob, reward in zip(action_probs, rewards):
            policy_loss.append(-log_prob * reward * args.lr)
        policy_loss = torch.sum(torch.stack(policy_loss))

        loss_bank.append(policy_loss) 

        # len(action_probs) = num words in episode
        lengths.append(len(action_probs)) 
        avg_lengths.append(sum(lengths)/len(lengths))

        # log at the end of the epoch
        if i == args.num_episodes - 1:
            print("Iteration %d, %s" % (j, sent))
            print("Average lengths: %2.2f" % (avg_lengths[-1]) )
            if args.validity:
                print("Validity_score: %2.2f" % (sum(policy_rewards)/len(policy_rewards)))
            elif args.sentiment:
                print("Sentiment score: %2.2f" % (sum(policy_rewards)/len(policy_rewards)))

    # backprop all of the episodes at end of epoch
    for policy_loss in loss_bank:
        model.zero_grad()
        policy_loss.backward()
        # crucial: this prevents the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

    
save(args.num_iter)  # save the current model as a checkpoint  

if args.sentiment:
    plt.plot(sentiment_scores)
    plt.title("Average Sentiment Score in an episode")
    plt.xlabel("Number of episodes")
    plt.ylabel("Average sentiment score in episode")
    plt.savefig('sentiment.jpg')

if args.short_sent:
    plt.plot(avg_lengths)
    plt.title("Reward : 0.1 if action is end of sentence, 0.0 otherwise")
    plt.xlabel("Number of episodes")
    plt.ylabel("Average length of sentences (in words)")
    plt.savefig('short_lengths.jpg')

if args.long_sent:
    plt.plot(avg_lengths)
    plt.title("Reward : 0.0 if action is end of sentence, 0.1 otherwise")
    plt.xlabel("Number of episodes")
    plt.ylabel("Average length of sentences (in words)")
    plt.savefig('long_lengths.jpg')

if args.validity:
    plt.plot(validity_scores)
    plt.title("Reward : 0.1 if action a valid word, -0.1 otherwise")
    plt.xlabel("Number of iterations")
    plt.ylabel("Average validity score in the iterations")
    plt.savefig('validity.jpg')
