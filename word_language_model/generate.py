###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import sys
import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--words', type=int, default='100',
                    help='number of words to generate')
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

model = torch.load(args.checkpoint, map_location={'cuda:0': 'cpu'})
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

sent = ''
num_words = args.words
with torch.no_grad():  # no tracking history
    for i in range(num_words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]

        # updating the value of input
        input.fill_(word_idx)

        word = corpus.dictionary.idx2word[word_idx]
        sent += word + ' '

print(sent)