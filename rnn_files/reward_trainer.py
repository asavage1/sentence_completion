""" Train with reward function
""" 
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# from train import *
from generate import *
from reward import compute_reward
from torch.autograd import Variable

def reward_train(inp, target, batch_size, chunk_len, lengths):
    hidden = decoder.init_hidden(batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    # loss = 0

    # for c in range(chunk_len):
    #     output, hidden = decoder(inp[:,c], hidden)
    #     loss += criterion(output.view(batch_size, -1), target[:,c])

    sent = generate(decoder, 'Lord', 100, cuda=args.cuda)
    reward = compute_reward(sent)
    lengths.append(len(sent))

    reward.backward()
    decoder_optimizer.step()

    return reward

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '_rewarded.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


# Run as standalone script
# if __name__ == '__main__':
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

decoder = torch.load("obama.pt")
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
file, file_len = read_file(args.filename)
start = time.time()
lengths = []
loss_total = 0

print("Training for %d epochs..." % args.n_epochs)
for epoch in tqdm(range(1, args.n_epochs + 1)):
    loss = reward_train(*random_training_set(args.chunk_len, args.batch_size), args.chunk_len, args.batch_size, lengths)
    loss_total += loss

    if epoch % args.print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss_total/epoch))
        sent = generate(decoder, 'Wh', 100, cuda=args.cuda)
        print(sent, '\n')

plt.plot(lengths)
plt.show()

print("Saving...")
save()

