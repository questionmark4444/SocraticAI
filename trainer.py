import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import json

# These constants are the hyperparameters
# The batch size is how many independent sequences will we process in parallel
BATCH_SIZE = 64
# The maximum context length for predictions
BLOCK_SIZE = 128
# Maximum interations of training
MAX_ITERS = 3000
# Every 100 iterations of training, print loss and save stage
EVAL_INTERVAL = 100
# The rate of learning
LEARNING_RATE = 3e-4
# I test this on my personal laptop which does not have a gpu
#  so the cpu will be used
DEVICE = 'cpu'
EVAL_ITERS = 200
# The dimension of embedding
N_EMBD = 384
# The number of self-attention heads
N_HEAD = 6
# The number of block layers
N_LAYER = 6
# This helps reduce overfitting by peridically setting some weights to zero
DROPOUT = 0.2

# The size of self-attention heads
HEAD_SIZE = N_EMBD // N_HEAD

# Open file containing output of using preprocessor
# Note this will likely be changed in a later commit
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

testing_file = open('testing.txt', 'w')


# print and write to testing file
def print_and_write_to_file(string_input):
    print(string_input)
    testing_file.write(f'{string_input}\n')


# Ask user if they want to retrain the models
mode = input('retrain the models (y/n): ')
testing_file.write(f'retrain the models (y/n): {mode}\n')
if (mode == 'y'):
    mode = 'training'
else:
    mode = 'None'

# Manually set seed for ai for consistent results during testing and such
torch.manual_seed(1337)

# Here are all the words that are common for every question and answer
# they will be converted to singular tokens
# to improve performance without increasing block size
words = [
    'question: "',
    'what',
    'why',
    'where',
    'how',
    'should',
    '?"\nanswer: "i don\'t know',
    'could',
    'you',
    'explain',
    'the',
    'concept',
    'of'
]
# each known letter, number, and special character as a token
chars = sorted(list(set(text)))
# all tokens
vocabulary = chars + words

# size of vocabuary
vocab_size = len(vocabulary)
# create encoding mapping from characters to integers
# and decoding mapping from integers to characters
vocab_to_int = {}
int_to_vocab = {}
for i, ch in enumerate(vocabulary):
    vocab_to_int[ch] = i
    int_to_vocab[i] = ch


# encoder: take a string, output a list of integers
def encode(string):
    encoding = []
    string_index = 0
    while string_index < len(string):
        word_index = 0

        # check if next part is a word in the list of words
        while word_index < len(words):
            word = words[word_index]
            if string[string_index: string_index + len(word)] == word:
                encoding.append(vocab_to_int[word])
                string_index += len(word)
                break
            word_index += 1

        # append single character token if no word was added
        if word_index == len(words):
            encoding.append(vocab_to_int[string[string_index]])
            string_index += 1
    return encoding


# decoder: take a encoded list of integers representing tokens
#  output the decoded string
def decode(tokens):
    decoded = ''.join([int_to_vocab[token] for token in tokens])
    return decoded


# Training data encoded and converted to tensor
train_data = torch.tensor(encode(text), dtype=torch.long)

# delete raw input data from memory
del text


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self):
        super().__init__()
        self.keys = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.queries = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.values = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # size of input (batch, time-step, channels)
        # size of output (batch, time-step, head size)
        key = self.keys(x)  # (B,T,hs)
        query = self.queries(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        weight = query @ key.transpose(-2, -1) * HEAD_SIZE**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weight = weight.masked_fill(
            self.tril[:BLOCK_SIZE, :BLOCK_SIZE] == 0,
            float('-inf')
        )  # (B, T, T)
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        value = self.values(x)  # (B,T,hs)
        output = weight @ value  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return output


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])
        self.project = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        self.LayerNormal = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        out = self.LayerNormal(x)
        out = torch.cat([h(out) for h in self.heads], dim=-1)
        out = self.dropout(self.project(out))
        return out


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        super().__init__()
        self.SelfAttention = MultiHeadAttention()
        self.FeedFoward = nn.Sequential(
            nn.LayerNorm(N_EMBD),
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.SelfAttention(x)
        x = x + self.FeedFoward(x)
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits
        #  for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.FinalLayerNormal = nn.LayerNorm(N_EMBD)  # final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        # better init, not covered in the original GPT video
        #  note watch newer video that explains to make a better comment
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx)  # (B,T,C)
        position_embed = self.position_embedding_table(
            torch.arange(BLOCK_SIZE, device=DEVICE)
        )  # (T,C)
        x = token_embed + position_embed  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.FinalLayerNormal(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        return logits

    def batch(self):
        # load a small batch of the training data for inputs idx and targets
        ix = torch.randint(len(train_data) - BLOCK_SIZE, (BATCH_SIZE,))

        # load block sized chunks into inputs idx and targets
        idx_data = []
        targets_data = []
        for i in ix:
            idx_data.append(train_data[i: i + BLOCK_SIZE])
            targets_data.append(train_data[i + 1: i + BLOCK_SIZE + 1])
        idx = torch.stack(idx_data)
        idx.to(DEVICE)
        targets = torch.stack(targets_data)
        targets.to(DEVICE)

        logits = self.forward(idx)

        logits = logits.view(BATCH_SIZE * BLOCK_SIZE, vocab_size)
        targets = targets.view(BATCH_SIZE * BLOCK_SIZE)
        loss = F.cross_entropy(logits, targets)

        return loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    # this property improves performance for this function
    @torch.no_grad()
    # estimate the model's loss
    def estimate_loss(self, step, model_number):
        self.eval()
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            losses[k] = self.batch().item()
        out = losses.mean()
        self.train()

        print_and_write_to_file(f'step {step}: loss {out:.4f}')

        # save model
        with open(f'model{model_number}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def model_information_printer(self, mode_index=None):
        # get number of parameters
        parameter_count = 0
        for p in self.parameters():
            parameter_count += p.numel()

        # print the number of parameters measured in millions in the model
        # and it's index
        print_and_write_to_file('')
        if mode_index is None:
            print_and_write_to_file(f'model has {parameter_count/1e6} million parameters')
        else:
            print_and_write_to_file(f'model {mode_index} has {parameter_count/1e6} million parameters')
        print_and_write_to_file('')

    def question_answerer(self, question_string):
        # encode question
        question = encode(
            f'question: "{question_string}"\nanswer: "i don\'t know'
        )
        # when the block is not at least BLOCK_SIZE, it will crash
        # add newline characters as placeholders, like the training data
        while len(question) < BLOCK_SIZE:
            question.insert(0, encode('\n')[0])

        # generate from the model
        context = torch.zeros(
            (1, len(question)),
            dtype=torch.long,
            device=DEVICE
        )
        for x in range(len(question)):
            context[0][x] = question[x]
        e = self.generate(context, max_new_tokens=25)
        for t in e:
            print_and_write_to_file('')
            print_and_write_to_file(decode(t.tolist()).replace('\n\n', ''))


def training_function():
    # train the model, moved into a function to free memory after finishing

    model_number = 0

    model = GPTLanguageModel()
    model.to(DEVICE)

    # print the number of parameters in the model
    model.model_information_printer()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):

        # every EVAL_INTERVAL evaluate the loss on train and val sets
        if iter % EVAL_INTERVAL == 0:
            model.estimate_loss(iter, model_number)
            model_number += 1

        # evaluate the loss
        model_loss = model.batch()
        optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        optimizer.step()

    # estimate the loss of the final model
    model.estimate_loss(MAX_ITERS, int(MAX_ITERS/EVAL_INTERVAL))


if mode == 'training':
    # train model
    training_function()

# ask user if they want to test the models
mode = input('test the models (y/n): ')
testing_file.write(f'test the models (y/n): {mode}\n')
if (mode == 'y'):
    mode = 'testing'
else:
    mode = 'None'

# input question from user for only last model
if mode != 'testing':
    # load model
    with open(f'model{int(MAX_ITERS/EVAL_INTERVAL)}.pkl', 'rb') as f:
        model = pickle.load(f)
    model.to(DEVICE)

    # print the number of parameters in the model
    model.model_information_printer()

    # input question from user and get answer from model
    user_question = input('question: ').lower()
    testing_file.write(f'question: {user_question}\n')
    model.question_answerer(user_question)

# test questions from file for the saved model stages
else:
    # load testing questions from file
    questions = json.load(open('testing.json', 'r'))

    # loop over each model file
    for model_index in range(0, int(MAX_ITERS/EVAL_INTERVAL)+1):
        # load model file
        with open(f'model{model_index}.pkl', 'rb') as f:
            model = pickle.load(f)
        model.to(DEVICE)

        # print the number of parameters in the model and it's index
        model.model_information_printer(model_index)

        # test model with each question
        for x in range(len(questions)):
            model.question_answerer(questions[x])

        print_and_write_to_file('')
