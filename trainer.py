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
# I test this on my personal laptop which does not have a gpu so the cpu will be used
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

# Open the files containing output of using preprocessor
text = open('input.txt', 'r', encoding='utf-8').read().lower()
validation = open('validation.txt', 'r', encoding='utf-8').read().lower()

# file that records testing
testing_file = open('testing.txt', 'w')


def print_and_write_to_file(string_input):
    """ print and write to testing file """

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
chars = sorted(list(set(text + validation)))
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


def encode(string):
    """ encoder: take a string, output a list of integers """

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
            # if character not part of vocabuary, skip it
            if string[string_index] in vocab_to_int:
                encoding.append(vocab_to_int[string[string_index]])
            string_index += 1
    return encoding


def decode(tokens):
    """ decoder: take a encoded list of integers representing tokens output the decoded string """

    decoded = ''.join([int_to_vocab[token] for token in tokens])
    return decoded


# Training and validation data encoded and converted to tensors
train_data = torch.tensor(encode(text), dtype=torch.long)
validation_data = torch.tensor(encode(validation), dtype=torch.long)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self):
        """  """

        super().__init__()
        self.keys = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.queries = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.values = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, head_input):
        """  """

        key = self.keys(head_input)  # (1,BLOCK_SIZE,HEAD_SIZE)
        query = self.queries(head_input)  # (1,BLOCK_SIZE,HEAD_SIZE)
        # compute attention scores ("affinities")
        weight = query @ key.transpose(-2, -1) * HEAD_SIZE**-0.5  # (1, BLOCK_SIZE, HEAD_SIZE) @ (B, HEAD_SIZE, BLOCK_SIZE) -> (1, BLOCK_SIZE, BLOCK_SIZE)
        weight = weight.masked_fill(
            self.tril[:BLOCK_SIZE, :BLOCK_SIZE] == 0,
            float('-inf')
        )  # (1, BLOCK_SIZE, BLOCK_SIZE)
        weight = F.softmax(weight, dim=-1)  # (1, BLOCK_SIZE, BLOCK_SIZE)
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        value = self.values(head_input)  # (1,BLOCK_SIZE,HEAD_SIZE)
        output = weight @ value  # (1, BLOCK_SIZE, BLOCK_SIZE) @ (1, BLOCK_SIZE, HEAD_SIZE) -> (1, BLOCK_SIZE, HEAD_SIZE)
        return output


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        """  """

        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])
        self.project = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        self.LayerNormal = nn.LayerNorm(N_EMBD)

    def forward(self, logits):
        """  """

        output = self.LayerNormal(logits)
        output = torch.cat([h(output) for h in self.heads], dim=-1)
        output = self.dropout(self.project(output))
        return output


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        """  """

        super().__init__()
        self.SelfAttention = MultiHeadAttention()
        self.FeedFoward = nn.Sequential(
            nn.LayerNorm(N_EMBD),
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, logits):
        """  """

        logits = logits + self.SelfAttention(logits)
        logits = logits + self.FeedFoward(logits)
        return logits


class GPTLanguageModel(nn.Module):

    def __init__(self):
        """  """

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
        """  """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context):
        """  """

        # context and targets are both (1,BLOCK_SIZE) tensor of integers
        token_embed = self.token_embedding_table(context)  # (1,BLOCK_SIZE,N_EMBD)
        position_embed = self.position_embedding_table(
            torch.arange(BLOCK_SIZE, device=DEVICE)
        )  # (BLOCK_SIZE,N_EMBD)
        logits = token_embed + position_embed  # (1,BLOCK_SIZE,N_EMBD)
        logits = self.blocks(logits)  # (1,BLOCK_SIZE,N_EMBD)
        logits = self.FinalLayerNormal(logits)  # (1,BLOCK_SIZE,N_EMBD)
        logits = self.lm_head(logits)  # (1,BLOCK_SIZE,vocab_size)

        return logits

    def batch(self, data):
        """ get a batch of training data """

        # load a small batch of the data for inputs context and targets
        data_batch = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

        # load block sized chunks into inputs context and targets
        context_data = []
        targets_data = []
        for data_index in data_batch:
            context_data.append(data[data_index: data_index + BLOCK_SIZE])
            targets_data.append(data[data_index + 1: data_index + BLOCK_SIZE + 1])
        context = torch.stack(context_data)
        context.to(DEVICE)
        targets = torch.stack(targets_data)
        targets.to(DEVICE)

        logits = self.forward(context)

        logits = logits.view(BATCH_SIZE * BLOCK_SIZE, vocab_size)
        targets = targets.view(BATCH_SIZE * BLOCK_SIZE)
        loss = F.cross_entropy(logits, targets)

        return loss

    def generate(self, context, max_new_tokens):
        """ generate tokens after  """

        # context is initally (1, BLOCK_SIZE) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop context to the last BLOCK_SIZE tokens
            context_crop = context[:, -BLOCK_SIZE:]
            # get the predictions
            logits = self.forward(context_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (1, vocab_size)
            # apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)  # (1, vocab_size)
            # sample from the distribution
            context_next_part = torch.multinomial(probabilities, num_samples=1)  # (1, 1)
            # append sampled index to the running sequence
            context = torch.cat((context, context_next_part), dim=1)  # (1, current context length + 1)

        return context

    # this property improves performance for this function
    @torch.no_grad()
    def estimate_loss(self, step, model_index):
        """ estimate the model's loss """

        self.eval()
        losses = torch.zeros(EVAL_ITERS)
        for looper in range(EVAL_ITERS):
            losses[looper] = self.batch(train_data).item()
        out_train = losses.mean()
        losses = torch.zeros(EVAL_ITERS)
        for looper in range(EVAL_ITERS):
            losses[looper] = self.batch(validation_data).item()
        out_validate = losses.mean()
        self.train()

        # print the step and losses
        print_and_write_to_file(f'step {step}: train loss {out_train:.4f}')
        # this is to align it with train loss during printing
        aligment_spacing = ' ' * len(str(step))
        print_and_write_to_file(f'  {aligment_spacing}  validate loss {out_validate:.4f}')

        # save model
        with open(f'model{model_index}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def model_information_printer(self, mode_index=None):
        """ print the model's parameters and index if in testing mode """

        # get number of parameters
        parameter_count = 0
        for parameter in self.parameters():
            parameter_count += parameter.numel()

        # print the number of parameters measured in millions in the model
        # and it's index
        print_and_write_to_file('')
        if mode_index is None:
            print_and_write_to_file(f'model has {parameter_count/1e6} million parameters')
        else:
            print_and_write_to_file(f'model {mode_index} has {parameter_count/1e6} million parameters')
        print_and_write_to_file('')

    def question_answerer(self, question_string):
        """ input question and generate answer """

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
        for looper in range(len(question)):
            context[0][looper] = question[looper]
        generated = self.generate(context, max_new_tokens=25)

        # blank line between separate sets of questions and answers
        print_and_write_to_file('')

        # generated only has 1 element, this is the string with the question and answer
        output_as_string = decode(generated[0].tolist())

        # make sure there are not multiple newlines between separate sets of questions and answers
        while output_as_string[0] == '\n':
            output_as_string = output_as_string[1:]
        while output_as_string[-1] == '\n':
            output_as_string = output_as_string[:-1]
        while '\n\n' in output_as_string:
            output_as_string = output_as_string.replace('\n\n', '')

        # print question and answer
        print_and_write_to_file(output_as_string)


def training_function():
    """ train the model """

    # index for stages
    model_index = 0

    # the model
    model = GPTLanguageModel()
    model.to(DEVICE)

    # print the number of parameters in the model
    model.model_information_printer()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iteration in range(MAX_ITERS):

        # every EVAL_INTERVAL evaluate the loss on train and val sets
        if iteration % EVAL_INTERVAL == 0:
            model.estimate_loss(iteration, model_index)
            model_index += 1

        # evaluate the loss
        model_loss = model.batch(train_data)
        optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        optimizer.step()

    # estimate the loss of the final model
    model.estimate_loss(MAX_ITERS, int(MAX_ITERS/EVAL_INTERVAL))


if mode == 'training':
    # train the model
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
        looper = 0
        while looper < int(len(questions)/2):
            model.question_answerer(questions[looper*2])
            looper += 1

        print_and_write_to_file('')
