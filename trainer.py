import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
device = 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
head_size = n_embd // n_head

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

# here are all the unique characters that occur in this text and common words
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
chars = sorted(list(set(text)))
chars += words

vocab_size = len(chars)
# create a mapping from characters to integers
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    # encoder: take a string, output a list of integers
    encoding = []
    c = 0
    while c < len(s):
        word_index = 0
        while word_index < len(words):
            if s[c:c+len(words[word_index])] == words[word_index]:
                encoding.append(char_to_int[words[word_index]])
                c += len(words[word_index])
                break
            word_index += 1
        if word_index == len(words):
            encoding.append(char_to_int[s[c]])
            c += 1
    return encoding

def decode(l):
    # decoder: take a list of integers, output a string
    decoded = ''.join([int_to_char[i] for i in l])
    return decoded

# Training data
train_data = torch.tensor(encode(text), dtype=torch.long)

del text

# data loading
def get_batch():
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(train_data) - block_size, (batch_size,))

    x_data = []
    y_data = []
    for i in ix:
        j = 0
        while decode([train_data[i+j+1].tolist()]) == "\n":
            j += 1
        while decode([train_data[i+j+block_size].tolist()]) == "\n":
            j -= 1
        x_data.append(train_data[i+j:i+j+block_size])
        y_data.append(train_data[i+j+1:i+j+block_size+1])
    x = torch.stack(x_data)
    x.to(device)
    y = torch.stack(y_data)
    y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        logits, loss = model.batch()
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self):
        super().__init__()
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.queries = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        key = self.keys(x)   # (B,T,hs)
        query = self.queries(x) # (B,T,hs)
        # compute attention scores ("affinities")
        weight = query @ key.transpose(-2,-1) * head_size**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[:block_size, :block_size] == 0, float('-inf')) # (B, T, T)
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        value = self.values(x) # (B,T,hs)
        output = weight @ value # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return output

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.project = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.LayerNormal = nn.LayerNorm(n_embd)

    def forward(self, x):
        out = self.LayerNormal(x)
        out = torch.cat([h(out) for h in self.heads], dim=-1)
        out = self.dropout(self.project(out))
        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.SelfAttention = MultiHeadAttention()
        self.FeedFoward = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.SelfAttention(x)
        x = x + self.FeedFoward(x)
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.FinalLayerNormal = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
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
        token_embed = self.token_embedding_table(idx) # (B,T,C)
        position_embed = self.position_embedding_table(torch.arange(block_size, device=device)) # (T,C)
        x = token_embed + position_embed # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.FinalLayerNormal(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits

    def batch(self):
        # sample a batch of data
        idx, targets = get_batch()

        logits = self(idx)

        logits = logits.view(batch_size * block_size, vocab_size)
        targets = targets.view(batch_size * block_size)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: loss {losses:.4f}")

    # evaluate the loss
    logits, loss = model.batch()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
question = encode('question: "why does death exist?"\nanswer: "i don\'t know')
while len(question) < block_size:
    question.insert(0, encode('\n')[0])
context = torch.zeros((1, len(question)), dtype=torch.long, device='cpu')
for x in range(len(question)):
    context[0][x] = question[x]
e = model.generate(context, max_new_tokens=25)
for t in e :
    print(decode(t.tolist()))
