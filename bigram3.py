# imports 
import torch
import torch.nn as nn 
from torch.nn import functional as F

# load data
with open("dutch.txt", "r", encoding="latin-1") as f:
    text = ""
    i=0
    for line in f:
        i+=1
        if i > 54:
            if line != "\n":
                text += line.strip() + "\n"

# construct dictionary
chars = sorted(list(set(text)))
vocab_size = (len(chars))
# encode each char by mapping char to integer, manny other methods are possible
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

# setting up cuda 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# encode and turn into tensor
data = torch.tensor(encode(text), dtype=torch.long)

# hyperparameters 
batch_size=64
block_size=256
learning_rate=3e-3
max_iter = 1_000
eval_iters = 200
torch.manual_seed(1337)
n_embd = 128
head_size = 16
num_heads = 8
dropout = 0.2

# splitting the data 
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data = train_data if split == "train_data" else val_data # which data are we looking at
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate random start sites for n_batches which are off-set by block_size otherwise system will break later due to insufficient positions after startsite
    x = torch.stack([data[i:i+block_size] for i in ix]) # the context
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # target word
    return x.to(device), y.to(device)

# the head model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # bias is typically not used
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # creating a tril variable which is not a parameter thus is assigned to a buffer
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x) 
        affin = q @ k.transpose(-1, -2) 
        affin = affin * C**-0.5 
        wei = affin.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return self.proj(x)
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # this can be done just before the influx from the residual connection
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# the model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads oof the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # nn.Embedding is a wrapper around the function that creates a matrix of vocab_size x embedding_size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(
            Block(n_embd, num_heads),
            Block(n_embd, num_heads),
            Block(n_embd, num_heads),
            nn.LayerNorm(n_embd),
        )
        # self.sa_heads = MultiHeadAttention(num_heads, n_embd//num_heads) # embedded vector is splitted into num_heads pieces 
        # self.ffwd = FeedForward(n_embd) # allows for more processing of the attention head outputs
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer that maps latent vector back to output vector with of length vocab_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and the targets are Batch(B) by Time (T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # pytorch handles this in such a manner that the matrix that follow take Batch x Time x Channels(embd_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # vector of length block_size containig positional embeddings
        x = tok_emb + pos_emb # compiles both embeddings into 1 
        x = self.block(x)
        logits = self.lm_head(x) # map the latent vector of size (B x T x embd_size) back to logits which is (BxTx vocab_size)
        # from the BTC matrix the word can be read        
        if targets is None:
            loss = None
        else:
            # we use cross-entropy loss which takes in a vector of BCT istead of BTC
            # how to transform the dimensions
            #logits = torch.movedim(logits, 2, 1)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens=256):
        # idx is BxT matrix so batch plus timesteps (characters)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens so that no more than block_size will come in
            idx_cond = idx[:, -block_size:]

            # generate probabilities 
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            p = F.softmax(logits, dim=-1)
            # sample from probabilities 
            idx_next = torch.multinomial(p, num_samples=1, replacement=True)

            # concatenate it to the the input BxT matrix so BxT+1
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

# declare model
m = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# loss estimation function in no_grad context 
@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y  = get_batch(split)
            logits, loss = m(X,Y)
            losses[k] = loss
        out[split] = losses.mean()
    m.train()
    return out

# training
for iter in range(max_iter):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"at iter{iter} train loss is: {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

init_context = torch.zeros((1,1), dtype=torch.long, device=device)
new_text = decode(m.generate(init_context, max_new_tokens=1000)[0].tolist())


with open("runs.txt", 'a') as f:
    f.write(new_text + "\n\n" + "#" * 100 + "\n\n")
        

