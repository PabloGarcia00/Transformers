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
batch_size=32
block_size=8
learning_rate=1e-2
max_iter = 10_000
eval_iters = 100
torch.manual_seed(1337)


# splitting the data 
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data = train_data if split == "train_data" else val_data # which data are we looking at
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate random start sites for n_batches which are off-set by block_size otherwise system will break later due to insufficient positions after startsite
    x = torch.stack([data[i:i+block_size] for i in ix]) # the context
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # target word
    return x.to(device), y.to(device)

# the model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads oof the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # nn.Embedding is a wrapper around the function that creates a matrix of vocab_size x vocab_size
    
    def forward(self, idx, targets=None):
        # idx and the targets are Batch(B) by Time (T) tensors of integers
        logits = self.token_embedding_table(idx) # pytorch handles this in such a manner that the matrix that follow take Batch x Time x Channels 
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
    
    def generate(self, idx, max_new_tokens=500):
        # idx is BxT matrix so batch plus timesteps (characters)
        for _ in range(max_new_tokens):
            # generate probabilities 
            logits, loss = self(idx)
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
print(decode(m.generate(init_context)[0].tolist()))