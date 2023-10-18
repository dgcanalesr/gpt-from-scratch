import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # Sequences processed in paralell
block_size = 256 # Maximum context length for pred
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Seed
torch.manual_seed(77)

# Data
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Characters mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
str_to_int_dict = {ch:i for i,ch in enumerate(chars)}
int_to_str_dict = {i:ch for i,ch in enumerate(chars)}

def encode(string):
    return [str_to_int_dict[char] for char in string]

def decode(list_of_int):
    return "".join([int_to_str_dict[i] for i in list_of_int])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split: str):
    
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        return
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Loss estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# One head self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        weigths = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,C) @ #(B,C,T) --> (B,T,T)
        weigths = weigths.masked_fill(self.tril[:T, :T] == 0, float("-inf")) #(B,T,T)
        weigths = F.softmax(weigths, dim=-1) #(B,T,T)
        weigths = self.dropout(weigths)

        v = self.value(x) #(B,T,C)
        out = weigths @ v #(B,T,T) @ (B,T,C) --> (B,T,C)
        return out


# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Linear layer and non linearity
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


# Transformer block
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


# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.sa_head = Head(n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dim sa -> n_embd dim
        #self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):

        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = token_emb + pos_emb #(B,T,C)
        #x = self.sa_heads(x) # one head of self-attention (B,T,C)
        #x = self.ffwd(x) #(B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) #(B,T,vocab_size) 
        
        if targets is None:
            loss = None
        else:     
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # get predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :] #(B,C)
            probs = F.softmax(logits, dim=-1)
            
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx
    

# Model
model = BigramLanguageModel()
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training
for iter in range(max_iters):

    # Evaluate loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
    
    # Data batch
    xb, yb = get_batch("train")

    # Loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
