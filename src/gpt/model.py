import torch
import torch.nn as nn
from torch.nn import functional as F


# One head self-attention
class Head(nn.Module):
    def __init__(
            self, 
            head_size: int, 
            n_embd: int, 
            block_size: int, 
            dropout: float
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
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
    def __init__(
            self, 
            num_heads: int, 
            head_size: int, 
            n_embd: int, 
            block_size: int,
            dropout: float
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Linear layer and non linearity
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)


# Transformer block
class Block(nn.Module):
    def __init__(
            self,
            n_embd: int, 
            n_head: int, 
            block_size: int, 
            dropout: float
    ):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head, 
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size, 
            dropout=dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# GPT model
class GPTModel(nn.Module):
    def __init__(
            self,
            vocab_size: int, 
            n_layer: int,
            n_embd: int, 
            n_head: int,
            block_size: int, 
            dropout: float,
            device: str
    ):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(
                n_embd=n_embd, 
                n_head=n_head, 
                block_size=block_size, 
                dropout=dropout
            ) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(
            self, 
            idx: torch.tensor, 
            targets: torch.tensor = None
        ) -> (torch.tensor, torch.tensor):

        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) #(T,C)
        x = token_emb + pos_emb #(B,T,C)
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
    
    def generate(self, idx: torch.tensor, max_new_tokens: int) -> torch.tensor:
        
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            
            # get predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :] #(B,C)
            probs = F.softmax(logits, dim=-1)
            
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)

        return idx

