import torch
import torch.nn as nn

from torch.nn import functional as F


# Bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(
            self, 
            idx: torch.tensor, 
            targets=None
    ) -> (torch.tensor, torch.tensor):
        
        logits = self.token_embedding_table(idx) #(B,T,C)
        
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
            
            # get predictions
            logits, loss = self(idx)
            
            # focus only on the last time step
            logits = logits[:, -1, :] #(B,C)
            probs = F.softmax(logits, dim=-1)
            
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)

        return idx
    



    



