import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch.amp import autocast
from transformers import GPT2Tokenizer

# Since the attention mechanism is quite complex, I did write some notes about it.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, offset: int = 0):
        B, T, _ = x.size()
        positions = torch.arange(T, device=x.device) + offset
        positions = positions.unsqueeze(0).expand(B, T)
        return x + self.pos_embedding(positions)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class FlashAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, max_len: int, dropout: int = 0.0, kv_caching: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.scaling = embed_dim ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.kv_caching = kv_caching
        self.v_cache = None
        self.k_cache = None
        self.max_len = max_len
        self.cache_index = 0

    def forward(self, x : torch.Tensor):
        B, T_in, D = x.shape
        H, head_dim = self.num_heads, D // self.num_heads

        # project & split
        qkv = self.qkv_proj(x).view(B, T_in, H, 3, head_dim)
        q, k, v = qkv[...,0,:], qkv[...,1,:], qkv[...,2,:]    # each (B, T_in, H, head_dim)
        """
        This section is quite interesting. So let's break it down:
        The goal is to split qkv into q, k, and v tensors.
        So the intruction qkv[...,i,:] means thet I'm accessing to the dimention corresponding to the i-th matrix.
        So qkv[...,0,:] means I'm accessing to the first matrix of the qkv tensor which is the query matrix and so on.

        I personally find this quite interesting since it is a clver way to access the matrices.
        This is way I used it here so I can remember it and use it in the future.
        ....
        """

        q, k, v = [t.permute(0,2,1,3) for t in (q,k,v)]       # now (B, H, T_in, head_dim)

        # As the name suggests, this is the KV cache part
        if self.kv_caching:
            if self.k_cache is None:
                # Initialize caches
                self.k_cache = torch.zeros(B, H, self.max_len, head_dim, device=x.device)
                self.v_cache = torch.zeros(B, H, self.max_len, head_dim, device=x.device)
                self.cache_index = 0

            end = self.cache_index + T_in
            if end <= self.max_len:
                self.k_cache[:, :, self.cache_index:end, :] = k
                self.v_cache[:, :, self.cache_index:end, :] = v
                self.cache_index = end
            else:
                shift = end - self.max_len
                self.k_cache = torch.roll(self.k_cache, -shift, dims=2)
                self.v_cache = torch.roll(self.v_cache, -shift, dims=2)
                self.k_cache[:, :, -T_in:, :] = k
                self.v_cache[:, :, -T_in:, :] = v
                self.cache_index = self.max_len

            k = self.k_cache[:, :, :self.cache_index, :]
            v = self.v_cache[:, :, :self.cache_index, :]


        T_k = k.size(2)    # total key/value length
        T_q = q.size(2)    # query length

        if x.device.type == 'cuda':
            # This is flash attention, you can also implement FlashAttention2 or FlashAttention3, or even XFormers
            # But since this is a simple implementation, I will remain simple and use FlashAttention
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
                attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout.p,
                    is_causal=True,
                    scale=self.scaling
                )
        else:
            # This is a simple implementation of attention since flash attention is not available on CPU
            mask = torch.triu(
                torch.full((T_q, T_k), -1e9, device=x.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0) 

            scores = torch.matmul(q, k.transpose(-2,-1)) * self.scaling 
            scores = scores + mask
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        T_out = attn.size(2) 
        attn = attn.permute(0,2,1,3).reshape(B, T_out, D)

        out = self.out_proj(self.dropout(attn))
        return out
    

    
class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout:int = 0.1):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.w2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, 4 * n_embd, bias=False)

    def forward(self, x : torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



class Block(nn.Module):
    def __init__(self, n_heads: int, n_embd: int, max_len: int, dropout: int = 0.1, kv_caching: bool = False):
        super(Block, self).__init__()
        self.attention = FlashAttention(n_heads, n_embd, max_len, dropout, kv_caching=kv_caching) 
        self.ff = MLP(n_embd, dropout)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        
    def forward(self, x: torch.Tensor):
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self,
                n_layers: int, 
                n_heads: int, 
                n_embd: int, 
                vocab_size: int, 
                max_len:int = 5000, 
                dropout:int = 0.1,
                kv_caching: bool = False):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = PositionalEncoding(n_embd, max_len)
        self.blocks = nn.Sequential(*[Block(n_heads, n_embd, max_len, dropout, kv_caching) for _ in range(n_layers)])
        self.ln_f = RMSNorm(n_embd)
        self.fc_out = nn.Linear(n_embd, vocab_size)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.max_length = max_len
        self.kv_caching = kv_caching
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights according to module type."""
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)

    def forward(self, x : torch.Tensor):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.blocks:
            if self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x) 
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids : torch.Tensor, max_new_tokens : int):
        """
        Generate new tokens given an input sequence, using KV-caching.
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)
        x = self.positional_encoding(x, offset=0)
        for i, block in enumerate(self.blocks):
            x = block(x)

        generated = input_ids
        for step in range(max_new_tokens):
            last_token = generated[:, -1:].to(device)
            offset     = generated.shape[1] - 1

            x = self.embedding(last_token)
            x = self.positional_encoding(x, offset=offset)

            for i, block in enumerate(self.blocks):
                x = block(x)

            x = self.ln_f(x)
            logits = self.fc_out(x)
            next_token = torch.argmax(logits[:, -1:], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.squeeze(0)], dim=1)

        return generated
    

def generate_texts(
        model: Transformer,
        tokenizer: GPT2Tokenizer, 
        prompts: str, 
        gen_len:int = 50, 
        temperature:float = 1.0, 
        device: str = 'cpu', 
        miwd_precision: bool = False):
    """"
    Generate text using the model.
    """
    model.eval()
    model.to(device)
    input_ids = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(gen_len):
            if miwd_precision:
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids)
            else:
                logits = model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated = torch.cat([generated, next_token], dim=1)
            if input_ids.size(1) > model.max_length:
                input_ids = input_ids[:, -model.max_length:]
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text

