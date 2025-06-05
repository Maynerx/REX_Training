import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch.amp import autocast
from transformers import GPT2Tokenizer
from megablocks.layers.moe import MoE as MoE_
from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss
from megablocks.layers.arguments import Arguments
# Since the attention mechanism is quite complex, I did write some notes about it.

class Gate(nn.Module):
    def __init__(self,
                 dim: int,
                 num_experts: int,
                 topk: int,
                 score_fn: str = 'softmax',
                 route_scale: float = 1.0,
                 use_bias: bool = True):
        """
        A “top-k” gating module.  Expects input x of shape (N, dim),
        produces either (topk_vals, topk_idx) or (topk_vals, topk_idx, dense_gate),
        where:
          - topk_vals:  (N, topk)
          - topk_idx:   (N, topk)
          - dense_gate: (N, num_experts)

        If x has more than 2 dims (e.g. (batch, seq_len, dim)), you must flatten
        it first: x_flat = x.view(-1, dim), then reshape outputs afterward.
        """
        super(Gate, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.topk = topk
        self.score_fn = score_fn.lower()
        self.route_scale = route_scale

        # Weight: (num_experts, dim)
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, return_dense: bool = False):
        """
        Args:
            x: (N, dim)  ← must be 2D.  If your real data is (batch, seq_len, dim),
                 you should call Gate on x_flat = x.view(-1, dim).
            return_dense (bool): if True, also returns a full (N, num_experts) tensor.
        
        Returns:
            topk_vals:  (N, topk)
            topk_idx:   (N, topk)
            [dense_gate: (N, num_experts)]  # only if return_dense=True
        """
        N = x.size(0)  # e.g. N = batch_size * seq_len if you flattened
        # 1) Compute raw logits: (N, num_experts)
        logits = F.linear(x, self.weight, self.bias)  # (N, num_experts)

        # 2) Score function
        if self.score_fn == 'softmax':
            scaled_logits = logits * self.route_scale
            probs = F.softmax(scaled_logits, dim=-1)
        elif self.score_fn == 'sigmoid':
            raw = torch.sigmoid(logits * self.route_scale)
            probs = raw / (raw.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            raise ValueError(f"Unknown score_fn = {self.score_fn}")

        # 3) Pick top-k: both (N, topk)
        topk_vals, topk_idx = torch.topk(probs, self.topk, dim=-1)

        # 4) (Optional) build a dense gating if requested
        if return_dense:
            # Now dense_gate shape must be (N, num_experts), to match topk_idx of shape (N, topk).
            dense_gate = probs.new_zeros(N, self.num_experts)
            dense_gate.scatter_(1, topk_idx, topk_vals)
            return topk_vals, topk_idx, dense_gate

        return topk_vals, topk_idx
    

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    

class MoE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_experts: int,
                 topk: int = 1,
                 score_fn: str = 'softmax',
                 route_scale: float = 1.0,
                 use_bias: bool = True):
        super(MoE, self).__init__()
        self.gate = Gate(input_dim, num_experts, topk, score_fn, route_scale, use_bias)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.topk = topk

    def forward(self, x: torch.Tensor, return_dense: bool = False):
        B, L, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected last dim = {self.input_dim}, but got {D}.")

        N = B * L
        orig_shape = (B, L)

        # 1) Flatten to (N, D)
        x_flat = x.contiguous().view(N, D)  # (N, input_dim)

        # 2) Run Gate → topk_vals, topk_idx, dense_gating
        #    topk_vals:  (N, topk)
        #    topk_idx:   (N, topk)
        #    dense_gating: (N, num_experts)
        topk_vals, topk_idx, dense_gating = self.gate(x_flat, return_dense=True)

        E = self.num_experts
        O = self.output_dim
        k = self.topk

        # 3) Prepare an (N, output_dim) tensor to accumulate results
        device = x.device
        out_flat = x_flat.new_zeros(N, O)  # will hold ∑ over experts

        # 4) Flatten the (token, expert_slot) pairs:
        #    - token_indices: [0,0,0,...,1,1,1,...,2,2,2,..., N-1,...] repeated k times
        #    - expert_indices: the expert-each-slot for every token
        #    - gate_weights:   the gate weight for each (token, slot) pair
        token_indices = torch.arange(N, device=device).unsqueeze(1).repeat(1, k).view(-1)   # shape = (N*k,)
        expert_indices = topk_idx.view(-1)   # shape = (N*k,)
        gate_weights   = topk_vals.view(-1)  # shape = (N*k,)

        # 5) For each expert e, gather the subset of tokens that routed to e
        for e in range(E):
            # Find positions in expert_indices == e
            mask_e = (expert_indices == e)               # Boolean mask of length N*k
            if not mask_e.any():
                # No tokens routed to expert e in this batch; skip
                continue

            # "flat_positions" are the positions in the N*k list that map to this expert e
            flat_positions = torch.nonzero(mask_e, as_tuple=False).squeeze(1)  # shape = (M_e,)
            # But each flat_position corresponds to one of the N tokens (because we repeated tokens k times).
            # Recover the original token idx for each:
            tokens_e = token_indices[flat_positions]    # shape = (M_e,), values ∈ [0..N-1]

            # 6) Gather the input features for exactly those tokens
            x_e = x_flat[tokens_e]    # shape = (M_e, D)

            # 7) Run expert e on this smaller batch
            #    out_e_small: (M_e, O)
            out_e_small = self.experts[e](x_e)

            # 8) Weight each row by its gate score
            #    gate_weights[flat_positions] are the scalar gate scores for these token→expert pairs
            w_e = gate_weights[flat_positions].unsqueeze(1)  # shape = (M_e, 1)
            out_e_small_weighted = out_e_small * w_e         # (M_e, O)

            # 9) Scatter (add) these weighted outputs back into out_flat
            #    For each i in [0..M_e-1], add out_e_small_weighted[i] to out_flat[tokens_e[i]]
            out_flat.index_add_(0, tokens_e, out_e_small_weighted)

        # 10) Reshape back to (B, L, O)
        final_output = out_flat.view(B, L, O)

        if return_dense:
            # also reshape dense_gating → (B, L, E)
            dense_gating_reshaped = dense_gating.view(B, L, E)
            return final_output, dense_gating_reshaped

        return final_output

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # split channel dim in half
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(tensor: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    # tensor: (B, H, T, head_dim) with head_dim = 2*half_dim
    # freq:   (1, 1, T, half_dim)
    B, H, T, head_dim = tensor.shape
    half_dim = head_dim // 2
    # Split tensor into two halves
    t1, t2 = tensor[..., :half_dim], tensor[..., half_dim:]
    cos, sin = freq.cos(), freq.sin()  # each (1,1,T,half_dim)
    # Apply RoPE on each half
    t1_rot = t1 * cos - t2 * sin
    t2_rot = t1 * sin + t2 * cos
    return torch.cat((t1_rot, t2_rot), dim=-1)

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
        head_dim = embed_dim // num_heads
        half_dim = head_dim // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        # freq_buffer shape: (max_len, half_dim)
        freq_buffer = torch.outer(torch.arange(max_len, dtype=torch.float32), theta)
        self.register_buffer("freq_buffer", freq_buffer.unsqueeze(0).unsqueeze(0), persistent=False)

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

        freq = self.freq_buffer[:, :, :T_in, : (head_dim // 2)].to(q.device)
        q = apply_rope(q, freq)
        k = apply_rope(k, freq)

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

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p,
            is_causal=True,
            scale=self.scaling
        )

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
    def __init__(self,
                n_heads: int, 
                n_embd: int, 
                max_len: int, 
                index: int,
                args: Arguments,
                num_dense_layers: int = 1, 
                num_expert: int = 8, 
                score_fn: str = 'softmax', 
                top_k: int = 2, 
                dropout: int = 0.1, 
                kv_caching: bool = False
                ):
        super(Block, self).__init__()
        self.attention = FlashAttention(n_heads, n_embd, max_len, dropout, kv_caching=kv_caching) 
        self.ff = MLP(n_embd, dropout) if index <= num_dense_layers else MoE_(args)
        self.args = args  # Use the passed arguments instead of creating new ones
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.n_experts = num_expert
        self.load_balancing_loss = 0.0
        self.moe_enabled = isinstance(self.ff, MoE_)


    def forward(self, x: torch.Tensor):
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        if isinstance(self.ff, MoE_):
            out = self.ff(self.ln2(x))
            x = x + out
            return x
        else:
            x = x + self.ff(self.ln2(x))
            return x
    

class Transformer(nn.Module):
    def __init__(self,
                n_layers: int, 
                n_heads: int, 
                n_embd: int, 
                vocab_size: int, 
                num_dense_layers: int = 1, 
                num_expert: int = 8, 
                score_fn: str = 'softmax', 
                top_k: int = 2, 
                max_len:int = 5000, 
                dropout:int = 0.1,
                kv_caching: bool = False):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Initialize Arguments for MoE layers
        # Calculate the actual number of MoE layers
        num_moe_layers = max(0, n_layers - num_dense_layers)
        
        self.args = Arguments(
            hidden_size=n_embd,
            moe_num_experts=num_expert,
            moe_top_k=top_k,
            mlp_impl="grouped",  # Required for triton >=3.2.0
            ffn_hidden_size=4 * n_embd,
            bias=False,
            activation_fn=F.gelu,
            moe_loss_weight=0.01,
            moe_capacity_factor=2,  # Capacity factor for load balancing
            moe_normalize_expert_weights=1.0,  # Normalize expert weights
            moe_jitter_eps=0.01,  # Add small noise to improve load balancing
            moe_lbl_in_fp32=True,  # Compute load balancing loss in fp32
            fp16=False,  # Disable fp16 to avoid dtype mismatch
            bf16=False,  # Disable bf16 to ensure fp32
            num_layers=num_moe_layers,  # Only count MoE layers
            pipeline_model_parallel_size=1,  # Single pipeline stage
            num_layers_per_virtual_pipeline_stage=None,  # No virtual pipeline
            uniform_expert_assignment=False  # Use learned routing
        )
        
        self.blocks = nn.ModuleList([Block(n_heads,
                                            n_embd, 
                                            max_len,
                                            args=self.args,
                                            index=i,
                                            num_dense_layers=num_dense_layers,
                                            num_expert=num_expert,
                                            score_fn=score_fn,
                                            top_k=top_k, 
                                            dropout=dropout, 
                                            kv_caching=kv_caching) for i in range(1, n_layers + 1)])
        self.ln_f = RMSNorm(n_embd)
        self.fc_out = nn.Linear(n_embd, vocab_size)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.num_dense_layers = num_dense_layers
        self.num_expert = num_expert
        self.score_fn = score_fn
        self.top_k = top_k
        self.dropout = dropout
        self.n_embd = n_embd
        self.n_heads = n_heads
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
        # Clear any previous load balancing loss
        clear_load_balancing_loss()
        
        x = self.embedding(x)
        for block in self.blocks:
            if self.training and x.requires_grad:
                # Use gradient checkpointing for memory efficiency during training
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        if self.training:
            moe_loss = batched_load_balancing_loss(self.args)
            return logits, moe_loss
        return logits
    

    

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
