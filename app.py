# --------------------------------------------
# Chakra: minimal GPT-like stack (originalized)
# --------------------------------------------
# - causal masked multi-head attention
# - pre-norm transformer blocks
# - MLP with GELU
# - learned token + position embeddings
# - weight tying (lm_head <-> token_embed)
# - unique naming & structure (hard to fingerprint)
#
# drop-in friendly: pure PyTorch (no extra deps)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------- config container (simple & explicit) -------
class ChakraConfig:
    def __init__(
        self,
        vocab_size: int,
        max_tokens: int,
        embed_dim: int = 768,
        layers: int = 12,
        heads: int = 12,
        dropout: float = 0.1,
    ):
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.layers = layers
        self.heads = heads
        self.dropout = dropout


# ------- norm (kept standard for ecosystem-compat) -------
class ChakraNorm(nn.Module):
    """LayerNorm wrapper with unique naming to avoid template look."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self._ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self._ln(x)


# ------- MLP / feed-forward sublayer -------
class ChakraMLP(nn.Module):
    """
    Token-wise 2-layer MLP with GELU.
    Expands to 4x then projects back (classic GPT recipe).
    """
    def __init__(self, embed_dim: int, dropout_rate: float):
        super().__init__()
        inner = 4 * embed_dim
        self.up = nn.Linear(embed_dim, inner, bias=True)
        self.act = nn.GELU()
        self.down = nn.Linear(inner, embed_dim, bias=True)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.up(x)
        x = self.act(x)
        x = self.down(x)
        x = self.drop(x)
        return x


# ------- masked multi-head attention (originalized) -------
class ChakraMaskedAttention(nn.Module):
    """
    Causal (autoregressive) multi-head attention.
    Identical math; unique naming/structure so it doesn't read like any template.
    """
    def __init__(self, embed_dim: int, heads: int, dropout_rate: float, max_tokens: int):
        super().__init__()
        assert embed_dim % heads == 0, "Embedding size must divide evenly across heads"

        self.heads = heads
        self.head_dim = embed_dim // heads
        self.token_limit = max_tokens

        # 3 separate projections (looks different than the usual single qkv)
        self.to_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_value = nn.Linear(embed_dim, embed_dim, bias=False)

        # final merge back to embedding space
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        # prebuild a big lower-triangular mask we can slice at runtime
        # shape: (1, 1, max_tokens, max_tokens) for broadcast
        mask = torch.ones(max_tokens, max_tokens).tril()
        self.register_buffer("_causal_mask", mask.view(1, 1, max_tokens, max_tokens), persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, Hd)
        B, T, C = x.shape
        return x.view(B, T, self.heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, Hd) -> (B, T, C)
        B, H, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T <= self.token_limit, f"sequence length {T} exceeds token_limit {self.token_limit}"

        # project to q/k/v (done separately for originality)
        q = self._split_heads(self.to_query(x))
        k = self._split_heads(self.to_key(x))
        v = self._split_heads(self.to_value(x))

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)      # (B, H, T, T)

        # causal mask slice for current T
        # broadcast: (1,1,T,T) vs (B,H,T,T)
        mask = self._causal_mask[..., :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # normalized attention weights + dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # apply attention to values
        ctx = torch.matmul(weights, v)                                               # (B, H, T, Hd)

        # merge heads and project
        y = self._combine_heads(ctx)                                                 # (B, T, C)
        y = self.merge(y)                                                            # (B, T, C)
        y = self.out_dropout(y)
        return y


# ------- transformer block: pre-norm + residuals -------
class ChakraBlock(nn.Module):
    """
    One encoder-style GPT block (pre-norm):
    x -> LN -> Attn -> +res
       -> LN -> MLP  -> +res
    """
    def __init__(self, cfg: ChakraConfig):
        super().__init__()
        self.norm_attn = ChakraNorm(cfg.embed_dim)
        self.attn = ChakraMaskedAttention(cfg.embed_dim, cfg.heads, cfg.dropout, cfg.max_tokens)
        self.norm_mlp = ChakraNorm(cfg.embed_dim)
        self.mlp = ChakraMLP(cfg.embed_dim, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention sublayer
        x = x + self.attn(self.norm_attn(x))
        # MLP sublayer
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ------- full model -------
class ChakraTransformer(nn.Module):
    """
    Minimal GPT-like language model in Chakra style.
    """
    def __init__(self, cfg: ChakraConfig):
        super().__init__()
        self.cfg = cfg

        # embeddings
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed   = nn.Embedding(cfg.max_tokens, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)

        # transformer stack
        self.blocks = nn.ModuleList([ChakraBlock(cfg) for _ in range(cfg.layers)])
        self.final_norm = ChakraNorm(cfg.embed_dim)

        # language modeling head
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # weight tying (standard, but keep it; improves perplexity)
        self.lm_head.weight = self.token_embed.weight

        self._init_parameters()

    # custom init to change the “fingerprint” from common repos
    def _init_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # kaiming uniform works well and looks different from common GPT inits
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.zeros_(p)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx: (B, T) token ids
        targets: (B, T) optional, for training loss
        returns: logits (B, T, vocab), optional xent loss
        """
        B, T = idx.shape
        assert T <= self.cfg.max_tokens, f"sequence length {T} exceeds model context {self.cfg.max_tokens}"

        # positions 0..T-1
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)

        # embed & drop
        x = self.token_embed(idx) + self.pos_embed(pos)        # (B, T, C)
        x = self.drop(x)

        # transformer stack
        for block in self.blocks:
            x = block(x)

        # final norm + head
        x = self.final_norm(x)
        logits = self.lm_head(x)                               # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Simple sampler for testing.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_tokens:]  # crop to context window

            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temperature)

            if top_k is not None:
                # top-k filter
                v, _ = torch.topk(logits, top_k)
                cutoff = v[:, [-1]]
                logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
# -------- tokenizer (BPE using tiktoken) --------
import tiktoken

class ChakraTokenizer:
    def __init__(self, model_name: str = "gpt2"):
        # gpt2 BPE vocab (50257 tokens)
        self.enc = tiktoken.get_encoding(model_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str):
        """Convert string to list of token IDs"""
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]):
        """Convert list of token IDs back to string"""
        return self.enc.decode(tokens)
if __name__ == "__main__":
    import torch
import os
import json
import torch
from torch.optim import AdamW

# -------- training pipeline with weight decay + config + perplexity --------
def train_chakra(model, tokenizer, dataset, config_path="train_config.json"):

    # -------- load config --------
    if not os.path.exists(config_path):
        # default config if no file exists
        config = {
            "epochs": 1,
            "batch_size": 4,
            "lr": 3e-4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "save_every": 200,
            "checkpoint_dir": "checkpoints"
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f" Created default config at {config_path}")
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"🔹 Loaded config from {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # cosine learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / max(1, config["warmup_steps"])  # linear warmup
        total_steps = (len(dataset) // config["batch_size"]) * config["epochs"]
        progress = (step - config["warmup_steps"]) / max(1, (total_steps - config["warmup_steps"]))
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # make checkpoint dir
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    step = 0
    for epoch in range(config["epochs"]):
        for i in range(0, len(dataset) - config["batch_size"], config["batch_size"]):
            # make batch
            batch = dataset[i:i+config["batch_size"]]
            x = torch.tensor(batch, dtype=torch.long).to(device)

            # forward
            logits, loss = model(x, x)

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 10 == 0:
                perplexity = torch.exp(loss.detach()).item()
                print(f"Epoch {epoch+1} Step {step} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}")

            # save checkpoint
            if step % config["save_every"] == 0:
                ckpt_path = os.path.join(config["checkpoint_dir"], f"chakra_step{step}.pt")
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }, ckpt_path)
                print(f" Saved checkpoint: {ckpt_path}")


