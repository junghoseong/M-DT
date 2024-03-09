import math
import torch
import torch.nn as nn
from src.pool import Pool

class ManeuverPool(Pool):

    def __init__(self, length=6, top_k=1, dropout_rate=0.0, log_mod_stats=False, exclude_k=False, n_layer=None,
                 exclude_v=False, exclude_ff=False, use_diversity_loss=True, state_history_length=20, **kwargs):
        self.use_diversity_loss = use_diversity_loss
        super().__init__(length=length, top_k=top_k, dropout_rate=dropout_rate, n_layer=n_layer, use_diversity_loss=use_diversity_loss, **kwargs)
        self.log_mod_stats = log_mod_stats
        self.exclude_k = exclude_k
        self.exclude_v = exclude_v
        self.exclude_ff = exclude_ff
        self.n_layer = n_layer

    def _setup_prompt(self):
        if self.n_layer is None:
            self.prompt = nn.Parameter(torch.randn((self.pool_size, self.length, self.embed_dim)))
            self.state_encoder = torch.nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.prompt = nn.Parameter(torch.ones((self.pool_size, self.n_layer, self.length, self.embed_dim)))
            self.state_encoder = torch.nn.Linear(self.embed_dim, self.embed_dim)
        if self.use_diversity_loss:
            self.overlap_bias = nn.Parameter(torch.randn(self.n_layer, self.length))

    def extract_prompt(self, idx):
        batched_prompt_raw = self.prompt[idx]

        if self.n_layer is None:
            batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, length, embed_dim).unsqueeze(1)
            vectors = [batched_prompt[:, :, 0], batched_prompt[:, :, 1], batched_prompt[:, :, 2:].flatten(-2)]
        else:
            batch_size, top_k, n_layer, length, embed_dim = batched_prompt_raw.shape
            if top_k == 1:
                batched_prompt = batched_prompt_raw.reshape(batch_size, n_layer, length, embed_dim)
            else:
                batched_prompt = batched_prompt_raw.reshape(batch_size, top_k, n_layer, length, embed_dim)
            vectors = [(p[:, :, 0], p[:, :, 1], p[:, :, 2:].flatten(-2)) for p in batched_prompt.split(1, dim=1)]

        stats = {}
        if self.log_mod_stats:
            for i, vec in enumerate(vectors):
                stats[f"mod_k_mean_{i}"] = round(vec[0].mean().item(), 3)
                stats[f"mod_k_std_{i}"] = round(vec[0].std().item(), 3)
                stats[f"mod_v_mean_{i}"] = round(vec[1].mean().item(), 3)
                stats[f"mod_v_std_{i}"] = round(vec[1].std().item(), 3)
                stats[f"mod_ff_mean_{i}"] = round(vec[2].mean().item(), 3)
                stats[f"mod_ff_std_{i}"] = round(vec[2].std().item(), 3)

        # turn off modulation vectors
        if any([self.exclude_v, self.exclude_k, self.exclude_ff]):
            # iterate each layers modulation vectors
            for i, vecs in enumerate(vectors):
                new_vecs = list(vecs)
                if self.exclude_v:
                    new_vecs[0] = None
                if self.exclude_k:
                    new_vecs[1] = None
                if self.exclude_ff:
                    new_vecs[2] = None
                vectors[i] = tuple(new_vecs)

        if self.use_diversity_loss is True:
            W = self.prompt.reshape(self.n_layer, self.length, self.pool_size, self.embed_dim)
            WT = self.prompt.reshape(self.n_layer, self.length, self.embed_dim, self.pool_size)
            WWT = torch.matmul(W, WT)
            I = torch.eye(self.pool_size).expand(self.n_layer, self.length, self.pool_size, self.pool_size).to(device='cuda:0')
            WWTI = torch.sub(WWT, I)
            norm = torch.norm(WWTI, p='fro', dim=[2, 3])
            loss = torch.sum(torch.max(norm - self.overlap_bias, torch.zeros_like(norm))) / (self.n_layer * self.length)
            stats['diversity_loss'] = loss

        return vectors, stats    

    def add_dropout(self, batched_prompt):
        return batched_prompt