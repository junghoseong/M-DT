import torch
import torch.nn as nn
from src.base_prompt import Prompt

class Pool(Prompt):
    def __init__(self, init_range=0.02, init_prompts="uniform", init_keys="uniform",
                 prompt_key=True, count_penalty=False, num_tasks=None, n_layer=None, n_head=None,
                 prefix_first_only=False, n_pretrain_keys=None,
                 eval_mode=False, continual_mode=False, pull_sim="cosine", pos_sim=True,
                 turn_off_count_penalty=False, use_diversity_loss=False, **kwargs):
        super().__init__(**kwargs)
        self.use_prompt_key = prompt_key
        self.init_prompts = init_prompts
        self.init_keys = init_keys
        self.init_range = init_range
        self.count_penalty = count_penalty
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_tasks = num_tasks
        self.prefix_first_only = prefix_first_only
        self.n_pretrain_keys = n_pretrain_keys
        self.eval_mode = eval_mode
        self.continual_mode = continual_mode
        self.pull_sim = pull_sim
        self.pos_sim = pos_sim
        self.turn_off_count_penalty = turn_off_count_penalty
        self._setup_prompt()
        self._setup_keys()
        self.task_to_mask = None
        if self.n_pretrain_keys and self.pretrain:
            self.pool_size = self.n_pretrain_keys
            self.register_buffer('counts', torch.zeros(self.n_pretrain_keys, requires_grad=False))
            self.register_buffer('inv_counts_so_far', torch.ones(self.n_pretrain_keys, requires_grad=False))
        self.use_diversity_loss = use_diversity_loss

    def _setup_prompt(self):
        if self.n_layer is None:
            self.prompt = nn.Parameter(torch.randn((self.pool_size, self.length, self.embed_dim)))
        else:
            if self.prefix:
                n_layer = self.n_layer * 2
                self.prompt = nn.Parameter(torch.randn((self.pool_size, n_layer, self.length, self.embed_dim)))
            else:
                self.prompt = nn.Parameter(torch.randn((self.pool_size, self.n_layer, self.length, self.embed_dim)))
        if self.init_prompts == "uniform":
            nn.init.uniform_(self.prompt, -1, 1)
        elif self.init_prompts == "normal":
            nn.init.normal_(self.prompt, 0, self.init_range)

    def _setup_keys(self):
        mult = 1 if self.agg_token != "concat" else 9
        if self.use_prompt_key:
            # if using learnable prompt keys
            self.prompt_key = nn.Parameter(torch.randn((self.pool_size, self.embed_dim * mult)))
            self._init_keys(self.prompt_key, init_keys=self.init_keys)
            if self.n_pretrain_keys:
                self.pretrain_key = nn.Parameter(torch.randn((self.n_pretrain_keys, self.embed_dim * mult)))
                self._init_keys(self.pretrain_key, init_keys=self.init_keys)

        else:
            prompt_mean = torch.mean(self.prompt, dim=1)
            if self.n_layer is not None:
                prompt_mean = self.prompt.mean(1)
            self.register_buffer('prompt_key', prompt_mean)

    @staticmethod
    def _init_keys(key, init_keys="uniform"): 
        if init_keys == "uniform":
            nn.init.uniform_(key, -1, 1)
        elif init_keys == "uniform_half": 
            nn.init.uniform_(key, -0.5, 0.5)
        elif init_keys == "uniform_double": 
            nn.init.uniform_(key, -2, 2)
        elif init_keys == "uniform_ten": 
            nn.init.uniform_(key, -10, 10)
        elif init_keys == "uniform_tenth": 
            nn.init.uniform_(key, -0.1, 0.1)
        elif init_keys == "zeros":
            nn.init.zeros_(key)
        elif init_keys == "ones":
            nn.init.ones_(key)
            
    def _setup_task_mask(self, device):
        if self.num_tasks is not None:
            self.task_to_mask = {}
            prompt_idx = torch.arange(self.pool_size, device=device)
            for i, ids, in enumerate(prompt_idx.split(self.pool_size // self.num_tasks)):
                self.task_to_mask[i] = torch.isin(torch.arange(self.pool_size, device=device), ids)

    def forward(self, x_embed, task_id=None, cls_features=None, attention_mask=None, tok_to_pos=None):
        if self.state_encoder:
             x_embed = self.state_encoder(x_embed)
             x_embed = torch.nn.functional.relu(x_embed)
        prompt_mask = None
        if task_id is not None and self.num_tasks is not None:
            if self.task_to_mask is None:
                self._setup_task_mask(x_embed.device)
            prompt_mask = self.task_to_mask[task_id.item()]
        
        if self.turn_off_count_penalty and self.counts_total > 50000: 
            self.count_penalty = False

        x_embed_mean = self.aggregate_embeds(x_embed, cls_features=cls_features, 
                                             attention_mask=attention_mask, tok_to_pos=tok_to_pos)
        key = self.prompt_key
        if self.n_pretrain_keys and self.pretrain:
            key = self.pretrain_key
        elif self.n_pretrain_keys and self.eval_mode:
            key = torch.cat([key, self.pretrain_key], dim=0)
        similarity, prompt_norm, x_embed_norm = self.compute_similarity(key, x_embed_mean)

        if prompt_mask is not None: 
            similarity[:, ~prompt_mask] = float('-inf')

        _, idx = torch.topk(similarity, k=self.top_k, dim=1)
        if self.training:
            self.add_counts(idx)
        if self.pretrain and not self.continual_mode:
            self.update_inv_counts()

        batched_key_norm = prompt_norm[idx.clone()]
        if self.n_pretrain_keys and self.pretrain:
            idx[:] = 0
        elif self.n_pretrain_keys and self.eval_mode:
            if (idx >= self.pool_size).any(): 
                return None
        batched_prompt, prompt_stats = self.extract_prompt(idx)

        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim = batched_key_norm * x_embed_norm
        if self.pull_sim == "euclidean":
            # negative MSE to make sure gets minimized
            reduce_sim  = -torch.norm(key[idx] - x_embed_mean.unsqueeze(1), dim=-1).mean()
        elif self.pull_sim == "manhattan": 
            reduce_sim = -torch.abs(key[idx] - x_embed_mean.unsqueeze(1)).mean()
        else:  
            reduce_sim = torch.sum(sim) / x_embed.shape[0]

        if self.n_pretrain_keys and self.pretrain and self.continual_mode:
            if isinstance(batched_prompt, (list, tuple)):
                if isinstance(batched_prompt[0], (list, tuple)):
                    batched_prompt = [[b.detach() for b in batch] for batch in batched_prompt]
                else:
                    batched_prompt = [b.detach() for b in batched_prompt]
            else:
                batched_prompt = batched_prompt.detach()
        elif self.n_pretrain_keys and self.pretrain:
            batched_prompt = None

        if self.use_diversity_loss:
            diversity_loss = prompt_stats['diversity_loss']
        else:
            diversity_loss = 0

        out = {
            "prompt_idx": idx, "prompt_norm": prompt_norm, "x_embed_norm": x_embed_norm,
            "prompt_key": self.prompt_key, "x_embed_mean": x_embed_mean,
            "sim": sim, "similarity": similarity, "selected_key": batched_key_norm, "reduce_sim": reduce_sim,
            "total_prompt_len": self.top_k * self.length, "diversity_loss": diversity_loss, **prompt_stats
        }
        return dict(prompt=batched_prompt, infos=out)

    def extract_prompt(self, idx):
        batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C

        if self.n_layer is None:
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C
        else:
            # [B, top_k, n_layer, length, C]
            batched_prompt_raw = batched_prompt_raw.permute(0, 2, 1, 3, 4)
            batch_size, n_layer, top_k, length, c = batched_prompt_raw.shape
            if self.prefix:
                # --> [n_layer, B, n_head, top_k * length, C]
                batched_prompt = batched_prompt_raw.reshape(batch_size, n_layer, top_k * length,
                                                            self.n_head, c // self.n_head)
                # permute + split up n_layers --> keys, values
                batched_prompt = batched_prompt.permute(1, 0, 3, 2, 4).split(2)
            else:
                # --> [B, n_layer, top_k * length, C]
                batched_prompt = batched_prompt_raw.reshape(batch_size, n_layer, top_k * length, c)
                #vectors = [(p[:,:,0], p[:,:,1], p[:,:,2].flatten(-2)) for p in batched_prompt.split(1, dim=1)] # change to adapt L2M

        return batched_prompt, {}
    
    def compute_similarity(self, key, x_embed_mean):
        # ensure similarity computation does not happen in fp16
        #with torch.autocast(device_type='cuda', enabled=False): # version check

        with torch.cuda.amp.autocast():
            if self.pull_sim == "euclidean":
                # don't norm here
                prompt_norm, x_embed_norm = key, x_embed_mean
                similarity = -torch.norm(prompt_norm.unsqueeze(0) - x_embed_norm.unsqueeze(1), dim=-1)
            elif self.pull_sim == "manhattan":
                prompt_norm, x_embed_norm = key, x_embed_mean
                similarity = -torch.abs(prompt_norm.unsqueeze(0) - x_embed_norm.unsqueeze(1)).sum(dim=-1)
            else: 
                prompt_norm = self.l2_normalize(key, dim=1)  # Pool_size, C
                x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C
                similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size
        
            if self.training and self.count_penalty and self.counts_total > 0 and (self.task_id > 0 or self.pretrain):
                if self.pull_sim == "cosine":
                    if self.pos_sim: 
                        # add +1 to similarity. cosine sim can be [-1, 1]. 
                        # for sim < 1, the penalty would otherwise decrease sim
                        similarity = (similarity + 1) / 2
                    similarity = similarity * self.inv_counts_so_far
                else: 
                    # for euliclidean, we need the actual counts for penalization
                    similarity = similarity * (1 - self.inv_counts_so_far)
                
        return similarity, prompt_norm, x_embed_norm
        
    def add_dropout(self, batched_prompt):
        if self.n_layer and self.prefix:
            batched_prompt = [self.dropout(p) for p in batched_prompt]
        else:
            batched_prompt = self.dropout(batched_prompt)
        return batched_prompt
