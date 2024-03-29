# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
from src.utils import flatten_prompt
import copy

class SequenceTrainer:

    def __init__(self, model, optimizer, batch_size, get_batch=None, loss_fn=None,
                 scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.start_time = time.time()


    def pure_train_iteration_mix(self, num_steps, no_pool=False, loss_component=False, config=None):

        train_losses = []
        logs = dict()
        train_start = time.time()
        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step_mix(no_pool, loss_component, config)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def train_step_mix(self, no_pool=False, loss_component=False, config=None):
        batch = self.get_batch()
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)
        if no_pool:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, pool=None
            )
        elif loss_component:
            state_preds, action_preds, reward_preds, prompt_infos = self.model.forward(
                states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, pool=no_pool
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if loss_component:
            reduce_sim = prompt_infos['reduce_sim']
            diversity_loss = prompt_infos['diversity_loss']

            if config is not None:
                loss = torch.mean(
                    (action_preds - action_target) ** 2) - config['alpha'] * reduce_sim + config['beta'] * diversity_loss
            else:
                loss = torch.mean(
                    (action_preds - action_target) ** 2) - 0.1 * reduce_sim + 10e-6 * diversity_loss

        else:
            loss = self.loss_fn(
                 None, action_preds, None,
                 None, action_target, None,
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def finetune_eval_iteration_multienv(self, get_prompt, get_batch, test_prompt_trajectories_list, test_trajectories_list, 
                                eval_episodes, env_name_list, info, 
                                variant, env_list, iter_num=0, print_logs=False, 
                                no_prompt=False, group='test-finetune',
                                finetune_opt=False):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()
        self.current_model_dict = copy.deepcopy(self.model.state_dict())

        eval_start = time.time()
        if finetune_opt:
            fintune_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=variant['finetune_lr'],
                weight_decay=1e-4,
            )
        else:
            fintune_optimizer = None
        for env_id, env_name in enumerate(env_name_list):
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            self.get_prompt = get_prompt(test_prompt_trajectories_list[env_id], info[env_name], variant)
            self.get_batch = get_batch(test_trajectories_list[env_id], info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(), batch_size=1) # one prompt for the whole batch now
            else:
                self.prompt = None

            self.model.train()
            # finetune the model on the data for this task 
            finetune_losses = []
            for _ in range(variant['finetune_steps']):
                finetune_loss = self.train_step(
                    batch_size_overwrite=variant['finetune_batch_size'],
                    optimizer=fintune_optimizer)
                finetune_losses.append(finetune_loss)

            self.model.eval()
            # need to sample eval_fn and prompt together 
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v
            
            self.model.load_state_dict(self.current_model_dict)

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs


    def train_step(self, batch_size_overwrite=None, optimizer=None):
        if batch_size_overwrite is not None:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(batch_size_overwrite)
        else:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=self.prompt
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        if optimizer is None:
            self.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        if optimizer is None:
            self.optimizer.step()
        else:
            optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def eval_iteration_multienv(self, eval_episodes, env_name_list, info,
                                variant, env_list, iter_num=0, print_logs=False, no_prompt=False, group='test'):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            print("env_id is {}, env_name is {}".format(env_id, env_name))
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            if not no_prompt:
                self.pool = True
            else:
                self.pool = None
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, pool=self.pool)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

 
    def save_model(self, env_name, postfix, folder):
        model_name = '/' + env_name + postfix
        torch.save(self.model.state_dict(),folder+model_name)  # model save
        print('model saved to ', folder+model_name)
