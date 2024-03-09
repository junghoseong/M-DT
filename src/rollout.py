import os
import numpy as np
from envs.AirCombatEnvironment import AirCombatEnvironment
import torch

def env_creator():
    env = AirCombatEnvironment()
    return env

env = env_creator()

def rollout(num_episodes=100, model=None, hyperparam_config=None):
    device = hyperparam_config['device']
    state = env.reset(seed=0)
    scale = 500.
    target_return = [1000]
    target_return[0] = target_return[0] / scale
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode, t = 0, 0, 0
    record = []

    result = {'you win': 1,
              'you lose': 2,
              'target destroyed': 3,
              'ownship destroyed': 4,
              'target alt down': 5,
              'ownship alt down': 6,
              'draw': 7}
    HP_record = {"own": [],
                 "tgt": []}

    WR = [0, 0, 0]

    while True:
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            prompt=t,
            eval_mode=True
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1
        )

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1
        )
        episode_return += reward
        t += 1
        if done:
            win = env.get_win()
            if win != -2:
                if win == result['you win'] or win == result['target alt down'] or win == result['target destroyed']:
                    WR[0] += 1
                elif win == result['draw']:
                    WR[1] += 1
                else:
                    WR[2] += 1
                record.append(episode_return)
                print("episode return is {}, {}".format(episode_return, WR))
                own_HP, tgt_HP = state[18] * 10000, state[20] * 10000
                if win != result['ownship alt down'] and win != result['target alt down']:
                    HP_record['own'].append(max(0, own_HP))
                    HP_record['tgt'].append(max(0, tgt_HP))
                    print(own_HP, tgt_HP)

            episode_return = 0
            episode += 1
            t = 0
            state = env.reset(seed=episode)
            target_return = [1000]
            target_return[0] = target_return[0] / scale
            states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)

            ep_return = target_return
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

            if episode == 100:
                print("average reward is {}, std is {}".format(sum(record) / episode, np.std(np.array([record]))))
                print(
                    "own_HP : {}, tgt_HP : {}".format(sum(HP_record['own']) / episode, sum(HP_record['tgt']) / episode))
                print("match record is {}".format(WR))
                break
    return record, WR


