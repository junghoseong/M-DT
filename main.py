import json, pickle, random, os, torch, argparse
from collections import namedtuple
from src.utils import get_env_list, process_total_data_mean, get_batch, process_info, eval_episodes
from src.maneuver_pool_decision_transformer import ManeuverPoolDecisionTransformer
from src.seq_trainer import SequenceTrainer
from src.rollout import rollout
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_config(base_path=None, task=None, hyperparam_path=None):
    if base_path is None:
        cur_dir = os.getcwd()
    else:
        cur_dir = base_path
    config_path = os.path.join(cur_dir, 'configs')
    config_data_params_path = os.path.join(config_path, 'data_params')
    hyperparam_config_path = os.path.join(config_path, 'hyperparameter')
    data_save_path = os.path.join(cur_dir, 'data')
    save_path = os.path.join(cur_dir, 'model_saved/')
    if not os.path.exists(save_path): os.mkdir(save_path)
    config_path_dict = {
        "All": "All/All.json",
        "HDP": "HDP/HDP.json",
        "Human": "Human/Human.json",
        "OnlineRL": "OnlineRL/OnlineRL.json",
        "Ablation-dataset-1": "Ablation-dataset-1/Ablation-dataset-1.json",
        "Ablation-dataset-2": "Ablation-dataset-2/Ablation-dataset-2.json",
        'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
        'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
        'ant_dir': "ant_dir/ant_dir_50.json",
    }

    if task is None:
        data_config_path = os.path.join(config_data_params_path, config_path_dict["All"])
    else:
        data_config_path = os.path.join(config_data_params_path, config_path_dict[task])

    with open(data_config_path, 'r') as f:
        data_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    if hyperparam_path is None:
        hyperparam_config_path = os.path.join(hyperparam_config_path, 'MDT.json')
    else:
        hyperparam_config_path = os.path.join(hyperparam_config_path, hyperparam_path)

    with open(hyperparam_config_path, 'r') as f:
        hyperparam_config = json.load(f)

    return data_config, hyperparam_config, data_save_path, save_path, config_data_params_path, hyperparam_config_path

def experiment_with_env(
        exp_prefix,
        variant,
):
    data_config, hyperparam_config, data_save_path, save_path, data_config_path, hyperparam_config_path = load_config(variant['base_path'],  variant['task'], variant['hyperparam_path'])
    device = hyperparam_config['device']

    train_env_name_list, test_env_name_list = [], []
    for task_ind in data_config.train_tasks:
        train_env_name_list.append(hyperparam_config['env'] + '-' + str(task_ind))
    for task_ind in data_config.test_tasks:
        test_env_name_list.append(hyperparam_config['env'] + '-' + str(task_ind))

    # training envs
    info, env_list = get_env_list(train_env_name_list, data_config_path, device)
    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, data_config_path, device)

    #print(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    #print(f'Env List: {env_list} \n\n Test Env List: {test_env_list}')

    # load training dataset
    if hyperparam_config["env"] is "AirCombatEnvironment":
        trajectories_list = []
        for env_name in train_env_name_list:
            dataset_path = data_save_path + f'/{hyperparam_config["env"]}/{env_name}.pkl'
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
            trajectories_list.append(trajectories)

        # load test dataset
        test_trajectories_list = []
        for env_name in test_env_name_list:
            dataset_path = data_save_path + f'/{hyperparam_config["env"]}/{env_name}.pkl'
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
            test_trajectories_list.append(trajectories)
    else:
        trajectories_list = []
        for env_name in train_env_name_list:
            dataset_path = data_save_path + f'/{hyperparam_config["env"]}/{env_name}-expert.pkl'
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
            trajectories_list.append(trajectories)

        # load test dataset
        test_trajectories_list = []
        for env_name in test_env_name_list:
            dataset_path = data_save_path + f'/{hyperparam_config["env"]}/{env_name}-expert.pkl'
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
            test_trajectories_list.append(trajectories)


    if hyperparam_config['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(trajectories_list))
        test_total = list(itertools.chain.from_iterable(test_trajectories_list))
        total_traj_list = train_total + test_total
        print("overall number of trajectories is : {}".format(len(total_traj_list)))
        total_state_mean, total_state_std = process_total_data_mean(total_traj_list, hyperparam_config['mode'])
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, hyperparam_config)
    # process test info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, hyperparam_config)
    exp_prefix = hyperparam_config['env']
    num_env = len(train_env_name_list)
    group_name = f'{exp_prefix}-{str(num_env)}-Env'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    print(group_name, exp_prefix)
    print(train_env_name_list, test_env_name_list)
    state_dim = test_env_list[0].observation_space.shape[0]
    act_dim = test_env_list[0].action_space.shape[0]

    model = ManeuverPoolDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=hyperparam_config['K'],
        max_ep_len=3000,
        hidden_size=hyperparam_config['embed_dim'],
        n_layer=hyperparam_config['n_layer'],
        n_head=hyperparam_config['n_head'],
        n_inner=4 * hyperparam_config['embed_dim'],
        activation_function=hyperparam_config['activation_function'],
        n_positions=1024,
        resid_pdrop=hyperparam_config['dropout'],
        attn_pdrop=hyperparam_config['dropout'],
        pool_size=hyperparam_config['M'],
        state_history_length=hyperparam_config['L'],
    )
    model = model.to(device=device)

    warmup_steps = hyperparam_config['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparam_config['learning_rate'],
        weight_decay=hyperparam_config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    env_name = train_env_name_list[0]
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=hyperparam_config['batch_size'],
        scheduler=scheduler,
        get_batch=get_batch(trajectories_list[0], info[env_name], hyperparam_config),
    )

    if variant['rollout'] and variant['ckpt_path'] is None:
        for iter in range(hyperparam_config['max_iters']):
            outputs = trainer.pure_train_iteration_mix(
                num_steps=hyperparam_config['num_steps_per_iter'],
                no_pool=hyperparam_config['no-pool'],
                loss_component=True,
                config=hyperparam_config,
            )
            #start evaluation
            if iter % hyperparam_config['test_eval_interval'] == 0:
                test_eval_logs = trainer.eval_iteration_multienv(
                    eval_episodes, test_env_name_list, test_info, hyperparam_config, test_env_list, iter_num=iter + 1,
                    print_logs=True, no_prompt=hyperparam_config['no-pool'], group='test')
                outputs.update(test_eval_logs)

            if iter % hyperparam_config['save_interval'] == 0:
                trainer.save_model(
                    env_name=hyperparam_config['env'],
                    postfix='_MCDT_Final_All_iter_small'+str(iter),
                    folder=save_path
                )
            outputs.update({"global_step": iter})

    else:
        if hyperparam_config["env"] is not "AirCombatEnvironment":
            model.load_state_dict(torch.load(variant['ckpt_path']))
            model.eval()
            model.to(device=device)
            test_eval_logs = trainer.eval_iteration_multienv(
                eval_episodes, test_env_name_list, test_info, hyperparam_config, test_env_list, iter_num= 1,
                print_logs=True, no_prompt=hyperparam_config['no-pool'], group='test')
        else:
            model.load_state_dict(torch.load(variant['ckpt_path']))
            model.eval()
            model.to(device=device)
            record, WR = rollout(num_episodes=100, model=model, hyperparam_config=hyperparam_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default=None)
    parser.add_argument('--task', type=str, default="ant_dir")
    parser.add_argument('--hyperparam_path', type=str, default="ant-MDT.json")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--rollout', type=bool, default=False)

    args = parser.parse_args()
    experiment_with_env('AirCombatEnvironment', vars(args))