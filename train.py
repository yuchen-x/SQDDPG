import numpy as np
from utilities.trainer import PGTrainer
import torch
import os
from utilities.util import *
from utilities.logger import Logger
import argparse

from utilities.gym_wrapper import GymWrapper
from marl_envs.particle_envs.make_env import make_env
from marl_envs.my_env.capture_target import CaptureTarget as CT
from marl_envs.my_env.small_box_pushing import SmallBoxPushing as SBP
from aux import Model, Strategy
import random


# parser = argparse.ArgumentParser(description='Test rl agent.')
# parser.add_argument('--save-path', type=str, nargs='?', default='./', help='Please input the directory of saving model.')
# argv = parser.parse_args()

def main(args):
    # if argv.save_path[-1] is '/':
    #     save_path = argv.save_path
    # else:
    #     save_path = argv.save_path+'/'

    # create save folders
    # if 'model_save' not in os.listdir(save_path):
    #     os.mkdir(save_path+'model_save')
    # if 'tensorboard' not in os.listdir(save_path):
    #     os.mkdir(save_path+'tensorboard')
    # if log_name not in os.listdir(save_path+'model_save/'):
    #     os.mkdir(save_path+'model_save/'+log_name)
    save_path = './'
    if args.logger:
        if args.save_dir not in os.listdir(save_path+'tensorboard/'):
            os.mkdir(save_path+'tensorboard/'+args.save_dir)
        else:
            path = save_path+'tensorboard/'+args.save_dir
            for f in os.listdir(path):
                file_path = os.path.join(path,f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    logger = Logger(save_path+'tensorboard/' + args.save_dir)

    model = Model[args.model_name]

    strategy = Strategy[args.model_name]
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)

    # create env
    if args.env_name.startswith('CT'):
        env_args = {'terminate_step': args.max_steps,
                     'n_target': args.n_target,
                     'n_agent': args.agent_num}
    elif args.env_name.startswith('BP') or args.env_name.startswith('SBP'):
        env_args = {'terminate_step': args.max_steps,
                    'random_init': args.random_init,
                    'small_box_only': args.small_box_only,
                    'terminal_reward_only': args.terminal_reward_only,
                    'big_box_reward': args.big_box_reward,
                    'small_box_reward': args.small_box_reward,
                    'n_agent': args.agent_num}
    else:
        env_args = {'max_epi_steps': args.max_steps,
                    'prey_accel': args.n_target,
                    'prey_max_v': args.prey_max_v,
                    'obs_r': args.obs_r,
                    'obs_resolution': args.obs_resolution,
                    'flick_p': args.flick_p,
                    'enable_boundary': args.enable_boundary,
                    'benchmark': args.benchmark,
                    'discrete_mul': args.discrete_mul,
                    'config_name': args.config_name}

    if args.env_name.startswith('CT'):
        env = CT(grid_dim=tuple(args.grid_dim), **env_args)
    elif args.env_name.startswith('SBP'):
        env = SBP(tuple(config.grid_dim), **env_args)
    else:
        env = make_env(args.env_name, 
                       discrete_action=True,
                       discrete_action_input=True,
                       **env_args)
        env.seed(args.seed)
        
    if strategy == 'pg':
        train = PGTrainer(args, model, env, logger, args.online)
    elif strategy == 'q':
        raise NotImplementedError('This needs to be implemented.')
    else:
        raise RuntimeError('Please input the correct strategy, e.g. pg or q.')

    stat = dict()

    for i in range(args.train_episodes_num):
        train.run(stat)
        if args.logger:
            train.logging(stat)
        # if i%args.save_model_freq == args.save_model_freq-1:
        # train.print_info(stat)
        #     torch.save({'model_state_dict': train.behaviour_net.state_dict()}, save_path+'model_save/'+log_name+'/model.pt')
        #     print ('The model is saved!\n')
        #     with open(save_path+'model_save/'+log_name +'/log.txt', 'w+') as file:
        #         file.write(str(args)+'\n')
        #         file.write(str(i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='sqddpg', type=str)
    parser.add_argument("--env_name", default='simple_spread', type=str)
    parser.add_argument("--agent_num", default=3, type=int)
    parser.add_argument("--hid_size", default=32, type=int)
    parser.add_argument("--continuous", action='store_true')
    parser.add_argument("--init_std", default=0.1, type=float)
    parser.add_argument("--policy_lrate", default=0.0001, type=float)
    parser.add_argument("--value_lrate", default=0.001, type=float)
    parser.add_argument("--max_steps", default=25, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--normalize_advantages", action='store_true')
    parser.add_argument("--entr", default=0.02, type=float)
    parser.add_argument("--entr_inc", default=0.0, type=float)
    parser.add_argument("--q_func", action='store_true')
    parser.add_argument("--train_episodes_num", default=5000, type=int)
    parser.add_argument("--replay", action='store_true')
    parser.add_argument("--replay_buffer_size", default=10000, type=int)
    parser.add_argument("--replay_warmup", default=0, type=int)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--grad_clip", action='store_true')
    parser.add_argument("--target", action='store_true')
    parser.add_argument("--target_lr", default=0.1, type=float)
    parser.add_argument("--behaviour_update_freq", default=100, type=int)
    parser.add_argument("--critic_update_times", default=10, type=int)
    parser.add_argument("--target_update_freq", default=200, type=int)
    parser.add_argument("--gumbel_softmax", action='store_true')
    parser.add_argument("--epsilon_softmax", action='store_true')
    parser.add_argument("--online", action='store_true')
    parser.add_argument("--reward_record_type", default='episode_mean_step', type=str)
    parser.add_argument("--shared_parameters", action='store_true')
    parser.add_argument("--sample_size", default=5, type=int)
    
    parser.add_argument('--grid_dim', nargs=2, default=[4,4], type=int)
    parser.add_argument("--n_target", default=1, type=int)
    parser.add_argument("--small_box_only", action='store_true')
    parser.add_argument("--terminal_reward_only", action='store_true')
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--small_box_reward", default=100.0, type=float)
    parser.add_argument("--big_box_reward", default=100.0, type=float)
    # particle envs args
    parser.add_argument("--prey_accel", default=4.0, type=float)
    parser.add_argument("--prey_max_v", default=1.3, type=float)
    parser.add_argument("--flick_p", default=0.0, type=float)
    parser.add_argument("--obs_r", default=2.0, type=float)
    parser.add_argument("--enable_boundary", action='store_true')
    parser.add_argument("--discrete_mul", default=1, type=int)
    parser.add_argument("--config_name", default="antipodal", type=str)
    parser.add_argument("--benchmark", action='store_true')
    parser.add_argument("--obs_resolution", default=8, type=int)
    
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--eval_num_epi", default=10, type=int)
    parser.add_argument("--run_idx", default=0, type=int)
    parser.add_argument("--save_dir", default='test', type=str)
    parser.add_argument("--save_rate", default=1000, type=int)
    parser.add_argument("--logger", action='store_true')

    config = parser.parse_args()

    # create the dirs to save results
    os.makedirs("./performance/" + config.save_dir + "/test", exist_ok=True)
    os.makedirs("./performance/" + config.save_dir + "/ckpt", exist_ok=True)

    main(config)
