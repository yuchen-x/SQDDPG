import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple



class SQDDPG(Model):

    def __init__(self, args, obs_size, action_dim, target_net=None):
        super(SQDDPG, self).__init__(args)
        self.obs_dim = obs_size
        self.act_dim = action_dim
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.sample_size = self.args.sample_size
        self.Transition = namedtuple('Transition', ('state', 'obs', 'action', 'reward', 'next_state', 'next_obs', 'done', 'last_step', 'valid'))

    def unpack_data(self, batch):
        batch_size = len(batch.obs)
        valids = cuda_wrapper(torch.tensor(batch.valid, dtype=torch.float), self.cuda_)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        obs = cuda_wrapper(prep_obs(list(zip(batch.obs))), self.cuda_)
        next_obs = cuda_wrapper(prep_obs(list(zip(batch.next_obs))), self.cuda_)
        return (valids, rewards, last_step, done, actions, state, obs, next_state, next_obs)

    def construct_policy_net(self):
        action_dicts = []
        if self.args.shared_parameters:
            l1 = nn.Linear(self.obs_dim, self.hid_dim)
            l2 = nn.Linear(self.hid_dim, self.hid_dim)
            a = nn.Linear(self.hid_dim, self.act_dim)
            for i in range(self.n_):
                action_dicts.append(nn.ModuleDict( {'layer_1': l1,\
                                                    'layer_2': l2,\
                                                    'action_head': a
                                                    }
                                                 )
                                   )
        else:
            for i in range(self.n_):
                action_dicts.append(nn.ModuleDict( {'layer_1': nn.Linear(self.obs_dim, self.hid_dim),\
                                                    'layer_2': nn.Linear(self.hid_dim, self.hid_dim),\
                                                    'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                                    }
                                                  )
                                   )
        self.action_dicts = nn.ModuleList(action_dicts)

    def construct_value_net(self):
        value_dicts = []
        if self.args.shared_parameters:
            l1 = nn.Linear( (self.obs_dim+self.act_dim)*self.n_, self.hid_dim )
            l2 = nn.Linear(self.hid_dim, self.hid_dim)
            v = nn.Linear(self.hid_dim, 1)
            for i in range(self.n_):
                value_dicts.append(nn.ModuleDict( {'layer_1': l1,\
                                                   'layer_2': l2,\
                                                   'value_head': v
                                                  }
                                                )
                                  )
        else:
            for i in range(self.n_):
                value_dicts.append(nn.ModuleDict( {'layer_1': nn.Linear( (self.obs_dim+self.act_dim)*self.n_, self.hid_dim ),\
                                                   'layer_2': nn.Linear(self.hid_dim, self.hid_dim),\
                                                   'value_head': nn.Linear(self.hid_dim, 1)
                                                  }
                                                )
                                  )
        self.value_dicts = nn.ModuleList(value_dicts)

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        actions = []
        for i in range(self.n_):
            h = torch.relu( self.action_dicts[i]['layer_1'](obs[:, i, :]) )
            h = torch.relu( self.action_dicts[i]['layer_2'](h) )
            a = self.action_dicts[i]['action_head'](h)
            actions.append(a)
        actions = torch.stack(actions, dim=1)
        return actions

    def sample_grandcoalitions(self, batch_size):
        seq_set = cuda_wrapper(torch.tril(torch.ones(self.n_, self.n_), diagonal=0, out=None), self.cuda_)
        grand_coalitions = cuda_wrapper(torch.multinomial(torch.ones(batch_size*self.sample_size, self.n_)/self.n_, self.n_, replacement=False), self.cuda_)
        individual_map = cuda_wrapper(torch.zeros(batch_size*self.sample_size*self.n_, self.n_), self.cuda_)
        individual_map.scatter_(1, grand_coalitions.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
        subcoalition_map = torch.matmul(individual_map, seq_set)
        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, self.n_, self.n_).contiguous().view(batch_size, self.sample_size, self.n_, self.n_) # shape = (b, n_s, n, n)
        return subcoalition_map, grand_coalitions

    def marginal_contribution(self, obs, act):
        batch_size = obs.size(0)
        subcoalition_map, grand_coalitions = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim) # shape = (b, n_s, n, n, a)
        act = act.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim).gather(3, grand_coalitions) # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)
        act_map = subcoalition_map.unsqueeze(-1).float() # shape = (b, n_s, n, n, 1)
        act = act * act_map
        act = act.contiguous().view(batch_size, self.sample_size, self.n_, -1) # shape = (b, n_s, n, n*a)
        obs = obs.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, 1, 1, n, o) -> (b, n_s, n, n, o)
        obs = obs.contiguous().view(batch_size, self.sample_size, self.n_, self.n_*self.obs_dim) # shape = (b, n_s, n, n, o) -> (b, n_s, n, n*o)
        inp = torch.cat((obs, act), dim=-1)
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dicts[i]['layer_1'](inp[:, :, i, :]) )
            h = torch.relu( self.value_dicts[i]['layer_2'](h) )
            v = self.value_dicts[i]['value_head'](h)
            values.append(v)
        values = torch.stack(values, dim=2)
        return values

    def get_loss(self, batch):
        batch_size = len(batch.obs)
        n = self.args.agent_num
        action_dim = self.act_dim
        valids, rewards, last_step, done, actions, state, obs, next_state, next_obs = self.unpack_data(batch)
        # do the argmax action on the action loss
        action_out = self.policy(obs)
        actions_ = select_action(self.args, action_out, status='train', exploration=False)
        shapley_values = self.marginal_contribution(obs, actions_).mean(dim=1).contiguous().view(-1, n)
        # do the exploration action on the value loss
        shapley_values_sum = self.marginal_contribution(obs, actions).mean(dim=1).contiguous().view(-1, n).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        # do the argmax action on the next value loss
        if self.args.target:
            next_action_out = self.target_net.policy(next_obs)
        else:
            next_action_out = self.policy(next_obs)
        next_actions_ = select_action(self.args, next_action_out, status='train', exploration=False)
        if self.args.target:
            next_shapley_values_sum = self.target_net.marginal_contribution(next_obs, next_actions_.detach()).mean(dim=1).contiguous().view(-1, n).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        else:
            next_shapley_values_sum = self.marginal_contribution(next_obs, next_actions_.detach()).mean(dim=1).contiguous().view(-1, n).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        assert shapley_values_sum.size() == next_shapley_values_sum.size()
        assert returns.size() == shapley_values_sum.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_shapley_values_sum[i].detach()
            else:
                next_return = next_shapley_values_sum[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - shapley_values_sum
        advantages = shapley_values
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        action_loss = -advantages
        action_loss = action_loss.mean(dim=0)
        value_loss = deltas.pow(2).mean(dim=0)
        return action_loss, value_loss, action_out

    def train_process(self, stat, trainer):
        info = {}
        obs = trainer.env.reset()
        state = [trainer.env.get_state()] * self.args.agent_num
        if self.args.reward_record_type is 'episode_mean_step':
            trainer.mean_reward = 0
            trainer.mean_success = 0
        for t in range(self.args.max_steps):
            obs_ = cuda_wrapper(prep_obs(obs).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            start_step = True if t == 0 else False
            obs_ = cuda_wrapper(prep_obs(obs).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_out = self.policy(obs_, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            _, actual = translate_action(self.args, action, trainer.env)
            actual = [np.argmax(a) for a in actual]
            _, next_obs, reward, done, valid, debug = trainer.env.step(actual)
            next_state = [trainer.env.get_state()] * self.args.agent_num
            if isinstance(done, list): done = np.sum(done)
            trans = self.Transition(state,
                                    obs,
                                    action.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    next_obs,
                                    done,
                                    done,
                                    np.array(valid)
                                    )
            self.transition_update(trainer, trans, stat)
            success = debug['success'] if 'success' in debug else 0.0
            trainer.steps += 1
            if self.args.reward_record_type is 'mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/trainer.steps*(success - trainer.mean_success)
            elif self.args.reward_record_type is 'episode_mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/(t+1)*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/(t+1)*(success - trainer.mean_success)
            else:
                raise RuntimeError('Please enter a correct reward record type, e.g. mean_step or episode_mean_step.')
            stat['mean_reward'] = trainer.mean_reward
            stat['mean_success'] = trainer.mean_success
            obs = next_obs
            state = next_state
        stat['turn'] = t+1
        trainer.episodes += 1
