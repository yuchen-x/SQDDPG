import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.ddpg import *



class MADDPG(Model):

    def __init__(self, args, target_net=None):
        super(MADDPG, self).__init__(args)
        self.rl = DDPG(self.args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()

    def reload_params_to_target(self):
        self.target_net.action_dict.load_state_dict( self.action_dict.state_dict() )
        self.target_net.value_dict.load_state_dict( self.value_dict.state_dict() )

    def update_target(self):
        params_target_action = list(self.target_net.action_dict.parameters())
        params_behaviour_action = list(self.action_dict.parameters())
        for i in range(len(params_target_action)):
            params_target_action[i] = (1 - self.args.target_lr) * params_target_action[i] + self.args.target_lr * params_behaviour_action[i]
        params_target_value = list(self.target_net.value_dict.parameters())
        params_behaviour_value = list(self.value_dict.parameters())
        for i in range(len(params_target_value)):
            params_target_value[i] = (1 - self.args.target_lr) * params_target_value[i] + self.args.target_lr * params_behaviour_value[i]
        # print ('traget net is updated!\n')

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'layer_1': nn.ModuleList( [ nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                           'layer_2': nn.ModuleList( [ nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                           'action_head': nn.ModuleList( [ nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_) ] )
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList( [ nn.Linear( (self.obs_dim+self.act_dim)*self.n_, self.hid_dim ) for _ in range(self.n_) ] ),\
                                          'layer_2': nn.ModuleList( [ nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                          'value_head': nn.ModuleList( [ nn.Linear(self.hid_dim, 1) for _ in range(self.n_) ] )
                                         }
                                       )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        actions = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['layer_1'][i](obs[:, i, :]) )
            h = torch.relu( self.action_dict['layer_2'][i](h) )
            a = self.action_dict['action_head'][i](h)
            actions.append(a)
        actions = torch.stack(actions, dim=1)
        return actions

    def value(self, obs, act):
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i]( torch.cat( ( obs.contiguous().view( -1, np.prod(obs.size()[1:]) ), act.contiguous().view( -1, np.prod(act.size()[1:]) ) ), dim=-1 ) ) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.rl.get_loss(batch, self, self.target_net)
        return action_loss, value_loss, log_p_a
