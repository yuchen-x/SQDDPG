import torch
from tester import *
from arguments import *


PATH='./exp1/simple.pt'
policy_net = torch.load(PATH)

test = Tester(env(), policy_net, args)
episodes = 10
render = True
test.run_game(episodes, render)