from collections import namedtuple



commnetArgs = namedtuple( 'commnetArgs', ['skip_connection', 'comm_iters'] )

ic3netArgs = namedtuple( 'ic3netArgs', ['comm_iters'] )

maddpgArgs = namedtuple( 'maddpgArgs', [] )

comaArgs = namedtuple( 'comaArgs', ['epsilon_softmax_init', 'epsilon_softmax_end'] )
