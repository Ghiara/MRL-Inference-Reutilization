from collections import OrderedDict
from smrl.trainers.transfer_function import TransferFunction

transfer_config = OrderedDict(
    obs_dim = 20,
    act_simple_dim = 1,
    act_complex_dim = 6,
    hidden_sizes = [64,64,64],
)

