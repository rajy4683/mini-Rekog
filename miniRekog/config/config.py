import torch

class Config(dict):
    def __init__(self, **kwargs):
        """
        Initialize an instance of this class.

        Args:

        """
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value):
        """
        Sets the value to the value.

        Args:
            key: (str):
            value:
        """
        self[key] = value
        setattr(self, key, value)

    def get_dict(self):
        return vars(self)

#### Default configuration and hyperparameters

config = Config(
    dropout = 0.1,
    batch_size = 512,
    test_batch_size=128,
    lr = 0.1,
    momentum = 0.9,
    no_cuda = False,
    seed = 1,
    epochs = 24,
    bias = False,
    sched_lr_gamma = 0.5, ## Gamma to apply for StepLR
    sched_lr_step= 1, ## After how many steps LR will be reduced
    start_lr = 0, ## After how many epochs to start StepLR scheduling
    weight_decay=0.0000, ### Weight decay value
    reg_l1=False, ## Enable L1 regularization 
    reg_l2=False, ## Enable L2 regularization 
    norm_strategy="BatchNorm", ## Which Normalization scheme to use i.e BN/GN/LN
    lr_decay_threshold=0.0,
    factor=0.0,
    project="news5",
    ocp_max_lr=0.5,
    final_div_factor=64,
    div_factor=128,
    anneal_strategy='linear',
    pct_start=0.208,
    cycle_momentum=False,
    lr_policy="ocp",
    split_pct=0.208,
    unfreeze_layer=3,
)
#    lr_decay_threshold=0.0,
#    factor=0.0,
#    project="news5",
#    ocp_max_lr=0.5,
#    final_div_factor=64,
#    div_factor=128,
#    anneal_strategy='linear',
#    pct_start=0.208,
#    cycle_momentum=False,
#    lr_policy="ocp",
#    split_pct=0.208,
#    unfreeze_layer=3,
