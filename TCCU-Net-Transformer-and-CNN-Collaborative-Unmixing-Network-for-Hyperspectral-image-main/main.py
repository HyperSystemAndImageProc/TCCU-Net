import random
import torch
import numpy as np
import TCCU_model

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nSelected device:", device, end="\n\n")

tmod = TCCU_model.Train_test(dataset='apex', device=device, skip_train=False, save=True)
tmod.run(smry=False)
