"""
tensorboardX

tensorboard --logdir runs
"""

import random
from tensorboardX.writer import SummaryWriter

writer = SummaryWriter()

for t in range(100):
    # generate noise
    r = random.random()
    writer.add_scalar("noise", r, t)

writer.close()
