import random
import math
from .ssd_config import ssd_config

def rand_chip():
    return random.randrange(ssd_config.num_chip)

def rand_channel():
    return random.randrange(ssd_config.num_channel)

def page_align_sz(data_sz):
    pg_sz = ssd_config.pg_sz_kb * 1e3
    return math.ceil(data_sz / pg_sz) * pg_sz