import torch

use_cuda = torch.cuda.is_available()

resize_size = 64

root = './data/'
raw_datafile = root + 'celebA/'
processed_datafile = root + 'resized_celebA/'

results_dir = './results/'