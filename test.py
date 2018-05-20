from dcgan import DCGAN 
from utils import provide_visual_samples
from constants import use_cuda

import torch
import json

if __name__ == "__main__":

    with open('params.json') as f:
        params = json.load(f)

    objective_type = params['objective_type']
    update_ratio = params['update_ratio']
    learning_rate = params['learning_rate']
    betas = (params['betas'][0], params['betas'][1])
    fake_input_dim = params['fake_input_dim']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']

    model = DCGAN(generator_type=params['generator_type'])

    # load pretrained model

    model.load_state_dict(torch.load(model.name + '.pt'))

    if use_cuda:
    	model = model.cuda()

    provide_visual_samples(model, fake_input_dim=fake_input_dim, batch_size=batch_size)