from dcgan import DCGAN
from constants import use_cuda
from utils import create_dataloader, imshow

import torch
from torch import nn
from torch import optim 
from torch.autograd import Variable

import json
import time

def train(model, dataloader, n_epochs=30, objective_type='gan', update_ratio=3, save_model=False,
          batch_size=128, fake_input_dim=100, learning_rate=0.0002, betas=(0.5, 0.999)):
    """
    Parameters
    ----------
    objective_type : string
        Objective we are maximizing using the model.
        Takes value in ['gan', 'wgan', 'lsgan']
        #TODO implement wgan objective    
        
    update_ratio : int
        Number of times we update the discriminator before updating the generator.
        Called 'k' in Goodfellow et al. 2014
    """
  
    if use_cuda:
        model = model.cuda()
    
    if objective_type == 'gan':
        criterion = nn.BCEWithLogitsLoss()
    elif objective_type == 'lsgan':
      criterion = nn.MSELoss()
    else:
        raise NotImplementedError ('Objective type "%s" not implemented yet'%objective_type)
      
    D_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=betas)
    G_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=betas)
    
    n_batches = len(dataloader)

    for epoch in range(n_epochs):

        start = time.time()
        for batch_idx, (inputs_real, _) in enumerate(dataloader):
      
            # Updating the discriminator
            model.discriminator.zero_grad()
            
            # feeding real data
            targets_real = torch.ones(inputs_real.shape[0])
            
            inputs_real, targets_real = Variable(inputs_real), Variable(targets_real)
            if use_cuda:
                inputs_real, targets_real = inputs_real.cuda(), targets_real.cuda()
            
            outputs_real = model(inputs_real, fake=False).squeeze()
            
            loss = criterion(outputs_real, targets_real)
            loss.backward()

            # feeding fake data
            inputs_fake = torch.randn(batch_size, fake_input_dim)
            targets_fake = torch.zeros(batch_size)
            
            inputs_fake, targets_fake = Variable(inputs_fake), Variable(targets_fake)
            if use_cuda:
                inputs_fake, targets_fake = inputs_fake.cuda(), targets_fake.cuda()
            
            outputs_fake = model(inputs_fake, fake=True).squeeze()       
            
            loss = criterion(outputs_fake, targets_fake)
            loss.backward()

            D_optimizer.step()

            if batch_idx % update_ratio == 0:

                # Updating the generator                        
                model.generator.zero_grad()        
                
                inputs_fake = torch.randn(batch_size, fake_input_dim)
                # We want to fool the discriminator. We label these fake examples as real
                targets_fake = torch.ones(batch_size)

                inputs_fake, targets_fake = Variable(inputs_fake), Variable(targets_fake)
                if use_cuda:
                    inputs_fake, targets_fake = inputs_fake.cuda(), targets_fake.cuda()
                
                outputs_fake = model(inputs_fake, fake=True).squeeze()    
                
                loss = criterion(outputs_fake, targets_fake)
                loss.backward()
                G_optimizer.step()

            # time utils
            rem_time = (time.time()-start) * (n_batches-batch_idx + 1) / (batch_idx + 1)
            
            rem_h = int(rem_time // 3600)
            rem_m = int(rem_time // 60 - rem_h * 60)
            rem_s = int(rem_time % 60)
            print("Batch : %d / %d ----- Time remaining for the epoch : %02d:%02d:%02d" % (batch_idx, n_batches, rem_h, rem_m, rem_s), end="\r")

        print()
        print("Epoch : %d" % epoch)
                
        #real_img
        real = inputs_real
        if use_cuda:
            real = real.cpu()

        #fake_img
        fake = Variable(torch.randn(batch_size, fake_input_dim))
        if use_cuda:
            fake = fake.cuda()
        fake = model.generator(fake).data
        if use_cuda:
          fake = fake.cpu()

        imshow(real, fake, show=False, save=True, epoch=epoch)

        if save_model:
            torch.save(model.state_dict(), model.name + '.pt')

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

    train(model, dataset, objective_type=objective_type, save_model=True, 
          update_ratio=update_ratio, fake_input_dim=fake_input_dim,
          learning_rate=learning_rate, betas=betas, n_epochs=n_epochs,
          batch_size=batch_size)
        