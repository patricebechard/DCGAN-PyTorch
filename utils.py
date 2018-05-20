from constants import use_cuda, processed_datafile, results_dir

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets


def provide_visual_samples(model, batch_size=32, fake_input_dim=100, title='Samples'):

    # use this function for question 5. (a)

    fake = Variable(torch.randn(batch_size, fake_input_dim))
    if use_cuda:
        fake = fake.cuda()
    fake = model.generator(fake).data
    if use_cuda:
        fake = fake.cpu()

    fake = fake[:6]

    fake = torchvision.utils.make_grid(fake, nrow=len(fake), padding=0)
    fake = fake.numpy().transpose((1, 2, 0))


    # Just a figure and one subplot
    f, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(fake)
    ax.set_title(title)
    ax.axis('off')

    plt.tight_layout()

    plt.savefig(results_dir + 'samples.png')

def changes_in_latent_space(model):
   
    sample = torch.randn(1, fake_input_dim)
    
    # we choose components of z for which we change the value completely 
    for i in range(3):
      
        interp = torch.zeros(11, fake_input_dim)
        var = np.linspace(-10, 10, 11)

        # we vary this component from (value - 5) to (value + 5)
        for j in range(len(var)):
            interp[j] = sample[0]
            interp[j, i] = interp[j, i] + var[j]

        #generating images from every point
        interp = Variable(interp)
        if use_cuda:
            interp = interp.cuda()
        interp = model.generator(interp).data
        if use_cuda:
          interp = interp.cpu()

        img = torchvision.utils.make_grid(interp, nrow=len(interp), padding=1)
        img = img.numpy().transpose((1, 2, 0))

        # plot samples
        f, ax = plt.subplots(figsize=(18, 6))
        ax.imshow(img)
        ax.set_title("Varying component %d" % i)
        ax.axis('off')

def interpolate_between_images(model):
  
    # use this function for question 5. (c)
    
    #choosing two random points
    z0 = torch.randn(1, fake_input_dim)
    z1 = torch.randn(1, fake_input_dim)
    
    #interpolate between the two points in latent space
    interp_z = torch.zeros(11, fake_input_dim)
    for i in range(11):
        alpha = 0.1 * i
        interp_z[i] = alpha * z0 + (1-alpha) * z1
    
    #generating images from every point
    interp_z = Variable(interp_z)
    if use_cuda:
        interp_z = interp_z.cuda()
    interp_z = model.generator(interp_z).data
    if use_cuda:
      interp_z = interp_z.cpu()
   
    img_z = torchvision.utils.make_grid(interp_z, nrow=len(interp_z), padding=1)
    img_z = img_z.numpy().transpose((1, 2, 0))
    
    # plot samples
    f, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(img_z)
    ax.set_title("Interpolation in z space")
    ax.axis('off')  

    # interpolating between two images in x space
    interp_x = torch.zeros_like(interp_z)
    x0 = interp_z[-1]
    x1 = interp_z[0]

    for i in range(11):
        alpha = 0.1 * i
        interp_x[i] = alpha * x0 + (1-alpha) * x1
        
    img_x = torchvision.utils.make_grid(interp_x, nrow=len(interp_z), padding=1)
    img_x = img_x.numpy().transpose((1, 2, 0))
      
    # plot samples
    f, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(img_x)
    ax.set_title("Interpolation from x space")
    ax.axis('off')  

def load_model(model, file_name='model.pt'):
    pass 

def save_model(model, file_name='model.pt'):
    pass     
 
def imshow(real, fake, show=True, save=False, epoch=0):
    """Imshow for Tensor."""

    #keep only 4 images for both real and fake inputs
    real = real[:4]
    fake = fake[:4]

    # Make a grid from batch
    real = torchvision.utils.make_grid(real, nrow=2, padding=0)
    fake = torchvision.utils.make_grid(fake, nrow=2, padding=0)

    #convert to numpy
    real = real.numpy().transpose((1, 2, 0))
    fake = fake.numpy().transpose((1, 2, 0))

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9,4))
    ax1.imshow(real)
    ax1.set_title('Real examples')
    ax1.axis('off')
    ax2.imshow(fake)
    ax2.set_title('Fake examples')
    ax2.axis('off')
    if show:
        plt.pause(0.001)
    if save:
        plt.savefig(results_dir + 'samples_%d.png' % epoch)
    plt.clf()

def create_dataloader(batch_size=64):

    data_transforms = transforms.Compose([transforms.ToTensor()])

    dataset =datasets.ImageFolder(processed_datafile, data_transforms)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
    
    return dataloader

if __name__ == "__main__":

    dataset = create_dataloader()

    for i in enumerate(dataset):

        print(i)

        break