# from google.colab import files

def train(model, objective_type='gan', update_ratio=3, n_episodes=2000, print_every=100,
          batch_size=128, fake_input_dim=100, learning_rate=0.0002, betas=(0.5, 0.999)):
    """
    Parameters
    ----------
    objective_type : string
        Objective we are maximizing using the model.
        Takes value in ['gan', 'wgan', 'lsgan']
        For the assignment, we only need to implement one
 ### NV - We actually need to implement 2 of them.       
   
    n_episodes : int
        Number of times we update the whole model.
        
    update_ratio : int
        Number of times we update the discriminator before updating the generator.
        Called 'k' in Goodfellow et al. 2014
    
    """
  
    # TODO implement 'gan', 'wgan' and 'lsgan' objective types
    # TODO implement Inception score and one other (Mode score, Wasserstein distance, Maximum Mean Discrepancy (MMD))
  
    if use_cuda:
        model = model.cuda()
    
    if objective_type == 'lsgan':
      criterion = nn.MSELoss()
    else:
      criterion = nn.BCEWithLogitsLoss()
      
    D_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=betas)
    G_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=betas)
    
    for episode in range(n_episodes):
      
        batch_loader = iter(train_dataloader)
        loss_episode = 0
      
        # Updating the discriminator
        for i in range(update_ratio):
          
            model.discriminator.zero_grad()
            
            # feeding real data
            inputs_real, _ = batch_loader.next()
            targets_real = torch.ones(batch_size)
            
            inputs_real, targets_real = Variable(inputs_real), Variable(targets_real)
            if use_cuda:
                inputs_real, targets_real = inputs_real.cuda(), targets_real.cuda()
            
            outputs_real = model(inputs_real, fake=False).squeeze()
            
            #if objective_type == 'lsgan':
              #outputs_real = F.sigmoid(outputs_real)            
            
            loss = criterion(outputs_real, targets_real)
            loss_episode += loss.data[0]
            
            loss.backward()

            # feeding fake data
            inputs_fake = torch.randn(batch_size, fake_input_dim)
            targets_fake = torch.zeros(batch_size)
            
            inputs_fake, targets_fake = Variable(inputs_fake), Variable(targets_fake)
            if use_cuda:
                inputs_fake, targets_fake = inputs_fake.cuda(), targets_fake.cuda()
            
            outputs_fake = model(inputs_fake, fake=True).squeeze()
            
            #if objective_type == 'lsgan':
              #outputs_fake = F.sigmoid(outputs_fake)            
            
            loss = criterion(outputs_fake, targets_fake)
            loss_episode += loss.data[0]
            loss.backward()

            D_optimizer.step()
        
        # Updating the generator                        
        model.generator.zero_grad()        
        
        inputs_fake = torch.randn(batch_size, fake_input_dim)
        # We want to fool the discriminator. We label these fake examples as real
    ### NV - (just a note to myself)
    ### NV - It's more like we use the Discriminator to learn the... ###
    ### NV - ...Generator's parameters adapting them in order to give real (1) ###
        targets_fake = torch.ones(batch_size)
        inputs_fake, targets_fake = Variable(inputs_fake), Variable(targets_fake)
        if use_cuda:
            inputs_fake, targets_fake = inputs_fake.cuda(), targets_fake.cuda()
        
        outputs_fake = model(inputs_fake, fake=True).squeeze()
        
        #if objective_type == 'lsgan':
          #outputs_fake = F.sigmoid(outputs_fake)        
        
        loss = criterion(outputs_fake, targets_fake)
        
        loss.backward()
        G_optimizer.step()
        
        
        if episode % print_every == 0:
          
            print("Episode : %d" % episode)
            print("Loss for discriminator: ", loss_episode)
            
            #for now, just show a couple examples of real examples vs fake examples
            
            #real_img
            real, _ = batch_loader.next()

            #fake_img
            fake = Variable(torch.randn(batch_size, fake_input_dim))
            if use_cuda:
                fake = fake.cuda()
            fake = model.generator(fake).data
            if use_cuda:
              fake = fake.cpu()

            imshow(real, fake)
            
            
#             torch.save(model.state_dict(), model.name + '.pt')

#             files.download(model.name + '.pt')
            
        