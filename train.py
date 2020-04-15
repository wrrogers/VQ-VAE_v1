import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import utils

from vqvae_laskin import VQVAE

from params import Parameters

from LungDataset import create_data_loader

from time import time
from time import gmtime

def get_time():
    hours = gmtime()[3]+2
    minutes = gmtime()[4]
    seconds = gmtime()[5]
    return "{}:{}:{}".format(hours, minutes, seconds)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'#,8,9'#,1,2,3'

"""
Hyperparameters
"""
args = Parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_loader = create_data_loader(args.batch_size, split=False)

'''
transform          = transforms.Compose([transforms.ToTensor()])
mnist_trainset     = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
args.trainingset_var    = torch.var(mnist_trainset.data.float())
mnist_testset      = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
args.testset_var        = torch.var(mnist_testset.data.float())

training_loader   = torch.utils.data.DataLoader(mnist_trainset,
                                             batch_size=256,
                                             shuffle=True,
                                             num_workers=0)
testing_loader    = torch.utils.data.DataLoader(mnist_trainset,
                                             batch_size=256,
                                             shuffle=True,
                                             num_workers=0)
'''

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

model = torch.nn.DataParallel(model).to(device)

"""
Set up optimizer and training loop
"""

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'embedding_loss':[],
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def train():

    for i in range(args.n_updates):
        x = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) #/ x_train_var
        loss = recon_loss + embedding_loss

        results["embedding_loss"].append(embedding_loss.cpu().detach().numpy())
        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        loss.sum().backward()
        optimizer.step()

        if i % args.print_interval == 0:
            print('Update #', i, 'Recon Error:',np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename)
                
            plots = [results["embedding_loss"],
                    results["recon_errors"],
                    results["perplexities"],
                    results["loss_vals"]]
            
            plt.figure(figsize=(16,3))
            for n, p in enumerate(plots):
                plt.subplot(1,4,n+1)
                plt.plot(p)
            plt.show()

    np.save('embedding_loss.npy', results['embedding_loss'])
    np.save('recon_loss.npy', results['recon_errors'])
    np.save('perplexities.npy', results['perplexities'])
    np.save('loss.npy', results['loss_vals'])
    
    print('Finished at:',get_time())
        
if __name__ == "__main__":
    train()
    





