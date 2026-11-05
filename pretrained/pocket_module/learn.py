import time
import torch
import torch.nn.functional as F
import sys

import dgl
import networkx as nx

#debug modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
##

if __name__ == '__main__':
    sys.path.append('../')  

from utils import dgl_to_nx
from decoy_utils import *


def send_graph_to_device(g, device):  
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)

    return g


def set_gradients(model, embedding=True, attributor=True): 
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
   # print("12121212")
    for param in model.named_parameters():
        name, p = param
        name = name.split('.')[0]
        if name in ['attributor', 'pool']:
            p.requires_grad = attributor
        if name == 'embedder':
            p.requires_grad = embedding
    pass

def print_gradients(model):  
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p.grad)
    pass
def test(model, test_loader, device, rec_lam=1):  
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """
    model.eval()  
    test_loss,  motif_loss_tot, recons_loss_tot = (0,) * 3     
    all_graphs = test_loader.dataset.dataset.all_graphs 
    test_size = len(test_loader)
    for batch_idx, (graph, K, idx) in enumerate(test_loader):
        # Get data on the devices
        #print (graph   K = K.to(device)
        K = torch.ones(K.shape).to(device) - K   
        graph = send_graph_to_device(graph, device)

        
      
        with torch.no_grad():
             embeddings = model(graph)

             loss = model.compute_loss(embeddings, K, rec_lam=rec_lam)
            

        del K
        del graph

        test_loss += loss.item()

        del loss

        

    return test_loss / test_size    
def train_model(model, criterion, optimizer, device, train_loader, test_loader, save_path,
                writer=None, num_epochs=25, wall_time=None,
                reconstruction_lam=1,
                early_stop_threshold=60):


    edge_map = train_loader.dataset.dataset.edge_map
    all_graphs = train_loader.dataset.dataset.all_graphs

    epochs_from_best = 0   

    start_time = time.time()   
    best_loss = sys.maxsize   
    best_epoch = 0 
    reconstruction_lam_orig = reconstruction_lam  

    batch_size = train_loader.batch_size         
    #if we delay attributor, start with attributor OFF


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Training phase
        model.train()     


        running_loss = 0.0

        time_epoch = time.perf_counter()

        num_batches = len(train_loader)   

        for batch_idx, (graph, K, idx) in enumerate(train_loader):

            # Get data on the devices
            #convert ints to one hots
            print ("***********")
            print (f"处理第 {idx} 个结构图") 
            graph = send_graph_to_device(graph, device)
            K = K.to(device)


            embeddings = model(graph)
            loss = model.compute_loss(embeddings, K,            
                                      rec_lam=reconstruction_lam)   
            # l = model.rec_loss(embeddings, K, similarity=False)
            # print(l)


            # Backward
            loss.backward()    
            optimizer.step()    
            model.zero_grad()   
 
            # Metrics           
            batch_loss = loss.item()   
            running_loss += batch_loss

            if batch_idx % 1 == 0:    
                time_elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Time: {:.2f}'.format(
                    epoch + 1,
                    (batch_idx + 1) * batch_size,
                    num_batches * batch_size,
                    100. * (batch_idx + 1) / num_batches,
                    batch_loss,
                    time_elapsed))                
                writer.add_scalar("Training batch loss", batch_loss,   ##epoch * num_batches + batch_idx 是 x 轴上的 step 值（全局 step）
                                  epoch * num_batches + batch_idx)
            del loss


        # Log training metrics
        train_loss = running_loss / num_batches   
        writer.add_scalar("Training epoch loss", train_loss, epoch)   

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss = test(model, test_loader, device, rec_lam=reconstruction_lam)
        print(">> test loss ", test_loss)

        writer.add_scalar("Test loss during training", test_loss, epoch)    
        
        
      
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch + 1
            epochs_from_best = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
            }, save_path)

        # Early stopping
        else:
            epochs_from_best += 1
            if epochs_from_best > early_stop_threshold:
                print('This model was early stopped')
                break

        # Sanity Check
        if wall_time is not None:
            # Break out of the loop if we might go beyond the wall time
            time_elapsed = time.time() - start_time
            if time_elapsed * (1 + 1 / (epoch + 1)) > .95 * wall_time * 3600:
                break

        with open("result_loss.txt", "a") as f:
            f.write(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | ")
            f.write(f"Best Loss: {best_loss:.6f} (Epoch {best_epoch})\n")
        del test_loss      

    return best_loss

if __name__ == "__main__":
    pass
