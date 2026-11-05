import argparse
import os, sys
import pickle

import numpy as np

cwd = os.getcwd()                #获取当前工作目录路径。
if cwd.endswith('learning'):        #
    sys.path.append('../')
else:
    sys.path.append('./')

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", default=True, help="If we don't want to run thing in parallel", action='store_false')
parser.add_argument("-da", "--annotated_data", default='out_dump_500')

parser.add_argument("-bs", "--batch_size", type=int, default=40, help="choose the batch size")      

parser.add_argument("-nw", "--workers", type=int, default=8, help="Number of workers to load data")   

parser.add_argument("-n", "--name", type=str, default='default_name', help="Name for the logs")

parser.add_argument("-ep", "--num_epochs", type=int, help="number of epochs to train", default=400)

parser.add_argument("-rl", "--reconstruction_lam", type=float, help="reconstruction lambda", default=1.0)

parser.add_argument('-ed','--embedding_dims', nargs='+', type=int, help='Dimensions for embeddings.', default=[16,16,16])   ##########
parser.add_argument('-sf','--sim_function', type=str, help='Node similarity function (Supported Options: R_1, IDF).', default="R_1")

parser.add_argument('-w', '--warm_start', type=str, default=None, help='Path to pre-trained model.')

parser.add_argument('-pw', '--pos_weight', type=int, default=0, help='Weight for positive examples.')

parser.add_argument('-rs', '--seed', type=int, default=5, help='Random seed to use (if > 0, else no seed is set).')

parser.add_argument('-kf', '--kfold', type=int, default=0, help='Do k-fold crossval and do decoys on each fold..')

parser.add_argument('-es', '--early_stop', type=int, default=60, help='Early stop epoch threshold (default=10)')

parser.add_argument('-cl', '--clustered',  action='store_true', default=False, help='Predict ligand cluster (default=False)')

parser.add_argument('-cn', '--num_clusts',  type=int, default=5, help='Number of clusters (default=8)')

args = parser.parse_args()

print("OPTIONS USED")
print("\n".join(map(str, zip(vars(args).items()))))  
# Torch impors
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

if args.seed > 0:     
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import learn as learn
from loader import *
from rgcn import *

from utils import mkdirs


print('Done importing')

'''
Hardware settings
'''

# torch.multiprocessing.set_sharing_strategy('file_system')
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# This is to create an appropriate number of workers, but works too with cpu
if args.parallel:
    used_gpus_count = torch.cuda.device_count()
else:
    used_gpus_count = 1



'''
Dataloader creation
'''

annotated_file = './data_processor_zhuo'
annotated_name = args.annotated_data
annotated_path = os.path.join(annotated_file,annotated_name)        #annotated_file, 
print(f'Using {annotated_path} as the pocket inputs')

batch_size = args.batch_size
num_workers = args.workers

dims = args.embedding_dims
#print (dims)


print(f'Using batch_size of {batch_size}')


loader = Loader(annotated_path=annotated_path,
                batch_size=batch_size, num_workers=num_workers,
                sim_function=args.sim_function,
                )


    

print('Created data loader')

'''
Model loading
'''




reconstruction_lam = args.reconstruction_lam


data = loader.get_data(k_fold=args.kfold)
for k, (train_loader, test_loader) in enumerate(data):
    model = Model(dims, device,
                  num_rels=loader.num_edge_types,
                  num_bases=-1,
                  pos_weight=args.pos_weight,
                 clustered=args.clustered,
                  num_clusts=args.num_clusts)

    #if pre-trained initialize matching layers
    if args.warm_start:   
        print("warm starting")
        m = torch.load(args.warm_start, map_location='cpu')['model_state_dict']
        #remove keys not related to embeddings
        for key in list(m.keys()):
            if 'embedder' not in key:
                print("killing ", key)
                del m[key]
        missing = model.load_state_dict(m, strict=False)
        print(missing)

    model = model.to(device)

    print(f'Using {model.__class__} as model')   

    '''
    Optimizer instanciation
    '''

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-7)   ###################
    #optimizer = optim.Adam(model.parameters())   #lr=0.001

    '''
    Experiment Setup
    '''

    name = f"{args.name}_{k}"
    print(name)
    result_folder, save_path = mkdirs(name)
    print(save_path)   #results/trained_models/default_name_0/default_name_0.pth
      
    writer = SummaryWriter(result_folder)  #tensorboard --logdir=results/trained_models/default_name_0
    print(f'Saving result in {result_folder}/{name}')  #results/trained_models/default_name_0/default_name_0





    all_graphs = np.array(test_loader.dataset.dataset.all_graphs)
    test_inds = test_loader.dataset.indices
    train_inds = train_loader.dataset.indices
 
    pickle.dump(({'test': all_graphs[test_inds], 'train': all_graphs[train_inds]}),
                    open(os.path.join(result_folder, f'splits_{k}.p'), 'wb'))

    '''
    Run
    '''
    num_epochs = args.num_epochs

    learn.train_model(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      save_path=save_path,
                      writer=writer,
                      num_epochs=num_epochs,
                      reconstruction_lam=reconstruction_lam,
                      early_stop_threshold=args.early_stop)
