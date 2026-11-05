# trainstage1.py
import torch,argparse,os
import net,config
from dataset_inter import *
from net_GNN import GNN
from torch.utils.tensorboard import SummaryWriter

# train stage one
from torch.utils.tensorboard import SummaryWriter

def train(args):
 
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current device:", DEVICE)

  
    data_root = './data_shang/out_finish'
    data_root_1 = './data_shang'
    data_dir = os.path.join(data_root)
    data_df = pd.read_csv(os.path.join(data_root_1, 'out_finish.csv'))

    toy_set = GraphDataset(data_dir, data_df, graph_type='ConBAP', dis_threshold=10, create=False)
    train_loader = ToyGraphDataset(toy_set)
    train_data = PLIDataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=2)   ####


    model = GNN(33, 128).to(DEVICE)
    lossLR = net.Loss(args.batch_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    os.makedirs(config.save_path, exist_ok=True)

 
    writer = SummaryWriter(log_dir=os.path.join(config.save_path, 'logs'))
    batch_count = 0
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss = 0

        for batch_idx, (imgL, imgR) in enumerate(train_data):
            
            imgL = [g.to(DEVICE) for g in imgL]
            imgR = [g.to(DEVICE) for g in imgR]

            pre_L = model(imgL)
            pre_R = model(imgR)

            loss = lossLR(pre_L, pre_R)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            print(f"epoch {epoch}, batch {batch_idx}, loss: {batch_loss:.6f}")

         
            writer.add_scalar("Training batch loss", batch_loss, batch_count) #(epoch - 1) * len(train_loader) + batch_idx
            batch_count += 1
            torch.cuda.empty_cache()

        epoch_loss = total_loss / len(train_loader) * args.batch_size
        print("epoch loss:", epoch_loss)

    
        writer.add_scalar("Training epoch loss", epoch_loss, epoch)

    
        with open(os.path.join(config.save_path, "stage_finish_loss.txt"), "a") as f:
            f.write(str(epoch_loss) + " ")

        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, f'model_epoch{epoch}.pth'))

  
    writer.close()



if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--batch_size', default=13, type=int, help='') 
    parser.add_argument('--max_epoch', default=23, type=int, help='')

    args = parser.parse_args()
    train(args)
    

    
    

