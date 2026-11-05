# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet50
#from resnet import *


# stage one ,unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=8):
        super(SimCLRStage1, self).__init__()

        #self.f = []


        self.resnet_dict = { "resnet18": generate_model(18)}

        self.backbone = self._get_basemodel('resnet18')
        #dim_mlp = self.backbone.layer4.out_channels
        #print(self.backbone.layer4.1.bn2.out_channels)
        dim_mlp=256
        n_classes=128

        # add mlp projection head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc= nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, n_classes))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# stage two ,supervised learning
class SimCLRStage2(torch.nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


class Loss(nn.Module):
    def __init__(self, batch_size, temperature=0.05, verbose=False):
        super().__init__()
       ##### #self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
        #temperature：温度系数，控制计算余弦相似度时的缩放。温度越低，正负样本的差异越显著；温度越高，相似度差异较小。

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        #分别是同一结构的两个增强样本的嵌入表示。
        """
        
        z_i = F.normalize(emb_i.squeeze(1), dim=1)   #归一化
        z_j = F.normalize(emb_j.squeeze(1), dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)   #拼接(2 * batch_size, feature_dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        N = z_i.shape[0]
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):

            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)   
            one_for_not_i = torch.ones((2 * N,), device=emb_i.device)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)
 

        loss = 0.0
        for k in range(0, N):  
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss



if __name__=="__main__":
    for name, module in resnet50().named_children():
        print(name,module)


