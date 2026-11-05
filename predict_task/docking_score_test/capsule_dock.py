import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(x, dim=-1):
    norm = torch.norm(x, dim=dim, keepdim=True)
    return (norm**2 / (1 + norm**2)) * (x / (norm + 1e-8))


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_dim, out_dim, num_routes,routing_iters):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.routing_iters = routing_iters

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_dim, in_dim))

    def forward(self, x, return_coeff=False):
        # x: [batch, num_routes, in_dim]
        x = x.unsqueeze(2).unsqueeze(-1)  # [batch, num_routes, 1, in_dim, 1]
        W = self.W.repeat(x.size(0), 1, 1, 1, 1)
        u_hat = torch.matmul(W, x).squeeze(-1)  # [batch, num_routes, num_capsules, out_dim]

        b = torch.zeros(x.size(0), self.num_routes, self.num_capsules, device=x.device)

        for it in range(self.routing_iters):    
            c = F.softmax(b, dim=2)  # [batch, num_routes, num_capsules]
           # print(c.mean(dim=0))
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)  # [batch, num_capsules, out_dim]
            v = squash(s)
            if it < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)
        
        if return_coeff:
            return v, c.detach(), u_hat.detach()
        else:
            return v
       

class FusionCapsuleRegression(nn.Module):
    def __init__(self, RGCN_dim, EGNN_dim, SE_dim, 
                 num_RGCN_caps=6, num_EGNN_caps=8, num_SE_caps=5,
                 num_out_caps=4,               
                 capsule_dim=64, dropout_p=0.1,routing_iters=2,reg_hidden_dim=128 ):
        super().__init__()

        # ---- Pre-LN ----
        
        self.RGCN_preLN = nn.LayerNorm(RGCN_dim)
        self.EGNN_preLN = nn.LayerNorm(EGNN_dim)
        self.SE_preLN   = nn.LayerNorm(SE_dim)

        # ---- projection ----
        self.RGCN_proj = nn.Linear(RGCN_dim, num_RGCN_caps * capsule_dim)
        self.EGNN_proj = nn.Linear(EGNN_dim, num_EGNN_caps * capsule_dim)
        self.SE_proj   = nn.Linear(SE_dim, num_SE_caps   * capsule_dim)
        
        self.modality_gate = nn.Parameter(torch.ones(3))     ######gate

        # ---- post-proj LN ----
        self.RGCN_ln = nn.LayerNorm(capsule_dim)
        self.EGNN_ln = nn.LayerNorm(capsule_dim)
        self.SE_ln   = nn.LayerNorm(capsule_dim)

        self.num_RGCN_caps = num_RGCN_caps
        self.num_EGNN_caps = num_EGNN_caps
        self.num_SE_caps   = num_SE_caps
        self.capsule_dim   = capsule_dim

        self.dropout = nn.Dropout(dropout_p)


        self.capsule_layer = CapsuleLayer(
            num_capsules=num_out_caps,    
            in_dim=capsule_dim,
            out_dim=capsule_dim,
            num_routes=num_RGCN_caps + num_EGNN_caps + num_SE_caps,
            routing_iters=routing_iters  
        )


        self.attn_fc = nn.Linear(capsule_dim, 1)

        # ---- post capsule LN ----
        self.post_caps_ln = nn.LayerNorm(capsule_dim)

        # ---- regressor ----
        self.regressor = nn.Sequential(
            nn.Linear(capsule_dim, reg_hidden_dim),
            nn.ReLU(),
            nn.Linear(reg_hidden_dim, 1)
        )
     
    def forward(self, RGCN_emb, EGNN_emb, SE_emb, return_coeff=False):
        B = RGCN_emb.size(0)

        # ---- Pre-LN ----
        RGCN_emb = self.RGCN_preLN(RGCN_emb)
        EGNN_emb = self.EGNN_preLN(EGNN_emb)
        SE_emb   = self.SE_preLN(SE_emb)

        # ---- Projection ----
        RGCN_caps = self.RGCN_proj(RGCN_emb).view(B, self.num_RGCN_caps, self.capsule_dim)
        EGNN_caps = self.EGNN_proj(EGNN_emb).view(B, self.num_EGNN_caps, self.capsule_dim)
        SE_caps   = self.SE_proj(SE_emb).view(B, self.num_SE_caps, self.capsule_dim)

        # ---- Post-proj LN ----
        RGCN_caps = self.RGCN_ln(RGCN_caps)
        EGNN_caps = self.EGNN_ln(EGNN_caps)
        SE_caps   = self.SE_ln(SE_caps)
        
        gate = torch.softmax(self.modality_gate, dim=0)  #####gate

        # ---- Fusion ----
        caps_input = torch.cat([RGCN_caps, EGNN_caps, SE_caps], dim=1)
        caps_input = self.dropout(caps_input)
        
        RGCN_caps = RGCN_caps * gate[0]
        EGNN_caps = EGNN_caps * gate[1]
        SE_caps   = SE_caps   * gate[2]
        caps_input = torch.cat([RGCN_caps, EGNN_caps, SE_caps], dim=1)
        caps_input = self.dropout(caps_input)
        

        # ---- Capsule routing ----
        caps_output, routing_c,u_hat = self.capsule_layer(caps_input, return_coeff=True)
        # caps_output: [B, num_out_caps, D]
        #print(routing_c[0])
        # ---- Capsule-level attention ----
        attn_score = self.attn_fc(caps_output)              # [B, K, 1]
        attn_weight = torch.softmax(attn_score, dim=1)     
        caps_feat = (attn_weight * caps_output).sum(dim=1) # [B, D]

        # ---- Post capsule ----
        caps_feat = self.post_caps_ln(caps_feat)
        caps_feat = self.dropout(caps_feat)

        pred = self.regressor(caps_feat).squeeze(1)

        if return_coeff:
            #return pred, routing_c, attn_weight.detach()
            return pred, routing_c, u_hat
        else:
            return pred




   
