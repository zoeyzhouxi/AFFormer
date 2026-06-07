import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange

class MultiScaleFusion(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1 + x2 + x3   
      

class ImportanceGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.multi_scale = MultiScaleFusion(input_channels, output_channels)
        # Add batch normalization to stabilize training
        self.bn = nn.BatchNorm2d(output_channels)
        # Add a learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        # Small constant to avoid numerical instability
        self.eps = 1e-8

    def forward(self, x):
        # Apply multi-scale fusion and batch norm
        x_multi = self.bn(self.multi_scale(x))  # (B, C, H, W)
        
        # Calculate probabilities with temperature scaling
        probs = F.softmax(x_multi / self.temperature, dim=1)
        
        # Calculate entropy with numerical stability
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 将entropy保存为一个一维 Numpy 数组并导出
        import numpy as np
        entropy_np = entropy.detach().cpu().numpy().flatten()
        np.save("entropy_values.npy", entropy_np)
 

        
        # # 将entropy中所有元素值画成一个箱型图，查看分布
        # import matplotlib.pyplot as plt

        # # 将entropy张量拉平为一维
        # entropy_flat = entropy.detach().cpu().numpy().flatten()

        # plt.figure(figsize=(4, 3))
        # plt.boxplot(entropy_flat, vert=False)
        # plt.title("Importance Generator Entropy Distribution")
        # plt.xlabel("Entropy values")
        # plt.tight_layout()
        # plt.savefig("entropy_boxplot.pdf", format='pdf')
        # plt.close()
        # print(entropy.mean())
        # Normalize entropy to [0,1] range for better stability
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + self.eps)

        # # 可视化entropy
        # # This block visualizes the entropy for the input batch and saves it as PDF.
        # import matplotlib.pyplot as plt
        # import os

        # # Take the first entropy map from the batch (B, 1, H, W) -> (H, W)
        # entropy_img = entropy[0, 0].detach().cpu().numpy()

        # plt.figure(figsize=(4, 3))
        # plt.imshow(entropy_img, cmap='hot')
        # plt.title('Importance Generator Entropy')
        # plt.colorbar()
        # plt.tight_layout()

        # # Save as PDF to current working dir or a subfolder as needed
        # pdf_save_path = os.path.join(os.getcwd(), "entropy_vis.pdf")
        # plt.savefig(pdf_save_path, format='pdf')
        # plt.close()
        
        # Compute importance with scaled negative exponential
        importance = torch.exp(-entropy * self.temperature)

        return importance


class ImportanceFusion(nn.Module):
    def __init__(self, input_channels, output_channels, num_embeddings):
        super().__init__()
        self.importance_generators = nn.ModuleList([
            ImportanceGenerator(input_channels, output_channels) for _ in range(num_embeddings)
        ])

    def forward(self, embeddings):
        importance_maps = [fgen(embedding) for fgen, embedding in zip(self.importance_generators, embeddings)]
        importance_maps = torch.cat(importance_maps, dim=1)  # (B, num_embeddings, H, W)
        importance_maps = F.softmax(importance_maps, dim=1)
        # Multiply each embedding with its importance map and sum them up
        fused_embedding = sum(imp * emb for imp, emb in zip(importance_maps.split(1, dim=1), embeddings))
        
        return fused_embedding


########################################################
###################### Feed Forward ####################
########################################################
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Gated-Dconv Feed-Forward Network (GDFN)
class GatedDconvFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, dropout):
        super().__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(self.dropout(x))
        return self.dropout(x)


########################################################
###################### Layer Norm ######################
########################################################        
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm_Conv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm(dim)

    def to_3d(self, x):
        return rearrange(x, 'b c h w -> b (h w) c')

    def to_4d(self, x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    def forward(self, x):
        # x: (B, C, H, W)
        h, w = x.shape[-2:]
        return self.to_4d(self.body(self.to_3d(x)), h, w)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, conv):
        super().__init__()
        if conv == True:
            self.norm = LayerNorm_Conv(dim)
        else:
            self.norm = nn.LayerNorm(dim)

        self.fn = fn

    def forward(self, x, **kwargs): 
        return self.fn(self.norm(x), **kwargs)



######################################################################
###################### Agent Temporal Attention ######################
######################################################################  
class AgentFusionAttention(nn.Module):
    def __init__(self, dim, heads, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, all_x, mask=None):
        # all_x: (B, T, H, W, L, C)
        # mask: (B, 1, 1, 1, 1, L)

        ego_x = all_x[:,:,:,:,0,:] # (B, T, H, W, C)

        q = self.to_q(ego_x).unsqueeze(-2) # (B, T, H, W, 1, C_inner)

        q = rearrange(q, 'b t h w l (m c) -> b m t h w l c', m=self.heads) # (B, M, T, H, W, 1, C_head)

        kv = self.to_kv(all_x).chunk(2, dim=-1) # [(B, T, H, W, L, C_inner) *2]
        k, v = map(lambda t: rearrange(t, 'b t h w l (m c) -> b m t h w l c', m=self.heads), kv) # (B, M, T, H, W, L, C_head)
        att_map = (q @ k.transpose(-2, -1)) * self.scale
        
        att_map = att_map.masked_fill(mask == 0, -float('inf')) # mask the padded agents
        
        # softmax
        att_map = att_map.softmax(dim=-1)

        out = att_map @ v
        out = rearrange(out, 'b m t h w l c -> b t h w l (m c)', m=self.heads) # (B, T, H, W, 1, C_inner)

        return self.to_out(out).squeeze(-2) # (B, T, H, W, C)

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, ego_x):
        # ego_x: (B, H, W, T, C)
        curr_x = ego_x[:,:,:,0,:] # (B, H, W, C)

        q = self.to_q(curr_x).unsqueeze(-2) # (B, H, W, 1, C_inner)
        q = rearrange(q, 'b h w t (m c) -> b m h w t c', m=self.heads) # (B, M, H, W, 1, C_head)

        kv = self.to_kv(ego_x).chunk(2, dim=-1) # [(B, H, W, T, C_inner) *2]

        k, v = map(lambda t: rearrange(t, 'b h w t (m c) -> b m h w t c', m=self.heads), kv) # (B, M, H, W, T, C_head)
        att_map = (q @ k.transpose(-2, -1)) * self.scale
       
        att_map = att_map.softmax(dim=-1)
        out = att_map @ v
        out = rearrange(out, 'b m h w t c -> b h w t (m c)', m=self.heads) # (B, H, W, 1, C_inner)

        out = self.to_out(out) # (B, H, W, 1, C)
        
        return out.squeeze(-2) # (B, H, W, C)

class AgentTemporalAttentionBlock(nn.Module):
    def __init__(self, cav_att_config, feed_config):
        super().__init__()

        depth = cav_att_config['depth']
        dim = cav_att_config['dim']
        heads=cav_att_config['heads']
        dim_head=cav_att_config['dim_head']
        dropout_a=cav_att_config['dropout']

        mlp_dim = feed_config['mlp_dim'] # 256
        dropout_f = feed_config['dropout'] # 0.3

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TemporalAttention(dim,
                                          heads=heads,
                                          dim_head=dim_head,
                                          dropout=dropout_a), False),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout_f), False)
            ]))

        self.agent_fusion_attn = nn.ModuleList([
            PreNorm(dim, AgentFusionAttention(dim, heads, dim_head, dropout_a), False),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout_f), False)
        ])
     
    def forward(self, x, mask): 
        # x: (B, L, T, H, W, C) -> (B, T, H, W, L, C)
        x = x.permute(0, 2, 3, 4, 1, 5) 
        # mask: (B, L) -> (B, 1, 1, 1, 1, L)
        mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        x = self.agent_fusion_attn[0](x, mask=mask) + x[:,:,:,:,0,:]
        x = self.agent_fusion_attn[1](x) + x # (B, T, H, W, C)

        # x: (B, T, H, W, C) -> (B, H, W, T, C)
        x = x.permute(0, 2, 3, 1, 4) 
        for attn, ff in self.layers:
            x = attn(x) + x[:,:,:,0,:] # (B, H, W, C)
            x = ff(x) + x # (B, H, W, C)  

        return x # (B, H, W, C)


###############################################################
###################### Spatial Attention ######################
###############################################################  
class HeightAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads

        self.scale = nn.Parameter(torch.ones(heads, 1, 1))
        
        self.to_qkv = nn.Conv2d(dim, inner_dim*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(inner_dim*3, inner_dim*3, kernel_size=3, stride=1, padding=1, groups=inner_dim)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        qkv = self.qkv_dwconv(self.to_qkv(x)).chunk(3, dim=1) # [(B, C_inner, H, W)*3]
       
        # q: (B, M, C_head, H*W)
        q, k, v = map(lambda t: rearrange(t, 'b (m c) h w -> b w m h c', m=self.heads), qkv)
        att_map = (q @ k.transpose(-2, -1)) * self.scale

        # softmax
        att_map = att_map.softmax(dim=-1)
        out = att_map @ v
        # out: (B, W, M, H, H)x(B, W, M, H, C)=(B, W, M, H, C)
        out = rearrange(out, 'b w m h c -> b (m c) h w') # (B, C_inner, H, W)

        out = self.to_out(out) # (B, C, H, W)

        return out # (B, C, H, W) 

class HeightAttentionBlock(nn.Module):
    def __init__(self, height_att_config, feed_config):
        super().__init__()

        depth = height_att_config['depth']
     
        dim = height_att_config['dim']
        heads=height_att_config['heads']
        dim_head=height_att_config['dim_head']
        ffn_expansion_factor = feed_config['ffn_expansion_factor'] # 2.66
        dropout_f = feed_config['dropout']
        dropout_a = height_att_config['dropout']
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, HeightAttention(dim,
                                          heads,
                                          dim_head,
                                          dropout_a), True),
                PreNorm(dim, GatedDconvFeedForward(dim, ffn_expansion_factor, dropout_f), True)
            ]))

    def forward(self, x): 
        # x: (B, C, H, W)
       
        for attn, ff in self.layers:
            x = attn(x) + x 
            x = ff(x) + x # (B, C, H, W)

        return x

class WidthAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads

        self.scale = nn.Parameter(torch.ones(heads, 1, 1))
        
        self.to_qkv = nn.Conv2d(dim, inner_dim*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(inner_dim*3, inner_dim*3, kernel_size=3, stride=1, padding=1, groups=inner_dim)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, H, W)
       
        qkv = self.qkv_dwconv(self.to_qkv(x)).chunk(3, dim=1) # [(B, C_inner, H, W)*3]
        
        # q: (B, H, M, W, C_head)
        q, k, v = map(lambda t: rearrange(t, 'b (m c) h w -> b h m w c', m=self.heads), qkv)

        att_map = (q @ k.transpose(-2, -1)) * self.scale

        # softmax
        att_map = att_map.softmax(dim=-1)
        out = att_map @ v
        # out: (B, H, M, W, W)x(B, H, M, W, C)=(B, H, M, W, C)
        out = rearrange(out, 'b h m w c -> b (m c) h w') # (B, C_inner, H, W)

        out = self.to_out(out) # (B, C, H, W)
       
        return out # (B, C, H, W) 

class WidthAttentionBlock(nn.Module):
    def __init__(self, width_att_config, feed_config):
        super().__init__()
        depth = width_att_config['depth']
        dim = width_att_config['dim']
        heads=width_att_config['heads']
        dim_head=width_att_config['dim_head']
        dropout_a=width_att_config['dropout']
        ffn_expansion_factor = feed_config['ffn_expansion_factor'] # 2.66
        dropout_f = feed_config['dropout']

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, WidthAttention(dim,
                                          heads,
                                          dim_head,
                                          dropout_a), True),
                PreNorm(dim, GatedDconvFeedForward(dim, ffn_expansion_factor, dropout_f), True)
            ]))
     
    def forward(self, x): 
        # x: (B, C, H, W)
        for attn, ff in self.layers:
            x = attn(x) + x 
            x = ff(x) + x # (B, C, H, W)  

        return x # (B, C, H, W)

class SpatialAttention(nn.Module):
    def __init__(self, width_att_config, height_att_config, feed_config):
        super().__init__()
        
        self.w_attention = WidthAttentionBlock(width_att_config, feed_config)
        self.h_attention = HeightAttentionBlock(height_att_config, feed_config)    

    def forward(self, x):
        # x: (B, C, H, W)
        x_w = self.w_attention(x) # (B, C, H, W)
        x_h = self.h_attention(x) # (B, C, H, W)
        fused_embedding = [x_w, x_h]
        
        return fused_embedding # (B, C, H, W)


#####################################################
###################### Encoder ######################
#####################################################
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        width_att_config = args['width_att_config']
        height_att_config = args['height_att_config']
        feed_config = args['feed_forward']

        self.agent_temporal_attn = AgentTemporalAttentionBlock(cav_att_config, feed_config)

        self.spatial_attn = SpatialAttention(width_att_config, height_att_config, feed_config)

        dim = width_att_config['dim']
        self.importance_fusion = ImportanceFusion(dim, dim, 2)

    def forward(self, x, mask):
        # x: (B, L, K, H, W, C)
        # mask: (B, L)
        
        x_at = self.agent_temporal_attn(x, mask) # (B, H, W, C)
        x_at = x_at.permute(0, 3, 1, 2)
        fused_embedding = self.spatial_attn(x_at) 
        x_s = self.importance_fusion(fused_embedding) # (B, C, H, W)
        
        return x_s, x_at


def main(args, device, model_name):
    
    if model_name == 'DualSA':
        model = SpatialAttention(args['width_att_config'], args['height_att_config'], args['feed_forward']).to(device).eval()
    else:
        from afformer.models.sub_modules.tomv_ablation import WHAttentionBlock
        model = WHAttentionBlock(args['width_att_config'], args['feed_forward']).to(device).eval()
    import argparse
    parser = argparse.ArgumentParser(description="Complexity benchmark: DualSA vs Standard SA")
    parser.add_argument("--batch",    type=int, default=1)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--height",   type=int, default=48)
    parser.add_argument("--width",    type=int, default=176)
    parser.add_argument("--plot",     action="store_true",
                        help="Save latency distribution histogram (requires matplotlib)")

    args = parser.parse_args() 

    B, C, H, W = args.batch, args.channels, args.height, args.width

    x = torch.randn(B, C, H, W).to(device)

    print(f"--- {model_name} ---")

    from afformer.utils.complexity_utils import profile_model, measure_latency, measure_memory
    profile_model(model, x)
    lats = measure_latency(model, x, device)
    mem_mb = measure_memory(model, B, C, H, W, device, model_name)

    return lats

if __name__ == '__main__':

    args={
         'cav_att_config':{
            'depth': 1,
            'dim': 256,
            'dim_head': 32,
            'dropout': 0.3,
            'eps': 1.0e-05,
            'heads': 8,
        'dim_in': 256,
        'dim_out': 64},

        'width_att_config':{
            'depth': 1,
            'dim': 256,
            'heads': 8,
            'dim_head': 32,
            'dropout': 0.3,
            'bias': True,
            },
        'height_att_config': {
            'depth': 1,
            'dim': 256,
            'heads': 8,
            'dim_head': 32,
            'dropout': 0.3,
            'bias': True,
        },
        'feed_forward': {
            'mlp_dim': 256,
            'dropout': 0.3,
            'bias': True,
            'ffn_expansion_factor': 2.66,
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    ds = main(args, device, 'DualSA')
    ss = main(args, device, 'TraditionalSA')

    from afformer.utils.complexity_utils import plot_latency_distribution_histogram
    plot_latency_distribution_histogram(ds, ss)