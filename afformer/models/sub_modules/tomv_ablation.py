import torch
import torch.nn as nn
from einops import rearrange
import numbers
import torch.nn.functional as F
from afformer.models.sub_modules.tomv_basic import AgentTemporalAttentionBlock, SpatialAttention, ImportanceFusion

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

class WHAttention(nn.Module):
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
        _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.to_qkv(x)).chunk(3, dim=1) # [(B, C_inner, H, W)*3]
        
        # q: (B, H, M, W, C_head)
        q, k, v = map(lambda t: rearrange(t, 'b (m c) h w -> b m (h w) c', m=self.heads), qkv)
        att_map = (q @ k.transpose(-2, -1)) * self.scale

        # softmax
        att_map = att_map.softmax(dim=-1)

        out = att_map @ v
        out = rearrange(out, 'b m (h w) c -> b (m c) h w', h=h, w=w) # (B, C_inner, H, W)

        out = self.to_out(out) # (B, C, H, W)
       
        return out # (B, C, H, W)

class WHAttentionBlock(nn.Module):
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
                PreNorm(dim, WHAttention(dim,
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

class AgentAttention(nn.Module):
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
        # all_x: (B, H, W, L, C)
        # mask: (B, 1, 1, 1, L)

        ego_x = all_x[:,:,:,0,:] # (B, H, W, C)

        q = self.to_q(ego_x).unsqueeze(-2) # (B, T, H, W, 1, C_inner)

        q = rearrange(q, 'b h w l (m c) -> b m h w l c', m=self.heads) # (B, M, T, H, W, 1, C_head)

        kv = self.to_kv(all_x).chunk(2, dim=-1) # [(B, T, H, W, L, C_inner) *2]
        k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c', m=self.heads), kv) # (B, M, T, H, W, L, C_head)
        att_map = (q @ k.transpose(-2, -1)) * self.scale
    
        att_map = att_map.masked_fill(mask == 0, -float('inf')) # mask the padded agents
        
        # softmax
        att_map = att_map.softmax(dim=-1)
        out = att_map @ v
        out = rearrange(out, 'b m h w l c -> b h w l (m c)', m=self.heads) # (B, T, H, W, 1, C_inner)

        return self.to_out(out).squeeze(-2) # (B, T, H, W, C)

class AgentAttentionBlock(nn.Module):
    def __init__(self, cav_att_config, feed_config):
        super().__init__()

        depth = cav_att_config['depth']
        dim = cav_att_config['dim']
        heads=cav_att_config['heads']
        dim_head=cav_att_config['dim_head']
        dropout_a=cav_att_config['dropout']

        mlp_dim = feed_config['mlp_dim'] # 256
        dropout_f = feed_config['dropout'] # 0.3

        self.agent_fusion_attn = nn.ModuleList([
            PreNorm(dim, AgentAttention(dim, heads, dim_head, dropout_a), False),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout_f), False)
        ])
     
    def forward(self, x, mask): 
        # x: (B, L, T, H, W, C)
        x = x[:,:,0,:,:,:] # (B, L, H, W, C)
        x = x.permute(0, 2, 3, 1, 4) # (B, H, W, L, C)
        # mask: (B, L) -> (B, 1, 1, 1, L)
        mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        x = self.agent_fusion_attn[0](x, mask=mask) + x[:,:,:,0,:]
        x = self.agent_fusion_attn[1](x) + x # (B, H, W, C)

        return x # (B, H, W, C)

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

class TemporalAttentionBlock(nn.Module):
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
     
    def forward(self, x, mask): 
        # x: (B, L, T, H, W, C) 
        x = x[:,0,:,:,:,:] # (B, T, H, W, C)
        # x: (B, T, H, W, C) -> (B, H, W, T, C)
        x = x.permute(0, 2, 3, 1, 4) 
        for attn, ff in self.layers:
            x = attn(x) + x[:,:,:,0,:] # (B, H, W, C)
            x = ff(x) + x # (B, H, W, C)  

        return x # (B, H, W, C)

class DualsaEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        width_att_config = args['width_att_config']
        feed_config = args['feed_forward']
        dim = width_att_config['dim']

        self.agent_temporal_attn = AgentTemporalAttentionBlock(cav_att_config, feed_config)       
        self.spatial_attn = WHAttentionBlock(width_att_config, feed_config)
        self.importance_fusion = ImportanceFusion(dim, dim, 2)

    def forward(self, x, mask):
        # x: (B, L, K, H, W, C)
        # mask: (B, L)       
        x_at = self.agent_temporal_attn(x, mask) # (B, H, W, C)
        x_at = x_at.permute(0, 3, 1, 2)
        x_s = self.spatial_attn(x_at) # (B, C, H, W)
  
        return x_s, x_at

class IatEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        temporal_att_config = args['temporal_att_config']
        width_att_config = args['width_att_config']
        feed_config = args['feed_forward']

        dim = width_att_config['dim']
        
        self.temporal_attn = TemporalAttentionBlock(temporal_att_config, feed_config)
        self.spatial_attn = SpatialAttention(width_att_config, feed_config)
        self.importance_fusion = ImportanceFusion(dim, dim, 2)

    def forward(self, x, mask):
        # x: (B, L, K, H, W, C)
        # mask: (B, L)

        x_at = self.temporal_attn(x, mask) # (B, H, W, C)
        x_at = x_at.permute(0, 3, 1, 2)
           
        x_w, x_h = self.spatial_attn(x_at) 
        fused_embedding = [x_w, x_h]
        x_s = self.importance_fusion(fused_embedding) # (B, C, H, W)

        return x_s, x_at

class TatEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        width_att_config = args['width_att_config']
        feed_config = args['feed_forward']
        dim = width_att_config['dim']
        self.agent_attn = AgentAttentionBlock(cav_att_config, feed_config)
        self.spatial_attn = SpatialAttention(width_att_config, feed_config)
        self.importance_fusion = ImportanceFusion(dim, dim, 2)


    def forward(self, x, mask):
        # x: (B, L, K, H, W, C)
        # mask: (B, L)

        x_at = self.agent_attn(x, mask) # (B, H, W, C)
        x_at = x_at.permute(0, 3, 1, 2)
        x_w, x_h = self.spatial_attn(x_at) 
        fused_embedding = [x_w, x_h]
        x_s = self.importance_fusion(fused_embedding) # (B, C, H, W)

        return x_s, x_at

class MataEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        width_att_config = args['width_att_config']
        feed_config = args['feed_forward']
        dim = width_att_config['dim']
        self.spatial_attn = SpatialAttention(width_att_config, feed_config)
        self.importance_fusion = ImportanceFusion(dim, dim, 2)

    def forward(self, x, mask):
        # x: (B, L, K, H, W, C)
        # mask: (B, L)
        
        x_a = torch.sum(x, dim=1, keepdim=True) # (B, K, H, W, C)
        x_t = torch.sum(x_a.squeeze(1), dim=1, keepdim=True) # (B, H, W, C)

        x_t = (x_t.squeeze(1)).permute(0, 3, 1, 2)
        x_w, x_h = self.spatial_attn(x_t) # (B, C, H, W)
        fused_embedding = [x_w, x_h]
        x_s = self.importance_fusion(fused_embedding) # (B, C, H, W)     

        return x_s, x_t

class UgfEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        width_att_config = args['width_att_config']
        feed_config = args['feed_forward']
        
        self.agent_temporal_attn = AgentTemporalAttentionBlock(cav_att_config, feed_config)
        self.spatial_attn = SpatialAttention(width_att_config, feed_config)


    def forward(self, x, mask):
        # x: (B, L, K, H, W, C)
        # mask: (B, L)

        x_at = self.agent_temporal_attn(x, mask) # (B, H, W, C)
        x_at = x_at.permute(0, 3, 1, 2)
        x_w, x_h = self.spatial_attn(x_at) 
        x_s = x_w + x_h
        
        return x_s, x_at