import math
import os

import torch
import numpy as np

from einops import rearrange
from afformer.utils.common_utils import torch_tensor_to_numpy
from typing import Sequence, Optional, Iterable, Dict, Any

def add_awgn_bev(
    x: torch.Tensor,
    snr_db: torch.Tensor,
    signal_dims: Sequence[int],
    eps: float = 1e-12,
    clamp: Optional[tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Add AWGN to real-valued BEV feature tensor.

    Args:
        x: real tensor, e.g. (B,N,C,H,W) or (B,N,T,C,H,W)
        snr_db: target SNR in dB. Can be:
            - scalar tensor ()
            - (B,) or (B,N) or any shape broadcastable to x with singleton dims.
        signal_dims: dims over which to compute signal power (e.g., C,H,W or T,C,H,W).
        eps: numerical stability.
        clamp: optional (min,max) to clamp output.

    Returns:
        y = x + n, where n ~ N(0, sigma^2) with sigma^2 = P_signal / SNR_lin
    """
    # Ensure float
    x = x.float()

    # Compute per-link signal power over signal_dims
    # Keep dims so it can broadcast back to x
    px = x.pow(2).mean(dim=signal_dims, keepdim=True).clamp_min(eps)  # same dtype/device

    snr_lin = (10.0 ** (snr_db / 10.0)).to(x.device).to(x.dtype)
    # snr_lin = (10.0 ** (snr_db / 10.0))
    # Make snr_lin broadcastable: add singleton dims if needed
    while snr_lin.ndim < x.ndim:
        snr_lin = snr_lin.unsqueeze(-1)

    sigma2 = (px / snr_lin).clamp_min(eps)
    noise = torch.randn_like(x) * torch.sqrt(sigma2)
    # print(noise)

    y = x + noise
    if clamp is not None:
        y = y.clamp(clamp[0], clamp[1])
    return y

def regroup(dense_feature, record_len, max_len, fading, fading_gain, SNR_db, enable_visualization=False):
    """
    Regroup the data based on the record_len and apply channel fading simulation.

    Parameters
    ----------
    dense_feature : torch.Tensor
        Input features of shape (N, C, H, W)
    record_len : torch.Tensor or list
        Length of each sample: [sample1_len, sample2_len, ...]
    max_len : int
        Maximum number of CAVs (Collaborative Autonomous Vehicles)
    fading_gain : torch.Tensor
        Fading gain values for each agent, shape (N,)
    SNR_db : torch.Tensor
        Signal-to-noise ratio in dB for each agent, shape (N,)
    enable_visualization : bool, optional
        Whether to visualize feature maps before and after fading (default: False)
    
    Returns
    -------
    regroup_feature : torch.Tensor
        Regrouped features of shape (B, L, C, H, W)
    agent_mask : torch.Tensor
        Agent mask of shape (B, L) indicating valid agents
    """
    # Split features and parameters by sample
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    split_fading_gain = torch.tensor_split(fading_gain, cum_sum_len[:-1])
    split_snr_db = torch.tensor_split(SNR_db, cum_sum_len[:-1])
    
    agent_mask = []
    regroup_features = []
    
    for batch_idx, split_feature in enumerate(split_features):
        # M, C, H, W where M is the number of agents in this sample
        M, C, H, W = split_feature.shape
        
        # Apply channel fading simulation to non-ego agents (skip index 0)
        if M > 1 and fading:
            # from afformer.visualization.vis_utils import visualize_feature_map
            # visualize_feature_map(split_feature[0], 'ego', False)
            # visualize_feature_map(split_feature[1], 'infra', False)
            agent_features = [split_feature[0:1]]  # ego, keep as-is
            for agent_idx in range(1, M):            
                # Out-of-place: apply fading gain then AWGN 
                corrupted = split_feature[agent_idx].mul(
                    split_fading_gain[batch_idx][agent_idx]
                )
               
                corrupted = add_awgn_bev(
                    corrupted,
                    snr_db=split_snr_db[batch_idx][agent_idx],
                    signal_dims=tuple(range(corrupted.ndim))
                )
                
                agent_features.append(corrupted.unsqueeze(0))
            split_feature = torch.cat(agent_features, dim=0)
    
            # Visualize degraded feature if enabled        
            # visualize_feature_map(split_feature[1], 'infra_fading', False)
        
        # Create padding for samples with fewer than max_len agents
        padding_len = max_len - M
        agent_mask.append([1] * M + [0] * padding_len)
        
        # Create padding tensor
        padding_tensor = torch.zeros(
            padding_len, C, H, W,
            device=split_feature.device,
            dtype=split_feature.dtype
        )
        
        # Concatenate features with padding
        split_feature = torch.cat([split_feature, padding_tensor], dim=0)
        
        # Reshape: (M, C, H, W) -> (1, M*C, H, W)
        split_feature = split_feature.view(-1, H, W).unsqueeze(0)
        
        regroup_features.append(split_feature)
    
    # Concatenate all batches: (B, M*C, H, W)
    regroup_features = torch.cat(regroup_features, dim=0)
    
    # Reshape: (B, M*C, H, W) -> (B, L, C, H, W) where L = max_len
    regroup_features = rearrange(regroup_features, 'b (l c) h w -> b l c h w', l=max_len)
    
    # Convert agent_mask to tensor: (B, L)
    agent_mask = torch.from_numpy(np.array(agent_mask)).to(regroup_features.device)
    
    return regroup_features, agent_mask