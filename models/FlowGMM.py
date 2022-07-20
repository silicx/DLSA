from typing import Optional
import torch
import torch.nn as nn
from math import log

from models.Flows import MAF


class FlowGMM(nn.Module):
    def __init__(self, args):
        super().__init__()

        feat_dim = args.feat_dim
        flow_args = args.gmflow.model

        n_components = flow_args.num_cluster
        center_std = flow_args.gmm_center_std
        
        self.nf = MAF(
            n_blocks = flow_args.n_blocks,
            input_size = feat_dim,
            hidden_size = feat_dim,
            n_hidden = flow_args.n_hidden,
            input_order = flow_args.input_order,
            batch_norm = flow_args.batch_norm
        )
        
        self.register_buffer("means", 
            torch.randn(1, n_components, feat_dim)*center_std ) #(1, K, dim)
        self.neg_half_d_log_2_pi = -0.5 * feat_dim * log(2*3.141593)
        self.n_components = n_components
        # self.neg_log_k = - log(float(n_components))

    def forward(self, x, cluster_mask: Optional[torch.BoolTensor]=None, return_lh_u=False):
        """
        x / u:  (bz, dim)
        log_p:  (bz, K)
        log_lh: (bz,)
        """
        u, sum_log_abs_det = self.nf(x)

        log_square = - 0.5 * (u.unsqueeze(1) - self.means).square()  # (bz, K, dim)
        
        log_p_u = self.neg_half_d_log_2_pi + log_square.sum(-1)  # (bz, K)
        log_p_x = log_p_u + sum_log_abs_det.sum(-1, keepdim=True).expand(-1, self.n_components)  # (bz, K)
        # sum_log_abs_det: (bz, dim)
        # log_square: (bz, K, dim)

        if cluster_mask is not None:
            masked_log_p_x = log_p_x[:, cluster_mask]
            masked_log_p_u = log_p_u[:, cluster_mask]
            log_p_x[:, ~cluster_mask] = 0
            log_p_u[:, ~cluster_mask] = 0
            n_component = cluster_mask.long().sum()
        else:
            masked_log_p_u = log_p_u
            masked_log_p_x = log_p_x
            n_component = self.n_components

        log_lh = torch.logsumexp(masked_log_p_x, -1) - log(float(n_component)) # (bz,)
        log_lh_u = torch.logsumexp(masked_log_p_u, -1) - log(float(n_component)) # (bz,)

        if return_lh_u:
            return u, log_p_x, log_lh, log_lh_u
        else:
            return u, log_p_x, log_lh

