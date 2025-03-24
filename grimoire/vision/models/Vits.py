import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Literal, Optional, Tuple


class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    
    
class DyT(nn.Module):
    def __init__(self,dim,init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)*init_alpha) #Learnable scalar (shared across dim)
        self.gamma = nn.Parameter(torch.ones(dim))          #Learnable per-channel scale
        self.beta = nn.Parameter(torch.zeros(dim))          #Learnable per-channel shift
        
        
    def forward(self,x):
        """

        Args:
            x: Input Tensor of Shape [B,T,C] or [*,C] for generic transformer input.

        Returns:
            Tensor of same shape as input after applying DyT
        """
        x = torch.tanh(self.alpha*x)
        
        return self.gamma*x + self.beta
    

class MLA(nn.Module):
    """
    Multi-Headed Latent Attention
    

    Args:
        dim (int): Dimension of the input tensor
        heads (int): Number of heads
        dim_head (int): Dimension of each head
        q_lora_rank (int): Rank of low-rank query tensor
        kv_lora_rank (int): Rank of low-rank key and value tensors
        qk_nope_head_dim (int): Dimension of the non-positional query/key projections
        qk_rope_head_dim (int): Dimension of the positional query/key projections
        qk_head_dim (int): Dimension of the query/key projections
        v_head_dim (int): Dimension of the value projections
        softmax_scale (float): Scale of the softmax operation
        
    """
    
    def __init__(self,args:ModelArgs):
        super().__init__()
        
    