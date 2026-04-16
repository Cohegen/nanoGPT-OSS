#importing required libraries
import os
import math
from dataclasses import dataclass
from torch.nn import nn

import torch 
import  torch.distributed as dist

@dataclass 
class ModelConfig:
    num_hidden_layers = 6
    hidden_size = 512
    num_attention_heads = 6
    head_dim = 64

    intermediate_size = 2048   # 4x hidden
    vocab_size = 50000
    dropout = 0.1
    
    #attention
    num_key_value_heads = 8

    #context
    max_pos_emb = 512

    #activation 
    activation = "swiglu"

    # disabled MOE
    num_experts = 1
    experts_per_token = 1

    sliding_window = None
    initial_context_length = 256 

    #positional encoding
    use_rope = True 
    rope_theta = 10000.0

    #norms 
    norm_type = "rmsnorm"
    eps = 1e-5

    #intialization 
    intializer_range = 0.02 

    #training configs
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 0.1

    betas = (0.9,0.95)
    grad_clip = 1.0

    warmup_steps = 500
    max_iters = 100_000

    lr_decay = True 
    min_lr  = 3e-5



class RMSNomm(nn.Module):
    def __init__(self,num_features:int,eps:float = 1e-5,device:torch.device | None = None):
        super().__init__()
        self.num_features = num_features 
        self.eps = eps
        self.scale = nn.Parameter(
            torch.ones(num_features,device=device,dtype=torch.float32)
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        ##validating whether there's dim mismatch
        assert x.shape[-1] == self.num_features 
        t,dtype = x.float(),x.dtype
        ##applying RMSNorm
        t = t*torch.rsqrt(torch.mean(t**2,dim=-1,keepdim=True)+self.eps)
        return (t*self.scale).to(dtype)

##helper function for rotarary embedding
def _apply_rotary_emb(x:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor)->torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1,x2 = torch.chunk(x,2,dim=-1)
    o1 = x1*cos -x2 *sin
    o2 = x2*cos +x1 *sin 
    return torch.cat((o1,o2),dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self,head_dim:int,base:int,dtype:torch.dtype,initial_context_length:int=256,scaling_factor:float = 1.0,ntk_alpha:float = 1.0,ntk_beta=32.0,device:torch.device | None =None,)->None:
        super().__init__()
        self.head_dim = head_dim 
        self.base = base 
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta 

    def _compute_concentration_and_inv_freq(self)->torch.Tensor:
        freq = self.base ** (
            torch.arange(0,self.head_dim,2,dtype=torch.float,device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            cocentration = (0.1*math.log(self.scaling_factor)+1.0)

            d_half = self.head_dim / 2

            #NTK by parts 
            low = (
            d_half
            * math.log(self.initial_context_length/(self.ntk_beta*2*math.pi))
            / math.log(self.base)
             )
 
            high = (
            d_half
            * math.log(self.initial_context_length /(self.ntk_alpha*2*math.pi))
            / math.log(self.base)
            )

            assert 0 < low <high < d_half -1 

        

            interpolation = 1.0 /(self.scaling_factor*freq)
            extrapolation = 1.0 / freq 

            ramp = (
            torch.arange(d_half,dtype=torch.float32,device=freq.device)-low
            ) / (high-low)
            mask = 1 - ramp.clamp(0,1)

            inv_freq = interpolation * (1-mask) +extrapolation * mask

       
        else:
            concentration = 1.0
            inv_freq = 1.0 /freq 

        return concentration,inv_freq 

    def _compute_cos_sin(self,num_tokens:int):
        pass
