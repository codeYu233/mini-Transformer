import torch
from einops import rearrange,einsum
from math import sqrt

class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        tensor = torch.empty((self.out_features, self.in_features), device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor,mean=0.0,std=sqrt(2/(self.in_features+self.out_features)),a=-sqrt(6/(self.in_features+self.out_features)),b=sqrt(6/(self.in_features+self.out_features))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, 'out_d in_d, ... in_d -> ... out_d')

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        tensor= torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNormLayer(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight=torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class PositionWiseFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.w1 = LinearLayer(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = LinearLayer(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = LinearLayer(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        gate = x1 * torch.sigmoid(x1)
        x3 = self.w3(x)
        return self.w2(gate * x3) 

class RoPELayer(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.freq=1/torch.pow(theta, torch.arange(0, d_k, 2,device= device).float()/d_k)
        position=torch.arange(max_seq_len, device=device).float()
        self.angle=torch.outer(position,self.freq)
        self.register_buffer(persistent=False, name='cos_buffer', tensor=self.angle.cos())
        self.register_buffer(persistent=False, name='sin_buffer', tensor=self.angle.sin())

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos=self.cos_buffer[token_positions]
        sin=self.sin_buffer[token_positions]

        if x.ndim > cos.ndim and cos.ndim >=3:
            cos=cos.unsqueeze(1)
            sin=sin.unsqueeze(1)

        cos=cos.to(x.dtype)
        sin=sin.to(x.dtype)

        x_even=x[...,::2]
        x_odd=x[...,1::2]
        x_rotated=torch.empty_like(x)
        x_rotated[...,::2]=cos*x_even-sin*x_odd
        x_rotated[...,1::2]=sin*x_even+cos*x_odd

        return x_rotated

class SoftmaxLayer(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim= dim

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        infeatures_max=in_features.max(dim=self.dim,keepdim=True).values
        in_features=in_features-infeatures_max
        exp_vals=torch.exp(in_features)
        return exp_vals/exp_vals.sum(dim=self.dim,keepdim=True)

class ScaledDotProductAttentionLayer(torch.nn.Module):
    def __init__(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        super().__init__()
        self.Q=Q
        self.K=K
        self.V=V
        self.mask=mask

    def forward(self) -> torch.Tensor:
        d_k=self.Q.size(-1)
        pre_softmax_values=einsum(self.Q, self.K, '... q_len d_k, ... k_len d_k -> ... q_len k_len')/sqrt(d_k)
        if self.mask is not None:
            pre_softmax_values=pre_softmax_values.masked_fill(~self.mask, -float('inf'))
        return einsum(SoftmaxLayer(-1)(pre_softmax_values), self.V, '... q_length k_length, ... k_length d_v -> ... q_length d_v')

class CausalMHSA(torch.nn.Module):
    def __init__(self,d_model: int, num_heads: int, theta: float, max_seq_len: int,token_positions: torch.Tensor=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.token_positions=token_positions
        self.W_Q=LinearLayer(d_model,d_model)
        self.W_K=LinearLayer(d_model,d_model)
        self.W_V=LinearLayer(d_model,d_model)
        self.W_O=LinearLayer(d_model,d_model)

    def forward(self, in_features: torch.Tensor):
        batch_size, seq_len, _ = in_features.size()
        Q=self.W_Q(in_features)
        K=self.W_K(in_features)
        V=self.W_V(in_features)

        Q=rearrange(Q,'b s (h d_k) -> b h s d_k', h=self.num_heads)
        K=rearrange(K,'b s (h d_k) -> b h s d_k', h=self.num_heads)
        V=rearrange(V,'b s (h d_k) -> b h s d_k', h=self.num_heads)

        if self.theta is not None:
            rope=RoPELayer(self.theta, self.d_k, self.max_seq_len, device=in_features.device)
            Q=rope(Q, self.token_positions)
            K=rope(K, self.token_positions)

        mask=torch.tril(torch.ones(seq_len,seq_len, dtype=torch.bool, device=in_features.device),diagonal=0)

        sdpa=ScaledDotProductAttentionLayer(Q,K,V,mask)
        attention_output=sdpa()

        return self.W_O(rearrange(attention_output, 'b h s d_k -> b s (h d_k)'))

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.mhsa=CausalMHSA(d_model, num_heads, theta, max_seq_len, None)
        self.rmsnormal1=RMSNormLayer(d_model)
        self.rmsnormal2=RMSNormLayer(d_model)
        self.ffn=PositionWiseFFN(d_model, d_ff)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        token_positions= torch.arange(in_features.size(1), device=in_features.device)
        rmsnormed=self.rmsnormal1(in_features)
        self.mhsa.token_positions=token_positions
        mhsa_output=self.mhsa(rmsnormed)
        in_features=in_features+mhsa_output
        rmsnormed=self.rmsnormal2(in_features)
        ffn_output=self.ffn(rmsnormed)
        return in_features+ffn_output

class TransformerModel(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, vocab_size: int, context_length: int,num_layers: int):
        super().__init__()
        self.TransformerBlocks=torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len) for _ in range(num_layers)])
        self.token_embedding=EmbeddingLayer(vocab_size, d_model)
        self.norm_final=RMSNormLayer(d_model)
        self.Linear_out=LinearLayer(d_model, vocab_size)
        self.softmax=SoftmaxLayer(-1)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        token_embeddings=self.token_embedding(in_indices)
        x=token_embeddings
        for block in self.TransformerBlocks:
            x=block(x)
        x=self.norm_final(x)
        logits=self.Linear_out(x)
        return logits
    