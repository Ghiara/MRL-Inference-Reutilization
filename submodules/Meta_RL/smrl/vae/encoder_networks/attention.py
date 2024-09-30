""" 
This module contains implementations for MDP-Encoders:
- ``GRUEncoder``
- ``MlpEncoder``
- ``AttentionEncoder``
See also: mdpvae.py

Note: This page
(https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
provides a nice-to-read code of a multihead-attention network / transformer encoder modules. 
Presumably, the PyTorch implementation functions simiarly.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-17
"""


import torch
from typing import Union, Tuple, List, Type

from smrl.utility.distributions import DiagonalMultivariateNormal
from ..mdpvae import MdpEncoder
from .util import batched


# NOTE: AttentionLayer is not used in the current implementation of Attention-
#   based encoders.
class AttentionLayer(torch.nn.Module):
    """
    A simple attention layer which computes outputs as follows:

    ```
    Q = W_q @ q
    K = W_k @ k
    V = W_v @ v
    output = Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    ```

    Parameters
    ----------
    embed_dim : int
        Dimension of the embeddings (values)
    key_dim : int, optional
        Dimension of the keys. If not provided, ``key_dim = embed_dim`` is
        assumed.
    query_dim : int, optional
        Dimension of the queries. If not provided, ``query_dim = embed_dim`` is
        assumed.
    
    References
    ----------
    * Vaswani et al. (2017), "Attention is All you Need"
    """
    def __init__(self, embed_dim: int, key_dim: int = None, query_dim: int = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim if key_dim is not None else embed_dim
        self.query_dim = query_dim if query_dim is not None else embed_dim

        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.key_dim, self.embed_dim)
        self.q_proj = torch.nn.Linear(self.query_dim, self.embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies attention to queries, keys, and values:

        ```
        Q = W_q @ q
        K = W_k @ k
        V = W_v @ v
        output = Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        ```

        Parameters
        ----------
        query : torch.Tensor
            Query tensor, shape (batch size, query length, query dim)
        key : torch.Tensor
            Key tensor, shape (batch size, seq. length, key dim)
        value : torch.Tensor
            Value tensor, shape (batch size, seq. length, embed. dim)
        """
        value = self.v_proj(value)
        key = self.k_proj(key)
        query = self.q_proj(query)
        dk = torch.tensor(key.shape[-1])

        score = torch.bmm(query, key.transpose(-1,-2)) / torch.sqrt(dk)
        attention = torch.softmax(score, dim=-1)
        new_value = torch.bmm(attention, value)
        return new_value, attention



class AttentionEncoder(MdpEncoder):
    """An MdpEncoder which uses
    - self-attention to preprocess the context data
    - query-based attention to extract features
    - linear layers to extract mean and variance of the output distribution.

    The query-based attention layers use data-independent, learnable queries
    which are multiplied with the data-dependent keys and values.

    Parameters
    ----------
    observation_dim : int
        Observation dim of the environment
    action_dim : int
        Action dim of the environment
    latent_dim : int
        Dimension of the latent space
    encoding_dim : int
        Dimensions of the encodings
    n_queries : int
        Number of queries (each query represents one feature vector)
    num_heads : int
        Number of parallel compuation heads
    self_attention_layers : int, optional
        Number of self-attention layers, by default 1
    query_layers : int, optional
        Number of query-attention layers, by default 1
    activation_function : Type[torch.nn.functional.relu], optional
        Activation function which is used after each attention layer,
        by default torch.nn.functional.relu
    context_size : int, optional
        Context size, by default None
    encoding_mode : str, optional
        Encoding mode, by default 'sample'
    """
    
    supports_variable_sequence_length = True

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: int,
        encoding_dim: int,
        n_queries: int,
        num_heads: int,
        self_attention_layers: int = 1,
        query_layers: int = 1,
        activation_function: Type[torch.nn.functional.relu] = torch.nn.functional.relu, 
        context_size: int = None,
        encoding_mode: str = 'sample',
        *args,
        **kwargs
    ) -> None:
        super().__init__(observation_dim, action_dim, latent_dim, encoding_dim, context_size, encoding_mode, *args, **kwargs)

        self.embed_dim = 2*observation_dim + action_dim + 2
        self.n_queries = n_queries
        self.num_heads = num_heads
        self.activation_function = activation_function

        # Self-attention layer for preprocessing
        self.self_attention_layers: List[torch.nn.MultiheadAttention] = torch.nn.ModuleList()
        for _ in range(self_attention_layers):
            self.self_attention_layers.append(
                torch.nn.MultiheadAttention(
                    self.embed_dim * num_heads,
                    num_heads,
                    batch_first=True,
                )
            )
            # self.self_attention_layers.append(AttentionLayer(self.embed_dim))

        # Query-attention layer, "Feature extraction"
        self.query_layers: List[torch.nn.MultiheadAttention] = torch.nn.ModuleList()
        self.queries = torch.nn.ParameterList()
        for _ in range(query_layers):
            self.query_layers.append(
                torch.nn.MultiheadAttention(
                    self.embed_dim * num_heads,
                    num_heads,
                    batch_first=True,
                )
            )
            self.queries.append(torch.nn.Parameter(
                torch.rand(self.n_queries, self.embed_dim * self.num_heads)
            ))
            # self.query_layers.append(AttentionLayer(self.embed_dim))

        # Attention results -> latent dim
        self.mean_layer = torch.nn.Linear(
            in_features = n_queries * self.embed_dim * self.num_heads,
            out_features = self.latent_dim,
        )
        self.var_layer = torch.nn.Linear(
            in_features = n_queries * self.embed_dim * self.num_heads,
            out_features = self.latent_dim,
        )

    @batched
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> torch.distributions.Distribution:
        x = torch.cat((observations, actions, rewards, next_observations, terminals), dim=-1)
        batch_size, sequence_length = x.shape[0], x.shape[1]

        # If context is empty, return default distribution
        if x.nelement() == 0:
            mean = torch.zeros(torch.Size([batch_size, self.latent_dim])).to(self.device)
            var = torch.ones(torch.Size([batch_size, self.latent_dim])).to(self.device)
            return DiagonalMultivariateNormal(mean, var)
    
        # Repeat x for num_heads in embedding dimension.
        # Apparently, PyTorch splits the embedding dimension into num_head
        # parallel compuations.
        if self.num_heads > 1:
            x = x.repeat(1, 1, self.num_heads)

        # Preprocess by self-attention, shape (batch_size, sequence_length, embed_dim*num_heads) -> shape (batch_size, sequence_length, embed_dim*num_heads)
        for self_attention_layer in self.self_attention_layers:
            x, attn = self_attention_layer.forward(x, x, x)
            x = self.activation_function(x)

        # Attention querying, shape (batch_size, sequence_length, embed_dim*num_heads) -> shape (batch_size, n_queries, embed_dim*num_heads)
        for query, query_layer in zip(self.queries, self.query_layers):
            x, attn = query_layer.forward(
                # query = torch.ones((batch_size, self.n_queries, self.embed_dim * self.num_heads), device=self.device),
                query = query.repeat(batch_size, 1, 1),
                key = x,
                value = x,
            )
            x = self.activation_function(x)

        # Feature processing, shape (batch_size, n_queries, embed_dim*num_heads) -> shape (batch_size, latent_dim)
        h = x.reshape([batch_size, self.n_queries * self.embed_dim * self.num_heads])
        mean = self.mean_layer.forward(h)
        var = self.var_layer.forward(h).exp()

        return DiagonalMultivariateNormal(mean, var)


class SelfAttentionEncoder(MdpEncoder):
    """An encoder which uses transformer networks to process MDP context data.

    References
    ----------
    * Ashish Vaswani et al. (2017), “Attention Is All You Need” 

    Parameters
    ----------
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of actions.
    latent_dim : int
        Dimension of latent representations.
    encoding_dim : int
        Dimension of latent encodings (can be different from latent_dim, 
        e.g. if we use 'mean_var' as encoding_mode)
    hidden_size : int
        Hidden dimension of the GRU network.
    num_layers : int
        Number of hidden layers in the GRU network.
    nhead : int, optional
        Number of attention heads for multi-headed attention, by default 1
    dropout : float, optional
        Dropout argument for torch.nn.TransformerEncoderLayer, by default 0.5
    """
    supports_variable_sequence_length = True
    def __init__(
        self, 
        observation_dim: int,
        action_dim: int,
        latent_dim: int, 
        encoding_dim: int,
        hidden_size: int,
        num_layers: int,
        nhead: int = 1,
        dropout: float = 0.5,
        *args,
        **kwargs
    ) -> None:
        super().__init__(observation_dim, action_dim, latent_dim, encoding_dim, *args, **kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        input_dim = 2*observation_dim + action_dim + 2

        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,  # sum of shapes: obs, next_obs, actions, reward, terminal
            nhead=nhead,
            dim_feedforward=hidden_size,
            batch_first=True,
            dropout=dropout
        )
        self._transformer_network = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_layers,
        )
        self._mean_layer = torch.nn.Linear(4*input_dim, latent_dim) # Outputs the mean vector of the distribution
        self._var_layer = torch.nn.Linear(4*input_dim, latent_dim)  # Outputs the variance vector of the distribution
        # GRU network accepts inputs of shape (batch_size, sequence_length, input_dim)
        # OPTION: Consider using GRUCell instead to keep track of history without passing it every time.

    @batched
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:

        x = torch.cat((observations, actions, rewards, next_observations, terminals), dim=-1)
        batch_size = x.shape[0] if x.shape[0] != 0 else 1

        # If context is empty, return default distribution
        if x.nelement() == 0:
            mean = torch.zeros(torch.Size([batch_size, self.latent_dim])).to(self.device)
            var = torch.ones(torch.Size([batch_size, self.latent_dim])).to(self.device)
            return DiagonalMultivariateNormal(mean, var)

        # Pass input sequence through attention network and extract features (mean, min, max, std)
        h: torch.Tensor = self._transformer_network(x)
        h = torch.cat([h.mean(dim=1), h.min(dim=1)[0], h.max(dim=1)[0], h.std(dim=1, unbiased=False)], dim=-1)


        # Apply output layers to obtain distribution parameters
        mean = self._mean_layer(h)
        var = self._var_layer(h).exp()
        return DiagonalMultivariateNormal(mean, var)