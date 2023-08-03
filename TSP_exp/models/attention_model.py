import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.distributions.normal import Normal
import math

from models.graph_encoder import GraphAttentionEncoder
from models.embed import StartEndEmbedding, PriorityEmbedding
from torch.nn import DataParallel

class AttentionModel(nn.Module):

    def __init__(self,
                 node_num,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 normalization='batch',
                 n_heads=8,
                 task='StartEnd'):
        super(AttentionModel, self).__init__()

        self.node_num = node_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers

        self.tanh_clipping = tanh_clipping


        self.n_heads = n_heads

        node_dim = 2   #for tsp on 2D plane
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        if (task == 'StartEnd'):
            self.node_context_embedder = StartEndEmbedding(node_num, embedding_dim)
        elif (task == 'Priority'):
            self.node_context_embedder = PriorityEmbedding(node_num, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, node_num)
        )

    def forward(self, input):
        """
        :param input: (batch_size, node_num, node_dim) input node features or dictionary with multiple tensors
        :return: (batch_size, n_samples, node_num, node_num) pre sinkhorn matrix
        """
        batch_size, node_num, _ = input.shape

        embedded_node = self.init_embed(input)
        node_context_emb = self.node_context_embedder(batch_size)
        embedded_node = embedded_node + node_context_emb
        
        encoded_node, _ = self.embedder(embedded_node)  # (batch_size, node_num, embedding_dim)
        pre_project_logits = self.decoder(encoded_node) # (batch_size, node_num, node_num)

        if self.tanh_clipping > 0:
            pre_project_logits = torch.tanh(pre_project_logits) * self.tanh_clipping

        return pre_project_logits