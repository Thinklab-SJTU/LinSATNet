import torch
from torch import nn

class StartEndEmbedding(nn.Module):
    '''
    The mebdedding for TSP-SE, where the first node in the sequence is the start node and the last node is the end node.
    '''
    def __init__(self, node_num, embedding_dim):
        super(StartEndEmbedding, self).__init__()

        self.node_num = node_num
        self.embedding_dim = embedding_dim

        self.start_embedding = nn.Parameter(torch.torch.distributions.Uniform(low=-1, high=1).sample((1, embedding_dim)))
        self.end_embedding = nn.Parameter(torch.torch.distributions.Uniform(low=-1, high=1).sample((1, embedding_dim)))
        self.middle_dummy = nn.Parameter(torch.zeros([node_num - 2, embedding_dim]))
        self.middle_dummy.require_grad = False
    
    def forward(self, batch_size):
        node_context_emb = torch.cat([self.start_embedding, self.middle_dummy, self.end_embedding], dim = 0)
        node_context_emb = node_context_emb[None, :, :].expand(batch_size, -1, -1)
        return node_context_emb

class PriorityEmbedding(nn.Module):
    '''
    The mebdedding for TSP-PRI, besides the start and end node, the second node is the priority node.
    '''
    def __init__(self, node_num, embedding_dim):
        super(PriorityEmbedding, self).__init__()

        self.node_num = node_num
        self.embedding_dim = embedding_dim

        self.start_embedding = nn.Parameter(torch.torch.distributions.Uniform(low=-1, high=1).sample((1, embedding_dim)))
        self.priority_embedding = nn.Parameter(torch.torch.distributions.Uniform(low=-1, high=1).sample((1, embedding_dim)))
        self.end_embedding = nn.Parameter(torch.torch.distributions.Uniform(low=-1, high=1).sample((1, embedding_dim)))
        self.middle_dummy = nn.Parameter(torch.zeros([node_num - 3, embedding_dim]))
        self.middle_dummy.require_grad = False
    
    def forward(self, batch_size):
        node_context_emb = torch.cat([self.start_embedding, self.priority_embedding, self.middle_dummy, self.end_embedding], dim = 0)
        node_context_emb = node_context_emb[None, :, :].expand(batch_size, -1, -1)
        return node_context_emb