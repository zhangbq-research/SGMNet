import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if retain_activation:
            self.block.add_module("ReLU", nn.ReLU(inplace=True))
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=4, stride=4, padding=0))

    def forward(self, x):
        out = self.block(x)
        return out

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)

def norm_layer(planes):
    return nn.BatchNorm2d(planes)

class CreatGraph(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.node_embedding = nn.Sequential(
          # ConvBlock(512, 512),
          ConvBlock(256, 256),
          # ConvBlock(512, 256),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, in_channels, 4*4))
        self.edge_encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_channels*3, out_features=in_channels, bias=False),
            # nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_channels, out_features=in_channels// 8, bias=False),
            # nn.BatchNorm1d(in_channels// 8),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=in_channels// 8, out_features=in_channels// 8, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        x_all = self.node_embedding(x)
        # x_all = self.node_embedding_1(x)
        # x_all = self.node_embedding_2(x_all)
        x_node = x_all[:, :self.in_channels, :, :]
        x_edge = x_all[:, self.in_channels:, :, :]
        # x_node = x_all
        # x_edge = x_all
        # x_edge = self.edge_embedding(x)
        m_batchsize, nc, nw, nh = x_edge.size()
        x_node = x_node.reshape(m_batchsize, nc, nw*nh).permute(0, 2, 1)
        x_edge = x_edge.reshape(m_batchsize, nc, nw * nh) + self.pos_embedding
        x_edge = x_edge.permute(0, 2, 1)

        m_batchsize, N, C = x_edge.size()
        inputs_left = torch.unsqueeze(x_edge, dim=2).expand(-1, -1, N, -1)
        inputs_right = torch.unsqueeze(x_edge, dim=1).expand(-1, N, -1, -1)

        features = torch.cat([inputs_left, inputs_right, inputs_left*inputs_right], dim=-1)
        nb, np_l, np_r, nd = features.shape

        adj = self.edge_encoder(features.reshape(nb*np_l*np_r, nd))
        adj = adj.reshape(nb, np_l, np_r, -1)
        # adj = inputs_left*inputs_right

        return x_node, adj



class PropagationLayers(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.propagation = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_channels*2+in_channels//8, out_features=in_channels*2, bias=False),
            nn.BatchNorm1d(in_channels*2),
            nn.ReLU(inplace=True),
            # # nn.Linear(in_features=in_channels*2, out_features=in_channels*2, bias=True),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_channels * 2, out_features=in_channels*2, bias=False),
            nn.BatchNorm1d(in_channels * 2),
            nn.Tanh(),
        )

    def forward(self, inputs, adj):
        m_batchsize, N, C = inputs.size()
        inputs_left = torch.unsqueeze(inputs, dim=2).expand(-1, -1, N, -1)
        inputs_right = torch.unsqueeze(inputs, dim=1).expand(-1, N, -1, -1)

        features = torch.cat([inputs_left, inputs_right, adj.reshape(adj.shape[0], N, N, -1)], dim=-1)
        nb, np_l, np_r, nd = features.shape
        features = self.propagation(features.reshape(nb*np_l*np_r, nd))
        features = features.reshape(nb, np_l, np_r, -1)
        features = torch.sum(features, dim=2)
        return features

class InteractionLayers(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-12)

    def forward(self, inputs1, inputs2):
        m_batchsize, N, C = inputs1.size()
        adj = torch.bmm(inputs1, inputs2.permute(0, 2, 1))
        adj1 = F.softmax(adj, dim=-1)
        adj2 = F.softmax(adj.permute(0, 2, 1), dim=-1)
        diff1 = inputs1 - torch.bmm(adj1, inputs2)
        diff2 = inputs2 - torch.bmm(adj2, inputs1)

        return diff1, diff2

class UpdateNodeLayers(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.update = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_channels*4, out_features=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=in_channels, out_features=in_channels, bias=True),
            nn.Tanh(),
        )


    def forward(self, inputs1, inputs2, inputs3):
        m_batchsize, N, C = inputs1.size()
        input = torch.cat([inputs1, inputs2, inputs3], dim=-1)
        nb, np, nd = input.shape

        output = self.update(input.reshape(nb*np, nd))
        output = output.reshape(nb, np, -1)
        return output

class GraphAgg(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, dim):
        """
        :param args: Arguments object.
        """
        super(GraphAgg, self).__init__()
        self.dim =dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim, self.dim))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix, gain=0.1)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.bmm(embedding, self.weight_matrix.unsqueeze(0).expand(embedding.shape[0], -1, -1)), dim=1)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.bmm(embedding, transformed_global.view(transformed_global.shape[0],-1,  1)))
        representation = torch.bmm(embedding.permute(0, 2, 1), sigmoid_scores)
        return representation.permute(0, 2, 1)

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        dim = 128
        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-4)
        # self.cosine = nn.Sequential(
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(in_features=dim*3, out_features=dim, bias=False),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(in_features=dim, out_features=dim, bias=True),
        #     # nn.ReLU(inplace=True),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(in_features=dim, out_features=1, bias=False),
        #     nn.Sigmoid(),
        # )
        self.creat_graph = CreatGraph(dim)
        self.propagation = PropagationLayers(dim)
        self.inter_action = InteractionLayers(dim)
        self.update_node = UpdateNodeLayers(dim)
        self.graph_agg = GraphAgg(dim)


    def forward(self, emb_support, emb_query, l_spt, l_qry, n_way):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        nb, ns, C, width, height = emb_support.size()
        _, nq, _, _, _ = emb_query.size()
        # emb_left = emb_query.unsqueeze(2).expand(-1, -1, ns, -1, -1, -1).reshape(-1, C, width*height).permute(0, 2, 1)
        # emb_right = emb_support.unsqueeze(1).expand(-1, nq, -1,  -1, -1, -1).reshape(-1, C, width*height).permute(0, 2, 1)
        emb_left = emb_query.unsqueeze(2).expand(-1, -1, ns, -1, -1, -1).reshape(-1, C, width, height)
        emb_right = emb_support.unsqueeze(1).expand(-1, nq, -1, -1, -1, -1).reshape(-1, C, width,height)
        emb_left, adj_left = self.creat_graph(emb_left)
        emb_right, adj_right = self.creat_graph(emb_right)
        feature_left = self.propagation(emb_left, adj_left)
        feature_right = self.propagation(emb_right, adj_right)
        diff_left, diff_right = self.inter_action(emb_left, emb_right)
        new_left = self.update_node(emb_left, feature_left, diff_left)
        new_right = self.update_node(emb_right, feature_right, diff_right)
        graph_vec_left = self.graph_agg(new_left)
        graph_vec_right = self.graph_agg(new_right)

        score = self.cosine(graph_vec_left, graph_vec_right)/2 + 0.5
        # features = torch.cat([graph_vec_left, graph_vec_right, graph_vec_left*graph_vec_right], dim=-1)
        # nb, np, nd = features.shape
        #
        # score = self.cosine(features.reshape(nb*np, nd))
        # score = score.reshape(nb, np, -1)


        # cal positive and negitive samples
        l_left = torch.unsqueeze(l_qry, dim=2).expand(-1, -1, l_spt.shape[1])
        l_right = torch.unsqueeze(l_spt, dim=1).expand(-1, l_qry.shape[1], -1)
        l_left = l_left.contiguous().view(l_left.shape[0], l_left.shape[1]*l_left.shape[2])
        l_right = l_right.contiguous().view(l_right.shape[0], l_right.shape[1] * l_right.shape[2])
        pos_neg_label = torch.eq(l_left, l_right).type(torch.float32)

        # cal query logits
        spt_one_hot = one_hot(l_spt.reshape(-1), n_way)
        spt_one_hot = spt_one_hot.view(-1, l_spt.shape[1], n_way)
        logits = torch.bmm(score.view(-1, l_qry.shape[1], l_spt.shape[1]), spt_one_hot) /(l_spt.shape[1]/n_way)

        return score, pos_neg_label, logits

    def graph_nodes(self, image):
        emb_right, _ = self.creat_graph(image)
        return emb_right

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies





