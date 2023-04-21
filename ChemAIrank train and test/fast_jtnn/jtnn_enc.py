import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from fast_jtnn.mol_tree import Vocab, MolTree
from fast_jtnn.nnutils import create_var, index_select_ND

class JTNNEncoder(nn.Module):
    """
    Description: This module implements the encoder for the Junction Trees.
    该模块实现了连接树的编码器。
    """
    def __init__(self, hidden_size, depth, embedding):
        """
        The constructor for the class.
        :param hidden_size: int
            The dimension of the hidden message vectors.
        :param depth: int
            The number of timesteps for which to implement the message passing.
        :param embedding: torch.embedding
            The embedding space for obtaining embedding of atom features vectors.
        """
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.embedding = embedding#nn.Embedding(vocab.size(), hidden_size)
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        #GRU 用于聚合来自子节点的消息向量
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope):
        """
           Args:
               fnode: torch.LongTensor() (shape: num_nodes)
                   The list of wids i.e. idx of the corresponding cluster vocabulary item, for the initial node of each edge.
                   每条边的初始节点的wids列表，即相应集群词汇项的idx。

               fmess: torch.LongTensor() (shape: num_edges)
                   The list of idx of the initial node of each edge.
                   每条边的初始节点的idx列表。

               node_graph: torch.LongTensor (shape: num_nodes x MAX_NUM_NEIGHBORS)
                   For each node, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.
                   对于每个节点，所有 "内向隐边信息向量 "的idxs列表，用于节点特征聚合。

               mess_graph: torch.LongTensor (shape: num_edges x MAX_NUM_NEIGHBORS)
                   For each edge, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.
                   对于每条边，所有 "内向隐藏边信息向量 "的idxs列表，用于节点特征聚合。

               scope: List[Tuple(int, int)]
                   The list to store tuples of (start_idx, len) to segregate all the node features, for a particular junction-tree.
                   列表中存储（start_idx, len）的图元，以隔离所有的节点特征，对于一个特定的结点树。

               mess_dict: Dict{Tuple(int, int): int}
                   The dictionary mapping edge in the form (x.idx, y.idx) to idx of message.
                   (x.idx, y.idx)形式的字典映射边缘到消息的idx。

           Returns:
               tree_vecs: torch.tensor (shape: batch_size x hidden_size)
                   The hidden vectors for the root nodes, of all the junction-trees, across the entire dataset.
                   在整个数据集中，所有结点树的根节点的隐藏向量。
               messages: torch.tensor (shape: num_edges x hidden_size)obtain the hidden vectors for all the edges
               消息：Torch.tensor (shape: num_edges x hidden_size)获得所有边的隐藏向量。
           """
        # creat Pytorch variables
        fnode = create_var(fnode)
        fmess = create_var(fmess)
        node_graph = create_var(node_graph)
        mess_graph = create_var(mess_graph)

        #hidden vectors for all the edges 所有边缘的隐藏向量
        messages = create_var(torch.zeros(mess_graph.size(0), self.hidden_size))#Tensor:(num_edges, hidden_size)

        # obtain node feature embedding
        fnode = self.embedding(fnode) #Tensor:(num_nodes, hidden_size)-->(num_nodes, 450)

        # for each edge obtain the embedding for the initial node每个边初始节点的embedding
        # 为每条边获得初始节点的嵌入。
        fmess = index_select_ND(fnode, 0, fmess) #Tensor:(num_edges, hidden_size)-->(num_edges, 450)

        # obtain the hidden vectors for all the edges using GRU
        # 使用GRU获得所有边缘的隐藏向量
        messages = self.GRU(messages, fmess, mess_graph)# torch.LongTensor:(num_edges, hidden_size)

        # for each node, obtain all the neighboring message vectors
        # 对于每个节点，获得所有邻近的信息向量
        mess_nei = index_select_ND(messages, 0, node_graph)#Tensor:(num_nodes, max_num_neighbors, hidden_size）

        # for each node, sum up all the neighboring message vectors
        # 对于每个节点，将所有相邻的信息向量相加。
        # for each node, cncatenate the node embedding feature and the sum of hidden beighbor message vectors
        # 对于每个节点，将节点嵌入特征和隐藏邻居信息向量的总和进行串联。
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)#Tensor:(num_nodes, hidden_size + hidden_size）

        #apply the neural network layer
        node_vecs = self.outputNN(node_vecs)#Tensor:(num_nodes, hidden_size）

        max_len = max([x for _, x in scope])
        # list to store feature vectors of the root node, for all the junction-trees, across the entire dataset
        # 列表用于存储根节点的特征向量，适用于整个数据集的所有结点树。
        batch_vecs = []
        for st, le in scope:# scope: List[Tuple(start_idx, len(node))]
            # root node is the first node  in the list of nodes of a junction-tree by design
            # 根据设计，根节点是结点树节点列表中的第一个节点。
            cur_vecs = node_vecs[st] #Root is the first node
            batch_vecs.append(cur_vecs)

        #stack the root tensors to from a 2-D tensor 叠加根张量，从一个二维张量开始
        tree_vecs = torch.stack(batch_vecs, dim=0)#batch_vecs: List[32个Tensor:(hidden_size,)]->Tensor:(batch_size32, hidden_size450)
        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):
        """
        :param tree_batch: List[MolTree]
            The list of junction-trees of all the molecular graphs in the dataset.
            数据集中所有分子图的结点树列表。
        :return:
        """
        # list to store junction-tree nodes, for all junction-trees, across the entire dataset
        # 列表，用于存储整个数据集中所有结点树的结点。
        node_batch = []
        #list to store tuples of (start_idx, len) to segregate(分離) all the node features, for a particular junction-tree
        # (start_idx, len)用于分离连接树节点特征的列表
        scope = []
        for tree in tree_batch:
            # （starting idx of collection of nodes for this junction-tree, the number of nodes in this junction-tree）
            # 这个结点树的结点集合的起始idx，这个结点树的结点数量
            scope.append( (len(node_batch), len(tree.nodes)) )#[(每个tree节点在node_batch的start_idx, 每个tree节点的个数)...]

            node_batch.extend(tree.nodes)#存储整个tree的节点

        return JTNNEncoder.tensorize_nodes(node_batch, scope)
    
    @staticmethod
    def tensorize_nodes(node_batch, scope):
        """
        Args:
            node_batch: List[MolJuncTreeNode]
                The list of junction-tree nodes, of all junction-trees, across the entire dataset.
                整个数据集的所有结点树的结点列表。
            scope: List[Tuple(int, int)]
                The list to store tuples of (start_idx, len) to segregate all the node features, for a particular junction-tree.
                列表中存储（start_idx, len）的图元，以隔离所有的节点特征，对于一个特定的结点树。

        Returns:fnode, fmess, node_graph, mess_graph, scope
            fnode: torch.LongTensor() (shape: num_edges)
                The list of wids i.e. idx of the corresponding cluster vocabulary item, for the initial node of each edge.
                每条边的初始节点的wids列表，即相应集群词汇项的idx。

            fmess: torch.LongTensor() (shape: num_edges)
                The list of idx of the initial node of each edge.
                每条边的初始节点的idx列表。

            node_graph: torch.LongTensor (shape: num_nodes x MAX_NUM_NEIGHBORS)
                For each node, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.
                对于每个节点，所有 "内向隐边信息向量 "的idxs列表，用于节点特征聚合。

            mess_graph: torch.LongTensor (shape: num_edges x MAX_NUM_NEIGHBORS)
                For each edge, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.
                对于每条边，所有 "内向隐藏边信息向量 "的idxs列表，用于节点特征聚合。

            scope: List[Tuple(int, int)]
                The list to store tuples of (start_idx, len) to segregate all the node features, for a particular junction-tree.
                列表中存储（start_idx, len）的图元，以隔离所有的节点特征，对于一个特定的结点树。

            mess_dict: Dict{Tuple(int, int): int}
                The dictionary mapping edge in the form (x.idx, y.idx) to idx of message.
                (x.idx, y.idx)形式的字典映射边缘到消息的idx。

        """
        #message: list to store all edges/messages, for all the junction-trees, asross the entire dataset. ensure messages are always 1-indexed
        # 列表存储所有结点树的所有边/消息，贯穿整个数据集。 确保消息总是以1为索引（用none占位）。
        #mess_dict: dictionary mapping edge in the form (x.idx, y.idx) to idx of message
        # 词典中的(x.idx, y.idx)形式的边缘映射到信息的idx上。
        messages, mess_dict = [None], {}

        # list to store wids of all the nodes, for all the junction-trees, across the entire dataset.
        # 列表中存储了整个数据集中所有结点树的WID。
        fnode = []
        for x in node_batch:
            fnode.append(x.wid)#存储该batch_size中所有节点的wid,即其在vocab中的idx
            for y in x.neighbors:
                mess_dict[(x.idx, y.idx)] = len(messages)#字典，{边(x.idx, y.idx): 该边在列表messages里的idx（idx从1开始）}
                messages.append((x, y))#存储该batch_size中所有tree的所有边，形式为（node_x, node_y）

        # list of lists, to store the idxs of all the "inward" message, for all the nodes, of all the junction-trees, across the entire dataset.
        # 列表，用于存储整个数据集中所有结点的所有 "内向 "信息的idxs，以及所有结点树的idxs
        node_graph = [[] for i in range(len(node_batch))]

        # list of lists, to store the idx of messages from "inward-edges", for all the edges, of all the junction-trees, across the entire dataset.
        # 列表，用于存储整个数据集中所有结点树的所有边的 "内向边 "的信息idx。
        mess_graph = [[] for i in range(len(messages))]

        # list to store the idx of initial node, for all the edges, of all the junction-trees, across the entire dataset.
        # 列表来存储整个数据集中所有结点树的所有边的初始节点的idx。
        fmess = [0] * len(messages)#用于存储所有边的初始节点的idx

        # iterate through the edges (x, y) 遍历边（x，y）。
        for x, y in messages[1:]:
            # retrieve the idx of the message vector for edge (x, y) 检索边（x，y）的信息向量的idx
            mid1 = mess_dict[(x.idx, y.idx)]

            # for the edge (x, y), node x is the initial node 对于边（x，y），节点x是初始节点
            fmess[mid1] = x.idx

            # for the node y, the message from edge (x, y) will be used in aggregation procedure
            # 对于节点y，来自边（x，y）的信息将被用于聚合程序中
            node_graph[y.idx].append(mid1)#对于所有的节点，存储以这个节点为尾节点的所有边在messages中的idx(从1开始)，即存储所有节点的“inward” message

            for z in y.neighbors:
                # ignore, if the neighbor node is x 忽略，如果邻居节点是x
                if z.idx == x.idx:
                    continue
                # for all edges of the form, (y, z), edge (x, y) is an "inward edge"
                # 对于所有形式的边，（y，z），边（x，y）是一个 "内向边"
                mid2 = mess_dict[(y.idx, z.idx)]
                mess_graph[mid2].append(mid1)#对于所有的边，存储每个边的所有前向边在messages中的idx,即存储所有边的“inward_edges”

        # the maxinum number of message vectors from "inward edges" for all "nodes", of all junction trees, across the entire dataset
        # 整个数据集的所有 "节点 "的 "内向边缘 "的信息向量的最大数量，即所有结点树的信息向量。
        max_len = max([len(t) for t in node_graph] + [1])#列表拼接
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len) #将列表node_graph里的所有列表都用0 padding到最大长度

        # the maxinum number of message vectors from "inward edges" for all "edges", of all junction trees, across the entire dataset
        # 整个数据集的所有 "边缘 "的 "内向边缘 "的信息向量的最大数量，即所有结点树的 "内向边缘"。
        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)#将列表mess_graph里的所有列表都用0 padding到最大长度

        mess_graph = torch.LongTensor(mess_graph)   #对于所有的边，存储每个边的所有前向边在messages中的idx,即存储所有边的“inward_edges”
        node_graph = torch.LongTensor(node_graph)   #对于所有的节点，存储以这个节点为尾节点的所有边在messages中的idx(从1开始)，即存储所有节点的“inward” message
        fmess = torch.LongTensor(fmess)             #用于存储所有边的初始节点的idx
        # fnode = torch.LongTensor(fnode)             #存储该batch_size中所有节点的wid,即其在vocab中的idx
        try:
            fnode = torch.LongTensor(fnode)
        except TypeError as e:
            print("类型错误：", e)
            fnode = None  # 可以选择使用 None 或其他默认值

        # 继续执行其他代码

        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict

class GraphGRU(nn.Module):
    """
    Description: This module implements the GRU for message aggregation for Tree Encoding purposes.
    该模块实现了用于树形编码的消息聚合的GRU。
    """
    def __init__(self, input_size, hidden_size, depth):
        """
            The constructor for the class.
            :param input_size: int
            :param hidden_size: int
                The dimension of the hidden message vectors.
            :param depth: int
                The number of timesteps for which to implement the message passing.
            """
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        # GRU weight matrices
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        """
        Args:
            h: torch.LongTensor (shape: num_message_vectors x hidden_size)
                The hidden message vectors for all the edge vectors.

            x: torch.LongTensor (shape: num_message_vectors x hidden_size)
                The embedding vector for initial nodes of all the edges.所有边初始节点的embedding

            mess_graph: torch.LongTensor (shape: num_message_vectors x MAX_NUM_NEIGHBORS)
                For each edge, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.
                对于每条边，所有 "内向隐藏边信息向量 "的idxs列表，用于节点特征聚合。
        Returns:
            h: torch.LongTensor (shape: num_message_vectors x hidden_size)
                The hidden message vectors for all the edge vectors. 所有边缘向量的隐藏信息向量。
        """
        mask = torch.ones(h.size(0), 1)#{Tensor:(num_edges, 1)}
        # the first hidden message vector is the padding vector i.e. vector of all zeros so we zero it out
        # #第一个隐藏信息向量是填充向量，即所有零的向量，所以我们把它清零。
        mask[0] = 0 #first vector is padding
        mask = create_var(mask)

        # implement message passing from timestep 0 to T (self.depth)-1
        for it in range(self.depth):
            # get "inward hidden message vectors" for all the edges
            # 获得所有边缘的 "内向隐藏信息向量"。
            h_nei = index_select_ND(h, 0, mess_graph)#[num_edges, max_num_neighbors, hidden_size]
            # sum the "inward hidden message vectors" for all the edges
            # 对所有边的 "内向隐藏信息向量 "进行求和
            sum_h = h_nei.sum(dim=1)#[num_edges, hidden_size] 对h_nei在第1维求和
            # concatenate the embedding vector for initial nodes and the sum of hidden message vectors
            # 将初始节点的嵌入向量和隐藏信息向量的总和连接起来
            z_input = torch.cat([x, sum_h], dim=1)#[num_edges, hidden_size + hidden_size]将所有边初始节点的embedding和它所有前向边的hidden vectors拼接

            # implement GRU operations
            #z = F.sigmoid(self.W_z(z_input))#[num_edges, hidden_size]
            z = torch.sigmoid(self.W_z(z_input))  # [num_edges, hidden_size]
            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)# [num_edges, hidden_size]->[num_edges, hidden_size]->[num_edges, 1, hidden_size]
            r_2 = self.U_r(h_nei)#[num_edges, max_num_neighbors, hidden_size]->[num_edges, max_num_neighbors, hidden_size]

            #r = F.sigmoid(r_1 + r_2)
            r = torch.sigmoid(r_1 + r_2)#[num_edges, max_num_neighbors, hidden_size]
            
            gated_h = r * h_nei#[num_edges, max_num_neighbors, hidden_size]
            sum_gated_h = gated_h.sum(dim=1)#[num_edges, hidden_size]
            h_input = torch.cat([x, sum_gated_h], dim=1)#[num_edges, hidden_size + hidden_size]
            #pre_h = F.tanh(self.W_h(h_input))
            pre_h = torch.tanh(self.W_h(h_input))#[num_edges, hidden_size]
            h = (1.0 - z) * sum_h + z * pre_h#[num_edges, hidden_size]
            h = h * mask

        return h


