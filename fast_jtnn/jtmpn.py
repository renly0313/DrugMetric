import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_jtnn.nnutils import create_var, index_select_ND
from fast_jtnn.chemutils import get_mol
import rdkit.Chem as Chem

# 原子列表
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']
# 原子类型 len(ELEM_LIST)
# 原子的度 ([0, 1, 2, 3, 4, 5]) 6
# 原子的形式电荷 ([-2, -1, 0, 1, 2]) 5
# 原子的芳香性 1
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
# 键类型的单热编码 ([单键、双键、三键、芳烃键、环内键])
BOND_FDIM = 5
# 最大原子数目（化学领域知识）。
MAX_NB = 30


def onek_encoding_unk(x, allowable_set):
    """
    Description: This function, given a categorical variable,
    returns the corresponding one-hot encoding vector.
    描述。这个函数，给定一个分类变量，返回相应的单热编码向量。

    Args:
        x: object
            The categorical variable to be one-hot encoded.
            要进行单热编码的分类变量。
        allowable_set: List[object]
            List of all categorical variables in consideration.
            考虑中的所有分类变量的清单。
    Returns:
         List[Boolean]
            The corresponding one-hot encoding vector.
            相应的单热编码向量。

    """

    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return torch.Tensor(
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        + [atom.GetIsAromatic()])


def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                         bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])


class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        """
        Constructor for the class.
        Args:
            hidden_size: Dimension of the hidden message vectors.
            depth: Number of timesteps for message passing.
        """
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope, tree_message):  # tree_message[0] == vec(0)
        """
        Description: Implements the forward pass for encoding the candidate molecular subgraphs, for the graph decoding step. (Section 2.5)
        为图的解码步骤，执行对候选分子子图进行编码的前向传递。(第2.5节)

        Args:
            fatoms: torch.tensor (shape: num_atoms x ATOM_FEATURE_DIM)
                Matrix of atom features for all the atoms, over all the molecules, in the dataset.
                数据集中所有分子的所有原子的原子特征矩阵。

            fbonds: torch.tensor (shape: num_bonds x ATOM_FEATURE_DIM + BOND_FEATURE_DIM)
                Matrix of bond features for all the bond, over all the molecules, in the dataset.
                数据集中所有分子上的纽带特征矩阵。

            a_graph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For every atom across the training dataset, this atom_graph gives the bond idxs of all the bonds
                in which it is present. An atom can at most be present in MAX_NUM_NEIGHBORS(= 6) bonds.
                对于训练数据集中的每一个原子，这个原子图给出了它所存在的所有化学键的键IDX。一个原子最多只能出现在MAX_NUM_NEIGHBORS(=6)个键上。

            b_graph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6))
                For every non-ring bond (cluster-node) across the training dataset, this bond_graph gives the bond idx of
                those non-ring bonds (cluster-nodes), to which it is connected in the "cluster-graph".
                对于整个训练数据集中的每一个非环状键（集群节点），这个bond_graph给出了这些非环状键（集群节点）的键idx，它在 "集群-图形 "中与之相连。

            scope: List[Tuple(int, int)]
                List of tuples of (total_atoms, num_atoms). Used to extract the atom features for a
                particular molecule in the dataset, from the atom_feature_matrix.
                (total_atoms, num_atoms)的tuples列表。用于从atom_feature_matrix中提取数据集中某个特定分子的原子特征。

        Returns:
            mol_vecs: torch.tensor (shape: num_candidate_subgraphs x hidden_size)
                The encoding of all the candidate subgraphs for scoring purposes. (Section 2.5)
                用于评分的所有候选子图的编码。(第2.5节)

        """
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        # apply ReLU activation for timestep, t = 0 应用ReLU激活时间步长，t=0
        graph_message = F.relu(binput)

        # implement message passing for timesteps, t = 1 to T (depth) 实现时间步数的消息传递，t=1到T（深度）。
        for i in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0)
            # obtain messages from all the "inward edges" 从所有 "内向型边缘 "获得信息
            nei_message = index_select_ND(message, 0, bgraph)

            # sum up all the "inward edge" message vectors 将所有 "内向型 "信息向量相加。
            nei_message = nei_message.sum(dim=1)  # assuming tree_message[0] == vec(0)

            # multiply with the weight matrix for the hidden layer 与隐藏层的权重矩阵相乘
            nei_message = self.W_h(nei_message)

            # message at timestep t + 1
            graph_message = F.relu(binput + nei_message)

        # neighbor message vectors for each from the message matrix 从信息矩阵中为每个人提供邻居信息向量
        message = torch.cat([tree_message, graph_message], dim=0)

        # neighbor message for each atom 每个原子的邻居信息
        nei_message = index_select_ND(message, 0, agraph)

        # neighbor maeesge for each atom 每个原子的相邻关系
        nei_message = nei_message.sum(dim=1)

        # concatenate atm feature vector and neighbor hidden message vector
        # 连接ATM特征向量和邻居隐藏信息向量
        ainput = torch.cat([fatoms, nei_message], dim=1)

        atom_hiddens = F.relu(self.W_o(ainput))

        # list to store the corresponding molecule vectors for each molecule
        # 列表来存储每个分子对应的分子向量
        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            # mol_vec = atom_hiddens[st: st + le].mean(dim=0)
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

    @staticmethod
    def tensorize(cand_batch, mess_dict):
        """
        Args:
            cand_batch: List[Tuple(str, List[MolJuncTreeNode], object: rdkit)]
                The list of candidate subgraphs to be scored, for the graph decoding step (Section 2.5).
                需要评分的候选子图列表，用于图解码步骤（第2.5节）。

            mess_dict: Dict{(MolJuncTreeNode, MolJuncTreeNode): torch.tensor (shape: hidden_size)}
                The dictionary containing edge messages from the tree-encoding step (Section 2.3 and Section 2.5 magic)
                包含来自树形编码步骤的边缘信息的字典（第2.3节和第2.5节魔法）。

        Returns:
            fatoms: torch.tensor (shape: num_atoms x ATOM_FEATURE_DIM)
                Matrix of atom features for all the atoms, over all the molecules, in the dataset.
                数据集中所有分子的所有原子的原子特征矩阵。

            fbonds: torch.tensor (shape: num_bonds x ATOM_FEATURE_DIM + BOND_FEATURE_DIM)
                Matrix of bond features for all the bond, over all the molecules, in the dataset.
                数据集中所有分子上的键特征矩阵。

            agraph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For every atom across the training dataset, this atom_graph gives the bond idxs of all the bonds
                in which it is present. An atom can at most be present in MAX_NUM_NEIGHBORS(= 6) bonds.
                对于训练数据集中的每一个原子，这个原子图给出了它所存在的所有化学键的键IDX。一个原子最多只能出现在MAX_NUM_NEIGHBORS(=6)个键上。

            bgraph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6))
                For every non-ring bond (cluster-node) across the training dataset, this bond_graph gives the bond idx of
                those non-ring bonds (cluster-nodes), to which it is connected in the "cluster-graph".
                对于整个训练数据集中的每一个非环状键（集群节点），这个bond_graph给出了这些非环状键（集群节点）的键idx，它在 "集群-图形 "中与之相连。

            scope: List[Tuple(int, int)]
                List of tuples of (total_atoms, num_atoms). Used to extract the atom features for a
                particular molecule in the dataset, from the atom_feature_matrix.
                (total_atoms, num_atoms)的tuples列表。用于从atom_feature_matrix中提取数据集中某个特定分子的原子特征。
        """
        # lists to store atom and bond feature vectors of candidate subgraphs to be encoded by message passing (Section 2.5)
        # 列表用于存储候选子图的原子和键的特征向量，通过信息传递进行编码（第2.5节）。
        fatoms, fbonds = [], []

        # in bonds: for each atom, the list of idx of all bonds, in which it is the terminal atom all_bonds: to store all the bonds (Tuple(MolJuncTreeNode, MolJuncTreeNode)), for all molecules, across the entire dataset.
        # in bonds：对于每个原子，所有键的idx列表，其中它是终端原子 all_bonds：存储所有的键（Tuple(MolJuncTreeNode, MolJuncTreeNode)），对于所有的分子，在整个数据集中。
        in_bonds, all_bonds = [], []

        # for each atom, of every molecule, across the dataset, we give it an idx 对于每个分子的每个原子，在整个数据集中，我们给它一个idx
        total_atoms = 0

        # the tensor at the 0th index is the padding vector 第0个索引的张量是填充矢量
        total_mess = len(mess_dict) + 1  # must include vec(0) padding 必须包括vec(0)的填充

        # to store tuples of (start_idx, len) to delinate atom features for a particular molecule.
        # 来存储(start_idx, len)的图元，以划定特定分子的原子特征。
        scope = []

        for smiles, all_nodes, ctr_node in cand_batch:
            # obtain the rdkit molecule object 获得rdkit分子对象
            mol = Chem.MolFromSmiles(smiles)

            # obtain the kekulized representation of the molecule object. The original jtnn version kekulizes. Need to revisit why it is necessary
            # 获得分子对象的kekulized表示。原始的jtnn版本kekulizes。需要重新审视为什么需要这样做
            Chem.Kekulize(mol)

            # number of atoms in this molecule, (for scope tuple) 该分子中的原子数，（对于范围元组）。
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            for atom in mol.GetAtoms():
                # obtain the feature vector for the given atom 获得给定原子的特征向量
                fatoms.append(atom_features(atom))
                # append an empty list of every molecule, to store idxs of all the bonds, in which it is the terminal atom.
                # 为每个分子添加一个空列表，以存储所有键的idxs，其中它是终端原子。
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()

                # offsetted begin and end atom idxs 偏移的开始和结束原子IDxs
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                # Here x_nid,y_nid could be 0 retrieve the idxs of the nodes in the junction tree, in whose corresponding clusters, the atoms are included
                # 这里x_nid,y_nid可以是0，检索出结点树中节点的idxs，在其对应的簇中，原子被包括在内。
                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                # obtain the feature vector for this bond. 获得该键的特征向量。
                bfeature = bond_features(bond)

                # bond idx offseted by total_mess 被total_mess抵消的键idx
                b = total_mess + len(all_bonds)  # bond idx offseted by total_mess 被total_mess抵消的键idx
                all_bonds.append((x, y))
                fbonds.append(torch.cat([fatoms[x], bfeature], 0))
                in_bonds[y].append(b)

                b = total_mess + len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(torch.cat([fatoms[y], bfeature], 0))
                in_bonds[x].append(b)

                # the weird message passing magic in (Section 2.5) 在（第2.5节）中奇怪的消息传递魔法
                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx)

            # append start_idx and len for delinating atom features for a particular molecule 附加start_idx和len，以确定某个特定分子的原子特征。
            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms, MAX_NB).long()
        bgraph = torch.zeros(total_bonds, MAX_NB).long()

        # for each atoms, the list of idxs of all the bonds, in which it is the terminal atom
        # 对于每个原子，所有键的idxs列表，其中它是终端原子
        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b

        # for each bond, the list of idx of "inward" "bonds", for message passing purposes
        # 对于每个键，"内向""键 "的idx列表，用于信息传递目的
        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                # b2 is offseted by total_mess b2被total_mess所抵消。
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bgraph[b1, i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)

