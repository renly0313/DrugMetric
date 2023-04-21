import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from fast_jtnn.nnutils import *
from fast_jtnn.chemutils import get_mol
# list of elements are were are considering in our problem domain
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

# one-hot encoding for the atom / element type
# one-hot encoding for the degree of the atom ([0, 1, 2, 3, 4, 5])
# one-hot encoding for the formal charge of the atom ([-2, -1, 0, 1, 2])
# one-hot enoding for the chiral-tag of the atom i.e. number of chiral centres ([0, 1, 2, 3])
# one-hot encoding / binary encoding whether atom is aromatic or not
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1

# 键类型的单热编码 ([单键、双键、三键、芳烃键、环内键])
# 键的立体构型的单热编码 ([0, 1, 2, 3, 4, 5])
BOND_FDIM = 5 + 6

# maximum number of an atom in a molecule (Chemistry Domain Knowledge) 原子的最大邻居数（化学领域知识）。
MAX_NB = 6

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
    """
    Description: This function, constructs the feature vector for the given atom.
    这个函数，为给定的原子构造特征向量。

    Args: (object: rdkit)
        The atom for which the feature vector is to be constructed.
        要构建特征向量的原子。

    Returns:
        torch.tensor (shape: ATOM_FEATURE_DIM)
    """
    return torch.Tensor(
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)# one-hot encode atom symbol/ element type
        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])# one-hot encode atom degree
        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])# one-hot encode formal charge of the atom
        + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])# one-hot encode the chiral tag of the atom 手性
        + [atom.GetIsAromatic()])# one-hot encoding / binary encoding whether atom is aromatic or not是否是芳香的

def bond_features(bond):
    """
    Description: This function, constructs the feature vector for the given bond.
    这个函数，为给定的键构建特征向量。

    Args:
        bond: (object: rdkit)
            The bond for which the feature vector is to be constructed.
            要构建特征向量的键。
    Returns:
        torch.tensor (shape: BOND_FEATURE_DIM)
    """
    # obtain the bond-type
    bt = bond.GetBondType()
    # obtain the stereo-configuration
    stereo = int(bond.GetStereo())
    # one-hot encoding the bond-type
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
             bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    # one-hot encoding the stereo configuration
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return torch.Tensor(fbond + fstereo)

class MPN(nn.Module):
    """
    Message Passing Network for encoding molecular graphs.
    """
    def __init__(self, hidden_size, depth):
        """
        Constructor for the class.
        Args:
            hidden_size: Dimension of the encoding space.
            depth: Number of timesteps for which to run the message passing
        Returns:
            The corresponding MessPassNet object.
        """
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # weight matrix for bond feature matrix
        # instead of W^g_1 x_u + W^g_2 x_uv
        # concatenate x_u and x_uv and use W_i x
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        # weight matrix for hidden layer
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        # weight matrix for output layer
        # instead of U^g_1 x_u + summation U^g_2 nu^(T)_vu
        # concatenate x_u and summation nu^(T)_vu and use W_o x
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope):
        """
        Description: Implements the forward pass for encoding the original molecular subgraphs, for the graph encoding step. (Section 2.2)
        实现对原始分子子图进行编码的前向传递，用于图编码步骤。

        Args:
            fatoms: torch.tensor (shape: num_atoms x ATOM_FEATURE_DIM)
                Matrix of atom features for all the atoms, over all the molecules, in the dataset.
                数据集中所有分子的所有原子的原子特征矩阵。

            fbonds: torch.tensor (shape: num_bonds x ATOM_FEATURE_DIM + BOND_FEATURE_DIM)
                Matrix of bond features for all the bond, over all the molecules, in the dataset.
                数据集中所有分子上的纽带特征矩阵。

            agraph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For every atom across the training dataset, this atom_graph gives the bond idxs of all the bonds in which it is present. An atom can at most be present in MAX_NUM_NEIGHBORS(= 6) bonds.
                对于训练数据集中的每一个原子，这个原子图给出了它所存在的所有化学键的键IDX。一个原子最多只能出现在MAX_NUM_NEIGHBORS(=6)键上。

            bgraph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6))
                For every non-ring bond (cluster-node) across the training dataset, this bond_graph gives the bond idx of those non-ring bonds (cluster-nodes), to which it is connected in the "cluster-graph".
                对于整个训练数据集中的每一个非环状键（集群节点），这个bond_graph给出了这些非环状键（集群节点）的键idx，它在 "集群-图形 "中与之相连。

            scope: List[Tuple(int, int)]
                List of tuples of (total_atoms, num_atoms). Used to extract the atom features for a particular molecule in the dataset, from the atom_feature_matrix.
                (total_atoms, num_atoms)的tuples列表。用于从atom_feature_matrix中提取数据集中某个特定分子的原子特征。

        Returns:
             mol_vecs: torch.tensor (shape: batch_size x hidden_size)
                The encoding vectors for all the molecular graphs in the entire dataset.
                整个数据集中所有分子图的编码向量。
        """
        fatoms = create_var(fatoms)#[num_atoms, ATOM_FEATURE_DIM(=39)]
        fbonds = create_var(fbonds)#[num_bonds, ATOM_FEATURE_DIM + BOND_FEATURE_DIM(=50)]
        agraph = create_var(agraph)#[num_atoms, MAX_NUM_NEIGHBORS(=6)]
        bgraph = create_var(bgraph)#[num_bonds, MAX_NUM_NEIGHBORS(=6)]

        binput = self.W_i(fbonds)#[num_bonds, ATOM_FEATURE_DIM + BOND_FEATURE_DIM]->[num_bonds, hidden_size]
        #apply ReLU activation for timestep, t = 0
        message = F.relu(binput)#[num_bonds, hidden_size]

        # implement message passing for timesteps, t = 1 to T (depth)
        for i in range(self.depth - 1):
            # obtain messages from all the "inward edges" 从所有 "内向型边缘 "获得信息
            nei_message = index_select_ND(message, 0, bgraph)#[num_bonds, MAX_NUM_NEIGHBORS(=6), hidden_size]
            # sum up all the "inward edge" message vectors 将所有 "内向型 "信息向量相加。
            nei_message = nei_message.sum(dim=1)#[num_bonds, hidden_size]
            # multiply with the weight matrix for the hidden layer 与隐藏层的权重矩阵相乘
            nei_message = self.W_h(nei_message)#[num_bonds, hidden_size]
            # message at timestep t +1
            message = F.relu(binput + nei_message)#[num_bonds, hidden_size]

        # neighbor message vectors for each node from the message matrix 从信息矩阵中获取每个节点的邻居信息向量
        nei_message = index_select_ND(message, 0, agraph)#[num_atoms, MAX_NUM_NEIGHBORS(=6), hidden_size]
        # neighbor message for each atom 每个原子的邻居信息
        nei_message = nei_message.sum(dim=1)#[num_atoms, hidden_size]
        # concatenate atom feature vector and neighbor hidden message vector 连接原子特征向量和邻居隐藏信息向量
        ainput = torch.cat([fatoms, nei_message], dim=1)#[num_atoms, ATOM_FEATURE_DIM + hidden_size]
        atom_hiddens = F.relu(self.W_o(ainput))#[num_atoms, hidden_size]

        max_len = max([x for _, x in scope])
        # list to store the corresponding molecule vectors for each molecule 列表来存储每个分子对应的分子向量
        batch_vecs = []
        for st, le in scope:
            # the molecule vector for molecule, is the mean of the hidden vectors for all the atoms of that molecule
            # 分子的分子向量，是该分子所有原子的隐藏向量的平均值
            cur_vecs = atom_hiddens[st: st + le].mean(dim=0)
            batch_vecs.append(cur_vecs)#List[32个Tensor: (hidden_size,)]

        mol_vecs = torch.stack(batch_vecs, dim=0)#{Tensor: (32, 450)}[batch_size, hidden_size]
        return mol_vecs 

    @staticmethod
    def tensorize(mol_batch):
        """
        Description: This method, given a batch of SMILES representations,
        constructs the feature vectors for the corresponding molecules
        这个方法，给定一批SMILES表征，为相应的分子构建特征向量

        Args:
            mol_batch: List[str]
                The batch of SMILES representations for the dataset. 数据集的一批SMILES表示法。

        Returns:(fatoms, fbonds, agraph, bgraph, scope)
            fatoms: torch.tensor (shape: batch_size x ATOM_FEATURE_DIM)
                The matrix containing feature vectors, for all the atoms, across the entire dataset.
                  包含所有原子的特征向量的矩阵，跨越整个数据集。

            fbonds: torch.tensor (shape: batch_size x ATOM_FEATURE_DIM + BOND_FEATURE_DIM)
                The matrix containing feature vectors, for all the bonds, across the entire dataset.
                  包含所有键的特征向量的矩阵，跨越整个数据集。

            agraph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire dataset, the idxs of all the neighboring atoms.
                对于每个原子，在整个数据集中，所有相邻原子的idxs。

            bgraph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6))
                For each bond, across the entire dataset, the idxs of the "inward bonds", for purposes of message passing.
                对于每个键，在整个数据集中，为了信息传递的目的，"内向键"的idxs。

            scope: List[Tuple(int, int)]
                The list to store tuples (total_bonds, num_bonds), to keep track of all the atom feature vectors,
                belonging to a particular molecule.
                存储图元（total_bonds, num_bonds）的列表，以跟踪所有原子特征向量，属于一个特定的分子。
        """
        # add zero-padding for the bond feature vector matrix 为键特征向量矩阵添加零填充
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)

        fatoms, fbonds = [], [padding] #Ensure that the bond features vectors are 1-indexed

        #对于分子中给定的atom_idx, in_bonds存储以该原子为尾节点的所有键在all_bonds中的idxs列表
        in_bonds, all_bonds = [], [(-1, -1)] #Ensure that the bonds are  1-indexed

        # list to store tuples of (start_index, len) for atoms feature vectors of each molecule
        # 用于存储每个分子的原子特征向量的（start_index, len）图组的列表
        scope = []
        # 在整个数据集中，所有分子的每个原子都有一个idx
        total_atoms = 0

        for smiles in mol_batch:
            mol = get_mol(smiles)
            #mol = Chem.MolFromSmiles(smiles)
            n_atoms = mol.GetNumAtoms()
            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                b = len(all_bonds) 
                all_bonds.append((x, y))
                fbonds.append(torch.cat([fatoms[x], bond_features(bond)], 0))
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(torch.cat([fatoms[y], bond_features(bond)], 0))
                in_bonds[x].append(b)
            
            scope.append((total_atoms, n_atoms))#(start_idx, len)存储每个分子在fatoms中的idx
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0)#[total_atoms, 39]
        fbonds = torch.stack(fbonds, 0)#[total_bonds, 50]
        agraph = torch.zeros(total_atoms, MAX_NB).long()#[total_atoms, 6]
        bgraph = torch.zeros(total_bonds, MAX_NB).long()#[total_bonds, 6]

        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b #agraph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6)).对每个原子，存储其所有邻居的idx
                # For each atom, across the entire dataset, the idxs of all the neighboring atoms.

        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                # given the bond (x, y), don't consider the same bond again i.e. (y, x)
                if all_bonds[b2][0] != y:
                    bgraph[b1, i] = b2#bgraph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6)).存储边b1的所有前向边的idx
                    # For each bond, across the entire dataset, the idxs of the "inward bonds", for purposes of message passing.
                    # 对于每个键，在整个数据集中，为了信息传递的目的，"内向键 "的idxs。
        return (fatoms, fbonds, agraph, bgraph, scope)

