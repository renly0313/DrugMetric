import rdkit
import rdkit.Chem as Chem
import copy

def get_slots(smiles):# 从SMILES获得一个分子中每一个原子的类型，电荷，连接的氢原子个数
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):
    """
        This class takes a list of SMILES strings, that correspond to the cluster vocabulary of the training dataset.
        这个类需要一个SMILES字符串的列表，它对应于训练数据集的集群词汇。
        """
    benzynes = ['C1=CC=CC=C1', 'C1=CC=NC=C1', 'C1=CC=NN=C1', 'C1=CN=CC=N1', 'C1=CN=CN=C1', 'C1=CN=NC=N1', 'C1=CN=NN=C1', 'C1=NC=NC=N1', 'C1=NN=CN=N1']
    penzynes = ['C1=C[NH]C=C1', 'C1=C[NH]C=N1', 'C1=C[NH]N=C1', 'C1=C[NH]N=N1', 'C1=COC=C1', 'C1=COC=N1', 'C1=CON=C1', 'C1=CSC=C1', 'C1=CSC=N1', 'C1=CSN=C1', 'C1=CSN=N1', 'C1=NN=C[NH]1', 'C1=NN=CO1', 'C1=NN=CS1', 'C1=N[NH]C=N1', 'C1=N[NH]N=C1', 'C1=N[NH]N=N1', 'C1=NN=N[NH]1', 'C1=NN=NS1', 'C1=NOC=N1', 'C1=NON=C1', 'C1=NSC=N1', 'C1=NSN=C1']

    def __init__(self, smiles_list):
        """
                This is the constructor for the ClusterVocab class.

                Args:
                    smiles_list: list of SMILES representations, that correspond to the cluster vocabulary over the training dataset.
                    SMILES表征的列表，对应于训练数据集上的集群词汇。

                Returns:
                    ClusterVocab object for the corresponding training dataset.
                    对应训练数据集的ClusterVocab对象。
                """
        #list of SMILES representations, corresponding to the cluster vocabulary 与集群词汇相对应的SMILES表示列表
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        #List[List[(原子符，化合價，鏈接的H原子數),(),()...],[],[]...]
        #感觉没有什么用
        Vocab.benzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 6] + ['C1=CCNCC1']
        Vocab.penzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 5] + ['C1=NCCN1','C1=NNCC1']

    def get_index(self, smiles):
        """
            This method gives the index corresponding to the given cluster vocabulary item.
            这种方法给出了与给定集群词汇项对应的索引。
            Args:
                smiles: SMILES representaion of a cluster vocabulary item.
            Returns:
                 Index of the corresponding cluster vocabulary item.
                 相应集群词汇项的索引。
        """
        try:
            return self.vmap[smiles]
        except KeyError:
            print("KeyError: '{}' is not a valid key in the vocabulary dictionary.".format(smiles))
            return None

    def get_smiles(self, idx):
        """
         This method returns the corresponding the SMILES representation for the cluster vocabulary item, given an index.
         该方法返回集群词汇项的相应SMILES表示，给定一个索引。
        Args:
            idx: index of the cluster vocabulary item
        Returns:
            The SMILES representation of the corresponding cluster vocabulary item.
            相应集群词汇项的SMILES表示。
        """
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

