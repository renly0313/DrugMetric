import rdkit
import rdkit.Chem as Chem
from fast_jtnn.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from fast_jtnn.vocab import *
import sys
import argparse

class MolTreeNode(object):
    """
    决策树每个节点的具体信息，node.的内部信息
    包括SMILES表示，mol分子表示（凯库勒表示），clique原子编号信息，neighbors相邻节点信息
    """
    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol): #original_mol是原始整个分子，而这的self是每个node的self
        '''
        给定原始分子，我们这步recover的节点是这个原始分子的一部分
        重构分子片段信息，该信息包括这个分子片段和所有临近的片段组成，如同目录下1.png所示
        :param original_mol:
        :return:
        '''

        clique = []
        clique.extend(self.clique) #把分子簇内的原子赋值给clique
        '''
        AtomMapNums被用作聚类标签
        将特定的簇的所有原子的atomMapNum设置为该簇的idx-nid
        
        如果这个节点不是叶节点，对于原子分子中这个簇包含的原子，把他们的atomMapNum设置为这个簇的idx
        注意，以下处理对象都是原始分子，设置的是原子的atomMapNum的值
        '''
        if not self.is_leaf: #对于小片段的非叶子子结构的原子，把其本来的编号，也就是clique里的编号，改为同一个nid的值，也就是键编号+1的值
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)
        '''
        同样对于邻居簇中的原子，把他们的atomMapNum的值设置为邻居簇的idx
        '''
        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #如果邻居点是叶子点，就不做标记了Leaf node, no need to mark
                continue
            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                '''
                第一个条件意思是，对于邻居簇中的原子（非共用原子外的），设置为邻居簇的idx
                第二个条件意思是，对于加的singleton节点，把他的原子（一个）设置为他的idx
                '''
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        #去除重复原子，得到一个大的分子片段，他是对应于当前节点（簇）和其邻居簇的集群的集合
        clique = list(set(clique))

        #获得这个大的分子片段的分子表示
        label_mol = get_clique_mol(original_mol, clique)

        #获得这个大的分子片段的SMILES表示，把他作为这个簇（节点）的label
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol))) #这个label是输入节点，也就是小分子片段和其相邻邻居的一个中型片段，并且对于

        #把原分子中的原子的AtomMapNum重置为0
        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label
    
    def assemble(self):
        '''


        :return:
        '''
        #获得这个节点（簇）的非singleton邻居节点（簇）
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1] #

        #按原子数的降序，排列邻居节点（簇），变成一个list
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)

        #获得为singleton的邻居节点的，加在邻居list的最前面
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        #获得所有可能的候选分子附着构型，对应于该簇的和邻居的所有可能的有效组合
        aroma: object
        cands,aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i,cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0: cands = new_cands

        if len(cands) > 0:
            self.cands, _ = list(zip(*cands))
            self.cands = list(self.cands)
        else:
            self.cands = []

class MolTree(object):
    """
    给一个分子的SMILES表示，该类可以构建该分子的连接树
    """

    def __init__(self, smiles):
        """
        MolTree的构建，参数为SMILES表示，返回为对应分子MolTree对象
        树节点为分子片段，分子片段是根据化学键和其相连的原子分割出来的
        注意：树的节点（分子片段）之间是有相交的原子的，两个节点之间如果连接，一定会有共用原子
        :param smiles:
        """
        self.smiles = smiles
        self.mol = get_mol(smiles) #用一个函数把smiles转化为mol，并把芳香键改为双键和单键

        #Stereo Generation (currently disabled)
        #mol = Chem.MolFromSmiles(smiles)
        #self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        #self.smiles2D = Chem.MolToSmiles(mol)
        #self.stereo_cands = decode_stereo(self.smiles2D)

        #该步骤为获得分子以键和其相连原子（分子片段）为构建节点基础的决策树过程
        # 获得该分子中的集群和决策树的稀疏表示，其中cliques表示为节点，每个list中的值代表该分子片段包含的原子序号，例如cliquees = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 10], [10, 11], [11, 12], [12, 13], [12, 14], [12, 15], [5, 6, 7, 8, 9], [1], [12]]
        # edges为结构树的稀疏表示，每个list里的值表示哪两个节点（分子片段）相连关系，例如edges = {list: 13} [(0, 12), (1, 2), (1, 12), (2, 3), (3, 4), (4, 11), (5, 6), (5, 12), (6, 7), (7, 13), (8, 13), (9, 13), (10, 13)]
        cliques, edges = tree_decomp(self.mol)

        #以下步骤为构建决策树完整的节点信息
        self.nodes = []
        root = 0
        for i,c in enumerate(cliques): #按每个键提取分子片段，c的值表示原子编号，i是节点编号,主要是创建决策树的节点
            cmol = get_clique_mol(self.mol, c)      #这步是把第i个节点和该节点表示的mol结构提取出来了，放入cmol
            node = MolTreeNode(get_smiles(cmol), c) #为节点定义一些初始的信息，包括clique包含的原子,mol分子形式，neighbors邻居,smiles表示四个
            self.nodes.append(node) #把每个节点node和其信息放在self.nodes下
            if min(c) == 0: root = i #意思就是，0号原子出现在哪一个节点，这个节点就作为分子树的根节点

        # 把相邻边的分子片段做一个相互加邻居，比如一个甲苯,就是把苯环作为一个node，相邻点不是甲基，而是smiles表示为'cc'，也就是说，他们共同拥有c
        for x,y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        # 对应上面的，如果根节点不是编号为0的节点，就把这个节点提到第一个来互换一下，把这个节点定义为0号节点,
        if root > 0:
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        # 对于每个非叶子节点，对于相应分子片段包含的原子，我们将这些原子的atomMapNum设置为该节点 / 分子片段的node_id / nid。
        for i,node in enumerate(self.nodes): #對於非葉子節點的原子進行一個標記，並且在nodes加入屬性 is_leaf bool類型
            node.nid = i + 1
            if len(node.neighbors) > 1: #Leaf node mol is not marked 对于非树叶结构，也就是不是一个邻居的，对其内部的所有原子进行同样标记为分子片段的id+1，可以通过atom.GetIdx() 獲取
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        """
        主要功能是恢复连接树中的所有节点
        例如，对于每一个节点（一个小结构），重构由该节点和其邻居节点组成的分子片段
        """
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        '''
        聚集连接树中的所有点节点，即对于每个节点，获得该节点和其相邻节点所有可能的分子链接配置
        :return:
        '''
        for node in self.nodes:
            node.assemble()


def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx:
            continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


def main_mol_tree(oinput, ovocab, MAX_TREE_WIDTH=50):
    cset = set()        #创建一个set，用来储存分子片段，set()不能重复
    with open(oinput, 'r') as input_file:
        for i, line in enumerate(input_file.readlines()): #枚举，i是输入SMILES的行数，line是每一行的SMILES
            smiles = line.strip().split()[0]
            alert = False
            mol = MolTree(smiles) #把随机的SMILES进行了树分解，并通过分子片段构建了新的决策树的节点，mol只有分子结构，结构树节点和smiles三个属性
            for c in mol.nodes:
                if c.mol.GetNumAtoms() > MAX_TREE_WIDTH: #如果一个分子片段超过50个原子，就会报错，除非整个分子就是一个点
                    alert = True
                cset.add(c.smiles) #把分子片段的smiles加到cset这个集合里
            if len(mol.nodes) > 1 and alert:
                sys.stderr.write('[WARNING]: %d-th molecule %s has a high tree-width.\n' % (i + 1, smiles))
    
    with open(ovocab, 'w') as vocab_file:
        for x in cset:
            vocab_file.write(x+'\n')


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    sys.stderr.write('Running tree decomposition on the dataset')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='../data/zinc/all.txt', dest="input")
    parser.add_argument("-v", "--vocab", default='../data/zinc/vocab_zinc.txt', dest="vocab")
    opts = parser.parse_args()

    main_mol_tree(opts.input, opts.vocab)
