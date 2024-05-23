import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from fast_jtnn.vocab import Vocab

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000

def set_atommap(mol, num=0):
    """
       Description: THhis function, given a molecule, sets the AtomMapNum of all atoms in the molecule, to the given num
       描述:这个函数，给定一个分子，将分子中所有原子的AtomMapNum设置为给定的数目（获得原子数目）
    Args:
        mol: (object: rdkit)
        """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_mol(smiles):
    """
    Description: This function, given the SMILES representation of a molecule,
    returns the kekulized molecule.
    输入smiles，返回凯库勒式
    Args:
        smiles: str
            SMILES representation of the molecule to be kekulized.

    Returns:
        mol: (object: rdkit)
            Kekulized representation of the molecule.
            凯库勒式表示分子
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol) #输出kekule形式，在符合4N+2规则的芳香体系中，通过使用双键代替小写的碳原子来表示芳香性，4N+2规则：也叫Hueckel规则，在闭环共轭体系中，当π电子数为4n+2时，才具有芳香性
    return mol

def get_smiles(mol):
    """
    Description: This function, given a molecule, returns the SMILES representation,
    which encodes the kekulized structure of the molecule
    输入分子，返回smiles，编码了分子的凯库勒结构
    Args:
        mol: (object: rdkit)
            The molecule to be kekulized.
        分子是凯库勒式
    Returns:
        SMILES: str
            SMILES representation, encoding the kekulized structure of the molecule
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def decode_stereo(smiles2D):
    """
       Description: This function, given the SMILES representation of a molecule, encoding its 2D structure,
       gives the list of SMILES representation of all stereoisomers of the molecule, encoding their 3D structure
       给定一个分子的smile表示，编码它的二维结构，
       给出了该分子所有立体异构体的smile表示列表，对其三维结构进行编码
       Args:
           smiles2D: str
               SMILES representation, encoding its 2D structure,

       Returns:
           smiles3D: List[str]
               The list of SMILES representation of all stereoisomers of the molecule, encoding their 3D structure
               分子的所有立体异构体的SMILES表示列表，编码其三维结构。
       """
    # convert to molecular representation, from the SMILES representation  从SMILES表示转换为分子表示
    mol = Chem.MolFromSmiles(smiles2D)
    # obtain all the stereoisomers of the molecule  得到分子的所有立体异构体
    dec_isomers = list(EnumerateStereoisomers(mol))
    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    # obtain SMILES representation of all stereoisomers, encoding their 3D structure
    # 获得所有立体异构体的smile表示，对其三维结构进行编码
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    # in organic chemistry, nitrogen atoms are common chiral centers  在有机化学中，氮原子是常见的手性中心
    # thus, we get the idx of all nitrogen atoms that are chiral  这样，我们就得到了所有手性氮原子的idx
    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    # if there are any chiral nitrogen centers, then we set their chiral tag to unspecified 如果有手性氮中心，则将其手性标记设为未指定
    # because we are not currently dealing with chirality of nitrogen atoms 因为我们现在不讨论氮原子的手性
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D

def sanitize(mol):
    """
        Description: This function, given a molecule, returns the kekulized representation of the same molecule.

        Args:
            mol: (object: rdkit)
                The molecule to be kekulized.

        Returns:
            mol: (object: rdkit)
                The kekulized molecule.
        """
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def copy_atom(atom):
    """
    Description: This function, given an atom, returns a new atom, which is a "deep copy" of the given atom
    这个函数，给定一个原子，返回一个新的原子，它是给定原子的一个 "深度拷贝"。
    Args:
        atom: (object: rdkit)
            The atom to be copied.

    Returns:
        new_atom: (object: rdkit)
            New copy of the atom.
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    """
    Description: This function, given a molecule, returns a new molecule, which is a "deep copy" of the given molecule

    Args:
        mol: (object: rdkit)
            The molecule to be copied.

    Returns:
        new_mol: (object: rdkit)
            New copy of the molecule.
    """
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(original_mol, atoms):
    """
        Description: This function, given the original molecule, and a cluster of atoms,
        returns the molecular fragment of the original molecule, corresponding to this cluster of atoms
        这个函数，给定原分子和一个原子簇，返回原分子的分子片段，对应于这个原子簇。

        Args:
            original_mol: (object: rdkit)
                The original molecule, of which this cluster is a part of. 原始分子，这个簇是其中的一部分。

            atoms: List[int]
                List of atom_idx in this cluster 这个群组中的atom_idx的列表

        Returns:
            mol: (object: rdkit)
                The valid molecular fragment corresponding to the given cluster of atoms. 对应于给定原子团的有效分子片段。
        """
    # get the SMILES representation of the given cluster of atoms in this molecule 获得该分子中给定原子簇的SMILES表示法
    smiles = Chem.MolFragmentToSmiles(original_mol, atoms, kekuleSmiles=True)
    #get the molecular fragment(片段) from the SMILES representation, corresponding to this cluster
    # 从SMILES表示法中得到与该团块相对应的分子片段（片段）。
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    #get a copy of the molecular fragment 获得一份分子片段的副本
    new_mol = copy_edit_mol(new_mol).GetMol()
    # obtain the kekulized representation of the molecular fragment 获得分子片段的凯库勒化表示法
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol



def tree_decomp(mol):
    '''
    树分解Tree Decomposition of Molecules
    此函数用于查找分子簇（molecular clusters）,一个cluster仅仅是该cluster中存在的所有原子列表。
    有3种cluster:
        1. Non-Ring Bond Clusters: Corresponding to non-ring bonds. (2 atoms in the cluster)
        2. Ring Clusters: Corresponding to rings. (More than 4 atoms in the cluster)
        3. Singleton Clusters: Corresponding to single atoms. These specific atoms are at the
               intersection of three or more bonds (can be both ring and non-ring bonds).
               The "cluster-subgraph", consisting of all the "cluster-nodes" containing this atom, "would form a clique".
               Thus, in an effort to reduce the number of cycles in the "cluster-graph", the authors have introduced the notion of singleton clusters.
               These singleton clusters consist of only one atom. These specific atoms are common to 3 or more clusters.
               In the "cluster-graph", an edge is created between the "singleton cluster nodes"
               and all the other "cluster-nodes" that contain the atom corresponding to that "singleton cluster".
               These edge are assigned a large weight, so as to ensure that these edges are included when a
               "maximum spanning tree" over the "cluster-graph" is being found.
               因此，为了减少 "集群图 "中的循环次数，作者引入了单子集群的概念。 这些单子集群只由一个原子组成。
               在 "集群图 "中，在 "单子集群节点 "和包含与该 "单子集群 "相对应的原子的所有其他 "集群节点 "之间建立了一条边。
               这些边被赋予了很大的权重，以确保在寻找 "集群图 "的 "最大生成树 "时，这些边被包括在内。
    Returns:
            clusters: list of clusters, of which the molecule is composed of.
            edges: adjacency list for the "maximum spanning tree", over the "cluster-graph" 最大生成树 "的邻接列表，在 "集群图 "上
    '''
    n_atoms = mol.GetNumAtoms()
    #如果只有一个原子，则[0]是唯一的cluster，adjacency list为空列表
    if n_atoms == 1: #special case 特例
        return [[0]], []

    cliques = []#存储cluster

    #找到分子中所有的无环键的原子索引，提取 non-ring bonds 非环键
    #非环键中的所有原子对有对应于一个non-ring bond clusters
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])#将bond cluster追加到clusters列表中
    #获取分子中所有单环的原子索引， 提取simple rings  单环
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    #对每个clique进行遍历，nei_list[[],[],[],...,[]], 存储每个原子atoms都在哪个cluster里出现过
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Merge Rings with intersection > 2 atoms，合并共享三个或更多原子的环，因为他们组成了bridged compounds
    #merge rings with having more than two atoms in common. i.e. bridged compounds 融合有两个以上共同原子的环，即桥接化合物。
    for i in range(len(cliques)):
        #ignore clusters corresponding to non-ring bounds and single atoms 忽略对应于非环形边界和单原子的团块
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:#如果clique是环，对该环中的每个原子进行遍历
            for j in nei_list[atom]:#j是对该环中的每个原子所存在的clique进行遍历，代表索引
                # i是按顺序对cliques进行遍历，避免重复，所以必须同时满足i<j和第j个cliques的长度大于2，是环（包含的原子数大于2）
                #只有在i后面的才是有可能与之相交的环
                # merge ring clusters in order of i < j
                # if len(clusters[cluster_idx_j]) <= 2 i.e. this clusters corresponds to a non-ring bond or single atoms, then don't merge
                if i >= j or len(cliques[j]) <= 2:
                    continue
                #find the number of common atoms between the two clusters 找出两个原子团之间的共同原子数
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j]) #合并两个环，去除重复项
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    #remove empty clusters
    cliques = [c for c in cliques if len(c) > 0]

    # 重新对每个clique进行遍历，nei_list[[],[],[],...,[]], 存储每个原子atoms都在哪个cluster里出现过
    #对于每个atom，获取包含它的所有cluster的cluster_idx
    #this is for constructing edges between clusters in the "cluster-graph", that share at least one atom
    # 这是为了在 "集群图 "中的集群之间构建边，这些集群至少共享一个原子。
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Build edges and add singleton clusters

    #adjacency list of "cluster-graph"
    edges = defaultdict(int) #当KEY不存在时，返回默认值0

    # atoms are common between at least 3 clusters, correspond to "singleton clusters"
    # some examples are as follows:
    # 1. number of bonds > 2
    # 2. number of bonds = 2 and number of rings > 1
    # 3. number of bonds = 1 and number of rings = 2 (not considered here)
    # 4. number of rings > 2
    for atom in range(n_atoms):
        #ignore if this atom alreagy belongs to a singleton cluster
        if len(nei_list[atom]) <= 1: 
            continue

        #list of idx, of all the clusters, in which this atom is included
        cnei = nei_list[atom] #表示1个atom存在的所有cluster索引列表

        #idx of clusters corresponding to non-ring bonds
        bonds = [c for c in cnei if len(cliques[c]) == 2]    #1个atom存在的键类cluster索引列表
        #idx of clusters corresponding to rings
        rings = [c for c in cnei if len(cliques[c]) > 4]     #1个atom存在的环类cluster索引列表

        # 一个atom存在3个及以上的键类cluster中，或者一个atom存在3个及以上的cluster中（其中包含2个键类cluster）
        # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
        # 一般情况下，如果len(cnei) >= 3，应该添加一个单元素，但目前没有处理1个键+ 2个环。
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            #append a singleton cluster, corresponding to this particular atom 附加一个单例簇，对应于这个特定的原子
            singleton_cluster = [atom]
            cliques.append(singleton_cluster)

            #obtain the idx of this new "singleton cluster"
            c2 = len(cliques) - 1 #C2是把相交的原子独立添加为cluster的索引
            # create edges between this singleton cluster and all the clusters in which this atom is present
            # 在这个单元素团簇和该原子所在的所有团簇之间创建边
            for c1 in cnei:
                edges[(c1, c2)] = 1

        #number of rings > 2
        elif len(rings) > 2: #Multiple (n>2) complex rings
            # append a singleton cluster, corresponding to this particular atom
            # 附加一个单子集群，对应于这个特定的原子
            singleton_cluster = [atom]
            cliques.append(singleton_cluster)

            # obtain the idx of this new "singleton cluster"
            # 获得这个新的 "单子集群 "的idx
            c2 = len(cliques) - 1
            # create edges between this singleton cluster and all the clusters in which this atom is present
            # 在这个单子集群和这个原子所在的所有集群之间建立边。
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1

        #build edges between all the clusters that share at least one atom
        # 在所有至少共享一个原子的集群之间建立边。
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]

                    #find number of common atoms between the two clusters
                    # 在所有至少共享一个原子的集群之间建立边。
                    inter = set(cliques[c1]) & set(cliques[c2])
                    #assign weight equal to number of common atoms, to the edge between the two clusters
                    # 给两个簇之间的边分配等于共同原子数的权重
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter) #cnei[i] < cnei[j] by construction 根据结构，cnei[i]< cnei[j]。

    #to obtain a maximum spanning tree of any given graph, we subtract(减去) the edge weights by a large value. and then simply find a minimum spanning tree of this new graph with modified edge weights
    # 为了得到任何给定图形的最大生成树，我们用一个大值减去（减去）边缘权重，然后简单地找到这个新图形的最小生成树，并修改边缘权重
    edges = [u + (MST_MAX_WEIGHT-v,) for u, v in edges.items()]
    #print(edges)
    if len(edges) == 0:
        return cliques, edges

    #Compute Maximum Spanning Tree 计算最大生成树
    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))

    #obtain the junction-tree for this molecule 得到这个分子的结点树
    junc_tree = minimum_spanning_tree(clique_graph)

    #obtain a sparse representation of this junction-tree 得到这个结点树的稀疏表示
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    # finally return the list of clusters and the adjacency list for the corresponding junction tree
    # 最后返回集群列表和相应结点树的邻接列表。
    return (cliques, edges)

def atom_equal(a1, a2):
    """
        Description: This function, given two atoms, checks if they have the same symbol and the same formal charge.
        这个函数，给定两个原子，检查它们是否有相同的符号和相同的形式电荷。
        Args:
            a1: (object: rdkit)
                The first atom
            a2: (object: rdkit)
                The second atom
        Returns:
            Boolean:
                Whether the two atoms have the same symbol and formal charge.
        """
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

#Bond type not considered because all aromatic (so SINGLE matches DOUBLE) 不考虑键的类型，因为都是芳香族（所以SINGLE匹配DOUBLE）。
def ring_bond_equal(b1, b2, reverse=False):
    """
        Description: This function, given two ring bonds, checks if they are the same or not.
        这个函数，给定两个环状键，检查它们是否相同。
        * bond type not considered because all bonds are aromatic i.e. ring bonds
        不考虑键的类型，因为所有的键都是芳香族的，即环状键。
        Args:
            b1: (object: rdkit)
                The first bond.
            b2: (object: rdkit)
                The second bond.
            reverse: Boolean
                Whether b2 has be to checked in reverse for equality.
                b2是否要反过来检查是否相等。
        Returns:
             Boolean:
                Whether the bonds are same or not
        """
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])

def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
    """
        Description: This function, given the center / current molecular fragment, neighbor nodes and atom_map_dict, constructs and returns the molecular attachment configuration as encoded in the atom_map_dict
        这个函数，给定中心/当前分子片段、相邻节点和atom_map_dict，构建并返回在atom_map_dict中编码的分子附件配置。

        Args:
            ctr_mol: (object: rdkit)
                The center / current molecular fragment 中心/当前分子片段
            neighbors: List[MolJuncTreeNode]
                The list of neighbor nodes in the junction tree of that node, to which the center / current molecular fragment corresponds to.
                该节点的结点树中，中心/当前分子片段所对应的邻居节点的列表。
            prev_nodes: List[MolJuncTreeNode]
                The list of nodes, already used in the center / current molecular fragment.
            neighbor_atom_map_dict: Dict{int: Dict{int: int}}# {nei_id: {nei_atom: ctr_atom}}
                A dictionary mapped to each neighbor node. For each neighbor node, the mapped dictionary
                further maps the atom idx of atom in neighbor node's cluster to the atom idx of atom in center / current molecular fragment.
                节点列表，已经在中心/当前分子片段中使用。
                neighbor_atom_map_dict:Dict{int:Dict{int: int}}# {nei_id:{nei_atom: ctr_atom}}一个映射到每个邻居节点的字典。
                对于每个邻居节点，映射的字典进一步将邻居节点簇中原子的idx映射到中心/当前分子片段中原子的idx。

        Returns:
            ctr_mol: (object: rdkit)
                The molecule attachment configuration as specified.
                指定的分子附件配置。
        """
    # nids of nodes previously used in the center/current molecular fragment
    # 中心/当前分子片段中以前使用的节点的nids
    prev_nids = [node.nid for node in prev_nodes]

    for nei_node in prev_nodes + neighbors:
        # 。
        nei_id, nei_mol = nei_node.nid, nei_node.mol
        # obtain the atom_map corresponding to the atoms of this neighbor node's molecular fragment
        # 获得与该邻居节点的分子片段的原子对应的原子图
        amap = nei_amap[nei_id]# {nei_id: {nei_atom_idx: ctr_atom_idx}} --> #{nei_atom_idx: ctr_atom_idx}

        for atom in nei_mol.GetAtoms():
            # if the atoms neighbor node's molecular fragment are not already present in the center/current molecular fragment, then add them
            # 如果原子邻居节点的分子片段还没有出现在中心/当前分子片段中，那么就把它们加进去
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        # if the neighbor node corresponds to a "singleton-cluster"
        # 如果邻居节点对应的是 "单子群"，则为 "单子群"
        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        # if the neighbor node corresponds to either a "ring-cluster" or a "bond-cluster"
        # 如果邻居节点对应的是 "环形集群 "或 "粘合集群"
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids: #father node overrides 父节点重写
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol

def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    """
        Description: This function, given the center / current molecular fragment, the current atom_map and multiple neighbor nodes,
        returns the molecular attachment configuration as encoded in the given atom_map.

        Args:
            ctr_mol: (object: rdkit)
                The center / current molecular fragment

            neighbors: List[MolJuncTreeNode]
                The list of neighbor nodes in the junction tree of that node, to which the center / current molecular fragment
                corresponds to.

            prev_nodes: List[MolJuncTreeNode]
                The list of nodes, already used in the center / current molecular fragment.

            amap_list: List[Tuple(int, int, int)]
                The atom_map encoding information about how the clusters corresponding to the neighbor clusters
                are attached to the current / center molecular fragment.

        * An atom_map, constructed with respect to a center / curent cluster, is a list of tuples of the form
        (neighbor_node.nid, idx of atom in center / current molecule, idx of atom in neighbor node's molecular fragment)

        Returns:
            ctr_mol: (object: rdkit)
                The molecule attachment configuration as specified in the atom_map.
        """
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei.nid: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom# {nei_id: {nei_atom: ctr_atom}}

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()

#This version records idx mapping between ctr_mol and nei_mol
def enum_attach(ctr_mol, nei_node, amap, singletons):
    """
        Description: This function, given the center / current molecular fragment, the current atom_map and neighbor node,
        enumerates all possible attachment configurations of the current molecular fragment
        with the neighbor node's molecular fragment.

        * An atom_map, constructed with respect to a center / curent cluster, is a list of tuples of the form
        (neighbor_node.nid, idx of atom in center / current molecule, idx of atom in neighbor node's molecular fragment).
        （邻居节点的id, 当前节点分子片段中原子的idx, 邻居节点分子片段中原子的idx）

        Args:
            ctr_mol: (object: rdkit)
                The center / current molecular fragment

            neighbors: List[MolJuncTreeNode]
                The list of neighbor nodes in the junction tree of that node, to which the center / current molecular fragment
                corresponds to.

            prev_nodes: List[MolJuncTreeNode]
                The list of nodes, already used in the center / current molecular fragment.

            amap: List[Tuple(int, int, int)]
                The atom_map encoding information about how the clusters corresponding to the neighbor clusters
                are attached to the current / center molecular fragments.

            singletons: List[int]
                The list of atom_idx of those atoms, which correspond to singleton clusters.

        Returns:
            att_confs: List[List[Tuple(int, int, int)]] 其中Tuple(邻居节点的id, 当前节点分子片段中原子的idx, 邻居节点分子片段中原子的idx)
             The list of atom_maps corresponding to all possible attachment configurations.
        """
    # obtain the neighbor node's molecular fragment and node id
    nei_mol, nei_idx = nei_node.mol, nei_node.nid

    # list for storing all possible attachment configurations
    att_confs = []
    #将对应于singleton-clusters的原子排除在考虑范围之外
    black_list = [atom_idx for nei_id, atom_idx, _ in amap if nei_id in singletons]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]
    ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

    # if the neighbors node corresponds to a "singleton-cluster"
    if nei_mol.GetNumBonds() == 0: #neighbor singleton
        #obtain the only atom of this singleton cluster
        nei_atom = nei_mol.GetAtomWithIdx(0)

        # obtain the idx of all atoms that have already been used in the current/center molecular fragment
        used_list = [atom_idx for _, atom_idx, _ in amap]
        for atom in ctr_atoms:
            if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
                att_confs.append(new_amap)
   # if the neighbor node corresponds to a simple "bond cluster"
    elif nei_mol.GetNumBonds() == 1: #neighbor is a bond
        # obtain the only bond of the neighbor node's molecular fragment
        bond = nei_mol.GetBondWithIdx(0)

        # obtain the bond valence(键价)
        bond_val = int(bond.GetBondTypeAsDouble())
        # obtain the beginning and ending atoms of the bond
        b1, b2 = bond.GetBeginAtom(), bond.GetEndAtom()

        for atom in ctr_atoms: 
            #Optimize if atom is carbon (other atoms may change valence)
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                continue
            if atom_equal(atom, b1):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
                att_confs.append(new_amap)
            elif atom_equal(atom, b2):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
                att_confs.append(new_amap)

    # the neighbor node corresponds to a "ring molecular fragment"
    else: 
        #intersection is an atom
        # if only one atom is commen between the center/current molecular fragment and the neighbor molecular fragment
        for a1 in ctr_atoms:
            for a2 in nei_mol.GetAtoms():
                if atom_equal(a1, a2):
                    #Optimize if atom is carbon (other atoms may change valence)
                    if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                        continue
                    new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
                    att_confs.append(new_amap)

        #intersection is an bond
        # if the intersection beween the center/current molecular fragment and the neighbor molecular fragment is a bond
        if ctr_mol.GetNumBonds() > 1:
            for b1 in ctr_bonds:
                for b2 in nei_mol.GetBonds():
                    if ring_bond_equal(b1, b2):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetBeginAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetEndAtom().GetIdx())]
                        att_confs.append(new_amap)

                    if ring_bond_equal(b1, b2, reverse=True):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetEndAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetBeginAtom().GetIdx())]
                        att_confs.append(new_amap)
    return att_confs

#Try rings first: Speed-Up 
def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
    """
       Description: This function, given a node in the junction tree and its neighbor nodes,
       returns all the possible molecular attachment configurations
       of this node's cluster to with its neighbor nodes' clusters.
       这个函数，给定结点树中的一个节点和它的邻居节点，返回这个节点的簇与它的邻居节点的簇之间所有可能的分子附着配置。

       * atom_maps incorporate information about how the clusters corresponding to nodes in the "cluster-graph"
       are attached together to form a valid molecular fragment
       atom_maps包含了关于 "团簇图 "中的节点所对应的团簇如何连接在一起以形成一个有效的分子片段的信息。

       Arguments:
           node: (object: MolJuncTreeNode)
               the node in the junction tree, whose molecular attachment configurations have to be enumerated.
               结点树中的节点，其分子连接配置必须被列举出来。
           neighbors: List[MolJuncTreeNode]
               The neighbors to be considered for molecular attachment.
               要考虑的分子附着的邻居。
           prev_nodes: List[MolJuncTreeNode]
               The nodes already considered for molecular attachment.
               已经考虑用于分子连接的节点。
           prev_amap: List[Tuple(int, int, int)]
               the previous atom map encoding information about the molecular attachment configuration with previously used neighbors.
               前面的原子图编码了关于分子附件配置与先前使用的邻居的信息。

       Returns:
           all_attach_confs: List[Tuple(str, object: rdkit, List[Tuple(int, int, int)])]
               List of tuples of all possible valid attachment configurations, of the form (smiles, molecule, atom_map).
               所有可能的有效附件配置的元组列表，其形式为（微笑，分子，原子图）。
       """
    # list of all possible valid, molecular attachment configurations of qiven node with its neighbor nodes
    # qiven节点与其邻居节点的所有可能的有效分子连接配置的列表
    all_attach_confs = []
    # get those "cluser-nodes" from the "neighbor-nodes" list that are "singleton-clusters"
    # 从 "邻居节点 "列表中获取那些 "单子集群 "的 "cluser-节点"。
    singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.mol.GetNumAtoms() == 1]

    # search for all possible molecular attachment configurations of this node's cluster to its neighbor node' clusters
    # 搜索这个节点的簇到其邻居节点的簇的所有可能的分子连接配置
    def search(cur_amap, depth):
        #为了提高效率，而考虑molecular attachment configurations的数量上限
        if len(all_attach_confs) > MAX_NCAND:
            return

     # if all the neighbor nodes have considered for building a molecular attachment configuration, then append this attachment configuration to attachment configuration list
     # 如果所有的邻居节点都考虑建立一个分子附件配置，那么就把这个附件配置附加到附件配置列表中。
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        #要附加到当前分子片段的下一个邻居节点
        nei_node = neighbors[depth]

        # return the list of possible atom_maps that encode information about how the above neighbor node can be attached to the current molecular fragment
        # 返回可能的原子图列表，这些原子图编码了关于上述邻居节点如何连接到当前分子片段的信息。
        cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)

        #set for storing SMILES representations of candidate molecular fragment configurations
        # 用于存储候选分子片段配置的SMILES表示的集合
        cand_smiles = set()
        # list for storing candidate atom_maps which encode the possible ways in which the neighbor node can be attached to the current molecular fragment
        # 用于存储候选原子图的列表，这些候选原子图编码了邻居节点与当前分子片段相连的可能方式。
        candidates = []
        for amap in cand_amap:
            # obtain a candidate molecular fragment in which the above neighbor node cluster has been attached to the current molecular fragment
            # 获得一个候选分子片段，在这个候选分子片段中，上述邻居节点簇已被连接到当前分子片段中
            cand_mol = local_attach(node.mol, neighbors[:depth+1], prev_nodes, amap)
            # obtain a kekulized representation of this candidate molecular fragment
            # 获得这个候选分子片段的Kekulized表示。
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue

            # obtain the SMILES representation of this molecule
            # 获得该分子的SMILES表示法
            smiles = get_smiles(cand_mol)
            if smiles in cand_smiles:
                continue

            # add the candidate SMILES string to the list 将候选的SMILES字符串添加到列表中。
            cand_smiles.add(smiles)
            # add the candidate atom_map to the list 将候选原子图添加到列表中。
            candidates.append(amap)

        # if no more candidates atom_maps are available, i.e. no more valid chemical intermediaries are possible, then stop searching
        # 如果没有更多的候选原子图，即不可能有更多的有效化学中间体，那么就停止搜索。
        if len(candidates) == 0:
            return
        # for each of the candidate atom_maps, search for more candidate atom_maps using more neighbor nodes
        # 对于每个候选原子图，使用更多的邻居节点搜索更多的候选原子图。
        for new_amap in candidates:
            search(new_amap, depth + 1)

    #search for candidate atom_maps with the previous atom_map
    # 用前一个原子图搜索候选原子图
    search(prev_amap, 0)

    # set for storing SMILES representation of candidate molecular fragment configurations
    # 用于存储候选分子片段配置的SMILES表示的集合
    cand_smiles = set()
    # list to store tuples of (SMILES representation, candidate molecular fragment, atom_map)
    # 列表用于存储（SMILES表示法、候选分子片段、原子图）的图组。
    candidates = []
    aroma_score = []

    for amap in all_attach_confs:
        # obtain the candidate molecular fragment 获得候选的分子片段
        cand_mol = local_attach(node.mol, neighbors, prev_nodes, amap)
        cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))

        #obtain the SMILES representation of this molecule 获得该分子的SMILES表示法
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles or check_singleton(cand_mol, node, neighbors) == False:
            continue

        cand_smiles.add(smiles)
        candidates.append((smiles, amap))
        aroma_score.append(check_aroma(cand_mol, node, neighbors))

    return candidates, aroma_score 

def check_singleton(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() > 2]
    singletons = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() == 1]
    if len(singletons) > 0 or len(rings) == 0: return True

    n_leaf2_atoms = 0
    for atom in cand_mol.GetAtoms():
        nei_leaf_atoms = [a for a in atom.GetNeighbors() if not a.IsInRing()] #a.GetDegree() == 1]
        if len(nei_leaf_atoms) > 1: 
            n_leaf2_atoms += 1

    return n_leaf2_atoms == 0

def check_aroma(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() >= 3]
    if len(rings) < 2: return 0 #Only multi-ring system needs to be checked 只有多环系统需要检查

    get_nid = lambda x: 0 if x.is_leaf else x.nid
    benzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.benzynes] 
    penzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.penzynes] 
    if len(benzynes) + len(penzynes) == 0: 
        return 0 #No specific aromatic rings 没有特定的芳香族环

    n_aroma_atoms = 0
    for atom in cand_mol.GetAtoms():
        if atom.GetAtomMapNum() in benzynes+penzynes and atom.GetIsAromatic():
            n_aroma_atoms += 1

    if n_aroma_atoms >= len(benzynes) * 4 + len(penzynes) * 3:
        return 1000
    else:
        return -0.001 

#Only used for debugging purpose
def dfs_assemble(cur_mol, global_amap, fa_amap, cur_node, fa_node):
    fa_nid = fa_node.nid if fa_node is not None else -1
    prev_nodes = [fa_node] if fa_node is not None else []

    children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
    neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors

    cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
    cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)

    cand_smiles,cand_amap = list(zip(*cands))
    label_idx = cand_smiles.index(cur_node.label)
    label_amap = cand_amap[label_idx]

    for nei_id,ctr_atom,nei_atom in label_amap:
        if nei_id == fa_nid:
            continue
        global_amap[nei_id][nei_atom] = global_amap[cur_node.nid][ctr_atom]
    
    cur_mol = attach_mols(cur_mol, children, [], global_amap) #father is already attached
    for nei_node in children:
        if not nei_node.is_leaf:
            dfs_assemble(cur_mol, global_amap, label_amap, nei_node, cur_node)

if __name__ == "__main__":
    import sys
    from fast_jtnn.mol_tree import MolTree
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    smiles = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1","O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2", "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3", "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1", 'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br', 'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1', "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1"]

    def tree_test():
        for s in sys.stdin:
            s = s.split()[0]
            tree = MolTree(s)
            print('-------------------------------------------')
            print(s)
            for node in tree.nodes:
                print((node.smiles, [x.smiles for x in node.neighbors]))

    def decode_test():
        wrong = 0
        for tot,s in enumerate(sys.stdin):
            s = s.split()[0]
            tree = MolTree(s)
            tree.recover()

            cur_mol = copy_edit_mol(tree.nodes[0].mol)
            global_amap = [{}] + [{} for node in tree.nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

            dfs_assemble(cur_mol, global_amap, [], tree.nodes[0], None)

            cur_mol = cur_mol.GetMol()
            cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
            set_atommap(cur_mol)
            dec_smiles = Chem.MolToSmiles(cur_mol)

            gold_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(s))
            if gold_smiles != dec_smiles:
                print((gold_smiles, dec_smiles))
                wrong += 1
            print((wrong, tot + 1))

    def enum_test():
        for s in sys.stdin:
            s = s.split()[0]
            tree = MolTree(s)
            tree.recover()
            tree.assemble()
            for node in tree.nodes:
                if node.label not in node.cands:
                    print((tree.smiles))
                    print((node.smiles, [x.smiles for x in node.neighbors]))
                    print((node.label, len(node.cands)))

    def count():
        cnt,n = 0,0
        for s in sys.stdin:
            s = s.split()[0]
            tree = MolTree(s)
            tree.recover()
            tree.assemble()
            for node in tree.nodes:
                cnt += len(node.cands)
            n += len(tree.nodes)
            #print(cnt * 1.0 / n)
    
    count()
