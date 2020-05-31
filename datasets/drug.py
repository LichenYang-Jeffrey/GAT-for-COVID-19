import os
import torch
from torch.utils.data import Dataset
from torch.nn import Module


import csv
import networkx
import pysmiles
from tqdm.auto import tqdm

import numpy as np
import scipy.sparse as sp

from utils.data import MolecularDataLoader


def convert_networkx(graph, types):
    atoms = graph.nodes(data='element')
    bonds = graph.adj
    n = len(atoms)

    atom_mask = torch.ones([n])
    atom_type = torch.zeros([n, len(types)])
    bond_mask = torch.zeros([n, n])
    bond_type = torch.zeros([n, n, 5])  # 0, 1, 2, 3, 4

    for idx, t in atoms:
        type_number = types[t]
        atom_type[idx, type_number] = 1
    
    for row, nbh_list in bonds.items():
        for col, attr in nbh_list.items():
            bond_mask[row, col] = 1
            if attr['order'] == 1.5:
                bond_type[row, col, 4] = 1
            else:
                bond_type[row, col, int(attr['order'])]    # 0, 1, 2, 3

    return {
        'atom_mask': atom_mask,
        'atom_type': atom_type,
        'bond_mask': bond_mask,
        'bond_type': bond_type
    }


class DrugDataset(Dataset):

    ATOM_TYPES = {
        'Ag': 0,
        'Al': 1,
        'As': 2,
        'B': 3,
        'Bi': 4,
        'Br': 5,
        'C': 6,
        'Ca': 7,
        'Cl': 8,
        'Co': 9,
        'F': 10,
        'Fe': 11,
        'Gd': 12,
        'H': 13,
        'Hg': 14,
        'I': 15,
        'K': 16,
        'Li': 17,
        'Mg': 18,
        'N': 19,
        'Na': 20,
        'Nd': 21,
        'O': 22,
        'P': 23,
        'Pb': 24,
        'Pt': 25,
        'S': 26,
        'Sb': 27,
        'Se': 28,
        'Si': 29,
        'Zn': 30,
    }

    def __init__(self, csv_file, force_reload=False):
        super().__init__()
        self.csv_file = csv_file
        self.processed_file = csv_file + '.pt'
        self.load(force_reload)

    def load(self, force_reload):
        if force_reload or not os.path.exists(self.processed_file):
            self.dataset = self.process()
        else:
            self.dataset = torch.load(self.processed_file)

    def process(self):
        bond_order = set()
        atom_types = set()
        processed = []
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)  # Ignore header
            for item in tqdm(reader):
                mol_graph = pysmiles.read_smiles(item[0], explicit_hydrogen=False)
                data = convert_networkx(mol_graph, self.ATOM_TYPES)
                data['simles'] = item[0]
                data['label'] = torch.LongTensor([int(item[1])])
                processed.append(data)
        torch.save(processed, self.processed_file)
        return processed

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_pairwise_difference(x, mask=None):
    """Compute pairwise difference vector.
    Args:
        x:    (B, N_max, F)
        mask: (B, N_max, N_max)
    Returns:
        diff: (B, N_max, N_max, F), where diff[b, i, j] = x_j - x_i
    """
    B, N_max, F = x.size()
    x_rep_j = x.unsqueeze(1).expand(B, N_max, N_max, F)
    x_rep_i = x.unsqueeze(2).expand(B, N_max, N_max, F)
    diff = x_rep_j - x_rep_i
    if mask is not None:
        diff = diff * mask.unsqueeze(-1)
    return diff


def get_pairwise_unit_direction(x, mask=None, eps=1e-6, use_random_for_zero=True):
    diff = get_pairwise_difference(x, mask=mask)    # (B, N_max, N_max, 3)
    diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=False)    # (B, N_max, N_max)
    zero_norm = torch.lt(diff_norm, eps)        # (B, N_max, N_max), BoolTensor

    if use_random_for_zero:
        rand_diff = torch.randn_like(diff)    # (B, N_max, N_max, 3)
        rand_dir = rand_diff / torch.norm(rand_diff, p=2, dim=-1, keepdim=True)
        rand_dir = rand_dir * zero_norm.unsqueeze(-1)

    # Set zero norm to one to avoid `inf`
    diff_norm = torch.where(zero_norm, torch.ones_like(diff_norm), diff_norm)

    unit_diff = diff / diff_norm.unsqueeze(-1)
    unit_diff = unit_diff * (~zero_norm).unsqueeze(-1)

    if use_random_for_zero:
        unit_diff = unit_diff + rand_dir

    return unit_diff


def get_pairwise_addition(x, mask=None):
    """
    Args:
        x:    (B, N_max, F)
        mask: (B, N_max, N_max)
    Returns:
        sun:  (B, N_max, N_max, F)
    """
    B, N_max, F = x.size()
    x_rep_j = x.unsqueeze(1).expand(B, N_max, N_max, F)
    x_rep_i = x.unsqueeze(2).expand(B, N_max, N_max, F)
    return x_rep_j + x_rep_i


def get_pairwise_distance(pos, mask=None):
    """Compute pairwise L2 distance.
    Args:
        pos:  (B, N_max, 3)
        mask: (B, N_max, N_max)
    Returns:
        dist: (B, N_max, N_max)
    """
    return torch.sqrt(get_pairwise_squared_distance(pos, mask=mask))


def get_pairwise_squared_distance(pos, mask=None):
    """Compute pairwise squared-L2 distance.
    Args:
        pos:  (B, N_max, 3)
        mask: (B, N_max, N_max)
    Returns:
        dist: (B, N_max, N_max)
    """
    dist = (get_pairwise_difference(pos)**2).sum(dim=-1)
    dist[dist != dist] = 0 # replace nan values with 0
    if mask is not None:
      dist = dist * mask
    return dist


def get_pairwise_mask(node_mask):
    """
    Args:
        node_mask: (B, N_max)
    Returns:
        pairwise_mask: (B, N_max, N_max)
    """
    n = node_mask.size(1)
    return node_mask.unsqueeze(1).repeat(1, n, 1) * node_mask.unsqueeze(2).repeat(1, 1, n)


def get_k_hop_pairwise_mask(adj, k, exclude_self=True):
    """
    Args:
        adj:  (B, N_max, N_max), Adjacency matrices.
    Returns:
        (B, N_max, N_max)
    """
    B, N, _ = adj.size()

    adj = adj.float()
    eye = torch.eye(N, device=adj.device).unsqueeze(0).expand_as(adj)

    k_hop_adj = adj + eye
    for _ in range(k-1):
        k_hop_adj = torch.bmm(k_hop_adj, adj) + eye

    k_hop_adj = torch.where(k_hop_adj > 0, torch.ones_like(k_hop_adj), torch.zeros_like(k_hop_adj))

    if exclude_self:
        mask = 1 - eye
        k_hop_adj = k_hop_adj * mask

    return k_hop_adj.long()


def remove_self_loops(adj):
    """
    Args:
        adj:  (B, N_max, N_max)
    Returns:
        (B, N_max, N_max)
    """
    B, N, _ = adj.size()
    eye = torch.eye(N).to(adj).unsqueeze(0).expand_as(adj)
    mask = 1 - eye
    return adj * mask


def aggregate(x, dim, aggr='add', mask=None, keepdim=False):
    """
    Args:
        x:    (..., A, ..., F), Features to be aggregated.
        mask: (..., A, ...)
    Returns:
        (...,  , ..., F), if keepdim == False
        (..., 1, ..., F), if keepdim == True
    """
    assert aggr in ('add', 'mean')

    if mask is not None:
        x = x * mask.unsqueeze(-1)

    y = torch.sum(x, dim=dim, keepdim=keepdim)
    if aggr == 'mean':
        if mask is not None:
            n = torch.sum(mask, dim=dim, keepdim=keepdim)
            n = torch.max(n, other=torch.ones_like(n))  # Avoid division by zero
        else:
            n = x.size(dim)
        y = y / n

    return y


class Aggregate(Module):

    def __init__(self, aggr='add'):
        super().__init__()
        assert aggr in ('add', 'mean')
        self.aggr = aggr

    def forward(self, x, dim, mask=None, keepdim=False):
        return aggregate(x, dim=dim, aggr=self.aggr, mask=mask, keepdim=keepdim)


def readout(x, mask, aggr='add'):
    """
    Args:
        x:    (B, N_max, F)
        mask: (B, N_max)
    Returns:
        (B, F)
    """
    return aggregate(x=x, dim=1, aggr=aggr, mask=mask, keepdim=False)


class Readout(Module):

    def __init__(self, aggr='add'):
        super().__init__()
        assert aggr in ('add', 'mean')
        self.aggr = aggr

    def forward(self, x, mask):
        return readout(x, mask=mask, aggr=self.aggr)


def gather_nodes(x, index):
    """
    Args:
        x:      (B, N, F) or (B, N)
        index:  (B, m)
    Returns:
        (B, m, F)
    """
    B, m = index.size()
    if len(x.size()) == 3:
        _, N, F = x.size()
        index = index.unsqueeze(-1).expand(B, m, F)
    elif len(x.size()) == 2:
        pass
    else:
        assert False, 'The size of `x` must either be (B, N, F) or (B, N).'

    return torch.gather(x, dim=1, index=index)

if __name__ == '__main__':
    dset = DrugDataset('../data_covid/ecoli.csv')
    dataloader = MolecularDataLoader(dataset=dset)
    idx_features_labels = []
    label_max = 0
    for idx, data in enumerate(dataloader):
        line = np.array([])
        amask = data['atom_mask']
        atype = data['atom_type']
        bmask = data['bond_mask']
        btype = data['bond_type']
        label = data['label'].item()
        x = readout(atype, amask)
        print(atype.size())
        print(bmask.size())

    # dset = DrugDataset('./data_covid/AID1706_binarized_sars.csv')
    """
    {0, 1, 2, 3, 1.5}
    {'P', 'Sb', 'Cl', 'Ca', 'As', 'Si', 'Na', 'Se', 'S', 'Bi', 'Li', 'C', 'N', 'Co', 'Pt', 'I', 'Zn', 'Al', 'Fe', 'K', 'Gd', 'F', 'Hg', 'Br', 'O', 'Pb'}
    290726it [05:50, 828.82it/s] 
    {0, 1, 2, 3, 1.5}
    {'P', 'Cl', 'Ca', 'As', 'Si', 'Na', 'Ag', 'Se', 'S', 'B', 'Li', 'C', 'N', 'Nd', 'Pt', 'I', 'Zn', 'H', 'Al', 'K', 'F', 'Br', 'Mg', 'O'}
    """
