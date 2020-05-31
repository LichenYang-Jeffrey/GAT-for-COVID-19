import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np


def check_connectivity(adj):
    bfs_queue = [0]
    visited = set(bfs_queue)
    while len(bfs_queue) > 0:
        current = bfs_queue.pop(0)
        for nxt, has_edge in enumerate(adj[current] > 0):
            if (not has_edge) or (nxt in visited):
                continue
            visited.add(nxt)
            bfs_queue.append(nxt)
    return len(visited) == adj.size(0)


def zero_pad(x, size):
    x_size = np.array(x.size(), dtype=np.int)
    target_size = np.array(size, dtype=np.int)
    pad_size = target_size - x_size
    pad = []
    for s in pad_size:
        pad = [0, s] + pad
    return F.pad(x, pad=pad, mode='constant', value=0)


def collate_molecules(molecules):
    
    for mol in molecules:
        n = mol['atom_type'].size(0)

    # Get maximun sizes of each attribute.
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in molecules[0].items() if isinstance(val, torch.Tensor)
    }
    for mol in molecules[1:]:
        for prop, val in mol.items():
            if not isinstance(val, torch.Tensor):
                continue
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )

    # Pad tensors with zeros and make batch
    batch = {
        prop: [] for prop, val in molecules[0].items() if isinstance(val, torch.Tensor)
    }
    for mol in molecules:
        for prop, val in mol.items():
            if not isinstance(val, torch.Tensor):
                continue
            batch[prop].append(zero_pad(val, max_size[prop]).unsqueeze(0))

    for prop in batch:
        batch[prop] = torch.cat(batch[prop], dim=0)

    return batch


class MolecularDataLoader(DataLoader):
    """A wrapper for the standard PyTorch DataLoader
    """
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=collate_molecules,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def batchify_mol(mol):
    return collate_molecules([mol])


def debatchify_mol(batch):
    data = {k:v for k, v in batch.items()}
    return data

