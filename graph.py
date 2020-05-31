import torch
from torch.nn import Module


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
