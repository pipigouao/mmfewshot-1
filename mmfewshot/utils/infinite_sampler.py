import itertools

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler


class InfiniteSampler(Sampler):
    """Return a infinite stream of index.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self, dataset, seed=0, shuffle=True):
        self.dataset = dataset
        self.seed = seed if seed is not None else 0
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()
            else:
                yield from torch.arange(self.size).tolist()

    def _indices(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), 0, None)

    def __iter__(self):
        for idx in self.indices:
            yield idx

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError


class InfiniteGroupSampler(Sampler):
    """Similar to `InfiniteSampler` all indices in a batch should be in the
    same group.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU. Default: 1.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self, dataset, samples_per_gpu=1, seed=0, shuffle=True):
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.seed = seed if seed is not None else 0
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # buffer used to save indices of each group
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), 0, None)

    def __iter__(self):
        # once batch size is reached, yield the indices
        for idx in self.indices:
            flag = self.flag[idx]
            group_buffer = self.buffer_per_group[flag]
            group_buffer.append(idx)
            if len(group_buffer) == self.samples_per_gpu:
                for i in range(self.samples_per_gpu):
                    yield group_buffer[i]
                del group_buffer[:]

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError


class DistributedInfiniteSampler(Sampler):
    """Similar to `BatchSampler` warping a `DistributedSampler.

    It is designed for `IterationBased` runner. The implementation logic
    is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        num_replicas (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.seed = seed if seed is not None else 0
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                indices = []
                for _ in range(self.num_replicas):
                    indices += torch.randperm(self.size, generator=g).tolist()
                yield from indices
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.num_replicas)

    def __iter__(self):
        for idx in self.indices:
            yield idx

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError


class DistributedInfiniteGroupSampler(Sampler):
    """Similar to `InfiniteSampler` but in distributed version.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU. Default: 1.
        num_replicas (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.seed = seed if seed is not None else 0
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # buffer used to save indices of each group
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                indices = []
                for _ in range(self.num_replicas):
                    indices += torch.randperm(self.size, generator=g).tolist()
                yield from indices
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.num_replicas)

    def __iter__(self):
        # once batch size is reached, yield the indices
        for idx in self.indices:
            flag = self.flag[idx]
            group_buffer = self.buffer_per_group[flag]
            group_buffer.append(idx)
            if len(group_buffer) == self.samples_per_gpu:
                for i in range(self.samples_per_gpu):
                    yield group_buffer[i]
                del group_buffer[:]

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError