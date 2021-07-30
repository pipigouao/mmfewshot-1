# Copyright (c) Open-MMLab. All rights reserved.
import json
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel.data_container import DataContainer
from torch.utils.data.dataloader import default_collate


def query_support_collate_fn(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    This is mainly used in query_support dataloader. The main
    difference with the :func:`collate_fn`  in mmcv is it
    can process list[list[DataContainer]].

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes

    Args:
        batch (list[list[:obj:`mmcv.parallel.DataContainer`]] |
            list[:obj:`mmcv.parallel.DataContainer`]): Data of
            single batch.
        samples_per_gpu (int): The number of samples of single GPU.
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    # This is usually a case in query_support dataloader, which
    # the :func:`__getitem__` of dataset return more than one images.
    # Here we process the support batch data in type of
    # List: [ List: [ DataContainer]]
    if isinstance(batch[0], Sequence):
        samples_per_gpu = len(batch[0]) * samples_per_gpu
        batch = sum(batch, [])

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [
            query_support_collate_fn(samples, samples_per_gpu)
            for samples in transposed
        ]
    elif isinstance(batch[0], Mapping):
        return {
            key: query_support_collate_fn([d[key] for d in batch],
                                          samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


class NumpyEncoder(json.JSONEncoder):
    """Save numpy array obj to json."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
