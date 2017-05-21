# -*- coding: utf-8 -*-
import numpy as np

HDF5_APPEND_MODE = 'a'


def append_batch_to_h5_dataset(hf, key, data, shape, **kwargs):
    for out in data:
        out = np.stack([out], axis=0)
        append_to_h5_dataset(hf, key, out, shape, **kwargs)


def append_to_h5_dataset(hf, key, data, shape, **kwargs):
    allowed_modes = [HDF5_APPEND_MODE, 'r+']
    if hf.mode not in allowed_modes:
        raise ValueError('H5 file mode should be "{}" but was "{}" instead.'.format(allowed_modes, hf.mode))

    maxshape = [None] + list(shape[1:])
    if key not in hf.keys():
        hf.create_dataset(
            key,
            data=data,
            maxshape=maxshape,
            compression='gzip',
            compression_opts=1,
            **kwargs
        )
    else:
        dset = hf[key]
        if data.shape[0] != 1:
            raise ValueError(('Expected shape[0] to be 1 (limitation of append implementation) but was found to be '
                              '{}. For bulk data, use append_batch_to_h5_dataset method.').format(data.shape))
        dset_size = len(dset)
        dset.resize([dset_size + 1] + list(shape[1:]))
        dset[dset_size] = data
