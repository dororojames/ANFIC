"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""

# TODO some comments needed about [..., -1] == 0

import torch

__version__ = '1.2.0'

# torchac can be built with and without CUDA support.
# Here, we try to import both torchac_backend_gpu and torchac_backend_cpu.
# If both fail, an exception is thrown here already.
#
# The right version is then picked in the functions below.
#
# NOTE:
# Without a clean build, multiple versions might be installed. You may use python setup.py clean --all to prevent this.
# But it should not be an issue.


import_errors = []
dtype = torch.int16


try:
    import torchac_backend_gpu
    CUDA_SUPPORTED = True
except ImportError as e:
    CUDA_SUPPORTED = False
    import_errors.append(e)

try:
    import torchac_backend_cpu
    CPU_SUPPORTED = True
except ImportError as e:
    CPU_SUPPORTED = False
    import_errors.append(e)


imported_at_least_one = CUDA_SUPPORTED or CPU_SUPPORTED


# if import_errors:
#     import_errors_str = '\n'.join(map(str, import_errors))
#     print(f'*** Import errors:\n{import_errors_str}')


if not imported_at_least_one:
    raise ImportError('*** Failed to import any torchac_backend! Make sure to install torchac with torchac/setup.py. '
                      'See the README for details.')


any_backend = torchac_backend_gpu if CUDA_SUPPORTED else torchac_backend_cpu


def gen_uniform_pmf(size, L):
    assert size[0] == 1
    histo = torch.ones(L, dtype=torch.float32) / L
    assert (1 - histo.sum()).abs() < 1e-5, (1 - histo.sum()).abs()
    extendor = torch.ones(*size, 1)
    pmf = extendor * histo
    return pmf


def pmf2cdf(pmf, precision=16):
    """
    :param pmf: NL
    :return: N(L+1) as int16 on CPU!
    """
    cdf = pmf.cumsum(dim=-1, dtype=torch.float64).mul_(2 ** precision)
    cdf = cdf.clamp_max_(2**precision - 1).round_()
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1)
    return cdf.to('cpu', dtype=dtype, non_blocking=True)


def range_index_encode(symbols, cdf, cdf_length, indexes):
    """
    :param symbols: symbols to encode, N
    :param cdf: cdf to use, NL
    :param cdf_length: cdf_length to use, N
    :param indexes: index to use, N
    :return: symbols encode to strings
    """
    if symbols.dtype != dtype:
        raise TypeError(symbols.dtype)
    # print(indexes.shape, symbols.shape)
    # print(symbols.flatten()[830:838])
    # print(symbols.flatten()[0], indexes.flatten()[0], cdf, cdf_length)
    # file_name = './test_{}.pt'.format("hp" if cdf.size(0) != 64 else 'cb')
    # torch.save(dict(s=symbols.flatten(), i=indexes.flatten(), c=cdf, l=cdf_length), file_name)
    # return b''
    # if cdf.size(0) == 64:
    #     print(cdf[11, :10])
    stream, outbound_stream = any_backend.encode_cdf_index(
        cdf, cdf_length, indexes, symbols)
    # print(len(stream), len(outbound_stream))

    return stream + b'\x46\xE2\x84\x91' + outbound_stream


def range_index_decode(strings, cdf, cdf_length, indexes):
    """
    :param strings: strings encoded by range_encode
    :param cdf: cdf to use, NL
    :param cdf_length: cdf_length to use, N
    :param indexes: index to use, N
    :return: decoded matrix as torch.int16, N
    """
    input_strings, outbound_strings = strings.split(b'\x46\xE2\x84\x91')
    # print('decode')
    # print(len(input_strings), len(outbound_strings))

    ret = any_backend.decode_cdf_index(
        cdf, cdf_length, indexes, input_strings, outbound_strings)
    # print(ret.flatten()[830:838])

    # file_name = './test_{}_d.pt'.format("hp" if cdf.size(0) != 64 else 'cb')
    # torch.save(dict(s=ret.flatten(), i=indexes.flatten(), c=cdf, l=cdf_length), file_name)
    return ret
