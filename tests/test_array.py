from __future__ import annotations

import numpy as np
import pytest
import zarr
import zarr.codecs
from zarr.registry import register_codec

from cuda_zarr import CuFileStore, ZstdGPU

register_codec("zstd", ZstdGPU)


@pytest.fixture(params=list(range(22)))
def level(request):
    return request.param


@pytest.fixture
def memory_array():
    return np.arange(100).reshape((20, 5))


@pytest.fixture(params=[True, False])
def checksum(request):
    return request.param


@pytest.fixture
def array_path(tmp_path, level, memory_array, checksum):
    with zarr.config.set(
        {
            "buffer": "zarr.core.buffer.cpu.Buffer",
            "ndbuffer": "zarr.core.buffer.cpu.NDBuffer",
        }
    ):
        z = zarr.create_array(
            tmp_path,
            shape=memory_array.shape,
            dtype=memory_array.dtype,
            compressors=[zarr.codecs.ZstdCodec(level=level, checksum=checksum)],
        )
        z[...] = memory_array
        return tmp_path


@pytest.fixture
def gpu_array_path(tmp_path, level, memory_array, checksum):
    z = zarr.create_array(
        tmp_path,
        shape=memory_array.shape,
        dtype=memory_array.dtype,
        compressors=[ZstdGPU(level=level, checksum=checksum)],
    )
    z[...] = memory_array
    return tmp_path


@pytest.fixture
def uncompressed_array_path(tmp_path, memory_array):
    z = zarr.create_array(
        tmp_path,
        shape=memory_array.shape,
        dtype=memory_array.dtype,
        compressors=None,
    )
    z[...] = memory_array
    return tmp_path


def test_roundtrip_from_cpu_with_gpu(array_path, memory_array):
    with zarr.config.set(
        {
            "codecs.zstd": f"{ZstdGPU.__module__}.{ZstdGPU.__name__}",
            "buffer": "zarr.core.buffer.gpu.Buffer",
            "ndbuffer": "zarr.core.buffer.gpu.NDBuffer",
        }
    ):
        z = zarr.open_array(CuFileStore(array_path))
        assert (z[...].get() == memory_array).all()


def test_roundtrip_from_gpu_with_gpu(gpu_array_path, memory_array):
    with zarr.config.set(
        {
            "codecs.zstd": f"{ZstdGPU.__module__}.{ZstdGPU.__name__}",
            "buffer": "zarr.core.buffer.gpu.Buffer",
            "ndbuffer": "zarr.core.buffer.gpu.NDBuffer",
        }
    ):
        z = zarr.open_array(CuFileStore(gpu_array_path))
        assert (z[...].get() == memory_array).all()


def test_roundtrip_uncompressed(uncompressed_array_path, memory_array):
    z = zarr.open_array(CuFileStore(uncompressed_array_path))
    assert (z[...].get() == memory_array).all()


@pytest.mark.xfail
def test_roundtrip_from_gpu_with_cpu(gpu_array_path, memory_array):
    with zarr.config.set(
        {
            "codecs.zstd": "zarr.codecs.zstd.ZstdCodec",
            "buffer": "zarr.core.buffer.cpu.Buffer",
            "ndbuffer": "zarr.core.buffer.cpu.NDBuffer",
        }
    ):
        z = zarr.open_array(gpu_array_path)
        assert (z[...].get() == memory_array).all()
