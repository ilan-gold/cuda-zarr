# cuda-zarr

[![PyPI](https://img.shields.io/pypi/v/cuda-zarr.svg)](https://pypi.org/project/cuda-zarr)
[![Downloads](https://static.pepy.tech/badge/cuda-zarr/month)](https://pepy.tech/project/cuda-zarr)
[![Downloads](https://static.pepy.tech/badge/cuda-zarr)](https://pepy.tech/project/cuda-zarr)
[![Stars](https://img.shields.io/github/stars/ilan-gold/cuda-zarr?style=flat&logo=github&color=yellow)](https://github.com/ilan-gold/cuda-zarr/stargazers)

A simple `zarr-python` [v3 compatible store](https://zarr.readthedocs.io/en/stable/user-guide/storage.html#developing-custom-stores) that uses the `CuFile` API: https://docs.rapids.ai/api/kvikio/stable/quickstart/
plus (at least one) codec(s).

## install

```shell
uv pip install cuda-zarr[cuda12]
```

## usage

Nvidia's documentation on how level/checksum are used in LZ4 (the only exported codec here) is quite sparse, but testing seems to show levels 1-22 all work. This codec only seems to work when used either roundtrip i.e., data is read and written using it, or only read. If you write data with this, it seems you can't read it back in with CPU data.

```python
from cuda_zarr import LZ4GPU, CuFileStore, RemoteCuFileStore
register_codec("zstd", LZ4GPU)
zarr.config.set({'codecs.zstd': f"{LZ4GPU.__module__}.{LZ4GPU.__name__}", "buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"})
store = CuFileStore('/path/to/store')
remote_store = RemoteCuFileStore.from_url("http://my_remote_data_server.com/path/to/the/store.zarr")
...
```

Untested in unit testing is the `RemoteCuFileStore` with s3 (although `http` is tested). Also `RemoteCuFileStore` only supports `get` and not `set` via `kvikio` (it will go through normal CPU based `fsspec` `io` in the `set` case).
