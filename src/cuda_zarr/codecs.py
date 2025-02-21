from functools import cached_property

import zarr
from kvikio.nvcomp_codec import NvCompBatchCodec


class ZstdGPU(zarr.codecs.ZstdCodec):
    """

    See https://zarr.readthedocs.io/en/stable/api/zarr/codecs/index.html#zarr.codecs.ZstdCodec

    Nvidia's documentation on how level/checksum are used is quite sparse, but testing seems to show levels 1-22 all work.
    This codec only seems to work when used either roundtrip i.e., data is read and written using it, or only read.
    If you write data with this, it seems you can't read it back in with CPU data.

    """

    @cached_property
    def _zstd_codec(self) -> NvCompBatchCodec:
        return NvCompBatchCodec("Zstd")
