import asyncio
from functools import cached_property

import zarr
from kvikio.nvcomp_codec import NvCompBatchCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer


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

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        b = await asyncio.to_thread(
            self._zstd_codec.decode, chunk_bytes.as_array_like()
        )
        return chunk_spec.prototype.buffer.from_array_like(b.ravel().astype("int8"))

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        b = await asyncio.to_thread(
            self._zstd_codec.encode, chunk_bytes.as_array_like()
        )
        return chunk_spec.prototype.buffer.from_bytes(b)
