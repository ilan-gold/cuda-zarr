from collections.abc import Iterable
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

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        chunks_and_specs_list = [cs for cs in chunks_and_specs]
        input_bytes = [
            chunk_bytes.as_array_like()
            for (chunk_bytes, _) in chunks_and_specs_list
            if chunk_bytes is not None
        ]
        iterable_decoded = iter(self._zstd_codec.decode_batch(input_bytes))
        return [
            chunk_spec.prototype.buffer.from_array_like(
                next(iterable_decoded).astype("int8")
            )
            if chunk_bytes is not None
            else None
            for (chunk_bytes, chunk_spec) in chunks_and_specs_list
        ]

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        chunks_and_specs_list = [cs for cs in chunks_and_specs]
        input_bytes = [
            chunk_bytes.as_array_like()
            for (chunk_bytes, _) in chunks_and_specs_list
            if chunk_bytes is not None
        ]
        iterable_encoded = iter(self._zstd_codec.encode_batch(input_bytes))
        return [
            chunk_spec.prototype.buffer.from_bytes(next(iterable_encoded))
            if chunk_bytes is not None
            else None
            for (chunk_bytes, chunk_spec) in chunks_and_specs_list
        ]
