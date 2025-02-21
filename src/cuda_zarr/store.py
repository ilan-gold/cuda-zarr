import asyncio
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import cupy as cp
import zarr
from kvikio import CuFile, RemoteFile
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import BufferPrototype
from zarr.core.buffer.gpu import Buffer
from zarr.storage import FsspecStore, LocalStore


def _get(
    path: Path | str,
    protocol: Literal["http", "file", "s3"],
    byte_range: ByteRequest | None,
) -> Buffer:
    if protocol == "file" and not path.exists():
        raise FileNotFoundError()
    with (
        CuFile(path)
        if protocol == "file"
        else getattr(
            RemoteFile, f"open_{protocol}{'_url' if protocol == "s3" else ''}"
        )(path) as handle
    ):
        nbytes = path.stat().st_size if protocol == "file" else handle.nbytes()
        if byte_range is None:
            size = nbytes
            b = cp.empty((size,), dtype="int8")
            handle.pread(b).get()
            return Buffer.from_array_like(b)
        if isinstance(byte_range, RangeByteRequest):
            if byte_range.end - byte_range.start < 0 or nbytes - byte_range.start < 0:
                return Buffer.create_zero_length()
            size = min(byte_range.end - byte_range.start, nbytes - byte_range.start)
            b = cp.empty((size,), dtype="int8")
            handle.pread(b, size=size, file_offset=byte_range.start).get()
            return Buffer.from_array_like(b)
        elif isinstance(byte_range, OffsetByteRequest):
            if nbytes - byte_range.offset < 0:
                return Buffer.create_zero_length()
            size = nbytes - byte_range.offset
            b = cp.empty((size,), dtype="int8")
            handle.pread(b, size=size, file_offset=byte_range.offset).get()
            return Buffer.from_array_like(b)
        elif isinstance(byte_range, SuffixByteRequest):
            if nbytes - byte_range.suffix < 0:
                return Buffer.create_zero_length()
            size = byte_range.suffix
            b = cp.empty((size,), dtype="int8")
            handle.pread(b, size=size, file_offset=nbytes - size).get()
            return Buffer.from_array_like(b)
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}.")


def _put(
    path: Path,
    value: Buffer,
    start: int | None = None,
) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with CuFile(path, flags="a" if path.exists() else "w") as handle:
        file_offset = start if start is not None else 0
        handle.pwrite(value.as_array_like(), file_offset=file_offset).get()
        return None


class CuFileStore(LocalStore):
    """See https://zarr.readthedocs.io/en/stable/api/zarr/abc/store/index.html#zarr.abc.store.Store"""

    def __init__(self, root: Path | str, *, read_only=False):
        super().__init__(root, read_only=read_only)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """See https://zarr.readthedocs.io/en/stable/api/zarr/abc/store/index.html#zarr.abc.store.Store.get"""
        default_buffer = zarr.config.get("buffer", None)
        default_ndbuffer = zarr.config.get("ndbuffer", None)
        if not (
            default_buffer == "zarr.core.buffer.gpu.Buffer"
            and default_ndbuffer == "zarr.core.buffer.gpu.NDBuffer"
        ):
            msg = f"Can only use this store with a GPU buffer, buffer config {default_buffer} or ndbuffer {default_ndbuffer}"
            raise ValueError(msg)
        if not self._is_open:
            await self._open()

        try:
            return await asyncio.to_thread(
                _get, self.root / key, protocol="file", byte_range=byte_range
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def set(self, key: str, value: Buffer) -> None:
        """See https://zarr.readthedocs.io/en/stable/api/zarr/abc/store/index.html#zarr.abc.store.Store.set"""
        if not self._is_open:
            await self._open()
        self._check_writable()

        if not isinstance(value, Buffer):
            raise TypeError(
                f"LocalStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )
        path = self.root / key
        await asyncio.to_thread(_put, path, value, start=None)


def _dereference_path(root: str, path: str) -> str:
    assert isinstance(root, str)
    assert isinstance(path, str)
    root = root.rstrip("/")
    path = f"{root}/{path}" if root else path
    return path.rstrip("/")


class RemoteCuFileStore(FsspecStore):
    """See https://zarr.readthedocs.io/en/stable/api/zarr/abc/store/index.html#zarr.abc.store.Store"""

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """See https://zarr.readthedocs.io/en/stable/api/zarr/abc/store/index.html#zarr.abc.store.Store.get"""
        default_buffer = zarr.config.get("buffer", None)
        default_ndbuffer = zarr.config.get("ndbuffer", None)
        if not (
            default_buffer == "zarr.core.buffer.gpu.Buffer"
            and default_ndbuffer == "zarr.core.buffer.gpu.NDBuffer"
        ):
            msg = f"Can only use this store with a GPU buffer, buffer config {default_buffer} or ndbuffer {default_ndbuffer}"
            raise ValueError(msg)
        if not self._is_open:
            await self._open()
        path = _dereference_path(self.path, key)
        has_s3_fs = find_spec("s3fs")
        if has_s3_fs:
            import s3fs

            if isinstance(self.fs, s3fs.core.S3FileSystem):
                path = "s3://" + path
        if path.startswith("http"):
            protocol = "http"
        elif path.startswith("s3"):
            if has_s3_fs:
                protocol = "s3"
            else:
                raise ValueError("Install `s3fs` if you wish to use `s3` as a prefix")
        else:
            raise ValueError(f"Bad protocol {path}")
        try:
            return await asyncio.to_thread(
                _get, path, protocol=protocol, byte_range=byte_range
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None
