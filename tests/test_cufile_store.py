from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import zarr
import zarr.codecs
from zarr.core.buffer.gpu import Buffer
from zarr.testing.store import StoreTests
from zarr.testing.utils import assert_bytes_equal

from cuda_zarr import CuFileStore

if TYPE_CHECKING:
    import pathlib


class TestCuFileStore(StoreTests[CuFileStore, Buffer]):
    store_cls = CuFileStore
    buffer_cls = Buffer

    async def get(self, store: CuFileStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store.root / key).read_bytes())

    async def set(self, store: CuFileStore, key: str, value: Buffer) -> None:
        parent = (store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir) -> dict[str, str]:
        return {"root": str(tmpdir)}

    def test_creates_new_directory(self, tmp_path: pathlib.Path):
        target = tmp_path.joinpath("a", "b", "c")
        assert not target.exists()

        store = self.store_cls(root=target)
        zarr.group(store=store)

    def test_invalid_root_raises(self):
        """
        Test that a TypeError is raised when a non-str/Path type is used for the `root` argument
        """
        with pytest.raises(
            TypeError,
            match=r"'root' must be a string or Path instance. Got an instance of <class 'int'> instead.",
        ):
            CuFileStore(root=0)

    async def test_get_with_prototype_default(self, store: CuFileStore):
        """
        Ensure that data can be read via ``store.get`` if the prototype keyword argument is unspecified, i.e. set to ``None``.
        """
        data_buf = self.buffer_cls.from_bytes(b"\x01\x02\x03\x04")
        key = "c/0"
        await self.set(store, key, data_buf)
        observed = await store.get(key, prototype=None)
        assert_bytes_equal(observed, data_buf)

    async def test_get_with_cpu_prototype(self, store: CuFileStore):
        """
        Ensure that data can be read via ``store.get`` if the prototype keyword argument is unspecified, i.e. set to ``None``.
        """
        zarr.config.set({"buffer": "zarr.core.buffer.cpu.Buffer"})
        with pytest.raises(
            ValueError, match=r"Can only use this store with a GPU buffer"
        ):
            await store.get("c/0", prototype=None)
