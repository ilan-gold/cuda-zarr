import cupy
from kvikio.utils import LocalHttpServer

from cuda_zarr import RemoteCuFileStore


async def test_url(tmp_path):
    a = cupy.arange(100, dtype="int8")
    a.tofile(tmp_path / "myfile")

    # Start a local server that serves files in `tmpdir`, Open remote file from a http url
    with (
        LocalHttpServer(root_path=tmp_path) as server,
        RemoteCuFileStore.from_url(server.url) as f,
    ):
        assert (
            a.get() == (await f.get("myfile", prototype=None)).as_numpy_array()
        ).all()
