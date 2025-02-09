from triton_mlir.dialects import tt, ttg, ttng, nvgpu, proton
from triton_mlir.ir import Attribute, Context


def test_smoke():
    assert "AddPtrOp" in dir(tt)
    assert "AsyncCommitGroupOp" in dir(ttg)
    assert "AsyncTMACopyGlobalToLocalOp" in dir(ttng)
    assert "ClusterArriveOp" in dir(nvgpu)
    assert "RecordOp" in dir(proton)

    lyout = "#ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>"
    with Context() as ctx:
        layout = Attribute.parse(lyout)
        assert str(layout) == lyout


if __name__ == "__main__":
    test_smoke()
