"""StreamNpy must produce files np.load reads back exactly."""
import numpy as np
import pytest

from npy_stream import StreamNpy


def test_flat_roundtrip(tmp_path):
    p = str(tmp_path / "a.npy")
    want = np.arange(1000, dtype=np.uint16)
    w = StreamNpy(p, np.uint16)
    for i in range(0, 1000, 37):
        w.append(want[i:i + 37])
    w.close()
    assert np.array_equal(np.load(p), want)
    assert np.load(p, mmap_mode="r").shape == (1000,)


def test_two_dim_roundtrip(tmp_path):
    p = str(tmp_path / "b.npy")
    want = np.arange(24 * 500, dtype=np.int16).reshape(500, 24)
    with StreamNpy(p, np.int16, cols=24) as w:
        for i in range(0, 500, 13):
            w.append(want[i:i + 13])
    got = np.load(p)
    assert got.shape == (500, 24) and np.array_equal(got, want)


def test_empty_is_valid(tmp_path):
    p = str(tmp_path / "c.npy")
    StreamNpy(p, np.int64).close()
    assert np.load(p).shape == (0,)


def test_header_width_is_stable_across_magnitudes(tmp_path):
    # The header is rewritten in place, so its byte width must not depend on the
    # number of digits in the shape -- otherwise the patch would shift the data.
    w = StreamNpy(str(tmp_path / "d.npy"), np.uint16)
    assert len(w._header(0)) == len(w._header(17_500_000_000))
    w.close()


def test_rejects_wrong_shape(tmp_path):
    with StreamNpy(str(tmp_path / "e.npy"), np.int16, cols=24) as w:
        with pytest.raises(ValueError):
            w.append(np.zeros((3, 23), dtype=np.int16))
