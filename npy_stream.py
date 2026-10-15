"""Append-as-you-go .npy writer, for arrays too large to hold in memory.

WHY THIS EXISTS
---------------
``build_web_pack.py`` collected every page into a Python list and called
``np.concatenate`` at the end. That is fine at 600k pages (13 GB) and impossible
at the full corpus (~17.5M pages, ~373 GB) on a 220 GB machine.

The obvious fixes both cost more than they save: a counting pre-pass doubles a
five-hour build, and writing per-shard packs then concatenating needs twice the
disk transiently, which we do not have.

HOW IT WORKS
------------
A ``.npy`` file is a small header followed by raw C-order bytes. The header
carries the shape, so it is the only thing that depends on the final length.
This writer emits a header whose dict is padded with spaces to a fixed width,
appends rows as they arrive, then seeks back and rewrites the header with the
true shape -- the same byte-width, so nothing moves and no copy happens.

Only the trailing dimension has to be known up front, which it always is here
(1 for a flat array, ``levels`` for the path table).

    w = StreamNpy("tokens.npy", np.uint16)
    w.append(np.asarray([1, 2, 3], dtype=np.uint16))
    w.close()                      # patches the header; the file is now loadable
"""
from __future__ import annotations

import numpy as np

#: Header dict padded to this width before the terminating newline. 256 bytes is
#: far more than any shape string needs and keeps the data 64-byte aligned.
_HEADER_BODY = 256
_MAGIC = b"\x93NUMPY\x01\x00"


class StreamNpy:
    """Write a .npy incrementally. Shape is patched into the header on close."""

    def __init__(self, path: str, dtype, cols: int | None = None) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.cols = cols                  # None -> 1-D
        self.rows = 0
        self.f = open(path, "wb")
        self.f.write(_MAGIC)
        self.f.write(len(self._header(0)).to_bytes(2, "little"))
        self._header_at = self.f.tell()
        self.f.write(self._header(0))

    def _header(self, rows: int) -> bytes:
        shape = f"({rows},)" if self.cols is None else f"({rows}, {self.cols})"
        body = f"{{'descr': '{self.dtype.str}', 'fortran_order': False, 'shape': {shape}, }}"
        if len(body) + 1 > _HEADER_BODY:
            raise ValueError(f"header {body!r} exceeds the reserved {_HEADER_BODY} bytes")
        return body.ljust(_HEADER_BODY - 1).encode("latin1") + b"\n"

    def append(self, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=self.dtype)
        if self.cols is None:
            if a.ndim != 1:
                raise ValueError(f"expected a 1-D array, got shape {a.shape}")
            self.rows += a.shape[0]
        else:
            if a.ndim != 2 or a.shape[1] != self.cols:
                raise ValueError(f"expected (n, {self.cols}), got shape {a.shape}")
            self.rows += a.shape[0]
        self.f.write(a.tobytes())

    def close(self) -> None:
        if self.f.closed:
            return
        self.f.flush()
        self.f.seek(self._header_at)
        self.f.write(self._header(self.rows))
        self.f.close()

    def __enter__(self) -> "StreamNpy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
