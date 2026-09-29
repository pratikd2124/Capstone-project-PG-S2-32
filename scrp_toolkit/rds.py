"""Reading RYA's R data file (P.RData) without installing anything extra.

RYA's R simulation kept its copy of the unreliability matrices in P.RData. Walkthrough Step 3b
compares it with the JSON copy the model uses. The usual Python package for R files (rdata)
couldn't be installed on some of our machines, so this is a small reader of our own.

It understands just what P.RData contains: a list of number matrices with names, in R's
standard compressed binary format. Anything else gets a clear error rather than a wrong answer.
On P.RData it gives exactly the same result as the rdata package (checked when it was written).
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import struct
from pathlib import Path

import numpy as np

_NILVALUE, _REF, _GLOBALENV, _EMPTYENV, _MISSINGARG, _BASEENV = 254, 255, 253, 242, 251, 241
_NIL, _SYM, _LIST, _CHAR, _LGL, _INT, _REAL, _STR, _VEC = 0, 1, 2, 9, 10, 13, 14, 16, 19


def _decompress(raw: bytes) -> bytes:
    """R files are usually gzip-compressed; handle the other formats R can use too."""
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


class _Reader:
    """Walks through R's binary format one object at a time."""

    def __init__(self, data: bytes):
        self.b, self.i, self.refs = data, 0, []

    def int(self) -> int:
        v = struct.unpack_from(">i", self.b, self.i)[0]
        self.i += 4
        return v

    def raw(self, n: int) -> bytes:
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def header(self) -> None:
        """Check the file starts like a binary R file, and skip the version information."""
        fmt = self.raw(2)
        if fmt != b"X\n":
            raise ValueError(f"Only XDR (binary) R serialization is supported, got {fmt!r}.")
        version = self.int()
        self.int(), self.int()                       # R version writer / minimum reader
        if version == 3:
            self.raw(self.int())                     # native encoding, e.g. "UTF-8"
        elif version != 2:
            raise ValueError(f"Unsupported R serialization version {version}.")

    def item(self):
        """Read the next R object: its type is in the low 8 bits of a flags number."""
        flags = self.int()
        typ = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))

        if typ in (_NILVALUE, _GLOBALENV, _EMPTYENV, _MISSINGARG, _BASEENV):
            return None
        if typ == _REF:
            idx = flags >> 8
            return self.refs[(idx if idx else self.int()) - 1]
        if typ == _SYM:
            name = self.item()                        # a CHARSXP
            self.refs.append(name)
            return name
        if typ == _LIST:                              # pairlist, used for attributes
            out = {}
            while True:
                attr = self.item() if has_attr else None
                tag = self.item() if has_tag else None
                out[tag] = self.item()
                nxt = self.int()
                if nxt & 0xFF == _NILVALUE:
                    return out
                self.i -= 4
                flags = self.int()
                has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if typ == _CHAR:
            n = self.int()
            return None if n == -1 else self.raw(n).decode("utf-8")

        n = self.int()
        if typ == _REAL:
            value = np.frombuffer(self.raw(8 * n), dtype=">f8").astype(float)
        elif typ in (_INT, _LGL):
            value = np.frombuffer(self.raw(4 * n), dtype=">i4").astype(int)
        elif typ == _STR:
            value = [self.item() for _ in range(n)]
        elif typ == _VEC:
            value = [self.item() for _ in range(n)]
        else:
            raise ValueError(f"R object type {typ} is not supported by this minimal reader.")

        attrs = self.item() if has_attr else {}
        return _apply_attrs(value, attrs or {})


def _apply_attrs(value, attrs: dict):
    """Give a vector its shape (dim) or turn a list into a dict (names), as R would."""
    dim = attrs.get("dim")
    if dim is not None and isinstance(value, np.ndarray):
        value = value.reshape(tuple(int(d) for d in dim), order="F")    # R is column-major
    names = attrs.get("names")
    if names is not None and isinstance(value, list):
        return dict(zip(names, value))
    return value


def read_rds(path: str | Path):
    """Read one R object from an .rds / single-object .RData file."""
    r = _Reader(_decompress(Path(path).read_bytes()))
    r.header()
    return r.item()
