"""Minimal, dependency-free reader for R's serialized .rds / single-object .RData files.

Written for RYA's ``P.RData`` (the unreliability matrices used by their R simulation): a named
list of numeric matrices, gzip-compressed, R serialization version 2 or 3. It supports only
what such files contain -- lists, numeric/integer/logical/character vectors, attributes (names,
dim), symbols and references -- and raises a clear error on anything else.

This replaces the optional ``rdata`` package, which some networks block pip from installing.
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
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


class _Reader:
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
