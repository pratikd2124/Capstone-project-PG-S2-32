import gzip
import struct

import numpy as np

from scrp_toolkit.rds import read_rds


def _i(v):
    return struct.pack(">i", v)


def _char(s):
    return _i(9) + _i(len(s)) + s.encode()


def test_reads_named_list_of_matrices_like_rya_p_rdata(tmp_path):
    """Hand-built R serialization of list(a = matrix(c(1, 2, 3, 4), 2)), R >= 3.5 format."""
    body = b"X\n" + _i(3) + _i(0x40300) + _i(0x30500) + _i(5) + b"UTF-8"
    body += _i(19 | 1 << 9) + _i(1)                                   # list of length 1, has attributes
    body += _i(14 | 1 << 9) + _i(4) + struct.pack(">4d", 1, 2, 3, 4)  # numeric, has attributes
    body += _i(2 | 1 << 10) + _i(1) + _char("dim") + _i(13) + _i(2) + _i(2) + _i(2) + _i(254)
    body += _i(2 | 1 << 10) + _i(1) + _char("names") + _i(16) + _i(1) + _char("a") + _i(254)
    f = tmp_path / "P.RData"
    f.write_bytes(gzip.compress(body))

    out = read_rds(f)
    assert list(out) == ["a"]
    assert np.array_equal(out["a"], np.array([[1.0, 3.0], [2.0, 4.0]]))   # R fills column by column
