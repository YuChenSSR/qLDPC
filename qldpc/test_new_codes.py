#!/usr/bin/env python3
import galois
import numpy as np
import qldpc

DEFAULT_FIELD_ORDER = 2


def CordaroWagner(length: int, field: int | None = None) -> qldpc.codes.ClassicalCode:
    """Construct Cordaro Wagner Code of length 4, 5, 6."""
    field = field or DEFAULT_FIELD_ORDER
    gf = galois.GF(field)
    if length == 4:
        gen = gf(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]))
    if length == 5:
        gen = gf(np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]))
    if length == 6:
        gen = gf(np.array([[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1]]))
    return ~qldpc.codes.ClassicalCode(gen, field=field)


def RepSum(length: int, field: int | None = None) -> qldpc.codes.ClassicalCode:
    """Construct punctured Hammming Codes [6,3,3] Code."""
    field = field or DEFAULT_FIELD_ORDER
    gf = galois.GF(field)
    if length == 5:
        gen = gf(np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]))
    if length == 6:
        gen = gf(np.array([[1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1]]))
    return ~qldpc.codes.ClassicalCode(gen, field=field)


def modified_hamming(length: int, field: int | None = None) -> qldpc.codes.ClassicalCode:
    """Modified Hammming qldpc.codes."""
    code = qldpc.codes.HammingCode(3, field=field)
    if length == 4:
        return code.shorten(2, 3).puncture(4)
    if length == 5:
        return code.shorten(2, 3)
    if length == 6:
        return code.shorten(3)
    raise ValueError(f"Unrecognized length for modified Hamming code: {length}")


gen = modified_hamming(4).generator
gen[:, [1, 3]] = gen[:, [3, 1]]  # permute columns/bits
assert np.array_equal(CordaroWagner(4).generator, gen)

gen = modified_hamming(5).generator
gen[:, [2, 4]] = gen[:, [4, 2]]  # permute columns/bits
assert np.array_equal(CordaroWagner(5).generator, gen)
assert np.array_equal(RepSum(5).generator, gen)

# gen = qldpc.codes.CordaroWagnerCode(6).generator
# assert np.array_equal(CordaroWagner(6).generator, gen)

gen = modified_hamming(6).generator
gen[:, [3, 5]] = gen[:, [5, 3]]  # permute columns/bits
assert np.array_equal(RepSum(6).generator, gen)
