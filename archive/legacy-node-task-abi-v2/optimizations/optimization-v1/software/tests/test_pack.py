from src.memory.pack import pack_exponents, unpack_exponents


def test_pack_unpack_exponents():
    exps = [1, -2, 3, -4]
    packed = pack_exponents(exps)
    unpacked = unpack_exponents(packed)
    assert unpacked == exps