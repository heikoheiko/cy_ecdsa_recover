# -*- coding: utf-8 -*-
import random
random.seed(12312421412)
from bitcoin import privtopub, encode_pubkey, ecdsa_raw_sign, ecdsa_raw_recover
import time
# import pyximport
# pyximport.install()
#from ecrecover import ecdsa_sign, ecdsa_verify, ecdsa_recover, test_big
from ecdsa_recover import ecdsa_raw_recover as c_ecdsa_raw_recover
from ecdsa_recover import ecdsa_raw_sign as c_ecdsa_raw_sign

priv = ''.join(chr(random.randint(0, 255)) for i in range(32))
pub = privtopub(priv)
msg = ''.join(chr(random.randint(0, 255)) for i in range(32))
vrs = ecdsa_raw_sign(msg, priv)
assert vrs == c_ecdsa_raw_sign(msg, priv)
p = ecdsa_raw_recover(msg, vrs)
p2 = c_ecdsa_raw_recover(msg, vrs)
assert p == p2
assert encode_pubkey(p, 'bin') == pub


def test_ecrecover(rounds=100):
    st = time.time()
    for i in range(rounds):
        p = ecdsa_raw_recover(msg, vrs)
    elapsed = time.time() - st
    print 'py took: %.2fsecs / %dμs per op / %d recoveries per sec' % \
        (elapsed, elapsed / rounds * 10**6, rounds / elapsed)


def test_cecrecover(rounds=100):
    st = time.time()
    for i in range(rounds):
        p = c_ecdsa_raw_recover(msg, vrs)
    elapsed = time.time() - st
    print 'cy took: %.2fsecs / %dμs per op  / %d recoveries per sec' % \
        (elapsed, elapsed / rounds * 10**6, rounds / elapsed)
    print 'c  takes: 300μs per op / 3000 recoveries per sec'  # c wraped in go, according to gustav


def profile():
    def test_profile():
        for i in range(100):
            p = c_ecdsa_raw_recover(msg, vrs)

    import pstats
    import cProfile

    cProfile.runctx("test_profile()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()


if __name__ == '__main__':
    # test_ecrecover()
    # test_cecrecover()
    # profile()
    test_ecrecover(1000)
    test_cecrecover(1000)
