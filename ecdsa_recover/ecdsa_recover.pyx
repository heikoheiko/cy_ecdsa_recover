# cython: profile=True

"""
cython + gmp  implementation of ecdsa recover

based on:
https://github.com/vbuterin/pybitcointools


"""

# Elliptic curve parameters (secp256k1)

P = 2**256 - 2**32 - 977
N = 115792089237316195423570985008687907852837564279074904382605163141518161494337
cdef int A = 0
cdef int B = 7
Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424
G = (Gx, Gy)





# #####################################

cdef extern from "gmp.h":
    ctypedef struct mpz_t:
        pass

    cdef void mpz_init(mpz_t)
    cdef void mpz_add(mpz_t, mpz_t, mpz_t)
    cdef void mpz_tdiv_q(mpz_t, mpz_t, mpz_t)

    cdef void mpz_add_ui(mpz_t, mpz_t, unsigned long int)
    cdef void mpz_sub_ui(mpz_t, mpz_t, unsigned long int)
    cdef void mpz_mul_ui(mpz_t, mpz_t, unsigned long int)
    cdef void mpz_mul_si(mpz_t, mpz_t, long int)
    cdef void mpz_addmul_ui(mpz_t, mpz_t, unsigned long int)
    cdef void mpz_submul_ui(mpz_t, mpz_t, unsigned long int)
    cdef unsigned long int mpz_get_ui(mpz_t)
    cdef void mpz_set(mpz_t, mpz_t)
    cdef void mpz_clear(mpz_t)
    cdef void mpz_set_ui(mpz_t, unsigned long int)
    cdef void mpz_set_si(mpz_t, long int)
    cdef void mpz_init_set_ui(mpz_t, unsigned long int)
    cdef void mpz_init_set(mpz_t, mpz_t,)
    cdef int mpz_cmp(mpz_t, mpz_t)
    cdef void mpz_mul_2exp(mpz_t, mpz_t, unsigned long int)
    cdef unsigned long int mpz_mod_ui(mpz_t, mpz_t, unsigned long int)
    cdef void mpz_divexact_ui(mpz_t, mpz_t, unsigned long int)
    cdef unsigned long int mpz_fdiv_ui(mpz_t, unsigned long int)
    cdef void mpz_fdiv_q(mpz_t q, const mpz_t n, const mpz_t d)

    cdef void mpz_mul(mpz_t, mpz_t, mpz_t)
    cdef void mpz_mod(mpz_t, mpz_t, mpz_t)
    cdef void mpz_powm(mpz_t, mpz_t, mpz_t, mpz_t)
    cdef void mpz_pow_ui(mpz_t, mpz_t, unsigned long int)
    cdef void mpz_sub(mpz_t, mpz_t, mpz_t)


ui32 = 2**32
ui32M1 = ui32 - 1
cdef mpz_t m_ui32
cdef mpz_t m_ui32M1
mpz_init_set_ui(m_ui32, 1)
mpz_mul_2exp(m_ui32, m_ui32, 32)  # Set rop to op1 times 2 raised to op2.
mpz_sub_ui(m_ui32M1, m_ui32, 1)
cdef mpz_t tmp_z
mpz_init(tmp_z)
cdef mpz_t mNull
mpz_init_set_ui(mNull, 0)


cdef class Mpz:
    cdef mpz_t z

    def __init__(self, l=0):
        self.from_pylong(l)

    def __cinit__(self):
        mpz_init(self.z)

    def __dealloc__(self):
        mpz_clear(self.z)

    def from_pylong(self, l):
        assert isinstance(l, (long, int))
        l = long(l)
        cdef unsigned long i = 0
        cdef unsigned long r
        mpz_init_set_ui(self.z, 0)
        while abs(l) > ui32M1:
            r = l % ui32
            mpz_set_ui(tmp_z, r)
            mpz_mul_2exp(tmp_z, tmp_z, 32 * i)
            mpz_add(self.z, self.z, tmp_z)
            l //= ui32
            i += 1
        mpz_set_si(tmp_z, l)
        mpz_mul_2exp(tmp_z, tmp_z, 32 * i)
        mpz_add(self.z, self.z, tmp_z)

    def as_pylong(self):
        cdef unsigned long int r
        cdef unsigned long int d = ui32
        l = 0
        i = 0
        mpz_set(tmp_z, self.z)
        is_signed = False
        if mpz_cmp(tmp_z, mNull) < 0:
            is_signed = True
            mpz_mul_si(tmp_z, tmp_z, -1)
        while mpz_cmp(tmp_z, m_ui32M1) > 0:
            r = mpz_fdiv_ui(tmp_z, d)
            l += r * ui32 ** i
            mpz_divexact_ui(tmp_z, tmp_z, d)
            i += 1
        r = mpz_get_ui(tmp_z)
        l += r * ui32**i
        if is_signed:
            l *= -1
        return l

cdef mpz_to_long(mpz_t m):
    mm = Mpz()
    mpz_set(mm.z, m)
    return mm.as_pylong()

cdef void set_mpz_from_long(mpz_t m, l):
    mm = Mpz(l)
    mpz_set(m, mm.z)


############################################
cdef mpz_t mOne
mpz_init_set_ui(mOne, 1)
cdef mpz_t mTwo
mpz_init_set_ui(mTwo, 2)
cdef mpz_t mysq
mpz_init(mysq)
cdef mpz_t mS
mpz_init(mS)
cdef mpz_t mM
mpz_init(mM)
cdef mpz_t mTmp
mpz_init(mTmp)
cdef mpz_t mTmp2
mpz_init(mTmp2)
cdef mpz_t mP
mpz_init(mP)
set_mpz_from_long(mP, P)
cdef mpz_t mN
mpz_init(mN)
set_mpz_from_long(mN, N)
cdef mpz_t mA
mpz_init(mA)
set_mpz_from_long(mA, A)
cdef int zero = 0

cdef mpz_t mU1
mpz_init(mU1)
cdef mpz_t mU2
mpz_init(mU2)
cdef mpz_t mS1
mpz_init(mS1)
cdef mpz_t mS2
mpz_init(mS2)
cdef mpz_t mH
mpz_init(mH)
cdef mpz_t mH2
mpz_init(mH2)
cdef mpz_t mH3
mpz_init(mH3)
cdef mpz_t mU1H2
mpz_init(mU1H2)
cdef mpz_t mR
mpz_init(mR)

cdef inv(a, n):
    if a == 0:
        return 0
    lm = 1
    hm = 0
    low = a % n
    high = n
    while low > 1:
        r = high // low
        nm = hm - lm * r
        new = high - low * r
        hm = lm
        lm = nm
        high = low
        low = new
    return lm % n




cdef class Jacobian:
    cdef mpz_t x
    cdef mpz_t y
    cdef mpz_t z

    def __cinit__(self):
        mpz_init(self.x)
        mpz_init(self.y)
        mpz_init(self.z)

    def __dealloc__(self):
        mpz_clear(self.x)
        mpz_clear(self.y)
        mpz_clear(self.z)


    cdef Jacobian copy(self):
        j = Jacobian()
        mpz_set(j.x, self.x)
        mpz_set(j.y, self.y)
        mpz_set(j.z, self.z)
        return j

    def equals(self, Jacobian other):
        return (mpz_cmp(self.x, other.x) == zero) and (mpz_cmp(self.y, other.y) == zero) \
            and (mpz_cmp(self.z, other.z) == zero)

    def from_point(self, xyz):
        set_mpz_from_long(self.x, xyz[0])
        set_mpz_from_long(self.y, xyz[1])
        mpz_set(self.z, mOne)

    def as_point(self):
        z = inv(mpz_to_long(self.z), P)
        return ((mpz_to_long(self.x) * z**2) % P, (mpz_to_long(self.y) * z**3) % P)


    cdef void jdouble(self):
        if mpz_cmp(self.y, mNull) == zero:
             mpz_set(self.x, mNull)
             mpz_set(self.z, mNull)
        else:
            # ysq = (self.y ** 2) % P
            mpz_pow_ui(mysq, self.y, 2)
            mpz_mod(mysq, mysq, mP)

            # S = (4 * self.x * ysq) % P
            mpz_mul(mS, self.x, mysq)
            mpz_mul_ui(mS, mS, 4)
            mpz_mod(mS, mS, mP)

            # M = (3 * self.x ** 2 + A * self.z ** 4) % P
            mpz_pow_ui(mM, self.x, 2)
            mpz_mul_ui(mM, mM, 3)
            mpz_pow_ui(mTmp, self.z, 4)
            mpz_mul(mTmp, mTmp, mA)
            mpz_add(mM, mM, mTmp)
            mpz_mod(mM, mM, mP)

            # self.x = (M**2 - 2 * S) % P
            mpz_pow_ui(self.x, mM, 2)
            mpz_mul_ui(mTmp, mS, 2) # this can be cached
            mpz_sub(self.x, self.x, mTmp)
            mpz_mod(self.x, self.x, mP)

            # self.z = (2 * self.y * self.z) % P # relies on old y
            mpz_mul_ui(self.z, self.z, 2)
            mpz_mul(self.z, self.z, self.y)
            mpz_mod(self.z, self.z, mP)

            # self.y = (M * (S - self.x) - 8 * ysq ** 2) % P
            mpz_sub(self.y, mS, self.x)
            mpz_mul(self.y, self.y, mM)
            mpz_pow_ui(mTmp, mysq, 2)
            mpz_mul_ui(mTmp, mTmp, 8)
            mpz_sub(self.y, self.y, mTmp)
            mpz_mod(self.y, self.y, mP)

    cdef void add(self,  Jacobian q):
        self._add(q.x, q.y, q.z)

    cdef void _add(self, mpz_t qx, mpz_t qy, mpz_t qz):
        if mpz_cmp(self.y, mNull) == zero:
             mpz_set(self.x, qx)
             mpz_set(self.y, qy)
             mpz_set(self.z, qz)
        elif mpz_cmp(qy, mNull) == zero:
            pass
        else:
            # U1 = (self.x * qz ** 2) % P
            mpz_pow_ui(mU1, qz, 2)
            mpz_mul(mU1, mU1, self.x)
            mpz_mod(mU1, mU1, mP)

            # U2 = (qx * self.z ** 2) % P
            mpz_pow_ui(mU2, self.z, 2)
            mpz_mul(mU2, mU2, qx)
            mpz_mod(mU2, mU2, mP)


            # S1 = (self.y * qz ** 3) % P
            mpz_pow_ui(mS1, qz, 3)
            mpz_mul(mS1, mS1, self.y)
            mpz_mod(mS1, mS1, mP)

            # S2 = (qy * self.z ** 3) % P
            mpz_pow_ui(mS2, self.z, 3)
            mpz_mul(mS2, mS2, qy)
            mpz_mod(mS2, mS2, mP)

            if mpz_cmp(mU1, mU2) == zero:
                if mpz_cmp(mS1, mS2) != zero:
                    mpz_set(self.x, mNull)
                    mpz_set(self.y, mNull)
                    mpz_set(self.z, mOne)
                else:
                    self.jdouble()
            else:
                # H = U2 - U1
                mpz_sub(mH, mU2, mU1)
                # R = S2 - S1
                mpz_sub(mR, mS2, mS1)
                # H2 = (H * H) % P
                mpz_mul(mH2, mH, mH)
                mpz_mod(mH2, mH2, mP)
                # H3 = (H * H2) % P
                mpz_mul(mH3, mH, mH2)
                mpz_mod(mH3, mH3, mP)

                # U1H2 = (U1 * H2) % P
                mpz_mul(mU1H2, mU1, mH2)
                mpz_mod(mU1H2, mU1H2, mP)

                # self.x = (R ** 2 - H3 - 2 * U1H2) % P
                mpz_pow_ui(self.x, mR, 2)
                mpz_sub(self.x, self.x, mH3)
                mpz_mul(mTmp, mU1H2, mTwo)
                mpz_sub(self.x, self.x, mTmp)
                mpz_mod(self.x, self.x, mP)

                # self.y = (R * (U1H2 - self.x) - S1 * H3) % P
                mpz_sub(self.y, mU1H2, self.x)
                mpz_mul(self.y, self.y, mR)
                mpz_mul(mTmp, mS1, mH3)
                mpz_sub(self.y, self.y, mTmp)
                mpz_mod(self.y, self.y, mP)

                # works
                # self.z = H * self.z * qz
                mpz_mul(self.z, self.z, mH)
                mpz_mul(self.z, self.z, qz)

    cdef void py_multiply(self, l):
        cdef mpz_t m
        mpz_init(m)
        set_mpz_from_long(m, l)
        self.multiply(m)
        mpz_clear(m)

    cdef void multiply(self, mpz_t n):
        cdef mpz_t x
        cdef mpz_t y
        cdef mpz_t z

        if mpz_cmp(self.y, mNull) == zero or mpz_cmp(n, mNull) == zero:
            mpz_set(self.x, mNull)
            mpz_set(self.y, mNull)
            mpz_set(self.z, mOne)
        # elif n == 1:        #
        elif mpz_cmp(n, mOne) == 0:
            pass
        # elif n < 0 or n >= N:
        elif mpz_cmp(n, mNull) < 0 or not (mpz_cmp(n, mN) < 0):
            # self.multiply(n % mN)
            mpz_mod(n, n, mN)
            self.multiply(n)
        # elif (n % 2) == 0:
        elif mpz_fdiv_ui(n, 2) == 0:
            mpz_fdiv_q(n, n, mTwo)
            self.multiply(n)
            self.jdouble()
        else:
            # elif (n % 2) == 1:
            mpz_init_set(x, self.x)
            mpz_init_set(y, self.y)
            mpz_init_set(z, self.z)
            mpz_fdiv_q(n, n, mTwo)
            self.multiply(n)
            self.jdouble()
            self._add(x, y, z)
            mpz_clear(x)
            mpz_clear(y)
            mpz_clear(z)

#############################################


def hash_to_int(msghash):
    assert len(msghash) == 32
    z = 0
    for c in msghash:
        z *= 256
        z += ord(c)
    return z

def ecdsa_raw_recover(msghash, vrs):
    v, r, s = vrs
    x = r
    beta = pow(x * x * x + A * x + B, (P + 1) // 4, P)
    y = beta if v % 2 ^ beta % 2 else (P - beta)
    z = hash_to_int(msghash)

    j = Jacobian()
    j.from_point(G)
    j.py_multiply((N - z) % N)

    j2 = Jacobian()
    j2.from_point((x, y))
    j2.py_multiply(s)

    j.add(j2)
    j.py_multiply(inv(r, N))

    Q = j.as_point()

    if ecdsa_raw_verify(msghash, vrs, Q):
        return Q
    return False


def ecdsa_raw_verify(msghash, vrs, pub):
    v, r, s = vrs
    w = inv(s, N)
    z = hash_to_int(msghash)
    u1, u2 = z*w % N, r*w % N
    pub = decode_pubkey(pub)
    j = Jacobian()
    j.from_point(G)
    j.py_multiply(u1)
    j2 = Jacobian()
    j2.from_point(pub)
    j2.py_multiply(u2)
    j.add(j2)
    x, y = j.as_point()
    return r == x


def ecdsa_raw_sign(msghash, priv):
    z = hash_to_int(msghash)
    k = deterministic_generate_k(msghash, priv)
    j = Jacobian()
    j.from_point(G)
    j.py_multiply(k)
    r, y = j.as_point()
    s = inv(k, N) * (z + r*decode_privkey(priv)) % N

    return 27+(y % 2), r, s



# wraper
from bitcoin import electrum_sig_hash, decode_sig, encode_pubkey, encode_sig
from bitcoin import decode_privkey, deterministic_generate_k, decode_pubkey

def ecdsa_sign(msg, priv):
    return encode_sig(*ecdsa_raw_sign(electrum_sig_hash(msg), priv))


def ecdsa_verify(msg, sig, pub):
    return ecdsa_raw_verify(electrum_sig_hash(msg), decode_sig(sig), pub)


def ecdsa_recover(msg, sig):
    return encode_pubkey(ecdsa_raw_recover(electrum_sig_hash(msg), decode_sig(sig)), 'hex')

