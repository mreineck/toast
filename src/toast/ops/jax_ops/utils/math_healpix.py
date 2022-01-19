# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from typing import NamedTuple
import numpy as np
import jax.numpy as jnp

TWOINVPI = 0.63661977236758134308
MACHINE_EPSILON = np.finfo(np.float).eps

# -------------------------------------------------------------------------------------------------
# JAX


class HealpixPixels_JAX(NamedTuple):
    """
    Encapsulate the information found in a HealpixPixels in a JAX compatible way
    NOTE: if we port all funciton to JAX, it might be better to drop the C++ class and always use a Python one
    """
    nside: np.int64
    npix: np.int64
    ncap: np.int64
    dnside: np.double
    twonside: np.int64
    fournside: np.int64
    nsideplusone: np.int64
    nsideminusone: np.int64
    halfnside: np.double
    tqnside: np.double
    factor: np.int64
    utab: np.array
    ctab: np.array

    @classmethod
    def from_HealpixPixels(cls, hpix):
        return cls(hpix.nside, hpix.npix, hpix.ncap, hpix.dnside, hpix.twonside, hpix.fournside,
                   hpix.nsideplusone, hpix.nsideminusone, hpix.halfnside, hpix.tqnside, hpix.factor,
                   # we convert lists into numpy array to be able to index into them with indices unknown at compile time
                   jnp.asarray(hpix.utab), jnp.asarray(hpix.ctab))


def xy2pix_jax(hpix, x, y):
    return hpix.utab[x & 0xff] \
        | (hpix.utab[(x >> 8) & 0xff] << 16) \
        | (hpix.utab[(x >> 16) & 0xff] << 32) \
        | (hpix.utab[(x >> 24) & 0xff] << 48) \
        | (hpix.utab[y & 0xff] << 1) \
        | (hpix.utab[(y >> 8) & 0xff] << 17) \
        | (hpix.utab[(y >> 16) & 0xff] << 33) \
        | (hpix.utab[(y >> 24) & 0xff] << 49)


def zphi2nest_jax(hpix, phi, region, z, rtz):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (double)
        region (int)
        z (double)
        rtz (double)

    Returns:
        pix (int)
    """
    phi = jnp.where(jnp.abs(phi) < MACHINE_EPSILON, 0.0, phi)

    tt = phi * TWOINVPI
    tt = jnp.where(phi < 0.0, tt+4.0, tt)

    # NOTE: we convert the branch into a jnp.where
    # if (jnp.abs(region) == 1)
    # then:
    temp1 = hpix.halfnside + hpix.dnside * tt
    temp2 = hpix.tqnside * z

    jp = jnp.int64(temp1 - temp2)
    jm = jnp.int64(temp1 + temp2)

    ifp = jp >> hpix.factor
    ifm = jm >> hpix.factor

    face_then = jnp.where(ifp == ifm,
                          jnp.where(ifp == 4, 4, ifp + 4),
                          jnp.where(ifp < ifm, ifp, ifm + 8))

    x_then = jm & hpix.nsideminusone
    y_then = hpix.nsideminusone - (jp & hpix.nsideminusone)
    # else:
    ntt = jnp.int64(tt)

    tp = tt - jnp.double(ntt)

    temp1 = hpix.dnside * rtz

    jp = jnp.int64(tp * temp1)
    jp = jnp.where(jp >= hpix.nside, hpix.nsideminusone, jp)

    jm = jnp.int64((1.0 - tp) * temp1)
    jm = jnp.where(jm >= hpix.nside, hpix.nsideminusone, jm)

    face_else = jnp.where(z >= 0, ntt, ntt + 8)
    x_else = jnp.where(z >= 0, hpix.nsideminusone - jm, jp)
    y_else = jnp.where(z >= 0, hpix.nsideminusone - jp, jm)
    # where
    face = jnp.where(jnp.abs(region) == 1, face_then, face_else)
    x = jnp.where(jnp.abs(region) == 1, x_then, x_else)
    y = jnp.where(jnp.abs(region) == 1, y_then, y_else)

    sipf = xy2pix_jax(hpix, jnp.int64(x), jnp.int64(y))
    pix = jnp.int64(sipf) + (face << (2 * hpix.factor))
    return pix


def zphi2ring_jax(hpix, phi, region, z, rtz, pix):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (double)
        region (int)
        z (double)
        rtz (double)

    Returns:
        pix (int)
    """
    phi = jnp.where(jnp.abs(phi) < MACHINE_EPSILON, 0.0, phi)

    tt = phi * TWOINVPI
    tt = jnp.where(phi < 0.0, tt+4.0, tt)

    # NOTE: we convert the branch into a jnp.where
    # if (jnp.abs(region) == 1)
    # then:
    temp1 = hpix.halfnside + hpix.dnside * tt
    temp2 = hpix.tqnside * z

    jp = jnp.int64(temp1 - temp2)
    jm = jnp.int64(temp1 + temp2)

    ir = hpix.nsideplusone + jp - jm
    kshift = 1 - (ir & 1)

    ip = (jp + jm - hpix.nside + kshift + 1) >> 1
    ip = ip % hpix.fournside

    pix_then = hpix.ncap + ((ir - 1) * hpix.fournside + ip)
    # else:
    tp = tt - jnp.floor(tt)

    temp1 = hpix.dnside * rtz

    jp = jnp.int64(tp * temp1)
    jm = jnp.int64((1.0 - tp) * temp1)
    ir = jp + jm + 1
    ip = jnp.int64(tt * jnp.double(ir))
    longpart = jnp.int64(ip / (4 * ir))
    ip = ip - longpart

    pix_pos = 2 * ir * (ir - 1) + ip
    pix_neg = hpix.npix - 2 * ir * (ir + 1) + ip
    pix_else = jnp.where(region > 0, pix_pos, pix_neg)
    # where
    pix = jnp.where(jnp.abs(region) == 1, pix_then, pix_else)

    return pix


def vec2zphi_jax(vec):
    """
    Args:
        vec (array, double) of size 3

    Returns:
        (phi, region, z, rtz)
        phi (double)
        region (int)
        z (double)
        rtz (double)
    """
    z = vec[2]
    za = jnp.abs(z)

    # region encodes BOTH the sign of Z and whether its
    # absolute value is greater than 2/3.
    itemps = jnp.where(z > 0.0, 1, -1)
    region = jnp.where(za <= 2./3., itemps, 2*itemps)

    work1 = 3.0 * (1.0 - za)
    rtz = jnp.sqrt(work1)

    work2 = vec[0]
    work3 = vec[1]
    phi = jnp.arctan2(work3, work2)

    return (phi, region, z, rtz)


def vec2nest_jax(hpix, vec):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        vec (double) of size 3

    Returns:
        pix (int)
    """
    (phi, region, z, rtz) = vec2zphi_jax(vec)
    pix = zphi2nest_jax(hpix, phi, region, z, rtz)
    return pix


def vec2ring_jax(hpix, vec):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        vec (array, double) of size 3

    Returns:
        pix (int)
    """
    (phi, region, z, rtz) = vec2zphi_jax(vec)
    pix = zphi2ring_jax(hpix, phi, region, z, rtz)
    return pix

# -------------------------------------------------------------------------------------------------
# NUMPY


def xy2pix_numpy(hpix, x, y):
    return hpix.utab[x & 0xff] \
        | (hpix.utab[(x >> 8) & 0xff] << 16) \
        | (hpix.utab[(x >> 16) & 0xff] << 32) \
        | (hpix.utab[(x >> 24) & 0xff] << 48) \
        | (hpix.utab[y & 0xff] << 1) \
        | (hpix.utab[(y >> 8) & 0xff] << 17) \
        | (hpix.utab[(y >> 16) & 0xff] << 33) \
        | (hpix.utab[(y >> 24) & 0xff] << 49)


def zphi2nest_numpy(hpix, phi, region, z, rtz):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (double)
        region (int)
        z (double)
        rtz (double)

    Returns:
        pix (int)
    """
    # machine epsilon
    eps = np.finfo(np.float).eps

    if np.abs(phi) < eps:
        phi = 0.0

    tt = phi * TWOINVPI
    if phi < 0.0:
        tt += 4.0

    if (np.abs(region) == 1):
        temp1 = hpix.halfnside + hpix.dnside * tt
        temp2 = hpix.tqnside * z

        jp = np.int64(temp1 - temp2)
        jm = np.int64(temp1 + temp2)

        ifp = jp >> hpix.factor
        ifm = jm >> hpix.factor

        if (ifp == ifm):
            face = 4 if ifp == 4 else ifp + 4
        elif (ifp < ifm):
            face = ifp
        else:
            face = ifm + 8

        x = jm & hpix.nsideminusone
        y = hpix.nsideminusone - (jp & hpix.nsideminusone)
    else:
        ntt = np.int64(tt)

        tp = tt - np.double(ntt)

        temp1 = hpix.dnside * rtz

        jp = np.int64(tp * temp1)
        jm = np.int64((1.0 - tp) * temp1)

        if (jp >= hpix.nside):
            jp = hpix.nsideminusone

        if (jm >= hpix.nside):
            jm = hpix.nsideminusone

        if (z >= 0):
            face = ntt
            x = hpix.nsideminusone - jm
            y = hpix.nsideminusone - jp
        else:
            face = ntt + 8
            x = jp
            y = jm

    sipf = xy2pix_numpy(hpix, np.int64(x), np.int64(y))
    pix = np.int64(sipf) + (face << (2 * hpix.factor))
    return pix


def zphi2ring_numpy(hpix, phi, region, z, rtz, pix):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (double)
        region (int)
        z (double)
        rtz (double)

    Returns:
        pix (int)
    """
    # machine epsilon
    eps = np.finfo(np.float).eps

    if (np.abs(phi) < eps):
        phi = 0.0

    tt = phi * TWOINVPI
    if (phi < 0.0):
        tt += 4.0

    if (np.abs(region) == 1):
        temp1 = hpix.halfnside + hpix.dnside * tt
        temp2 = hpix.tqnside * z

        jp = np.int64(temp1 - temp2)
        jm = np.int64(temp1 + temp2)

        ir = hpix.nsideplusone + jp - jm
        kshift = 1 - (ir & 1)

        ip = (jp + jm - hpix.nside + kshift + 1) >> 1
        ip = ip % hpix.fournside

        pix = hpix.ncap + ((ir - 1) * hpix.fournside + ip)
    else:
        tp = tt - np.floor(tt)

        temp1 = hpix.dnside * rtz

        jp = np.int64(tp * temp1)
        jm = np.int64((1.0 - tp) * temp1)
        ir = jp + jm + 1
        ip = np.int64(tt * np.double(ir))
        longpart = np.int64(ip / (4 * ir))
        ip -= longpart

        pix = (2 * ir * (ir - 1) + ip) if (region >
                                           0) else (hpix.npix - 2 * ir * (ir + 1) + ip)

    return pix


def vec2zphi_numpy(vec):
    """
    Args:
        vec (array, double) of size 3

    Returns:
        (phi, region, z, rtz)
        phi (double)
        region (int)
        z (double)
        rtz (double)
    """
    z = vec[2]
    za = np.abs(z)

    # region encodes BOTH the sign of Z and whether its
    # absolute value is greater than 2/3.
    itemps = 1 if (z > 0.0) else -1
    region = itemps if (za <= 2./3.) else 2*itemps

    work1 = 3.0 * (1.0 - za)
    rtz = np.sqrt(work1)

    work2 = vec[0]
    work3 = vec[1]
    phi = np.arctan2(work3, work2)

    return (phi, region, z, rtz)


def vec2nest_numpy(hpix, vec):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        vec (array, double) of sise 3

    Returns:
        pix (int)
    """
    (phi, region, z, rtz) = vec2zphi_numpy(vec)
    pix = zphi2nest_numpy(hpix, phi, region, z, rtz)
    return pix


def vec2ring_numpy(hpix, vec):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        vec (array, double) of sise 3

    Returns:
        pix (int)
    """
    (phi, region, z, rtz) = vec2zphi_numpy(vec)
    pix = zphi2ring_numpy(hpix, phi, region, z, rtz)
    return pix

# -------------------------------------------------------------------------------------------------
# C++


# see toast_math_healpix.cpp
"""
void toast::HealpixPixels::vec2zphi(int64_t n, double const * vec,
                                    double * phi, int * region, double * z,
                                    double * rtz) const 
{
    toast::AlignedVector <double> work1(n);
    toast::AlignedVector <double> work2(n);
    toast::AlignedVector <double> work3(n);

    for (int64_t i = 0; i < n; ++i) 
    {
        int64_t offset = 3 * i;

        // region encodes BOTH the sign of Z and whether its
        // absolute value is greater than 2/3.

        z[i] = vec[offset + 2];

        double za = ::fabs(z[i]);

        int itemp = (z[i] > 0.0) ? 1 : -1;

        region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;

        work1[i] = 3.0 * (1.0 - za);
        work3[i] = vec[offset + 1];
        work2[i] = vec[offset];
    }

    toast::vfast_sqrt(n, work1.data(), rtz);
    toast::vatan2(n, work3.data(), work2.data(), phi);
}

void toast::HealpixPixels::zphi2nest(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const 
{
    double eps = std::numeric_limits <float>::epsilon();
    
    for (int64_t i = 0; i < n; ++i) 
    {
        double ph = phi[i];
        if (fabs(ph) < eps) 
        {
            ph = 0.0;
        }
        double tt = (ph >= 0.0) ? ph * TWOINVPI : ph * TWOINVPI + 4.0;

        int64_t x;
        int64_t y;
        double temp1;
        double temp2;
        int64_t jp;
        int64_t jm;
        int64_t ifp;
        int64_t ifm;
        int64_t face;
        int64_t ntt;
        double tp;

        if (::abs(region[i]) == 1) 
        {
            temp1 = halfnside_ + dnside_ * tt;
            temp2 = tqnside_ * z[i];

            jp = static_cast <int64_t> (temp1 - temp2);
            jm = static_cast <int64_t> (temp1 + temp2);

            ifp = jp >> factor_;
            ifm = jm >> factor_;

            face;
            if (ifp == ifm) 
            {
                face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
            } 
            else if (ifp < ifm) 
            {
                face = ifp;
            } 
            else 
            {
                face = ifm + 8;
            }

            x = jm & nsideminusone_;
            y = nsideminusone_ - (jp & nsideminusone_);
        } 
        else 
        {
            ntt = static_cast <int64_t> (tt);

            tp = tt - static_cast <double> (ntt);

            temp1 = dnside_ * rtz[i];

            jp = static_cast <int64_t> (tp * temp1);
            jm = static_cast <int64_t> ((1.0 - tp) * temp1);

            if (jp >= nside_) 
            {
                jp = nsideminusone_;
            }
            if (jm >= nside_) 
            {
                jm = nsideminusone_;
            }

            if (z[i] >= 0) 
            {
                face = ntt;
                x = nsideminusone_ - jm;
                y = nsideminusone_ - jp;
            } 
            else 
            {
                face = ntt + 8;
                x = jp;
                y = jm;
            }
        }

        uint64_t sipf = xy2pix_(static_cast <uint64_t> (x), static_cast <uint64_t> (y));
        pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
    }
}

void toast::HealpixPixels::zphi2ring(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const {
    double eps = std::numeric_limits <float>::epsilon();
    
    for (int64_t i = 0; i < n; ++i) {
        double ph = phi[i];
        if (fabs(ph) < eps) {
            ph = 0.0;
        }
        double tt = (ph >= 0.0) ? ph * TWOINVPI : ph * TWOINVPI + 4.0;

        double tp;
        int64_t longpart;
        double temp1;
        double temp2;
        int64_t jp;
        int64_t jm;
        int64_t ip;
        int64_t ir;
        int64_t kshift;

        if (::abs(region[i]) == 1) {
            temp1 = halfnside_ + dnside_ * tt;
            temp2 = tqnside_ * z[i];

            jp = static_cast <int64_t> (temp1 - temp2);
            jm = static_cast <int64_t> (temp1 + temp2);

            ir = nsideplusone_ + jp - jm;
            kshift = 1 - (ir & 1);

            ip = (jp + jm - nside_ + kshift + 1) >> 1;
            ip = ip % fournside_;

            pix[i] = ncap_ + ((ir - 1) * fournside_ + ip);
        } else {
            tp = tt - floor(tt);

            temp1 = dnside_ * rtz[i];

            jp = static_cast <int64_t> (tp * temp1);
            jm = static_cast <int64_t> ((1.0 - tp) * temp1);
            ir = jp + jm + 1;
            ip = static_cast <int64_t> (tt * (double)ir);
            longpart = static_cast <int64_t> (ip / (4 * ir));
            ip -= longpart;

            pix[i] = (region[i] > 0) ? (2 * ir * (ir - 1) + ip)
                        : (npix_ - 2 * ir * (ir + 1) + ip);
        }
    }
}

void toast::HealpixPixels::vec2nest(int64_t n, double const * vec,
                                    int64_t * pix) const 
{
    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());
    zphi2nest(n, phi.data(), region.data(), z.data(), rtz.data(), pix);
}

void toast::HealpixPixels::vec2ring(int64_t n, double const * vec,
                                    int64_t * pix) const {
    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());
    zphi2ring(n, phi.data(), region.data(), z.data(), rtz.data(), pix);
}
"""
