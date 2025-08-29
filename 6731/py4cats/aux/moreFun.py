""" A 'random' collection of some useful(?) mathematical functions:

sindg       sine   for angle x given in degrees
cosdg       cosine for angle x given in degrees
asindg      arcsin with angle returned in degrees
acosdg      arccos with angle returned in degrees
Sinc        Sinc(x) = sin(x)/x
j2_over_xx  spherical Bessel function of the first kind divided by x**2
nexprl      relative error exponential for negative exponent:  (exp(-x)-1)/x
erf         error function (using rational approximation)
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

import math

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

pi      = math.pi
dg2pi   = pi/180. # conversion factor degree to radian
sqrt2   = math.sqrt(2.)
sqrt2pi = math.sqrt(2.*pi)
one     = 1.0
half    = 0.5

# select the functions to be imported  (thus 'from moreFun import *' will not import the constants!)
__all__ = 'sindg cosdg  asindg acosdg  Sinc  j2_over_xx  nexprl  erf'.split()

####################################################################################################################################

# NOTE:  the next two functions are also available as 'Convenience Functions' in scipy.special

def sindg(x):
	""" Compute sine for angle x given in degrees. """
	return np.sin(x*dg2pi)

def cosdg(x):
	""" Compute cosine for angle x given in degrees. """
	return np.cos(x*dg2pi)


def asindg(x):
	""" Compute arcsin and return angle alpha given in degrees. """
	return np.arcsin(x)/dg2pi
def acosdg(x):
	""" Compute arccos and return angle alpha given in degrees. """
	return np.arccos(x)/dg2pi


####################################################################################################################################

def Sinc(x):
	""" Sinc(x) = sin(x)/x
	    For small arguments x<1e-3 Taylor expansion is used.

	    NOTE:  numpy defines the 'cardinal sinc' with an extra factor pi:
	           sinc = sinc(pi*x)/(pi*x)
	    """
	return np.where(abs(x)>1e-3, np.sin(x)/x, 1.0-x**2/6.)


####################################################################################################################################

def j2_over_xx(x):
	""" Spherical Bessel function of the first kind divided by x**2
	j_2 = sqrt(0.5*pi/x) J_{5/2} = spherical Bessel function of the first kind,
	http://dlmf.nist.gov/10.49.E3
	"""
	return np.where(abs(x)>1e-8, ((3.0-x**2)*np.sin(x)-3.0*x*np.cos(x))/x**5, 1.0/15.0)


####################################################################################################################################

r3, r4, r5, r6, r7 = 1.0/3.0, 0.25, 0.2, 1.0/6.0, 1.0/7.0

def nexprl(x):
	""" Relative error exponential  (EXP(-x)-1)/x  for negative exponent.
	    see also:  http://www.netlib.no/netlib/slatec/fnlib/exprel.f and dexprl.f """
	# NOTE: Slatec defines a relative error exponential exprel(x) = (EXP(+X)-1)/X
	ax = abs(x)

	if isinstance(x, np.ndarray):
		ree = np.where(ax<1.0, one-half*ax*(one-r3*ax* (one-r4*ax* (one-r5*ax* (one-r6*ax* (one-r7*ax))))),
		                       (one-np.exp(-ax))/ax )
	else:
		if ax<1.0:   ree = one-half*ax*(one-r3*ax* (one-r4*ax* (one-r5*ax* (one-r6*ax* (one-r7*ax)))))
		else:        ree = (one-np.exp(-ax))/ax

	# use reflection formula for negative x, see Slatec
	return  np.where(x>0, ree, np.exp(-x) * ree)


####################################################################################################################################

##### Error function for real argument
##### NOTE:  scipy.special also has several functions for erf, erfc, erfcx, ...

PE =  0.3275911
A1, A2, A3, A4, A5 =  0.254829562, -0.284496736, 1.421413741, -1.453152027, 1.061405429

def erf(x):
	""" Compute the error function using a rational approximation for the exp scaled complementary error function.
	    erf(x)   = 1 - exp(-x^2)*erfcx(x)
	    erfcx(x) = exp(x^2) * erfc(x) = exp(x^2) * [1 - erf(x)]
            Accuracy 1.5e-7
            Literature:  Abramowitz&Stegun 1964: 7.1.26 """
	t  = one/(one+PE*x)
	erfce = t * (A1 + t*(A2 + t*(A3 + t*(A4 + t*A5))))
	return one - np.exp(-x**2)*erfce

def erfcx(x):
	""" Compute the exp scaled complementary error function using a rational approximation.
	    erfcx(x) = exp(x^2) * erfc(x) = exp(x^2) * [1 - erf(x)]
            Accuracy 1.5e-7
            Literature:  Abramowitz&Stegun 1964: 7.1.26 """
	t  = one/(one+PE*x)
	return t * (A1 + t*(A2 + t*(A3 + t*(A4 + t*A5))))


####################################################################################################################################

def quadratic_polynomial (x,y):
        """ Set up quadratic polynomial for first triplet of points (x[0],y[0]),  (x[1],y[1]),  (x[2],y[2]). """
        if x.shape==y.shape==(3,):
                DeltaML   = x[1] - x[0]
                DeltaRL   = x[2] - x[0]
                DeltaRM   = x[2] - x[1]
                SumML     = x[1] + x[0]
                SumRL     = x[2] + x[0]
                SumRM     = x[2] + x[1]
                ProductML = x[1] * x[0]
                ProductRL = x[2] * x[0]
                ProductRM = x[2] * x[1]
                #
                TermLft   =  y[0] / (DeltaML*DeltaRL)
                TermMid   = -y[1] / (DeltaML*DeltaRM)
                TermRgt   =  y[2] / (DeltaRL*DeltaRM)
                # coefficients of polynomial    a*x^2 + b*x +c
                a = TermLft + TermMid + TermRgt
                b = -(SumRM*TermLft + SumRL*TermMid + SumML*TermRgt)
                c = ProductRM*TermLft + ProductRL*TermMid + ProductML*TermRgt

                return a, b, c
        else:
                print ('ERROR  quadratic_polynomial:  input arrays x and y do NOT have shape (3,)')
                return
