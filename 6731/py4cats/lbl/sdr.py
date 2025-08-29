""" Speed dependent Voigt (SDV) and Rautian (SDR) functions K(x,y,...)

    The generalizations of the Voigt function K(x,y) with one and/or two additional arguments q and/or r.

    ARGUMENTS:
    x = sqrtLn2*(vGrid-vLine)/gammaG      (vGrid = wavenumber (grid);   vLine = actual line position (incl. shift))
    y = sqrtLn2*gammaL/gammaG             (gammaL = gamma0 speed averaged broadening at actual p,T)
    q = sqrtLn2*gamma2/gammaG             (gamma2 = quadratic dependence of broadening parameter at p,T)
    r = sqrtLn2*nu_vc/gammaG              (frequency of velocity-changing collisions (Dicke narrowing)  (= zeta in Varghese&Hanson))


    REFERENCES:
    BWB = Boone, Walker, Bernath:
    An efficient analytical approach for calculating line mixing in atmospheric remote sensing applications.
    JQSRT 112(6), pp. 8=980-089, 2011

    TNH = Tran, Ngo, Hartmann: Efficient computation of some speed-dependent isolated line profiles.
    JQSRT 129, pp. 199–203, 2013. (Erratum: JQSRT 134, 104 (2014))

    FS:    Computational aspects of speed-dependent Voigt profiles. JQSRT 187, pp. 44-53, 2017
    FS+PH: Computational aspects of speed-dependent Voigt and Rautian profiles. JQSRT 258, 107385, 2021
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy failed!')

# try:                        from scipy.special import wofz
# except ImportError as msg:  raise SystemExit (str(msg) + '\nimport scipy failed!')

sqrtPi   = np.sqrt(np.pi)

from .. aux.racef import zpf16p                                        ### zpf16h(x,y) <---> zpf16p(x+1j*y) used by sdv/sdr only !!!

# public functions (i.e. do not import the constants):
__all__ = 'rautian sdv sdr'.split()


####################################################################################################################################
####################################################################################################################################

def rautian (x, y, r=0.0, cerf=zpf16p):
	""" Evaluate Rautian function  (ignoring pressure induced shift).

	    The generalization of the Voigt function K(x,y) with one additional argument
	     x = sqrtLn2*(vGrid-vLine)/gammaD         (vGrid = wavenumber (grid);   vLine = actual line position (incl. shift))
	     y = sqrtLn2*gammaL/gammaD                (gammaL = gamma0 speed averaged broadening at actual p,T)
	     r = sqrtLn2*nu_vc/gammaD  = zeta (Varghese&Hanson)
	    """

	ww    = cerf(x+1j*(y+r))
	ratio = ww / (1.0 - sqrtPi*r*ww)

	# and return the !complex! function (imag required for line mix!)
	return  ratio


####################################################################################################################################

def sdv (x, y, q=0.0, cerf=zpf16p):
	""" Evaluate speed-dependent Voigt function  (ignoring pressure induced shift).
	    The generalization of the Voigt function K(x,y) with one additional arguments q.

	    ARGUMENTS:
	     x = sqrtLn2*(vGrid-vLine)/gammaD         (vGrid = wavenumber (grid);   vLine = actual line position (incl. shift))
	     y = sqrtLn2*gammaL/gammaD                (gammaL = gamma0 speed averaged broadening at actual p,T)
	     q = sqrtLn2*gamma2/gammaD                (gamma2 = quadratic dependence of broadening parameter at p,T):w

	    RETURNS:
	     sdv = the complex "generalization" of the complex error function w

	    This is a 'naive' version without check if the two arguments zPlus, zMinus are in the same region.
	    Complex arithmetic similar to TNH, but with cancellation-safe evaluation of the square root difference
	    """

	if isinstance(q,(int,float)) and q<=0.0:  return cerf(x+1j*y)

	# arguments for the complex error function
	recQ     = 1/q
	bigX     = (y - 1j*x)*recQ - 1.5   # alfa+1j*beta
	sqrtBigY = 0.5*recQ                # sqrtDelta
	bigY     = sqrtBigY**2
	sqrtXY   = np.sqrt(bigX+bigY)
	zPlus    = sqrtXY + sqrtBigY
	zMinus   = bigX / zPlus

	# evaluate the difference terms
	wPlus  = cerf(1j*zPlus)
	wMinus = cerf(1j*zMinus)

	# evaluate the ratio
	wDelta   = wMinus-wPlus

	# and return the !complex! function (imag required for line mix!)
	return wDelta


####################################################################################################################################

def sdr (x, y, q=0.0, r=0.0, cerf=zpf16p):
	""" Evaluate speed-dependent Rautian function  (ignoring pressure induced shift).
	    The generalization of the Voigt function K(x,y) with two additional arguments q and r.

	    ARGUMENTS:
	     x = sqrtLn2*(vGrid-vLine)/gammaD         (vGrid = wavenumber (grid);   vLine = actual line position (incl. shift))
	     y = sqrtLn2*gammaL/gammaD                (gammaL = gamma0 speed averaged broadening at actual p,T)
	     q = sqrtLn2*gamma2/gammaD                (gamma2 = quadratic dependence of broadening parameter at p,T):w
	     r = sqrtLn2*nu_vc/gammaD  = zeta (Varghese&Hanson)

	    RETURNS:
	     sdr = the complex "generalization" of the complex error function w

	    This is a 'naive' version without check if the two arguments zPlus, zMinus are in the same region.
	    Complex arithmetic similar to TNH, but with cancellation-safe evaluation of the square root difference
	    """

	if isinstance(q,(int,float)) and q<=0.0:
		if isinstance(r,(int,float)) and r<=0.0:  return cerf(x+1j*y)
		else:                                     rautian (x, y, r, cerf)

	# arguments for the complex error function
	recQ     = 1/q
	bigX     = (y+r - 1j*x)*recQ - 1.5   # alfa+1j*beta
	sqrtBigY = 0.5*recQ                  # sqrtDelta
	bigY     = sqrtBigY**2
	sqrtXY   = np.sqrt(bigX+bigY)
	zPlus    = sqrtXY + sqrtBigY
	#zMinus   = sqrtXY - sqrtBigY
	zMinus   = bigX / zPlus

	# evaluate the difference terms
	wPlus  = cerf(1j*zPlus)
	wMinus = cerf(1j*zMinus)

	# evaluate the ratio
	wDelta   = wMinus-wPlus
	sdrFunction = wDelta / (1.0 - sqrtPi*r*wDelta)

	# and return !complex! profile (imag required for line mix!)
	return sdrFunction
