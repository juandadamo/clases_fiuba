"""
Line profile functions normalized to 1.0 with wavenumber (array) and position, strengths, and width(s) as input variables.
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

# some math constants
from math import pi, sqrt

from .. aux.ir import h, c, k, sqrtPi, ln2, sqrtLn2, recSqrtPi
from .. aux.racef import hum1wei24, hum2wei32, zpf16h      ### zpf16h(x,y) <---> zpf16p(x+1j*y) used by sdv/sdr only
try:  from py4cats.lbl.sdr import sdv, sdr                 ### returns the complex function (Im required for line mixing)!
except ImportError as msg:  print (str(msg) + '\nimport sdr failed, no speed dependent profiles')

recSqrt2      = 1./sqrt(2.0)
recSqrtLn2    = 1./sqrtLn2
sqrtLn2overPi = sqrtLn2/sqrtPi

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')


####################################################################################################################################

def voigtWidth (gammaL=1.0, gammaD=1.0):
	""" Half width half maximum (HWHM) of Voigt profile (Whiting's approximation). """
	return 0.5*(gammaL+np.sqrt(gammaL**2+4.0*gammaD**2))


####################################################################################################################################

def Lorentz (vGrid, vLine=0.0, strength=1.0, gammaL=1.0):
	""" Pressure broadening: Lorentz profile normalized to one, multiplied with line strength. """
	return (strength*gammaL/pi)/((vGrid-vLine)**2+gammaL**2)


def LorentzMix (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, yMix=0.0):
	""" Pressure broadening: Lorentz profile incl. line mixing, normalized to one, multiplied with line strength.
	    See D. Edwards, SPIE Vol. 928 (1986), Eq. (4.5) """
	deltaV = vGrid-vLine
	return (strength/pi)*(gammaL + yMix*deltaV)/(deltaV**2+gammaL**2)


####################################################################################################################################

def vanVleckWeisskopf (vGrid, vLine=0.0, strength=1.0, gammaL=1.0):
	""" Pressure broadening: VanVleck-Weisskopf lineshape without tanh prefactor, multiplied with line strength.
	    ARTS manual, eq. (3.12)
	    ARTS --- Buehler, Eriksson et al., JQSRT Vol. 91 (2005), Eq. (7)
	    See D. Edwards, SPIE Vol. 928 (1986), Eq. (4.3)
	"""
	factor = strength * gammaL/pi
	return (vGrid/vLine)**2 * (factor / ((vGrid-vLine)**2+gammaL**2) + factor / ((vGrid+vLine)**2+gammaL**2))


####################################################################################################################################

def vanVleckHuber (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, temp=250.0):
	""" Pressure broadening: VanVleck-Huber lineshape with tanh prefactor, multiplied with line strength.
	    See D. Edwards, SPIE Vol. 928 (1986), Eq. (4.3)
	    ARTS --- Buehler, Eriksson et al., JQSRT Vol. 91 (2005), Table 2
	"""
	if not isinstance(temp,float):  raise SystemExit('ERROR --- vanVleckHuber: no temperature specified!')
	factor = strength * gammaL/pi
	hc2k   = h*c/(2.0*k)
	tv0    = np.tanh(hc2k*vLine/temp)
	return vGrid * np.tanh(hc2k*vGrid/temp)/(vLine*tv0) * \
	       (factor / ((vGrid-vLine)**2+gammaL**2) + factor / ((vGrid+vLine)**2+gammaL**2))


####################################################################################################################################

### NOTE:  there is also a `Gauss` function defined in the srf.py module (with two arguments only)

def Gauss (vGrid, vLine=0.0, strength=1.0, gammaD=1.0):
	""" Doppler broadening:  Gauss profile normalized to one, multiplied with line strength. """
	t = ln2*((vGrid-vLine)/gammaD)**2
	# don't compute exponential for very large numbers > log(1e308)=709.
	# (in contrast to python.math.exp the numpy exp overflows!!!)
	return strength * sqrtLn2/(sqrtPi*gammaD) * np.where(t<200.,np.exp(-t),0.0)


####################################################################################################################################

def Voigt (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0):
	""" Voigt profile normalized to one, multiplied with line strength. """
	rgD   = sqrtLn2 / gammaD
	xGrid = rgD * (vGrid-vLine)
	y     = rgD * gammaL
	if y>1e-6:  vgt = recSqrtPi*rgD*strength * hum1wei24(xGrid,y).real  # Voigt profile
	else:       vgt = recSqrtPi*rgD*strength * hum2wei32(xGrid,y).real  # Voigt profile
	return vgt


def VoigtMix (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0, yMix=0.0):
	""" Voigt profile (normalized to one) with Rosenkranz first order approximation, multiplied with line strength.

	    Boone, Walker, Bernath (JQSRT 112, 980-989,  2011):  Eq. (8)
	"""
	rgD   = sqrtLn2 / gammaD
	xGrid = rgD * (vGrid-vLine)
	y     = rgD * gammaL
	cef   = hum1wei24(xGrid,y)
	return recSqrtPi*rgD*strength * (cef.real+yMix*cef.imag)


####################################################################################################################################

def SpeedVoigt (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0, gamma2=1.0):
	""" Speed-dependent Rautian profile normalized to one, multiplied with line strength.

	    Boone, Walker, Bernath (JQSRT 105, 2007): Speed-dependent Voigt profilefor water vapor in infrared ...
	    Tennyson et al. (Pure & Applied Chemistry 86, 2014):  Recommended isolated line profile ... (IUPAC Technical Report)
	    Schreier (JQSRT 187, 2017): Computational Aspects of Speed-Dependent Voigt Profiles
	"""

	# the standard Voigt parameters
	rgD   = sqrtLn2 / gammaD
	xGrid = rgD*(vGrid-vLine)
	y     = rgD*gammaL
	# the beyond Voigt parameter
	q     = rgD*gamma2

	sdvFunction = sdv(xGrid, y, q)
	return strength*(sqrtLn2*recSqrtPi/gammaD)*sdvFunction.real


def SpeedVoigtMix (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0, gamma2=1.0, yMix=0.0):
	""" Speed-dependent Voigt profile normalized to one, multiplied with line strength.

	    Boone, Walker, Bernath (JQSRT 105, 2007): Speed-dependent Voigt profilefor water vapor in infrared ...
	    Tennyson et al. (Pure & Applied Chemistry 86, 2014):  Recommended isolated line profile ... (IUPAC Technical Report)
	    Schreier (JQSRT 187, 2017): Computational Aspects of Speed-Dependent Voigt Profiles
	"""

	# the standard Voigt parameters
	xGrid = sqrtLn2*(vGrid-vLine)/gammaD
	y = sqrtLn2*gammaL/gammaD
	# the beyond Voigt parameter
	q = sqrtLn2*gamma2/gammaD

	sdvFunction = sdv(xGrid, y, q)
	return strength*(sqrtLn2*recSqrtPi/gammaD)*(sdvFunction.real + yMix*sdvFunction.imag)


####################################################################################################################################

def Rautian (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0, gammaN=1.0):
	""" Rautian profile normalized to one, multiplied with line strength.
	    Broadening by thermal motion and state-perturbing collisions;
	    hard collision model for velocity-changing collisions; collisional perturbations uncorrelated

	    Philip L. Varghese and Ronald K. Hanson:
	    Collisional narrowing effects on spectral line shapes measured at high resolution [AO 23(14), 2376-2385, 1984]
	"""
	rgD    = sqrtLn2 / gammaD
	xGrid  = rgD * (vGrid-vLine)
	y      = rgD * gammaL
	zeta   = rgD * gammaN
	cef    = zpf16h(xGrid,   (y+zeta))   ### Note that this (horner) version of zpf16 uses two real arguments
	wRatio = cef / (1.0-sqrtPi*zeta*cef)
	rtn    = recSqrtPi*rgD*strength * wRatio.real
	return rtn


def RautianMix (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0, gammaN=1.0, yMix=0.0):
	""" Rautian profile including Rosenkranz line mixing ( normalized to one), multiplied with line strength.
	    Broadening by thermal motion and state-perturbing collisions;
	    hard collision model for velocity-changing collisions; collisional perturbations uncorrelated

	    Philip L. Varghese and Ronald K. Hanson:
	    Collisional narrowing effects on spectral line shapes measured at high resolution [AO 23(14), 2376-2385, 1984]
	"""
	rgD    = sqrtLn2 / gammaD
	xGrid  = rgD * (vGrid-vLine)
	y      = rgD * gammaL
	zeta   = rgD * gammaN
	cef    = hum2wei32(xGrid,y+zeta)
	wRatio = cef / (1.0-sqrtPi*zeta*cef)
	if abs(yMix):  return recSqrtPi*rgD*strength * (wRatio.real + yMix*wRatio.imag)
	else:          return recSqrtPi*rgD*strength *  wRatio.real


####################################################################################################################################

def SpeedRautian (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0, gamma2=1.0, gammaN=1.0):
	""" Speed-dependent Rautian profile normalized to one, multiplied with line strength.

	    Boone, Walker, Bernath (JQSRT 105, 2007): Speed-dependent Voigt profilefor water vapor in infrared ...
	    Tennyson et al. (Pure & Applied Chemistry 86, 2014):  Recommended isolated line profile ... (IUPAC Technical Report)
	    Schreier (JQSRT 187, 2017): Computational Aspects of Speed-Dependent Voigt Profiles
	"""

	# the standard Voigt parameters
	rgD   = sqrtLn2 / gammaD
	xGrid = rgD*(vGrid-vLine)
	y = rgD*gammaL
	# the beyond Voigt parameters
	q = rgD*gamma2
	r = rgD*gammaN

	sdrFunction = sdr(xGrid, y, q, r)
	return strength*(sqrtLn2*recSqrtPi/gammaD)*sdrFunction.real


def SpeedRautianMix (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0, gamma2=1.0, gammaN=1.0, yMix=0.0):
	""" Speed-dependent Rautian profile normalized to one, multiplied with line strength.

	    Boone, Walker, Bernath (JQSRT 105, 2007): Speed-dependent Voigt profilefor water vapor in infrared ...
	    Tennyson et al. (Pure & Applied Chemistry 86, 2014):  Recommended isolated line profile ... (IUPAC Technical Report)
	    Schreier (JQSRT 187, 2017): Computational Aspects of Speed-Dependent Voigt Profiles
	"""

	# the standard Voigt parameters
	rgD   = sqrtLn2 / gammaD
	xGrid = rgD*(vGrid-vLine)
	y     = rgD*gammaL
	# the beyond Voigt parameters
	q     = rgD*gamma2
	r     = rgD*gammaN

	sdrFunction = sdr(xGrid, y, q, r)
	return strength*(sqrtLn2*recSqrtPi/gammaD)*(sdrFunction.real + yMix*sdrFunction.imag)

####################################################################################################################################

def Voigt_Kuntz_Humlicek1 (vGrid, vLine=0.0, strength=1.0, gammaL=1.0, gammaD=1.0):
	""" Voigt profile normalized to one, multiplied with line strength.
	    Rational approximation for asymptotic region with large |x|+y.
	    Real part of w(z) using Kuntz (JQSRT, 1997) implementation with Ruyten's (JQSRT, 2003) correction. """
	xx = (sqrtLn2 * (vGrid-vLine)/ gammaD)**2
	y  = sqrtLn2 * gammaL / gammaD
	yy = y*y
	kha1 = 0.5+yy
	kha2 = kha1 + yy*yy; khb2 = 2.*yy-1.0
	return  (sqrtLn2overPi/gammaD)*strength * y*recSqrtPi * (kha1+xx) / (kha2 + xx*(khb2+xx))
