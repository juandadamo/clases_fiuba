#!/usr/bin/env python3

"""
  srf  ---  spectral response functions
                                       a.k.a.
  ils  ---  instrument line shape

  Evaluate normalized srf/ils on a wavenumber grid
  integral srf(v) dv = 1

  usage:
  srf  [options]  srfValue

  * Gaussian etc response function:
    srfValue = width = HWHM = half width @ half maximum
  * FTS (Fourier transform spectrometer):
    srfValue is interpreted as MOPD
    L = MOPD = maximum optical path difference
    the first zero is at 1/2L
    ===> hwhm = 1/4L approximately

  -h               help
  -c     char      comment character(s) used in output file (default '#')
  -o     string    output file for saving of vGrid and srfValues
  -t     string    type of spectral response function:
                   Gauss (default), Lorentz, Hyperbolic, Triangle, FTS
  -a     char      apodization (for FTS only)
                   default no apodization, i.e. sinc
                   c       cosine
		   g       gaussian
		   q       quartic (Connes)
		   t|b     triangular / Bartlett
		   w|m|s   weak|medium|strong Norton-Beer
		   H       Hamming
		   h       Hanning
  -s     int       sampling rate:  grid point spacing = delta = width/sample
                   default 5
  -x     float     wavenumber grid extension in units of half widths
                   default 10.0
"""

_LICENSE_ = """\n
This file is part of the Py4CAtS package.

Authors:
Franz Schreier
DLR Oberpfaffenhofen
Copyright 2002 - 2019  The Py4CAtS authors

Py4CAtS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Py4CAtS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

####################################################################################################################################
####################################################################################################################################

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

try:                        from scipy.integrate import quad
except ImportError as msg:  raise SystemExit (str(msg) + '\nfrom scipy.integrate import quad failed!')

cos = np.cos
sqrt2 = np.sqrt(2.0)

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.ir import pi, recPi, sqrtPi, ln2, sqrtLn2
from py4cats.aux.misc import trapez
from py4cats.aux.moreFun import j2_over_xx

def Sinc(x):
	""" Sinc(x) = sin(x)/x
	    !!!  WARNING:  numpy.sinc = sin(pi*x)/(pi*x)  WARNING !!!
	"""
	return np.sinc(recPi*x)


# some constants required for the super- and hyper-Gauss
recGammaQuarter = 0.27581566283020931  # 1/Gamma(1/4)
recGammaSixth   = 0.17965203550789732  # 1/Gamma(1/6)
recGammaEighth  = 0.13273264557288264  # 1/Gamma(1/8)
sqrtSqrtLn2     = np.sqrt(sqrtLn2)

####################################################################################################################################
# for more fts apodized response functions, see Sanderson in Rao: Molecular Spectroscopy - Modern Research Vol. I
#                                               http://mathworld.wolfram.com/ApodizationFunction.html
#                                               Table 3.3 in E.Weisz PhD Thesis "Temperature Profiling by ... IASI ..." (Graz, 2001)
#                                               /home/donau101/mirror/IR/documents/iasi/EW-IGAMWissBer-No11-174p-y2001.pdf
####################################################################################################################################

# constants for Norton-Beer (JOSA, 1976) apodization functions paper 1977)
#cw0, cw1, cw2 = 0.5480, -0.833,    0.5353
#cm0, cm1, cm2 = 0.26,   -0.154838, 0.894838
#cs0, cs2, cs4 = 0.09,    0.5875,   0.3225
# constants for Norton-Beer (Errata 1977) apodization functions
cw0, cw1, cw2 = 0.384093, -0.087577, 0.703484
cm0, cm1, cm2 = 0.152442, -0.136176, 0.9837340
cs0, cs2, cs4 = 0.045335,  0.554883, 0.399782

# some special functions required for the Norton-Beer apodized response functions (taken from the Errata paper 1977)
def _q1(x):
	x2i = 1.0/x**2
	return 2.0 * (Sinc(x) - cos(x)) * x2i

def _q2(x):
	x2i = 1.0/x**2
	return -8.0 * ( (1.0-3.0*x2i)*Sinc(x) + 3.0*x2i*cos(x) ) * x2i

def _q4(x):
	x2i = 1.0/x**2
	return  384.0 * ( (1.0-45.0*x2i+105.0*x2i*x2i)*Sinc(x) +5.0*x2i*(2.0-21.0*x2i)*cos(x) ) * x2i*x2i

# expansion to be used for small arguments                 ### TODO:  rewrite using Horner scheme !!!
def _q1a(xx):
	return  (2.0/3.0) * (1.0 - xx/10.0 + xx**2/120.0 - xx**3/5040.)

def _q2a(xx):
        return  (8.0/15.0) * (1.0 - xx/14.0 + xx**2/504.0 - xx**3/33264.)

def _q4a(xx):
	return  (384.0/945.0) * (1.0 - xx/22.0 + xx**2/1144.0 - xx*3/102960.)

def gauss4quad(x):
	""" Gaussian required for the integral defining the Gaussian apodization function, to be solved by quadpack. """
	return np.exp(-0.5*x*x)

def hamming(t):
	""" Hamming instrument function for apodization (without the MOPD prefactor!). """
	return (1.08-0.64*t**2) * Sinc(2.0*pi*t) / (1.0-4.0*t**2)  # 1.08=27./25. and 0.64=16./25.

def hanning1(t):
	""" Hanning instrument function for apodization (without the MOPD prefactor!). """
	return Sinc(2.0*pi*t) / (1.0-4.0*t**2)
def hanning2(t):
	""" Hanning instrument function for apodization (without the MOPD prefactor!). """
	return Sinc(2.0*pi*t) + 0.5*Sinc(2.0*pi*t-pi)  + 0.5*Sinc(2.0*pi*t+pi)


def fts (vGrid, mopd=1.0, apodize='', vShift=0.0, sigma=1.0):
	""" Instrument Line Shape (spectral response function) for Fourier transform spectrometer.
	    Default:      ils = 2 mopd sinc(2 pi mopd v) = sin(2 pi mopd v)/(pi v) unapodized;

	    ARGUMENTS:
	    ----------
	    vGrid:       wavenumber grid
	    mopd:        L = mopd = maximum optical path difference [cm]
	    apodize:     type of apodization function, default None, i.e. ILS = 2*L*sinc(2 pi L vGrid)
	                 triangle, cosine, weak|medium|strong Norton-Beer, Hamming, Hanning, quartic, Gaussian, ...
	                 In most cases case-insensitive, the first letter is sufficient (e.g. 't' for triangular)
	                 except for:
	                 "H"    Hamming
	                 "h"    hanning
	    vShift:      wavenumber shift (default 0.0)
	    sigma:       width for Gaussian apodization

	    RETURNS:
	    --------
	    ils:         a ndarray of response function values with size=len(vGrid)

	    REFERENCES:
	    -----------
	    Weisstein, Eric W. "Apodization Function." From MathWorld -- A Wolfram Web Resource.
	        http://mathworld.wolfram.com/ApodizationFunction.html
            R.H. Norton and C.P. Rinsland. New apodozing functions in Fourier spectrometry.
	        J. Opt. Soc. Am., 66:259-264, 1976. doi: 10.1364/JOSA.66.000259.
            R.H. Norton and R. Beer. Errata: New apodozing functions in Fourier spectrometry.
                J. Opt. Soc. Am., 67:419, 1977. doi: 10.1364/JOSA.67.000419.
	"""

	tpl = 2.0*pi*mopd

	if apodize.lower().startswith('t') or apodize.lower().startswith('b'):  # triangular, Bartlett
		ils = mopd*Sinc(pi*mopd*(vGrid-vShift))**2
	elif apodize.lower().startswith('s'):  # strong Norton-Beer apodization
		x   = tpl*(vGrid-vShift)
		ils = 2.0*mopd * np.where(abs(x)>0.1, cs0*Sinc(x) + cs2*_q2(x) + cs4*_q4(x),  # large x
		                                      (cs0*(1.0-x*x/6.0) + cs2*_q2a(x*x) + cs4*_q4a(x*x)) )
	elif apodize.lower().startswith('m'):  # medium Norton-Beer apodization
		x   = tpl*(vGrid-vShift)
		ils = 2.0*mopd * np.where(abs(x)>0.1, cm0*Sinc(x) + cm1*_q1(x) + cm2*_q2(x),
		                                      cm0*(1.0-x*x/6.0) + cm1*_q1a(x*x)  + cm2*_q2a(x*x) )
	elif apodize.lower().startswith('w'):  # weak Norton-Beer apodization
		x   = tpl*(vGrid-vShift)
		ils = 2.0*mopd * np.where(abs(x)>0.1, cw0*Sinc(x) + cw1*_q1(x) + cw2*_q2(x),
		                                      cw0*(1.0-x*x/6.0) + cw1*_q1a(x*x)  + cw2*_q2a(x*x) )
	elif apodize=='W':  # "new" Norton-Beer weak apodization, avoids multiple sinc and cos calls and uses Horner
		x   = tpl*(vGrid-vShift)
		x2i = 1/x**2
		ils = (cw0 - (3*cw1-15*cw2+45*cw2*x2i)*x2i)*Sinc(x) - 3*x2i*(cw1+15*cw2*x2i)*cos(x)
		ils *= 2.0*mopd
	elif apodize.lower().startswith('c'):                    # cosine apodization
		ils = 4.0*mopd*recPi * cos(tpl*(vGrid-vShift)) / (1.0 - (4.0*mopd*(vGrid-vShift))**2)
	elif apodize.lower().startswith('q'):                    # quartic (Connes) apodization
		ils = 16.0*mopd * j2_over_xx(tpl*(vGrid-vShift))
	elif apodize=='H' or apodize.lower().startswith('ham'):  # hamming apodization
		print('hamming')
		ils = mopd * hamming(mopd*(vGrid-vShift))
	elif apodize=='h' or apodize.lower().startswith('han'):  # hanning apodization
		print('hanning')
		ils = mopd * hanning1(mopd*(vGrid-vShift))
	elif apodize.lower().startswith('g'):  # gaussian apodization
		#ils = 2.0*np.array([quad(gauss4quad, 0.0,mopd, weight='cos', wvar=2.0*pi*v)[0] for v in vGrid])  # shift=0, sigma=1
		#print 'fts_apodize_gauss', mopd, vShift, sigma
		vsGrid = vGrid-vShift
		ils = 2.0*sigma*np.array([quad(gauss4quad, 0.0,mopd/sigma, weight='cos', wvar=2.0*pi*sigma*v)[0] for v in vsGrid])
	else:
		ils = 2.*mopd*Sinc(tpl*(vGrid-vShift))

	return ils


####################################################################################################################################

### NOTE:  there is also a `Gauss` function defined in the lineshapes.py module (with four arguments!)

def Gauss (vGrid, gamma=1.0):
	""" Gauss profile normalized to one (vGrid is wavenumber/frequency;  gamma is HWHM). """
	t = ln2*(vGrid/gamma)**2
	# don't compute exponential for very large numbers > log(1e308)=709.
	# (in contrast to python.math.exp the numpy exp overflows!!!)
	return sqrtLn2/(sqrtPi*gamma) * np.where(t<200.,np.exp(-t),0.0)


####################################################################################################################################

def SuperGauss (vGrid, gamma=1.0):
	""" 'super-Gauss' profile with power of four in the exponent normalized to one (gamma is HWHM). """
	recWidth = 1.0/gamma
	t = ln2*(vGrid*recWidth)**4
	return 2.0*sqrtSqrtLn2 * recWidth * recGammaQuarter * np.where(t<100.,np.exp(-t),0.0)


####################################################################################################################################

def HyperGauss (vGrid, gamma=1.0):
	""" 'hyper-Gauss' profile with power of six in the exponent normalized to one (gamma is HWHM). """
	recWidth = 1.0/gamma
	t = ln2*(vGrid*recWidth)**6
	return 3.0*sqrtLn2**(1/3) * recWidth * recGammaSixth * np.where(t<100.,np.exp(-t),0.0)


####################################################################################################################################

def UltraGauss (vGrid, gamma=1.0):
	""" 'ultra-Gauss' profile with power of eight in the exponent normalized to one (gamma is HWHM). """
	recWidth = 1.0/gamma
	t = ln2*(vGrid*recWidth)**8
	return 4.0*sqrtLn2**0.25 * recWidth * recGammaEighth * np.where(t<100.,np.exp(-t),0.0)


####################################################################################################################################

def Lorentz (vGrid, gamma=1.0):
	""" Lorentz profile normalized to one (gamma is HWHM). """
	return gamma*recPi / (vGrid**2 + gamma**2)


####################################################################################################################################

def Hyperbolic (vGrid, gamma=1.0):
	""" Hyperbolic profile normalized to one (gamma is HWHM). """
	return sqrt2 / (pi*gamma**3 *(vGrid**4 + gamma**4))


####################################################################################################################################

def Triangle (xGrid, hwhm=1.0):
	""" Triangular profile normalized to one (the triangle goes from -2*hwhm to +2*hwhm). """
	xLeft  = -2*hwhm
	xRight =  2*hwhm
	mask      = np.logical_and(xGrid>=xLeft, xGrid<=xRight)
	triangle  = np.where(mask, 1-0.5*abs(xGrid)/hwhm, 0.0) / (2*hwhm)
	return triangle


####################################################################################################################################

def srf (vGrid, srFunction='Gauss', value=1.0):
	""" Evaluate spectral response function on given wavenumber grid and return array of data values.

	vGrid:       wavenumber grid
	srFunction:  name of the spectral reponse function (default Gauss)
	             G = Gauss | L = Lorentz | H = Hyperbolic | T = Triangle | S = SuperGauss | HG = HyperGauss
		     FTS = Fourier transform spectrometer  (apodization selected by second 'word' in this string)
	value        hwhm OR mopd characterizing the spectral response function:
	             hwhm (half width @ half maximum) [1/cm] of (super/hyper) Gauss / Lorentz / Hyperbolic / Triangle
	             mopd (maximum optical path difference) [cm] of FTS

	NOTE:        FTS default without apodization, i.e. sinc
		     add a letter indicating type of apodization function, e.g. "FTS T" for triangular apodization
		     (see function fts documentation)
	"""

	srFun = srFunction.split()
	srFct = srFun[0].upper()

	if srFct.startswith('G'):
		yValues = Gauss(vGrid, value)
	elif srFct.startswith('S'):
		yValues = SuperGauss(vGrid, value)
	elif srFct.startswith('HG') or (srFct.startswith('H') and 'G' in srFct):
		yValues = HyperGauss(vGrid, value)
	elif srFct.startswith('U'):
		yValues = UltraGauss(vGrid, value)
	elif srFct.startswith('L'):
		yValues = Lorentz(vGrid, value)
	elif srFct.startswith('H'):
		yValues = Hyperbolic(vGrid, value)
	elif srFct.startswith('T'):
		yValues = Triangle(vGrid, value)
	elif srFct.startswith('FTS'):
		mopd = value
		if len(srFun==1):  yValues = fts (vGrid, mopd)
		else:              yValues = fts (vGrid, mopd, srFun[1])  # second word selects apodization
	else:
		raise SystemExit ("ERROR --- srf:  unknown/invalid type of spectral response function")

	return yValues


####################################################################################################################################

def _srf_ (srfValue, outFile=None, commentChar='#', srFunction='Gauss', extension=10.0, apodize='', sample=5, checkNorm=False):
	""" Evaluate spectral response function and save grid and data values. """

	if srFunction.upper().startswith('F'):
		srFunction = 'FTS'
		mopd  = srfValue
		width = 0.25/mopd
		print("INFO:  fts type spectral response with L = MOPD = %fcm\n       half width = %fcm-1 (approx)" %
				(mopd, width))
		comments = ['FTS mopd %8.2fcm           ' % mopd,
		            '    hwhm %8.4fcm-1 (approx)' % width]
		if apodize:  comments.append ('    apodization: ' + apodize)
	else:
		width = srfValue
		mopd  = None
		if len(apodize)>0:  print("WARNING:  ignoring apodization,  no meaning for non-FTS response function!")
		comments = ['SRF = %s %10.2fcm-1 hwhm' % (srFunction, width)]

	# grid point spacing (approximately)
	delta = width/sample
	# limits of wavenumber grid
	vMin, vMax = -width*extension,+width*extension
	# wavenumber (or ...) grid
	vGrid =  np.arange(vMin, vMax+delta, delta)

	# evaluate response function
	if srFunction.upper().startswith('F'):  srfValues = srf (vGrid, srFunction, mopd)
	else:                                   srfValues = srf (vGrid, srFunction, width)

	# ... and optionally check the norm
	if checkNorm:
		norm = trapez(vGrid, srfValues)
		infoMsg = 'Norm = integral srf(v) dv = %f' % norm
		comments.append(infoMsg);  print(infoMsg)

	# finally save data to file
	awrite ((vGrid, srfValues), outFile, commentChar=commentChar, comments=comments)


####################################################################################################################################

if __name__ == "__main__":
	from py4cats.aux.aeiou  import awrite
	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       dict(ID='about'),
	       dict(ID='a', name='apodize', type=str, default='',
	                    constraint='apodize.lower()[0] in "bctsmwgqmnh"'),
	       dict(ID='t', name='srFunction', type=str, default='Gauss'),
	       dict(ID='s', name='sample', type=int, default=5,   constraint='sample>0'),
	       dict(ID='x', name='extension',  type=float, default=10.0, constraint='extension>0.0'),
	       dict(ID='n', name='checkNorm')
	       ]

        # parse the command, return (ideally) one float (grid maximum) and some options
	srfValue, options, commentChar, outFile = parse_command (opts,1)

	if 'h'     in options:  raise SystemExit (__doc__ + "\n End of srf help")
	if 'about' in options:  raise SystemExit (_LICENSE_)
	if 'checkNorm' in options:  options['checkNorm']=True

	try:
		srfValue  = float(srfValue[0])
	except ValueError as msg:
		raise SystemExit ("ERROR:  srf function value (width or mopd)\n" + str(msg))

	_srf_ (srfValue, outFile, commentChar, **options)
