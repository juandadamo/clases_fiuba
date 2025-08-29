""" Rational Approximations for the Complex Error Function
    w(z)  =  (i/pi) integral exp(-t**2) / (z-t) dt  =  K(x,y) + iL(x,y)

J. Humlicek.
An efficient method for evaluation of the complex probability function
J. Quant. Spectrosc. & Radiat. Transfer, 21, 309­313, 1979.

J. Humlicek.
Optimized computation of the Voigt and complex probability function
J. Quant. Spectrosc. & Radiat. Transfer, 27, 437­444, 1982.

J.A.C. Weideman.
Computation of the Complex Error Function.
SIAM J. Num. Anal., 31, 1497­1518, 1994.

F. Schreier.
Optimized Implementations of Rational Approximations for the Voigt and Complex Error Function.
J. Quant. Spectrosc. & Radiat. Transfer, 112(6), 1010-1025, 2011;  doi 10.1016/j.jqsrt.2010.12.010

F. Schreier.
The Voigt and complex error function: Humlicek's rational approximation generalized.
Mon. Not. Roy. Astron. Soc., 479(3), 3068-3075, 2018;  doi 10.1093/mnras/sty1680
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

from math import sqrt

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

from . ir import recSqrtPi  # 1.0/np.sqrt(np.pi) = 0.5641895835477563

####################################################################################################################################

# public functions (i.e. do not import the constants):
__all__ = 'weideman24a weideman32a  hum1wei24 hum2wei32  zpf16h hum1zpf16m hum2zpf16m'.split()

####################################################################################################################################

L24 = sqrt(24.)/2**0.25  # sqrt(N)/2^(1/4)
a24 = np.array([0.000000000000e+00,
               -1.5137461654527820e-10,  4.9048217339494231e-09,  1.3310461621784953e-09, -3.0082822751372376e-08,
               -1.9122258508123359e-08,  1.8738343487053238e-07,  2.5682641345559087e-07, -1.0856475789744469e-06,
               -3.0388931839363019e-06,  4.1394617248398666e-06,  3.0471066083229116e-05,  2.4331415462599627e-05,
               -2.0748431511424456e-04, -7.8166429956141891e-04, -4.9364269012799368e-04,  6.2150063629501625e-03,
                3.3723366855316406e-02,  1.0838723484566790e-01,  2.6549639598807695e-01,  5.3611395357291292e-01,
                9.2570871385886777e-01,  1.3948196733791201e+00,  1.8562864992055403e+00,  2.1978589365315413e+00])
w24polynom = np.poly1d(a24)

def weideman24a (x,y):
	""" Complex error function using Weideman's rational approximation:
	    J.A.C. Weideman (SIAM-NA 1994); equation (38.I) for N=24 and table I.
	    Maximum relative error: 2.6e-1 for 0<x<25 and 1e-8<y<1e2
	                            2.6e-3 for 0<x<25 and 1e-6<y<1e2
				    2.6e-5 for 0<x<25 and 1e-4<y<1e2. """
	iz  = 1j*x - y
	lpiz = L24 + iz  # wL - y + x*complex(0.,1.)
	lmiz = L24 - iz  # wL + y - x*complex(0.,1.)
	recLmiZ  = 1.0 / lmiz
	Z       = lpiz * recLmiZ
	# Horner scheme and numpy.poly1d are equivalent in speed
	w24 = (recSqrtPi + 2.0*recLmiZ*w24polynom(Z)) * recLmiZ
	#w24 = recLmiZ * (recSqrtPi + 2.0*recLmiZ*(a24[24]+(a24[23]+(a24[22]+(a24[21]+(a24[20]+(a24[19]+(a24[18]+(a24[17]+(a24[16]+(a24[15]+(a24[14]+(a24[13]+(a24[12]+(a24[11]+(a24[10]+(a24[9]+(a24[8]+(a24[7]+(a24[6]+(a24[5]+(a24[4]+(a24[3]+(a24[2]+a24[1]*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r))
	return w24


####################################################################################################################################

L32 = sqrt(32./sqrt(2.)) #  sqrt(N)/2^(1/4) = 4.1195342878142354
a32 = np.array([  0.0,
                 -1.3031797863050087e-12,  3.7425618160114027e-12,  8.0313811157139980e-12, -2.1542712058675306e-11,
                 -5.5441734536643139e-11,  1.1658311815931910e-10,  4.1537467909691372e-10, -5.2310171472225164e-10,
                 -3.2080150445812361e-09,  8.1248959990509739e-10,  2.3797557087250343e-08,  2.2930439611554623e-08,
                 -1.4813078890306988e-07, -4.1840763665439336e-07,  4.2558331384172769e-07,  4.4015317319188437e-06,
                  6.8210319440019865e-06, -2.1409619200778492e-05, -1.3075449254551185e-04, -2.4532980269917430e-04,
                  3.9259136070117972e-04,  4.5195411053501472e-03,  1.9006155784845689e-02,  5.7304403529837913e-02,
                  1.4060716226893769e-01,  2.9544451071508926e-01,  5.4601397206393498e-01,  9.0192548936480166e-01,
                  1.3455441692345453e+00,  1.8256696296324824e+00,  2.2635372999002676e+00,  2.5722534081245696e+00])
w32polynom = np.poly1d(a32)

def weideman32a (x,y):
	""" Complex error function using Weideman's rational approximation:
	    J.A.C. Weideman (SIAM-NA 1994); equation (38.I) for N=32 and table I.
	    Maximum relative error: 2.6e-4 for 0<x<25 and 1e-8<y<1e2
	                            2.6e-6 for 0<x<25 and 1e-6<y<1e2. """
	iz  = 1j*x - y
	lpiz    = L32 + iz  # wL - y + x*complex(0.,1.)
	lmiz    = L32 - iz  # wL + y - x*complex(0.,1.)
	recLmiZ = 1.0 / lmiz
	Z       = lpiz * recLmiZ
	w       = (recSqrtPi  +  2.0 * w32polynom(Z) * recLmiZ)  *  recLmiZ
	return w


####################################################################################################################################

L40 = sqrt(40./sqrt(2.)) #  sqrt(N)/2^(1/4) = 4.1195342878142354
a40 = np.array([  0.0,
                 -1.7289454976371382e-15,  1.6557302589815397e-15,  1.1731887704363968e-14, -5.0037190226678157e-15,
                 -7.0485405495744577e-14,  1.3964447037065922e-14,  4.5367930883081557e-13,  1.2029823135314783e-13,
                 -2.9077185104142701e-12, -2.7263524771115047e-12,  1.7715251487970820e-11,  3.4728309117326715e-11,
                 -9.0550589249005505e-11, -3.5632288231113309e-10,  2.1086039581064143e-10,  3.0177809806630764e-09,
                  3.2497466717629210e-09, -1.8315616101549639e-08, -6.3517734399942688e-08,  1.4198643161211778e-08,
                  5.9121369553882812e-07,  1.4835661137925159e-06, -1.0660138983004151e-06, -1.8007447144285152e-05,
                 -5.5913092642348801e-05, -3.9393631453655214e-05,  4.3980701598771805e-04,  2.7054056330749685e-03,
                  1.0048186242784123e-02,  2.9202916471242974e-02,  7.1823617790743685e-02,  1.5504263802479584e-01,
                  2.9989437996150059e-01,  5.2665289882771060e-01,  8.4721745765938250e-01,  1.2563815675765149e+00,
                  1.7253830848179783e+00,  2.2015137948783128e+00,  2.6160541527618602e+00,  2.8996245093897057e+00])
w40polynom = np.poly1d(a40)

def weideman40a (x,y):
	""" Complex error function using Weideman's rational approximation:
	    J.A.C. Weideman (SIAM-NA 1994); equation (38.I) for N=40 and table I.
	    Maximum relative error: 9e-7 for 0<x<25 and 1e-8<y<1e2. """
	iz  = 1j*x - y
	lpiz    = L40 + iz  # wL - y + x*complex(0.,1.)
	lmiz    = L40 - iz  # wL + y - x*complex(0.,1.)
	recLmiZ = 1.0 / lmiz
	Z       = lpiz * recLmiZ
	w       = (recSqrtPi  +  2.0 * w40polynom(Z) * recLmiZ)  *  recLmiZ
	return w


####################################################################################################################################
#####                                                                                                                          #####
#####    Combinations of Humlicek (1982) asymptotic approximations with Weideman rational approximation                        #####
#####                                                                                                                          #####
####################################################################################################################################

def hum1wei24 (x,y):
	""" Complex error function combining Humlicek's and Weideman's rational approximations:

	    |x|+y>15:  Humlicek (JQSRT, 1982) rational approximation for region I;
	    else:      J.A.C. Weideman (SIAM-NA 1994); equation (38.I) and table I.

	    F. Schreier, JQSRT 112, pp. 1010-1025, 2011:  doi: 10.1016/j.jqsrt.2010.12.010 """

	# For safety only. Probably slows down everything. Comment it if you always have arrays (or use assertion?).
	# if isinstance(x,(int,float)):  x = np.array([x])  # %timeit ===> 84 us ± 400 ns per loop  ! same time without check !
	# x = np.atleast_1d(x)                              # %timeit ===> 83.9 us ± 267 ns per loop

	t = y - 1j*x
	w = t * recSqrtPi / (0.5 + t*t)  # Humlicek (1982) approx 1 for s>15

	if y<15.0:
		mask = abs(x)+y<15.       # returns true for interior points
		iz  = -t[np.where(mask)]  # returns small complex array covering only the interior region
		# the following five lines are only evaluated for the interior grid points
		lpiz = L24 + iz  # wL - y + x*complex(0.,1.)
		lmiz = L24 - iz  # wL + y - x*complex(0.,1.)
		recLmiZ  = 1.0 / lmiz
		Z       = lpiz * recLmiZ
		w24 = (recSqrtPi + 2.0*recLmiZ*w24polynom(Z)) * recLmiZ
		# replace asympotic Humlicek approximation by Weideman rational approximation in interior center region
		np.place(w, mask, w24)
	return w


def hum2wei32 (x,y):
	""" Complex error function combining Humlicek's and Weideman's rational approximations:

	    |x|+y>10.0: Humlicek (JQSRT, 1982) rational approximation for region II;
	    else:       J.A.C. Weideman (SIAM-NA 1994); equation (38.I) and table I.

	    F. Schreier, JQSRT 112, pp. 1010-1025, 2011:  doi: 10.1016/j.jqsrt.2010.12.010 """

	# For safety only. Probably slows down everything. Comment it if you always have arrays (or use assertion?).
	# if isinstance(x,(int,float)):  x = np.array([x])
	# x = np.atleast_1d(x)

	z  = x+1j*y
	zz = z*z
	w  = 1j* (z * (zz*recSqrtPi-1.410474)) / (0.75 + zz*(zz-3.0))

	if y<10.0:
		mask = abs(x)+y<10.0           # returns true for interior points
		iz  = 1j*x[np.where(mask)]-y  # returns small complex array covering only the interior region
		# the following five lines are only evaluated for the interior grid points
		lpiz = L32 + iz  # wL - y + x*complex(0.,1.)
		lmiz = L32 - iz  # wL + y - x*complex(0.,1.)
		recLmiZ  = 1.0 / lmiz
		Z       = lpiz * recLmiZ
		w32 = (recSqrtPi + 2.0*recLmiZ*w32polynom(Z)) * recLmiZ
		# replace asympotic Humlicek approximation by Weideman rational approximation in interior center region
		np.place(w, mask, w32)
	return w


##########################    the following two versions are slower, but can be used for contour plots    ##########################

def hum1wei24w (x,y, sCut=15.0):
	""" Complex error function combining Humlicek's and Weideman's rational approximations:
	    Version to be used for contour plots (e.g. error plots)

	    |x|+y>15.0: Humlicek (JQSRT, 1982) rational approximation for region I;
	    else:       J.A.C. Weideman (SIAM-NA 1994); equation (38.I) and table I. """

	t = y - 1j*x
	return np.where(abs(x)+y>=sCut,
	                t * recSqrtPi / (0.5 + t*t),  # Humlicek (1982) approx 1 for large s
	                weideman24a(x,y))             # Weideman (1994) n=24 approximation for small s


def hum2wei32w (x,y, sCut=10.0):
	""" Complex error function combining Humlicek's and Weideman's rational approximations:
	    Version to be used for contour plots (e.g. error plots)

	    |x|+y>10.0: Humlicek (JQSRT, 1982) rational approximation for region II;
	    else:       J.A.C. Weideman (SIAM-NA 1994); equation (38.I) and table I.

	    F. Schreier, JQSRT 112, pp. 1010-1025, 2011:  doi: 10.1016/j.jqsrt.2010.12.010 """

	z  = x+1j*y
	return np.where(abs(x)+y>sCut,
	                1j* (z * (z*z*recSqrtPi-1.410474)) / (0.75 + z*z*(z*z-3.0)),
	                weideman32a(x,y))

####################################################################################################################################
#####                                                                                                                          #####
#####    Optimized (single fraction) Humlicek (JQSRT 1979) region I rational approximation for n=16 and delta=1.31183          #####
#####                                                                                                                          #####
####################################################################################################################################

aa  = np.array([ 41445.0374210222,
                 -136631.072925829j,
                 -191726.143960199,
                 268628.568621291j,
                 173247.907201704,
                 -179862.56759178j,
                 -63310.0020563537,
                 56893.7798630723j,
                 11256.4939105413,
                 -9362.62673144278j,
                 -1018.67334277366,
                 810.629101627698j,
                 44.5707404545965,
                 -34.5401929182016j,
                 -0.740120821385939,
                 0.564189583547714j])  # identical to 1/sqrt(pi) except for last two digits

bb  = np.array([ 7918.06640624997, 0.0,
                 -126689.0625,     0.0,
                 295607.8125,      0.0,
                 -236486.25,       0.0,
                 84459.375,        0.0,
                 -15015.0,         0.0,
                 1365.0,           0.0,
                 -60.0,            0.0,
                 1.0])

# The poly1d class assumes that the very first coefficient corresponds to the highest power of z and the last coefficient to z**0
numPoly16 = np.poly1d(np.flipud(aa))
denPoly16 = np.poly1d(np.flipud(bb))        # %timeit denPoly16(z)  10000 loops, best of 3:  28 microseconds per loop
denPoly8e = np.poly1d(np.flipud(bb[::2]))   # %timeit denPoly8(z*z) 100000 loops, best of 3: 16.5 microseconds per loop

def zpf16p (z):                                                ### one complex argument (required by rautian, sdv, sdr functions) !!!
	""" Humlicek (JQSRT 1979) complex probability function for n=16 and delta=1.31183.
	    Generalization described in MNRAS 479(3), 3068-3075, 2018, doi: 10.1093/mnras/sty1680

	    Optimized rational approximation using numpy.poly1d  (applicable for all z=x+iy). """
	Z  = z + 1.31183j
	ZZ = Z*Z
	return numPoly16(Z)/denPoly8e(ZZ)    # %timeit zpf16p    10000 loops, best of 3: 173 microseconds per loop


def zpf16h (x,y):                                                                                         ### two real arguments !!!
	""" Humlicek (JQSRT 1979) complex probability function for n=16 and delta=1.31183.
	    Generalization described in MNRAS 479(3), 3068-3075, 2018, doi: 10.1093/mnras/sty1680

	    Optimized rational approximation with Horner scheme  (applicable for all z=x+iy). """
	Z  = x +1j*(y + 1.31183)
	ZZ = Z*Z
	return (((((((((((((((aa[15]*Z+aa[14])*Z+aa[13])*Z+aa[12])*Z+aa[11])*Z+aa[10])*Z+aa[9])*Z+aa[8])*Z+aa[7])*Z+aa[6])*Z+aa[5])*Z+aa[4])*Z+aa[3])*Z+aa[2])*Z+aa[1])*Z+aa[0]) \
            / ((((((((ZZ+bb[14])*ZZ+bb[12])*ZZ+bb[10])*ZZ+bb[8])*ZZ+bb[6])*ZZ+bb[4])*ZZ+bb[2])*ZZ+bb[0])


####################################################################################################################################
#####  Humlicek (1979,1982) combinations (with real arguments x,y and Horner scheme for zpf16)                                 #####
#####  !! these versions are useful for line-by-line 'number crunching' (each line has its own xGrid, but a common y) !!       #####
#####  !! (To compute a "matrix" of complex error function values (e.g. for contour plot of errors) use where instead of mask) #####
####################################################################################################################################

def hum1zpf16m (x, y, s=15.0):
	""" Complex error function w(z)=w(x+iy) combining Humlicek's rational approximations:

	    |x|+y>15:  Humlicek (JQSRT, 1982) rational approximation for region I;
	    else:      Humlicek (JQSRT, 1979) rational approximation with n=16 and delta=y0=1.31183

	    Version using a mask and np.place;  two real arguments x,y.  """

	# For safety only. Probably slows down everything. Comment it if you always have arrays (or use assertion?).
	# if isinstance(x,(int,float)):  x = np.array([x])

	t = y-1j*x
	w = t * recSqrtPi / (0.5 + t*t)  # Humlicek (1982) approx 1 for s>15

	if y<s:
		mask  = abs(x)+y<s                      # returns true for interior points
		Z     = x[np.where(mask)]+ 1j*(y+1.31183)  # returns small complex array covering only the interior region
		ZZ    = Z*Z
		numer = (((((((((((((((aa[15]*Z+aa[14])*Z+aa[13])*Z+aa[12])*Z+aa[11])*Z+aa[10])*Z+aa[9])*Z+aa[8])*Z+aa[7])*Z+aa[6])*Z+aa[5])*Z+aa[4])*Z+aa[3])*Z+aa[2])*Z+aa[1])*Z+aa[0])
		denom = (((((((ZZ+bb[14])*ZZ+bb[12])*ZZ+bb[10])*ZZ+bb[8])*ZZ+bb[6])*ZZ+bb[4])*ZZ+bb[2])*ZZ+bb[0]
		np.place(w, mask, numer/denom)
	return w


def hum2zpf16m (x, y, s=10.0):
	""" Complex error function w(z)=w(x+iy) combining Humlicek's rational approximations:

	    |x|+y>10:  Humlicek (JQSRT, 1982) rational approximation for region II;
	    else:      Humlicek (JQSRT, 1979) rational approximation with n=16 and delta=y0=1.31183

	    Version using a mask and np.place;  two real arguments x,y. """

	# For safety only. Probably slows down everything. Comment it if you always have arrays (or use assertion?).
	# if isinstance(x,(int,float)):  x = np.array([x])

	z  = x+1j*y
	zz = z*z
	w  = 1j* (z * (zz*recSqrtPi-1.410474)) / (0.75 + zz*(zz-3.0))

	if y<s:
		mask  = abs(x)+y<s                         # returns true for interior points
		Z     = x[np.where(mask)]+ 1j*(y+1.31183)  # returns small complex array covering only the interior region
		ZZ    = Z*Z
		numer = (((((((((((((((aa[15]*Z+aa[14])*Z+aa[13])*Z+aa[12])*Z+aa[11])*Z+aa[10])*Z+aa[9])*Z+aa[8])*Z+aa[7])*Z+aa[6])*Z+aa[5])*Z+aa[4])*Z+aa[3])*Z+aa[2])*Z+aa[1])*Z+aa[0])
		denom = (((((((ZZ+bb[14])*ZZ+bb[12])*ZZ+bb[10])*ZZ+bb[8])*ZZ+bb[6])*ZZ+bb[4])*ZZ+bb[2])*ZZ+bb[0]
		np.place(w, mask, numer/denom)
	return w
