"""
Rayleigh cross section, absorption coefficient, and optical depth.
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

from py4cats.aux.ir import k as kBoltzmann
from py4cats.aux.aeiou import join_words
from py4cats.art.absCo import acArray
from py4cats.art.oDepth import odArray

# frequently used constants
e4 = 1e4;     em4 = 1e-4;     em28 = 1e-28

# select the functions to be imported  (thus 'from rayleigh import *' will not import the constants!)
__all__ = [xs+'_crossSection' for xs in 'bodhaine bucholtz nicolet CO2 H2'.split()] + 'totalAirOpticalDepth rayleighx'.split()

####################################################################################################################################

a0=3.9729066e0;  a1=4.6547659e0;  a2=4.5055995e-4;  a3=2.3229848e-5

def bad_bates_crossSection (vGrid):
	""" Rayleigh scattering cross sections.

	    Empirical fit by D.R. Bates "Rayleigh Scattering by Air" PSS 32, 785-790 (1984)
	    (Sara Seager: Exoplanet Atmospheres - Physical Processes, Section 8.4, Eq. (8.63)  (Princeton University Press, 2010)
	    Accuracy 0.3% for 0.205 < lambda < 1.05mue

	    WARNING:  clear discrepancy of Bates compared to all other approximations
	"""
	recLambda  = em4*vGrid                                 # wavenumber[cm-1] ---> 1/wavelength[mue]
	recLambda2 = recLambda**2
	xsRay      = em28 * recLambda2**2 * (a0 + recLambda2*(a1 + recLambda2*(a2+recLambda2*a3)))
	return xsRay


####################################################################################################################################

n0 = 1.0456;  nv = -341.29e-8;   nl = -0.9023       # numerator constants
d0 = 1.0;     dv = 0.002706e-8;  dl = -85.9686      # denominator constants

def bodhaine_crossSection (vGrid):
	""" Rayleigh scattering cross sections.
	    Empirical fit by Bodhaine et al., JAOT, 16(11), pp. 1854-1861, 1999
	    Implementation Eq. (2) of Yan et al., Int. J. Astrobiology, 14(2), pp. 255-266, 2015 """
	lambda2 = (e4/vGrid)**2                             # wavenumber[cm-1] ---> wavelength[mue]
	xsRay   = em28 * ((n0 + nv*vGrid**2 + nl*lambda2) / (d0 + dv*vGrid**2 + dl*lambda2))
	return xsRay


####################################################################################################################################

aL=4.01061e-28;  bL=3.99668;  cL=11.0298;  dL=2.71393e-6   # NOTE:  cL and dL include a factor 10^4 for mue -> cm

def bucholtz_crossSection (vGrid):
	""" Rayleigh scattering cross sections.
	    Empirical fit by A. Bucholtz (Appl.Opt. 34(15), pp.2765-2773, 1995, Eq. (8)
	    !!! longwave approximation only for nu<20000cm-1 or lambda>0.5mue
	"""
	x     = bL + cL/vGrid + dL*vGrid  # longwave exponent
	xsRay = aL * (em4*vGrid)**x       # wavelength ~ 1/wavenumber:  1/lambda[mue] = nu[cm-1]/10^4
	return xsRay


####################################################################################################################################

def nicolet_crossSection (vGrid):
	""" Rayleigh scattering cross sections.
	    Empirical fit by M. Nicolet, PSS 32, 1467-1468 (1984)
	    !!! longwave approximation only for nu<18181cm-1 or lambda>0.55mue
	"""
	xsRay = 4.02e-28 * (em4*vGrid)**4.004
	return xsRay


####################################################################################################################################

def CO2_crossSection (vGrid):
	""" Rayleigh scattering cross sections for a CO2 atmosphere.

	    E. Marcq et al. (Icarus, 2011, doi 10.1016/j.icarus.2010.08.021)       xs_CO2 [cm2] = 12.4e-27 * (532.24/lambda[nm])**4
	"""
	xsRay = 12.4e-27 * (532.24e-3*em4*vGrid)**4
	return xsRay


####################################################################################################################################

def H2_crossSection (vGrid):
	""" Rayleigh scattering cross sections for a H2 atmosphere.

	    Sara Seager: Exoplanet Atmospheres - Physical Processes, Section 8.4, Eq. (8.64)  (Princeton University Press, 2010)
	    Dalgarno&Williams: Rayleigh Scattering by Molecular Hydrogen (Astrophys. J., 136, 690-692, 1962)
	    see also A. Caldas et al. A&A 2019 or http://de.arxiv.org/abs/1901.09932,  appendix D
	    see also K. Heng, A cloudiness index for transiting exoplanets ...., APJL, 826:L16 (6pp), 2016 July 20
	    see also P. Cubillos: PyRatBay, arxiv 2105.05598 --> MNRAS
	"""
	# Ray = 8.14e-13/lambdaNM**4 + 1.28e-6/lambdaNM**6 + 1.61/lambdaNM**8  # lambdaNM = 1e8/vGrid
	# Ray = 8.14e-45*vGrid**4 + 1.28e-56*vGrid**6 + 1.61e-64*vGrid**8
	vv = vGrid**2
	xsRay = vv**2 * (8.14e-45 + vv*(1.28e-56 + vv*1.61e-64))
	return xsRay


####################################################################################################################################
####################################################################################################################################

def totalAirOpticalDepth (vGrid):
	""" Earth atmosphere total optical depth.
	Zdunkowski, Trautmann, Bott:  Radiation in the atmosphere (Cambridge 2007): Eq. (11.12). """
	lambdaMueSq = (e4/vGrid)**2  # wavelength [mue] squared
	return 0.008569/lambdaMueSq**2 * (1.0 + 0.0113/lambdaMueSq + 0.00013/lambdaMueSq**2)


####################################################################################################################################

# WARNING:  there is also a `rayleigh` function in numpy.random !!!
#########>  renamed the main function below rayleigh ---> rayleighx

rxsDict = {'bodhaine':  bodhaine_crossSection,
           'bucholtz':  bucholtz_crossSection,
           'nicolet':   nicolet_crossSection,
	   'CO2':       CO2_crossSection,
	   'H2':         H2_crossSection}

def rayleighx (data, factor=1.0, model='nicolet'):                                                         # rayleigh eXtinction
	""" Compute Rayleigh molecular cross section and add extinction to optical depth(s) or absorption coefficient(s).

	    If a list of ac or od is given, add Rayleigh individually and return the corresponding list.
	    Optionally the extinction can be scaled by a "Rayleigh enhancement factor" (default 1.0).
	"""

	if model in rxsDict.keys():
		rayFunction = rxsDict[model]
	else:
		raise KeyError ("ERROR --- rayleigh:  invalid model selected, choose from " + join_words(rxsDict.keys()))

	if  not (isinstance(factor,(int,float)) and factor>0):
		raise ValueError ('rayleighx --- expected a positive number as Rayleigh enhancement factor')

	if   isinstance(data,(list,tuple)):
		# recursive call
		return [rayleighx(spec, factor, model) for spec in data]
	elif isinstance(data,acArray):
		vGrid = data.grid()
		ac    = data.base
		air   = data.p/(kBoltzmann*data.t)  # data.molec['air']
		return acArray(ac+factor*air*rayFunction(vGrid), data.x, data.z, data.p, data.t, data.molec)
	elif isinstance(data,odArray):
		vGrid    = data.grid()
		od       = data.base
		column   = data.N
		return odArray(od+factor*column*rayFunction(vGrid), data.x, data.z, data.p, data.t, data.N)
	else:
		raise SystemExit ("ERROR --- rayleigh:  incorrect data type,\n" +
		                  "                      expected absorption coefficient, optical depth, or list thereof")


####################################################################################################################################
#################################################  comparison of optical depths  ###################################################

#from rayleigh import totalAirOpticalDepth, rayleighx

#mls=atmRead('/data/atm/15/mls.xy', zToA=25)

#vRange = Interval(5400,6600)
#hwhm = 1.0  # spectral response function
#dll = higstract('/data/hitran/86/lines', vRange+5*hwhm+5, 'main')

#dodl = lbl2od(mls,dll,vRange+hwhm) # monochrom in the extended interval
#tod  = dod2tod(dodl)  # total atmosphere opt depth
## convolve with Gauss
#todg = tod.convolve(hwhm,'G')

#uGrid = tod.grid()
#vGrid = todg.grid()

#todgr = rayleighx(todg)  # bad trapez quadrature estimate for entire atmosphere

#rayZTB = totalAirOpticalDepth(vGrid)  # Zdunkowski, Trautmann, Bott estimate

#dodlr = rayleighx(dodl)   # rayleigh layer by layer
#todgr2 = dod2tod(dodlr).convolve(hwhm,'G')

#odPlot ((todg,todgr,todgr2))
#semilogy(vGrid, todg.base+rayZTB, '--')
#legend(['lbl opt depth', 'lbl plus Rayleigh-Bodhaine total', 'lbl plus Rayleigh-Bodhaine by layer','lbl+ZTB'])
#title ('MLS with ToA 25km, Hitran86')

#################################################  comparison of cross sections  ###################################################

#vGrid = linspace(1000.0,20000.0,191)
#xsBates = bad_bates_crossSection(vGrid)
#xsBodhaine = bodhaine_crossSection(vGrid)
#xsBucholtz = bucholtz_crossSection(vGrid)
#xsNicolet = nicolet_crossSection(vGrid)
#xsCO2 = CO2_crossSection(vGrid)
#xsH2 = H2_crossSection(vGrid)

#plot(vGrid,xsBates, 'r+', label='bates')
#plot(vGrid,xsBodhaine, 'b--', label='bodhaine')
#plot(vGrid,xsBucholtz, 'g-.', label='bucholtz')
#plot(vGrid,xsNicolet, 'k:', label='nicolet')
#plot(vGrid,xsCO2, 'c:', label='CO2')
#plot(vGrid,xsH2, 'm:', label='H2')
#legend()

# ===> clear discrepancy of Bates compared to all other approximations
#      Bodhaine, Bucholtz, and Nicolet (visually) agree for v>5000cm-1 (Bodhaine deviates for smaller v)
