#!/usr/bin/env python3

"""  od2ri
  (absorption) optical depth to radiation intensity --- Schwarzschild solver

  usage:
  od2ri   [options]  opticalDepthFile

  -h              help
  -c    char      comment character(s) used in input,output file (default '#')
  -o    string    output file for saving of radiances (if not given: write to StdOut)

 --BoA  float     bottom-of-atmosphere altitude [km]  (read opt.depth only for levels above)
 --ToA  float     top-of-atmosphere altitude [km]     (read opt.depth only for levels below)
                  NOTE:  no interpolation, i.e. integration starts/stops at the next level above/below BoA/ToA
  -C              flag indicating that input optical depth is cumulative (default: difference/delta od)
  -q   char       quadrature method:  trapezoid with default 'T' for "B linear in tau" or 'E' for "B exponential in tau"
  -g   float      ground / surface altitude [km]  (default 0.0)
  -T   float      surface temperature [K]  (default 0)
  -v              flag: verbose
  -x   Interval   lower,upper wavenumbers (comma separated pair of floats [no blanks!],
                                           default set according to range of optical depth in datafiles)
  -S   float      sun zenith angle, SZA (sun at zenith: 0dg, sun at horizon: 90dg)
  -z   float      observer zenith angle (uplooking: 0dg, downlooking 180dg)
                  NOTE:  a horizontal view, i.e. angle=90dg, is not implemented!!!


  WARNING:
  od2ri  does not know the type of optical depth (delta or accumulated ...) given in the input file!
         ===> use the -C flag if you have (ac)cumulated optical depth
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################
####################################################################################################################################

from os.path import isfile, abspath, dirname, join
from math import log as ln

try:   import numpy as np
except ImportError as msg:  raise SystemExit(str(msg) + '\nimport numeric python failed!')

if __name__ == "__main__":
	import sys
	catsDir = dirname(dirname(abspath(__file__)))
	sys.path.append(dirname(catsDir))

from py4cats.aux.pairTypes import Interval, PairOfFloats
from py4cats.aux.cgsUnits import lengthUnits, cgs
from py4cats.aux.aeiou import loadxy, join_words
from py4cats.aux.ir import pi, recPi, radiusSun
from py4cats.aux.moreFun import nexprl, cosdg
from py4cats.art.oDepth import odArray, odRead, od_list2matrix, oDepth_altitudes, cod2dod, oDepth_zpT, odInfo
from py4cats.art.planck import Planck_Wavenumber
from py4cats.art.radInt import riArray, riSave

earth2sun = lengthUnits['au']


####################################################################################################################################
####################################################################################################################################

def _schwarzschild_BexpTau (vGrid, dodMatrix, temperatures, zenithAngle=0.0, omega=0.0):
	""" Quadrature of Schwarzschild integral over optical depth.
	    (equivalent to Planck exponential in optical depth approximation)

	    For input and output arguments, see schwarzschild_BlinTau

	    omega = single scattering albedo = scattering/extinction
	    omega nonzero coresponds to the 'Absorption Approximation' Eq. (3)
	    C. Rathke & J. Fischer
	    Evaluation of four approximate methods for calculating infrared radiances in cloudy atmospheres
	    JQSRT 75, 297--321, 2002, doi: 10.1016/S0022-4073(02)00012-2
	"""

	# NOTE: this function expects an nFreqs*nLayers matrix of delta optical depths,
	# i.e. all optical depths have already been interpolated to a common (the densest) wavenumber grid

	nLevels = dodMatrix.shape[1]+1  # nLayers+1

	# initialize (evtl. to be replaced by surface radiance as function input)
	radiance = 0.0
	# observation angle  (Rathke&Fischer: zenith viewing angle with +mue upwelling, -mue downwelling radiance)
	mue=abs(cosdg(zenithAngle));  secans = 1./mue

	coAlbedo = 1.0-omega  # 1-singleScatteringAlbedo

	if zenithAngle<90.:
		# resort for uplooking geometry:  optical depth starts with 0 at observer
		temperatures = np.flipud(temperatures)
		dodMatrix    = np.fliplr(dodMatrix)
		print(' schwarzschild_BexpTau:  uplooking: flip levels upDown')

	bbLo = Planck_Wavenumber (vGrid, temperatures[0])
	for l in range(1,nLevels):
		print('%2i %6.1fK  %s %10g %10g %8g' %
		      (l, temperatures[l], 'od', min(dodMatrix[:,l-1]), max(dodMatrix[:,l-1]), np.mean(dodMatrix[:,l-1])), end=' ')
		if abs(temperatures[l]-temperatures[l-1])<0.1:
			dodMatrix[:,l] = dodMatrix[:,l-1] + dodMatrix[:,l]
			print('level', l, '  identical temp', temperatures[l],temperatures[l-1], '  merging!')
			continue  # with next level, i.e. skip rest of loop body
		bbHi       = Planck_Wavenumber (vGrid, temperatures[l])
		beta       = np.log(bbLo/bbHi)/dodMatrix[:,l-1]
		transLayer = np.exp(-secans*coAlbedo*dodMatrix[:,l-1])
		radiance   = radiance*transLayer + coAlbedo*(bbLo*transLayer-bbHi)/(mue*beta-coAlbedo)
		print('-->  I  %10g %10g %8g' % (min(radiance), max(radiance), np.mean(radiance)))
		bbLo     = bbHi.copy()

	# sum optical depth over all layers
	transmission = np.exp(-dodMatrix.sum(1))

	return radiance, transmission


####################################################################################################################################

def _schwarzschild_BlinTau (vGrid, dodMatrix, temperatures, zenithAngle=0.0):
	""" Evaluate Schwarzschild integral equation:  integrate Planck * exp(-tau) d_tau  with tau=opticalDepth.

	    ARGUMENTS:
	    ----------
	    vGrid:          wavenumber (frequency) grid
	    dodMatrix:      delta/differential/layer optical depth "matrix"
	    temperatures:   temperature at the levels (layer bounds)
	    zenithAngle:    viewing angle: 0dg=uplooking, 180dg=downloooking/nadir

	    RETURNS:
	    --------
	    radiance:      intensity in erg/s/(cm^2 sr cm-1) as a function of wavenumber
	    transmission:  total transmission as a function of wavenumber
	    """

	# NOTE: this function expects an nFreqs*nLayers matrix of delta optical depths,
	# i.e. all optical depths have already been interpolated to a common (the densest) wavenumber grid

	if zenithAngle>90.:  # nadir view - downlooking
		temperatures = np.flipud(temperatures)
		dodMatrix   = np.fliplr(dodMatrix)
		mue = -cosdg(zenithAngle)
	elif 0.0<=zenithAngle<90.:  # zenith view - uplooking
		mue = cosdg(zenithAngle)
	else:
		raise SystemExit ('ERROR --- schwarzschild_BlinTau:  invalid observer zenith angle = 90dg (no "horizontal view")')

	if abs(mue-1.0)>0.001:
		dodMatrix = dodMatrix/mue

	# initialize
	bbLo = Planck_Wavenumber (vGrid, temperatures[0])
	bbHi = Planck_Wavenumber (vGrid, temperatures[-1])
	radiance    = bbLo - bbHi*np.exp(-dodMatrix.sum(1))
	sumOptDepth = np.zeros_like(vGrid)

	# loop over all layers
	for l in range(dodMatrix.shape[1]):
		bbHi         = Planck_Wavenumber (vGrid, temperatures[l+1])
		deltaBB      = bbHi - bbLo
		radiance    += deltaBB * nexprl(dodMatrix[:,l]) * np.exp(-sumOptDepth)
		sumOptDepth += dodMatrix[:,l]
		bbLo         = bbHi.copy()

	# total transmission
	transmission = np.exp(-sumOptDepth)

	return radiance, transmission


####################################################################################################################################

def _schwarzschild_ (vGrid, dodMatrix, temperatures, obsAngle, sunAngle, tSurface, rSurface, irrad, omega, verbose=False):
	""" Evaluate Schwarzschild integral equation:  integrate Planck * exp(-tau) d_tau  with tau=opticalDepth.
	    Optionally add attenuated surface and/or space radiation. """

	if isinstance(omega,float):
		radiance, transmission = _schwarzschild_BexpTau (vGrid, dodMatrix, temperatures, obsAngle, omega)
	else:
		radiance, transmission = _schwarzschild_BlinTau (vGrid, dodMatrix, temperatures, obsAngle)

	if obsAngle>90.0:
		if verbose:  print ('\nupwelling thermal emission:  %19.3g <= I <= %10g' % (min(radiance), max(radiance)))
		if tSurface>0.0:
			deltaRad = (1.0-rSurface) * transmission * Planck_Wavenumber(vGrid,tSurface)
			# add surface thermal emission to atmospheric signal
			radiance +=  deltaRad
			if verbose:  print ('plus %5.1fK surface (e=%6.3f):       %10.3g <= I <= %10g' %
			                    (tSurface, 1.0-rSurface, min(deltaRad), max(deltaRad)))
		if rSurface>0.0:
			# atmospheric thermal emission downwelling to the surface
			if isinstance(omega,float):
				radDown = _schwarzschild_BexpTau (vGrid, dodMatrix, temperatures, 180-obsAngle, omega)[0]
			else:
				radDown = _schwarzschild_BlinTau (vGrid, dodMatrix, temperatures, 180-obsAngle)[0]
			# specular reflection
			deltaRad  = rSurface * transmission*radDown
			radiance += deltaRad
			if verbose:  print ('plus downwelling reflected emission:  %10.3g <= I <= %10g' %
			                    (min(deltaRad), max(deltaRad)))
			# add solar irradiance attenuated on double path ToA-surface-observer
			if isinstance(irrad,np.ndarray):
				trans2sun = np.exp(-dodMatrix.sum(1)/cosdg(sunAngle))  # downwelling to surface
				deltaRad  = rSurface*cosdg(sunAngle)*recPi *irrad * transmission*trans2sun
				radiance += deltaRad
				if verbose: print ('plus irradiance:   %29.3g <= I <= %10g   with trans sun2earth: %10.5f <= T <= %10.5f' %
                                                   (min(deltaRad), max(deltaRad), min(trans2sun), max(trans2sun)))
	else:
		if isinstance(irrad,np.ndarray):
			if verbose:  print ('\ndownwelling thermal emission:  %10.3g <= I <= %10g' % (min(radiance), max(radiance)))
			deltaRad  = irrad*transmission
			radiance += deltaRad
			if verbose:
				print ('irradiance @ ToA:  %10.3g <= I <= %10g' % (min(irrad), max(irrad)))
				print ('plus irradiance:   %10.3g <= I <= %10g' % (min(deltaRad), max(deltaRad)))
	return radiance, transmission


####################################################################################################################################

def _beer_ (vGrid, dodMatrix, obsAngle, sunAngle, tSurface, rSurface, irradiance, verbose=False):
	""" Evaluate Beer transmission and compute radiance as attenuated surface and/or space radiation. """
	if obsAngle>90.0:
		totalOptDepth = +dodMatrix.sum(1)                              # vertical
		transUp   = np.exp(+totalOptDepth/cosdg(obsAngle))             # upwelling (note the plus sign because cosdg<0.0)
		if rSurface>0 and isinstance(irradiance,np.ndarray):           # double path ToA --> surface --> observer
			transDown    = np.exp(-totalOptDepth/cosdg(sunAngle))  # downwelling
			transmission = transUp*transDown
			radiance     = rSurface*cosdg(sunAngle)*recPi * irradiance * transmission
			if verbose:
				print ('\nirradiance @ ToA:  %26.3g <= I <= %10g' % (min(irradiance), max(irradiance)))
				print ('reflected attenuated sun:  %18.3g <= I <= %10g' % (min(radiance), max(radiance)))
				print ('trans obs2earth:   %26.5f <= T <= %10.5f' % (min(transUp),   max(transUp)))
				print ('trans sun2earth:   %26.5f <= T <= %10.5f' % (min(transDown), max(transDown)))
			if tSurface>0:
				deltaRad  = transUp* (1.0-rSurface) * Planck_Wavenumber(vGrid, tSurface)
				radiance += deltaRad
				if verbose:  print ('plus %5.1fK surface:      %19.3g <= I <= %10g' %
				                    (tSurface, min(deltaRad), max(deltaRad)))
		elif rSurface>=0 and tSurface>0:                               # single path surface --> observer
			transmission = transUp
			radiance     = transUp * (1.0-rSurface) * Planck_Wavenumber(vGrid, tSurface)
			if verbose:  print ('\nupwelling attenuated %5.1fK surface emission only (%.2f emissivity)' %
			                    (tSurface, 1.0-rSurface))
		else:
			raise ValueError ("ERROR --- _beer_: Beer radiance, but zero surface and space emission")
	else:
		if not irradiance:
			raise ValueError ("ERROR --- _beer_: downwelling radiance for uplooking observer, but zero space emission")
		transmission = np.exp(-dodMatrix.sum(1)/cosdg(obsAngle))
		radiance = transmission*irradiance
	return radiance, transmission


####################################################################################################################################

def oDepth_subLayer (dodList, zSurface):
	""" 'Interpolate' bottom layer optical depth and return a (true) copy of the opt depth list. """

	zGrid = oDepth_altitudes(dodList)

	if zSurface<250.0:
		zSurface = cgs('km', zSurface)
		print('\nWARNING --- od2ri:  zSurface very small, assuming kilometer units')
	elif zSurface<zGrid[0]:   raise ValueError ("ERROR --- dod2ri:  ground/bottom altitude below BoA!")
	elif zSurface>zGrid[-1]:  raise ValueError ("ERROR --- dod2ri:  ground/bottom altitude above ToA!")
	elif zSurface>zGrid[-2]:  print ("WARNING --- dod2ri:  surface (ground/bottom)altitude very large, near ToA")

	if abs(zGrid[0]-zSurface)<1000.0:  return dodList  # almost BoA (<10m)

	# locate appropriate layer and determine linear interpolation weight
	lSurface = np.searchsorted(zGrid, zSurface)  # returns upper level, i.e. l with zGrid[l-1]<zSurface<=zGrid[l]
	q = (zSurface-zGrid[lSurface-1])/(zGrid[lSurface]-zGrid[lSurface-1])

	if abs(q-1.0)<0.001:  lSurface+=1

	dodList = dodList[lSurface-1:]  # probably "suboptimal" w.r.t. extra memory

	if 0.0<q<1.0:  # replace data at layer bottom level
		tSurface =        (1.0-q)*dodList[0].t.left     + q*dodList[0].t.right
		pSurface = np.exp((1.0-q)*ln(dodList[0].q.left) + q*ln(dodList[0].q.right))
		ySurface = (1.0-q) * dodList[0].base     # crude approximation, maybe some better "interpolation" scheme!?!
		dodList[0] = odArray(ySurface,
		                     dodList[0].x,
		                     PairOfFloats(zSurface,dodList[0].z.right),
		                     PairOfFloats(pSurface,dodList[0].p.right),
		                     PairOfFloats(tSurface,dodList[0].t.right),
				     dodList[0].N)
	print ('dod2ri  zSurface=%.2fkm    l=%i    q=%f' % (cgs('!km',zSurface), lSurface, q), end=' ---> ')
	odInfo(dodList[0])

	return dodList


####################################################################################################################################

def unpack_oDepths (dodList, zSurface=None):
	""" Unpack opt. depth list, evtl. skip bottom layer(s) and 'interpolate' lowest layer. """

	if   isinstance(zSurface,(int,float)):
		zGrid = oDepth_altitudes(dodList)
		# check surface height and 'consistency' with opt. depth layer bounds
		if zSurface<250.0:
			zSurface = cgs('km', zSurface)
			print('\nWARNING --- od2ri:  zSurface very small, assuming kilometer units')
		if   abs(zGrid[0]-zSurface)<1000.0:  pass
		elif zSurface<zGrid[0]:   raise ValueError ("ERROR --- dod2ri:  ground/bottom altitude below BoA!")
		elif zSurface>zGrid[-1]:  raise ValueError ("ERROR --- dod2ri:  ground/bottom altitude above ToA!")
		elif zSurface>zGrid[-2]:  print ("WARNING --- dod2ri:  surface (ground/bottom)altitude very large, near ToA")
		# locate appropriate layer and determine linear interpolation weight
		lSurface = np.searchsorted(zGrid, zSurface)  # returns upper level, i.e. l with zGrid[l-1]<zSurface<=zGrid[l]
		q = (zSurface-zGrid[lSurface-1])/(zGrid[lSurface]-zGrid[lSurface-1])
		if abs(q-1.0)<0.001:  lSurface+=1
	elif zSurface is None:
		zSurface=0.0
		lSurface=1
		q =0
	else:
		raise ValueError ('ERROR --- dod2ri:  invalid surface (ground) altitude %s' % zSurface)

	# unpack opt. depth list, evtl. skipping bottom layer(s)
	# 1. interpolate all data to common, densest grid (use this call sequence because of some initial checks)
	vGrid, dodMatrix = od_list2matrix (dodList[lSurface-1:])
	# 2. extract "attributes"
	zGrid, pGrid, tData = oDepth_zpT(dodList[lSurface-1:])

	if 0.0<q<1.0:  # replace data at layer bottom level
		zGrid[0] = zSurface
		tData[0] =        (1.0-q)*tData[0] + q*tData[1]
		pGrid[0] = np.exp((1.0-q)*ln(pGrid[0]) + q*ln(pGrid[1]))
		dodMatrix[:,0] *= (1.0-q)

	return zGrid, pGrid, tData, vGrid, dodMatrix


####################################################################################################################################

def dod2ri (dodList, obsAngle=0.0, tSurface=None, rSurface=0.0, zSurface=None, sunAngle=0.0, space=None,
            omega=None, mode=None, verbose=False):
	""" Evaluate Schwarzschild or Beer integral equation(s), optionally add surface or space radiation and return radiance.

	    ARGUMENTS:
	    ----------

	    dodList:       delta (layer) optical depths (a list of odArray's)
	    obsAngle:      observer viewing zenith angle: 0dg=uplooking (default) ... 180dg=downloooking/nadir
                           NOTE:  a horizontal view with 90dg is not implemented!!!
	    tSurface:      surface temperature [K], default None
	                   ===> Schwarzschild mode: no background contribution, only atmospheric thermal emission
	                        Beer mode:          use BoA temperature
	                   To set tSurface to the bottom-of-atmos temperature, set tSurface=-1 or tSurface='BoA'
	    rSurface:      surface reflectivity, equivalent to one-emissivity:  R=1-E; default 0.0 (i.e. full emissivity)
	    zSurface:      ground/bottom altitude (default None, i.e. use BoA altitude)
	    sunAngle:      solar zenith angle: 0dg=zenith (default) ... 90dg=horizon
	    space:         temperature [K] or irradiance spectrum, default None
	                   * positive floats:  simply assume I_space = Planck (v,space)
	                   * negative float (approx -6000):  convert to irradiance as
			     I_space = pi*(sunRadius/sun2earth)**2*planck(v,-space)
	                   * file(name):  read extraterrestrial irradiance spectrum from file (e.g. Kurusz)
	                   * 'one':   assume I_space=1.0 to compute sun-normalized radiance
	    omega:         single scattering albedo (default None)
	                   if 0<=omega<=1 automatically select mode Schwarzschild with B exponential in opt.depth
			   To-Do:  replace by scattering coefficient!
	    mode:          Schwarzschild | Beer | ...
	                   default None, i.e. choose automatically
	                   Schwarzschild for v<=2500cm-1   (lambda>=4.00mue)
	                   Beer              v> 3000cm-1   (lambda< 3.33mue)

	    RETURNS:
	    --------
	    riArray:       a subclassed numpy array with radiance along with some attributes

	    NOTE: ?!? outdated ?!?
	    -----
	    dod2ri  evaluates the Schwarzschild (and Beer) integral from the first to the last altitude,
	    i.e. lowest to highest for an uplooking observer (zenith<90dg) and highest to lowest for nadir view.
	    For a surface located between atmospheric levels (grid points), a "primitive" interpolation is done.
	    If you want radiance for an observer somewhere in between, give only the subset of relevant odArray's.
	    However, dod2ri does not consider an observer between the levels (layer bounds).
	    In particular, no support for an airborne observer (esp. downlooking with surface reflection in NIR/SWIR).
	 """

	# some initial argument checks (dodList ist checked soon)
	if not (0<=obsAngle<90.0 or 90.0<obsAngle<=180.0): raise ValueError ("ERROR --- dod2ri:  invalid observer zenith angle!")

	# surface reflectivity
	if not 0.0<=rSurface<=1.0:  raise ValueError ('ERROR --- dod2ri:  invalid surface reflectivity outside [0,1]')

#####   under construction --- BAUSTELLE --- Work in Progress --- vägarbete pågår!
# old   if   isinstance(zSurface,(int,float)):  dodList = oDepth_subLayer(dodList, zSurface)
# old   elif zSurface is None:                  zSurface=0

	# 1. interpolate all data to common, densest wavenumber grid
	# 2. extract "attributes"
	# 3. optionally (for zSurface>0) skip some bottom layer(s) and 'interpolate' the lowest layer
	zGrid, pGrid, tData, vGrid, dodMatrix = unpack_oDepths(dodList, zSurface)

	# Schwarzschild or Beer or ....
	if mode:
		mode = mode.capitalize().strip()
	else:
		if   vGrid.max()<2500.0:  mode = 'Schwarzschild'
		elif vGrid.min()>3000.0:  mode = 'Beer'
		else:  raise ValueError ('ERROR ---  dod2ri:  undefined mode in transition region, select "Schwarzschild|Beer|..."')

	# surface temperature
	if isinstance(tSurface,(int,float)):
		if 100.0<tSurface<=180.0 and obsAngle>180.0:
			print ("\nWARNING --- dod2ri:  did you mix up background temperature %f and zenith angle %f?!?"
			       % (tSurface, obsAngle))
		if 1e2<tSurface<=1e3:          pass
		elif abs(tSurface+1.0)<0.001:  tSurface = tData[0]
		else:                          raise ValueError ('ERROR --- dod2ri: surface extremely cold (<100K) or hot (>1000K)')
	elif isinstance(tSurface,str) and  tSurface.lower()=='boa':
		tSurface = tData[0]
	else:
		tSurface = 0.0

	# space (solar,stellar) temperature or spectrum
	if isinstance(space,(int,float)):
		if space>0.0:  # for example microwave background radiation 2.725K
			irradiance = Planck_Wavenumber(vGrid,space)
		elif -8000.0<=space<=-3000.0:  # assume Planck with appropriate radiance-irradiance conversion
			irradiance = pi*(radiusSun/earth2sun)**2 * Planck_Wavenumber(vGrid,abs(space))
		else:
			raise ValueError ('ERROR --- dod2ri:  invalid space temperature %s' % space)
	elif isinstance(space,str):
		if   isfile(space):                    irradiance = np.interp(vGrid, *loadxy(space))  # loadxy returns 2 arrays x,y
		elif space.lower() in ['one', 'snr']:  irradiance = np.ones_like(vGrid)               # sun normalized radiance
		else:  raise ValueError ("ERROR --- dod2ri:  `space` string, expected either a file or `one`")
	elif space is None:
		irradiance=0.0
	else:
		raise ValueError ('ERROR --- dod2ri:  invalid space temperature %s' % space)

	if verbose:
		print ("\ndod2ri %s:   %.1fdg,   T_surface=%.1fK,   r=%.2f" % (mode, obsAngle, tSurface, rSurface))
		for l in range(dodMatrix.shape[1]):  print ('%3i %8.1fkm  %12g <=dod<= %12g   <dod> %11g' %
		                    (l, cgs('!km',zGrid[l+1]), min(dodMatrix[:,l]), max(dodMatrix[:,l]), np.mean(dodMatrix[:,l])))

	if   mode.startswith("S"):
		# bExpTau = 'e' in mode.lower() or isinstance(omega,float)
		if isinstance(omega,(int,float)):
			if 0.0<=omega<=1.0:  omega=float(omega)
			else:                raise ValueError ("ERROR --- dod2ri:  bad single scattering albedo, not in [0,1]")
		else:
			if 'E' in mode.upper() or 'X' in mode.upper():  omega=0.0  # good luck, no 'E' or 'X' in "Schwarzschild"

		radiance, transmission =  _schwarzschild_ (vGrid, dodMatrix, tData, obsAngle, sunAngle,
		                                           tSurface, rSurface, irradiance, omega, verbose)
	elif mode.startswith("B"):
		mode = "Beer"
		radiance, transmission =  _beer_ (vGrid, dodMatrix, obsAngle, sunAngle, tSurface, rSurface, irradiance, verbose)
	else:
		raise SystemExit ('ERROR --- dod2ri:  undefined/invalid/unknown mode, select "Schwarzschild|Beer|..."')

	print ('dod2ri:  %10s  ===> %19.3g <= I <= %10g   with %10.3g <= T <= %8f  (and  %i points)' %
	       (mode, min(radiance), max(radiance), min(transmission), max(transmission), len(vGrid)))

	return riArray (radiance, Interval(vGrid[0],vGrid[-1]),  tSurface, obsAngle,
	                Interval(zGrid[0],zGrid[-1]), Interval(pGrid[0],pGrid[-1]), Interval(tData.min(),tData.max()))


####################################################################################################################################
####################################################################################################################################

def _od2ri_ (odFile, radFile=None, commentChar='#', cumOD=False,
             zToA=0.0, zBoA=0.0, xLimits=None,
             obsAngle=0., tSurface=None, rSurface=None, zSurface=0.0, sunAngle=0.0, space=None, omega=None,
             mode='Schwarzschild', verbose=False):

	# read optical depth including some attributes, nb. temperature
	odList = odRead (odFile, zToA, zBoA, xLimits, commentChar, verbose)

	if cumOD: odList = cod2dod(odList)     # Subtract consecutive cumulated optical depths to delta (layer) optical depths

	# if omega:  raise SystemExit('od2ri --- sorry, single scattering albedo not implemented')

	# assumes delta/layer optical depth as input
	radiance = dod2ri (odList, obsAngle, tSurface, rSurface, zSurface, sunAngle, space, omega, mode, verbose)

	# save radiance intensity
	riSave (radiance, radFile, commentChar='#')


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions

	# parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       dict(ID='about'),
	       dict(ID='C', name='cumOD'),
	       dict(ID='g', name='zSurface', type=float, constraint='zSurface>0.0'),  # ground altitude
	       dict(ID='BoA', name='zBoA', type=float, constraint='zBoA>0.0'),
	       dict(ID='ToA', name='zToA', type=float, constraint='zToA>0.0'),
	       dict(ID='x', name='xLimits', type=Interval, constraint='xLimits.lower>=0.0'),
	       dict(ID='m', name='mode', type=str, default='Schwarzschild'),
	       dict(ID='T', name='tSurface', type=float, default=None, constraint='tSurface>0.0'),
	       dict(ID='r', name='rSurface', type=float, default=0.0, constraint='0.0<rSurface=1.0'),
	       dict(ID='w', name='omega', type=float, default=0.0, constraint='omega>=0.0'),
	       dict(ID='z', name='obsAngle', type=float, default=0., constraint='0<=obsAngle<=180. and abs(obsAngle-90.)>0.1'),
	       dict(ID='S', name='sunAngle', type=float, default=0., constraint='0<=sunAngle<90.'),
	       dict(ID='v', name='verbose')]

	inFiles, options, commentCharacter, outFile = parse_command (opts,1)

	if 'h' in options:  raise SystemExit (__doc__ + "\n End of od2ri help")
	if 'about' in options:
		raise SystemExit (join_words(open(join(catsDir,'license.txt')).readlines()))

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID']=='h')]
	for key in boolOptions:                   options[key] = key in options

	_od2ri_ (inFiles[0], outFile, commentCharacter, **options)
