#!/usr/bin/env python3

"""  od2limb

  vertical optical depth to effective height

  usage:
  od2limb   [options]  opticalDepthFile

  -h              help
  -c    char      comment character(s) used in input,output file (default '#')
  -o    string    output file for saving of radiances (if not given: write to StdOut)

  -C              flag indicating that input optical depth is cumulative (default: difference/delta/layer od)
  -R              planetary radius (default Earth 6371.23e5cm)
  -s              flag: include surface limb path (i.e. zero tangent point altitude) with zero transmission
  -v              flag: verbose
  -x   Interval   lower,upper wavenumbers (comma separated pair of floats [no blanks!],
                                           default set according to range of optical depth in datafiles)

  WARNING:
  od2limb  does not know the type of optical depth (delta or accumulated ...) given in the input file!
           ===> use the -C flag if you have (ac)cumulated optical depth
"""

_LICENSE_ = """\n
This file is part of the Py4CAtS package.

Authors:
Franz Schreier
DLR-IMF Oberpfaffenhofen
Copyright 2002 - 2021  The Py4CAtS authors

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


try:   import numpy as np
except ImportError as msg:  raise SystemExit(str(msg) + '\nimport numeric python failed!')

if __name__ == "__main__":
	from os.path import abspath, dirname
	import sys
	catsDir = dirname(dirname(abspath(__file__)))
	sys.path.append(dirname(catsDir))

from py4cats.aux.ir import radiusEarth  # 6371.23e5 cm
from py4cats.aux.pairTypes import Interval, PairOfFloats
from py4cats.aux.cgsUnits import cgs
from py4cats.aux.aeiou import awrite
from py4cats.aux.misc import monotone
from py4cats.art.oDepth import odArray, odRead, cod2dod, od_list2matrix, flipod, oDepth_zpT
from py4cats.aux.convolution import convolveBox, convolveTriangle, convolveGauss

####################################################################################################################################
####################################################################################################################################

def _dod2limbOD_ (zGrid, dodMatrix, lTangent, rEarth):
	""" Transform vertical layer optical depth to corresponding limb optical depth and sum over all layers.

	    ARGUMENTS:
	    ----------
	    zGrid:         altitude grid
	    dodMatrix:     delta/differential/layer optical depth "matrix"
	    lTangent:      index of tangent point in altitude grid array

	    RETURNS:
	    --------
	    optDepth:      total optical depth along horizontal line-of-sight
	    """

	# NOTE: this function expects a nFreqs*nLayers matrix of delta optical depths,
	# i.e. all optical depths have already been interpolated to a common (the densest) wavenumber grid

	# horizontal path steps
	zTangent = zGrid[lTangent]
	sqrtDeltaRadii = np.sqrt(zGrid[lTangent:]-zTangent)
	sqrtSumRadii   = np.sqrt(2.0*rEarth+zGrid[lTangent:]+zTangent)
	deltaS = sqrtSumRadii[:-1] * np.ediff1d(sqrtDeltaRadii)

	# loop over all layers above tangent
	lod = np.zeros(dodMatrix.shape[0])
	for l in range(lTangent,dodMatrix.shape[1]):
		lod += 2*dodMatrix[:,l] * deltaS[l-lTangent] / (zGrid[l+1]-zGrid[l])

	print ('dod2limbOD: tangent %8.1fkm @ level # %3i  ---> path length %9.1fkm ===> %g <od< %g' %
	       (cgs('!km',zTangent), lTangent, cgs('!km',sum(deltaS)),min(lod), max(lod)))

	return lod


####################################################################################################################################

def dod2limb (dodList, zTangent=None, rEarth=radiusEarth):
	""" Given vertical optical depths, compute limb optical depth.

 	    ARGUMENTS:
 	    ----------

 	    dodList:       delta (layer) optical depths (a list of odArray's)
 	    zTangent:      tangent altitude
	                   Default: None ===> compute od for all tangent points = levels
                           NOTE:  values <=200 are interpreted as kilometer
 	    rEarth:        radius [cm] of Earth (default 6371.23e5cm) or planet

 	    RETURNS:
 	    --------
 	    odArray:       a subclassed numpy array with optical depth along with some attributes
	                   or a list of odArrays (for a list of tangents)

	    NOTE:
	    -----
	    * dod2limb assumes a symmetric path space-tangent-space
	    * tangent points inside layers, i.e. different from levels, not supported (no interpolation etc.)
 	"""

 	# initial checks
	if not (isinstance(dodList,(list,tuple)) and all([isinstance(od,odArray) for od in dodList])):
		raise SystemExit ("ERROR --- dod2limb:  expected a list of odArray's, but ...")

	# extract "attributes"
	zGrid, pGrid, temperature = oDepth_zpT(dodList)

	# viewing direction
	if monotone(zGrid)>0 and monotone(pGrid)<0:
		print('\n INFO --- dod2limb:  dodList (%i layers) sorted from BoA @ %.1f -> ToA @ %.1fkm' %
		      (len(dodList), cgs('!km',zGrid[0]), cgs('!km',zGrid[-1])))
	elif monotone(zGrid)<0 and monotone(pGrid)>0:
		print('\n INFO --- dod2limb:  dodList (%i layers) sorted from ToA @ %.1f -> BoA @ %.1fkm\n%s\n' %
		      (len(dodList), cgs('!km',zGrid[0]), cgs('!km',zGrid[-1]), ' flip up<-->down!'))
		dodList = flipod(dodList)
		zGrid   = np.flipud(zGrid)
		pGrid   = np.flipud(pGrid)
		temperature = np.flipud(temperature)
	else:
		raise SystemExit ("ERROR --- dod2lt:  both z and p grid monotone in or decreasing!")

	# interpolate all data to common, densest grid
	vGrid, dodMatrix = od_list2matrix (dodList)

	if isinstance(zTangent,(int,float)):
		if zTangent<200:
			print(' INFO --- od2limb:  zTangent =', zTangent, 'very small, probably km, converted to', end=' ')
			zTangent = cgs('km',zTangent);  print(zTangent, 'cm')
		if zTangent<zGrid[0]:
			raise ValueError ("%s %f %s %f" %
			              ('tangent point', cgs('!km',zTangent), 'below lowest altitude grid point', cgs('!km',zGrid[0])))
		if zTangent>zGrid[-2]:
			raise ValueError ("%s %f %s %f" %
			    ('tangent point', cgs('!km',zTangent), 'above second highest altitude grid point', cgs('!km',zGrid[-2])))

		# locate the tangent point
		lTangent = zGrid.searchsorted(zTangent)
		if abs(zTangent-zGrid[lTangent])<1000:
			print ('tangent', zTangent, lTangent, zGrid[lTangent], pGrid[lTangent])
		else:
			print ("\nlayer optical depths for", len(zGrid), "altitude grid points", cgs('!km',zGrid))
			raise ValueError ("%s %.2fkm %s" % ('tangent point', cgs('!km',zTangent), 'not an altitude grid point!'))

		limbOD = _dod2limbOD_ (zGrid, dodMatrix, lTangent, rEarth)
		return odArray(limbOD, Interval(vGrid[0],vGrid[-1]),
		               PairOfFloats(zGrid[lTangent],zGrid[-1]),
		               PairOfFloats(pGrid[lTangent],pGrid[-1]),
			       PairOfFloats(temperature[lTangent],temperature[-1]), -1.0)     # VCD yet undefined
	else:
		lodList = []
		for k,zT in enumerate(zGrid[:-1]):
			if zT<=0.0:  continue  # only "true" limb paths with tangent altitude zT>0 allowed
			limbOD = _dod2limbOD_ (zGrid, dodMatrix, k, rEarth)
			lodList.append(odArray(limbOD, Interval(vGrid[0],vGrid[-1]),
			                       PairOfFloats(zT,zGrid[-1]),
			                       PairOfFloats(pGrid[k+1],pGrid[-1]),
			                       PairOfFloats(temperature[k+1],temperature[-1]), -1.0))     # VCD yet undefined
		return lodList

####################################################################################################################################

def dod2eh (dodList, rEarth=radiusEarth, surface=False, wavenumbers=False, kilometer=False, srf=None, hwhm=1.0):
	""" Given vertical optical depths, compute effective height spectrum.

	    ARGUMENTS:
 	    ----------

 	    dodList:       delta (layer) vertical optical depths (a list of odArray's)
 	    rEarth:        radius of Earth (default 6371.23e5cm) or planet
	    surface:       flag (default False):  add limb path with zero transmission hitting surface
	    wavenumbers:   flag (default False):  return wavenumber grid and effective height
	    kilometer:     flag (default False):  return effective height in km (instead of cm)
	    srf:           spectral response function, default None, choices: B[ox], T[riangle], G[auss]
	    hwhm:          half width (at half maximum) of srf

 	    RETURNS:
 	    --------
	   [vGrid:          optional, only when `wavenumbers' flag is set]
 	    effHgt:         a numpy array of effective heights
	"""

	lodList = dod2limb(dodList, rEarth=rEarth)
	zTangents = np.array([od.z.left for od in lodList])
	print (len(lodList), "optDepths for zTangent", zTangents)

	if surface:
		# lodList[0] is the very first "true" optical depth with nonzero tangent height zTangents[0]
		# the first/lowest limb path hits the surface, hence zero transmission
		effHgt = (1.0-0.5*np.exp(-lodList[0].base)) * zTangents[0]
		for k,od in enumerate(lodList[:-1]):
			odNext  = lodList[k+1]
			effHgt += (1.0-0.5*(np.exp(-od.base)+np.exp(-odNext.base))) * (zTangents[k+1]-zTangents[k])
	else:
		# only "true" limb paths with positive tangent point altitudes are considered
		effHgt = np.zeros_like(lodList[0].base)
		#for k,od in enumerate(lodList[:-1]):
		#	odNext  = lodList[k+1]
		#	effHgt += (1.0-0.5*(np.exp(-od.base)+np.exp(-odNext.base))) * (zTangents[k+1]-zTangents[k])
		absLower = 1.0 - np.exp(-lodList[0].base)
		for k,od in enumerate(lodList[1:]):
			absUpper = 1.0 - np.exp(-od.base)
			effHgt  += 0.5*(absLower+absUpper) * (zTangents[k+1]-zTangents[k])
			absLower = absUpper

	if isinstance(srf,str):
		if   srf.lower().startswith('b'):
			wGrid, eh  =  convolveBox(lodList[0].grid(), effHgt, hwhm)
		elif srf.lower().startswith('t'):
			wGrid, eh  =  convolveTriangle(lodList[0].grid(), effHgt, hwhm)
		elif srf.lower().startswith('g'):
			wGrid, eh  =  convolveGauss(lodList[0].grid(), effHgt, hwhm)
		else:
			raise SystemExit ("ERROR --- limb2eh:  unknown/invalid srf!")
		effHgt = eh
	else:
		wGrid = lodList[0].grid()  # only needed when wavenumber is returned, too

	if kilometer:    effHgt *= cgs('!km')

	if wavenumbers:  return wGrid, effHgt
	else:            return effHgt


####################################################################################################################################
####################################################################################################################################

def _od2limb_ (odFile, ehFile=None, commentChar='#', cumOD=False,
               xLimits=None, rEarth=radiusEarth, surface=False, wavenumbers=True, kilometer=True, verbose=False):

	# read optical depth including some attributes, nb. temperature
	zToA = zBoA = None
	odList = odRead (odFile, zToA, zBoA, xLimits, commentChar, verbose)

	if cumOD:  # Subtract consecutive cumulated optical depths to delta (layer) optical depths
		odList = cod2dod(odList)

	# assumes delta/layer optical depths as input
	vGrid, effHgt = dod2eh (odList, rEarth, surface, wavenumbers, kilometer)

	awrite ((vGrid,effHgt), ehFile, commentChar=commentChar)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions

	# parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
               dict(ID='about'),
	       dict(ID='C', name='cumOD'),
	       dict(ID='x', name='xLimits', type=Interval, constraint='xLimits.lower>=0.0'),
               dict(ID='R', name='rEarth', type=float, default=radiusEarth, constraint='rEarth>0.0'),
               dict(ID='v', name='verbose'),
               dict(ID='s', name='surface'),
               dict(ID='km', name='kilometer')
               ]

	inFiles, options, commentChar, outFile = parse_command (opts,1)

	if 'h' in options:  raise SystemExit (__doc__ + "\n End of od2limb help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID']=='h')]
	for key in boolOptions:                   options[key] = key in options

	_od2limb_ (inFiles[0], outFile, commentChar, **options)
