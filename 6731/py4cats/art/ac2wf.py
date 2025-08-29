#!/usr/bin/env python3

"""  ac2wf
  computation of weighting funtions given absorption coefficients along with atmospheric data (altitudes)

  usage:
  ac2wf  [options]  absCoFile

  -h               help
  -c     char      comment character(s) used in input,output file (default '#')
  -o     string    output file for saving of optical depth (if not given: write to StdOut)
                   (if the output file's extension ends with ".nc", ".ncdf" or ".netcdf",
                    a netcdf file is generated, otherwise the file is ascii tabular)

  -a     float     observer angle [dg]
                   uplooking = 0dg ... nadir downlooking = 180dg (default)
  -z     float     observer altitude [km]
                   NOTE:  no interpolation, i.e. integration starts/stops at the next level above/below BoA/ToA
  -t               save transposed weighting functions, i.e. distance (altitude) vs WF
                   (default:  save wavenumber grid (first column) vs WF (following columns))
  -x     Interval  lower,upper wavenumbers (comma separated pair of floats [no blanks!],
                                            default set according to range of lines in datafiles)
 --fov   string    type of field-of-view function:  Gauss | Box | Rectangle
  -w     float     width of field-of-view, HWHM in dg, default 1.0

"""

_LICENSE_ = """\n
This file is part of the Py4CAtS package.

Authors:
Franz Schreier
DLR-IMF Oberpfaffenhofen
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

##############################################################################################################

from math  import sqrt

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.ir import recSqrtPi
from py4cats.aux.cgsUnits import cgs
from py4cats.aux.pairTypes import Interval
from py4cats.aux.aeiou import minmaxmean, cstack
from py4cats.aux.moreFun import cosdg
from py4cats.aux.misc import monotone

from py4cats.art.absCo import acArray, acRead, ac_list2matrix
from py4cats.art.oDepth import od_list2matrix, oDepth_altitudes
from py4cats.art.wgtFct import wfArray, wfSave

####################################################################################################################################

def ac2wf (acList, angle=180., zObs=None, FoV=None):
	""" Absorption coefficient to weighting functions.

	    ARGUMENTS:
	    ----------
	    acList:      list of (molecular) absorption coefficients [1/cm]  (acArray's with x, z, p, T as attributes)
	    angle:       zenith viewing angle [dg]:  0dg=uploooking ... 180dg downlooking default
	    zObs:        observer altitude [cm] (if very small, assume km and scale by 1e5 to get cm)
	                 if unspecified: assume ToA for angle>90dg, i.e. nadir corresponding to the default angle
			                 assume BoA for angle<90dg
	    FoV:         string giving field-of-view type and width (HWHM in dg), e.g. "Gauss 1.0" (default None)
	                 (computed by sum of three pencil beams)
	    RETURNS:
	    --------
	    wgtFct       a numpy array matrix along with attributes

	    NOTE:
	    -----
	    the shape of the weighting function array returned depends on the number of levels above/below the observer
	"""

	if isinstance(FoV,str):
		# Weighting functions for finite field-of-view evaluated recursively
		try:
			fovType, fovWidth = FoV.split()
			fovWidth = float(fovWidth)
		except ValueError as msg:
			raise SystemExit (str(msg) +
			                 'ERROR --- ac2wf:  parsing FoV specification failed, expecting string and float\n')
		else:
			print('INFO --- ac2wf:  FoV type', fovType, 'width HWHM [dg]', fovWidth)

		# for an odd number of pencil beams the center beam node is simply zero
		if   fovType.lower().startswith('gau'):  # 2D Gaussian (Gauss-Hermite quadrature: AbSt64:(25.4.46))
			node    = +1.22474487
			weights = [recSqrtPi * 1.18163590, recSqrtPi * 0.29540898]
		elif fovType.lower().startswith('box'):  # 2D box (Gauss-Chebyshev II quadrature: AbSt64:(25.4.40))
			node    = +1.0/sqrt(2.0)
			weights = [0.5, 0.25]
		elif fovType.lower().startswith('rec'):  # 1D box
			node    = +1.0
			weights = [2./3., 1./6.]
		else:
			raise SystemExit ('ERROR --- ac2wf:  invalid/unknown type of field-of-view function')

		# center pencil beam
		print('ac2wf --- center', weights[1])
		wfCenter = ac2wf (acList, angle, zObs)
		# off-center pencil beams
		print('ac2wf --- lower',  node, angle+node*fovWidth, weights[1])
		wfPlus = ac2wf (acList, angle+node*fovWidth, zObs)
		print('ac2wf --- upper', -node, angle-node*fovWidth, weights[1])
		wfMinus = ac2wf (acList, angle-node*fovWidth, zObs)
		# combine the three beams
		sGrid = wfCenter.grid()    ### possibly not required ?!?
		wgtFct = weights[0]*wfCenter.base + weights[1]*(wfPlus.base+wfMinus.base)
	else:
		# convert list of acArray's to matrix and also return wavenumber grid; retrieve altitudes, too
		zGrid = np.array([ac.z for ac in acList])
		vGrid, absCo = ac_list2matrix (acList)

		if not isinstance(zObs,(int,float)):        # use ToA or BoA if no observer altitude is given
			if   angle>90.0:  zObs = zGrid[-1]  # downlooking
			elif angle<90.0:  zObs = zGrid[0]   # uplooking
			else:             raise SystemExit ('ERROR --- ac2wf:  observer angle 90dg from zenith, i.e. horizontal')
		elif zObs<0.:
			raise SystemExit ('ERROR --- ac2wf:  negative observer height')
		elif zObs<250.:
			zObs = cgs('km',zObs)
			print('WARNING --- ac2wf:  zObs very small, assuming kilometer units')
		elif zObs>max(zGrid):
			raise SystemExit ('%s %f' % ('ERROR --- ac2wf:  observer above ToA @', zGrid[-1]))

		print (' %s %i*%i %s %.1f-%.2fkm %s %.2fkm %s %.2fdg' %
		       ('ac2wf --- pencilbeam: ',   absCo.shape[0],absCo.shape[1], 'wavenumbers * levels with zGrid',
		       cgs('!km',zGrid[0]), cgs('!km',zGrid[-1]), '   observer @', cgs('!km',zObs),'with', angle))

		# find index of level next to observer
		lObs = np.argmin(abs(zGrid-zObs))
		if abs(zGrid[lObs]-zObs)>10e3:
			print('WARNING --- ac2wf:  observer more than 1000cm away from next atmospheric level')
			print('                    computing weighting function w.r.t. ', 1e-5*zGrid[lObs], 'km')

		# get rid of 'unwanted' data above downlooking or below uplooking observer
		if   angle>90.0:  # downlooking
			if lObs<1: raise SystemExit ('ERROR --- ac2wf:  downlooking, but observer very low')
			if angle>180.0: raise SystemExit ('ERROR --- ac2wf:  viewing angle > 180dg')
			zGrid  = zGrid[:lObs+1]
			sGrid  = (zGrid[::-1]-zGrid[lObs]) / cosdg(angle)  # distance from observer (flip and subtract)
			ac     = np.fliplr(absCo[:,:lObs+1])                    # also flip up-down
			deltaS = np.ediff1d(sGrid)
		elif angle<90.0:  # uplooking
			if lObs>len(zGrid)-1: raise SystemExit ('ERROR --- ac2wf:  uplooking, but observer very high')
			if angle<0.0: raise SystemExit ('ERROR --- ac2wf:  viewing angle < 0dg')
			sGrid  = zGrid[lObs:] / cosdg(angle)
			deltaS = np.ediff1d(sGrid)
			ac     = absCo[:,lObs:]
		else:
			raise SystemExit ('ERROR --- ac2wf:  sorry, horizontal path not (yet) implemented')

		# accumulative optical depth and transmission from uplooking observer up to ToA or downlooking observer down to BoA
		dod = 0.5*deltaS * (ac[:,1:]+ac[:,:-1])
		cod = np.add.accumulate(cstack(np.zeros(dod.shape[0]),dod),1)
		trans = np.exp(-cod)
		wgtFct = np.array([trans[:,j]*ac[:,j] for j in range(trans.shape[1])]).T

	return wfArray (wgtFct, Interval(vGrid[0],vGrid[-1]), sGrid, zObs, angle)


####################################################################################################################################

def dod2wf (dodList, angle=180., zObs=None):
	""" Optical depth to weighting functions (using 2-point finite difference).

	    Arguments:
	    ----------
	    dodList:     differential optical depth list
	    angle:       viewing angle [dg]:  0dg=uploooking ... 180dg downlooking (default)
	    zObs:        observer altitude [cm] (if very small, assume km and scale by 1e5 to get cm)
	                 if unspecified: assume ToA for angle>90dg, i.e. nadir corresponding to the default angle
			                 assume BoA for angle<90dg

	    RETURNS:
	    --------
	    vGrid        the spectral grid (wavenumbers)
	    sGrid        the spatial grid (altitudes (scaled by 1/cos(angle)) relative to observer)
	    wgtFct       numpy array matrix with absCo.shape[0] rows and ??? columns for the altitude(s)

	    WARNING:
	    --------
	    probably more accurate to compute finite diff of optical depth and multiply with transmission
	    OR even better "analytically" from transmission*absCoefficient (see function ac2wf)
	"""

	# 1. interpolate all data to common, densest grid, 2. extract heights (use this sequence because of some initial checks)
	vGrid, dodMatrix = od_list2matrix (dodList)
	zGrid = oDepth_altitudes (dodList)

	# check if zGrid is monotonically increasing !?!
	if monotone(zGrid)<=0:  raise ValueError ('ERROR --- dod2wf:  zGrid is not monotonically increasing!')

	if not zObs:                                # use ToA or BoA if no observer altitude is given
		if   angle>90.0:  zObs = zGrid[-1]  # downlooking
		else:             zObs = zGrid[0]   # uplooking
	elif zObs<250.:
		zObs = cgs('km',zObs)
		print('WARNING --- dod2wf:  zObs very small, assuming kilometer units')
	elif zObs>zGrid[-1]:
		raise ValueError ('%s %f' % ('ERROR --- dod2wf:  observer above ToA @', zGrid[-1]))

	# find index of level next to observer
	lObs = np.argmin(abs(zGrid-zObs))
	if abs(zGrid[lObs]-zObs)>10e3:
		print('WARNING --- dod2wf:  observer more than 1000cm away from next atmospheric level')
		print('                     computing weighting function w.r.t. ', 1e-5*zGrid[lObs], 'km')

	# get rid of 'unwanted' data above downlooking or below uplooking observer
	if   angle>90.0:  # downlooking
		if lObs<1: raise ValueError ('ERROR --- dod2wf:  downlooking, but observer very low')
		sGrid  = (zGrid[lObs::-1]-zGrid[lObs]) / cosdg(angle)  # distance from observer (flip and subtract)
		#deltaS = np.ediff1d(sGrid) # used for two-point finite difference only
		od     = -np.fliplr(dodMatrix[:,:lObs]) / cosdg(angle)
	elif angle<90.0:  # uplooking
		if lObs>len(zGrid)-1: raise ValueError ('ERROR --- dod2wf:  uplooking, but observer very high')
		sGrid = zGrid[lObs:] / cosdg(angle)
		#deltaS = np.ediff1d(sGrid) # used for two-point finite difference only
		od    = dodMatrix[:,lObs:] / cosdg(angle)
	else:
		raise SystemExit ('ERROR --- dod2wf:  sorry, horizontal path not implemented')

	# now compute cumulative optical depth from observer and exponentiate to transmission
	#od = cstack(np.zeros(od.shape[0]),np.add.accumulate(od,1))
	cod = np.add.accumulate(cstack(np.zeros(od.shape[0]),od),1)
	trans = np.exp(-cod)

	#return -(trans[:,1:]-trans[:,:-1]) / deltaS # two-point finite difference
	dTdS = np.empty_like(trans)
	for i in range(trans.shape[0]):  dTdS[i,:] = np.gradient(trans[i,:])/np.gradient(sGrid)

	# turn vGrid, sGrid, -dTdS
	return wfArray (-dTdS, Interval(vGrid[0],vGrid[-1]), sGrid, zObs, angle)


####################################################################################################################################
####################################################################################################################################

def _ac2wf_ (acFile, outFile=None, commentChar='#', zObs=None, zBoA=None, zToA=None, angle=180.0, fov=None, width=1.0,
             xLimits=None, transposeWF=False, verbose=False):
	""" Read absorption coefficients along with atmospheric data (esp. altitude), compute weighting functions, and save them. """

	try:
		absCoList = acRead (acFile, zToA, zBoA, xLimits, commentChar)
	except ValueError as errMsg:
		raise SystemExit ('ERROR --- _ac2wf_:  reading absorption coefficient file failed\n' +
	                          '(probably no atmospheric data in file header)\n' + str(errMsg))
	else:
		if isinstance(absCoList,(list,tuple)) and all([isinstance(ac,acArray) for ac in absCoList]):
			print("INFO --- _ac2wf_:  got a list of acArray's with ", len(absCoList), " absorption coefficients")
		else:
			raise SystemExit ("ERROR --- _ac2wf_:  reading absorption coefficient file failed\n" +
			                  "                    expected a list of acArray's")
		if verbose:
			for ac in absCoList:  minmaxmean(ac,'ac')

	if isinstance(fov,str) and isinstance(width,float):  fovTypeWidth = '%s %f' % (fov, width)
	else:                                                fovTypeWidth = None

	# ToDo:  implement subclassed array for weighting functions
	vGrid, sGrid, wgtFct = ac2wf (absCoList, zObs, angle, fovTypeWidth)
	if verbose:  minmaxmean(wgtFct, 'wf')

	# convert altitude, pressure, temperature into a structured array (similar to standard atmos1D, but without gases)
	atm_zpT = np.array([(ac.z,ac.p,ac.t) for ac in absCoList], dtype={'names': 'z p T'.split(), 'formats': 3*[np.float]})

	wfSave (vGrid, sGrid, wgtFct, outFile, atm_zpT, transposeWF, commentChar)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'about'},
               {'ID': 'a', 'name': 'angle', 'type': float, 'constraint': '0.0<=angle<=180.'},
               {'ID': 'w', 'name': 'width', 'type': float, 'constraint': 'width>0.0', 'default': 1.0},
               {'ID': 'z', 'name': 'zObs', 'type': float, 'constraint': 'zObs>0.0'},
               {'ID': 'ToA', 'name': 'zToA', 'type': float, 'constraint': 'zToA>0.0'},
               {'ID': 'BoA', 'name': 'zBoA', 'type': float, 'constraint': 'zBoA>=0.0'},
	       {'ID': 'x', 'name': 'xLimits', 'type': Interval, 'constraint': 'xLimits.lower>=0.0'},
               {'ID': 'fov', 'name': 'fov', 'type': str, 'constraint': 'fov.lower()[0] in "gbt"'},
               {'ID': 't',   'name': 'transposeWF'},
               {'ID': 'v',   'name': 'verbose'}]

	acFile, options, commentChar, outFile = parse_command (opts,1)

	if 'h' in options:      raise SystemExit (__doc__ + "\n end of ac2wf help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID'] in ['h','help','about'])]
	for key in boolOptions:  options[key] = key in options

	_ac2wf_ (acFile[0], outFile, commentChar, **options)
