#!/usr/bin/env python3

"""  ac2od

  Read absorption coefficient file and integrate over vertical path through atmosphere.

  usage:
  ac2od  [options]  ac_file

  -h               help
  -c     char      comment character(s) used in input,output file (default '#')
  -o     string    output file for saving of optical depth (if not given: write to StdOut)
                   (if the output file's extension ends with ".nc", ".ncdf" or ".netcdf",
                    a netcdf file is generated, otherwise the file is ascii tabular or pickled)

  -m     char      mode: 'c' ---> cumulative optical depth
                         'd' ---> difference (delta) optical depth (default)
                         'r' ---> reverse cumulative optical depth
                         't' ---> total optical depth

 --BoA   float     bottom-of-atmosphere altitude [km]  (compute opt.depth only for levels above)
 --ToA   float     top-of-atmosphere altitude [km]     (compute opt.depth only for levels below)
  -x     Interval  lower,upper wavenumbers (comma separated pair of floats [no blanks!],
                                            default set according to wavenumber range of absorption coefficients)

  -i     char      interpolation method   [default: '2' two-point Lagrange,  choices are one of "234lqc"]
 --nm              on output write optical depth versus wavelength [nm] (default: wavenumber 1/cm)
 --xFormat string  format to be used for wavenumbers,   default '%12f'   (only for ascii tabular)
 --yFormat string  format to be used for optical depth, default '%11.5f' (only for ascii tabular)
                   (if xFormat or yFormat is an empty string, netcdf or pickled format will be used)
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

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.ir import k  # Boltzmann constant
from py4cats.aux.cgsUnits import cgs
from py4cats.aux.pairTypes import PairOfFloats
from py4cats.art.oDepth import odArray, odSave, dod2tod, dod2cod
from py4cats.art.absCo import acInfo


####################################################################################################################################

def ac2dod  (acList, verbose=True):
	""" Integrate absorption coefficient along vertical path thru atmosphere layer-by-layer and return delta optical depths.

	ARGUMENTS
	---------

	acList:      a list of subclassed numpy arrays with the absorption coefficients,
                     including z, p, T, and wavenumber grid information as attributes (similar to xsArray)
	verbose      boolean flag

	RETURNS:
	--------
	dodList      a list of delta (layer) optical depths
	             (instances of odArray: subclassed numpy arrays with the layer optical depths
                     including z, p, T, and wavenumber grid information as attributes (similar to xsArray)

	NOTE:        the optDepth list has one element less than the acList:
	             each ac is defined for a level (altitude, pressure, ...)
		     each od is defined for a layer (altitude interval, ...)
		     Accordingly, the z, p, ... attributes of od are actually pairs (z and p intervals etc.)
	"""

	if verbose:  acInfo(acList)

	# check monotonicity of altitude and pressure levels
	nLevels = len(acList)
	zIncreasing = all(np.ediff1d(np.array([ac.z for ac in acList]))>0.0)
	pDecreasing = all(np.ediff1d(np.array([ac.p for ac in acList]))<0.0)
	if   zIncreasing and pDecreasing:
		print('\n INFO --- ac2dod:', nLevels, ' abs.coefficients with altitudes increasing and pressures decreasing')
	elif not zIncreasing and not pDecreasing:
		print('\n INFO --- ac2dod:', nLevels, ' abs.coefficients with altitudes decreasing and pressures increasing!?!')
	else:
		print('\n WARNING --- ac2dod:  both altitudes and pressures increasing / decreasing !?!', zIncreasing, pDecreasing)
		if not verbose:  acInfo(acList)

	# initialize
	dodList = []
	acLast  = acList[0]
	totalColumn = 0.0

	# loop over second, third, .... levels
	for l,ac in enumerate(acList[1:]):
		deltaZ = abs(acLast.z-ac.z)
		if not acLast.x==ac.x:
			print('level # %3i  @ %7.2fkm   wavenumber %s' % (l-1, cgs('!km',acLast.z), acLast.x),
			    '\n      # %3i  @ %7.2fkm              %s' % (l,   cgs('!km',ac.z),     ac.x))
			raise SystemExit ('ERROR --- ac2dod:  inconsistent wavenumber regions')
		if len(acLast)>len(ac):
			dod = 0.5*deltaZ * (acLast+ac.regrid(len(acLast)))
		elif len(acLast)<len(ac):
			dod = 0.5*deltaZ * (acLast.regrid(len(ac))+ac)
		else:
			dod = 0.5*deltaZ * (acLast+ac)
		# add the new layer to the optical depth list and update the previous absorption coefficient
		# use PairOfFloats instead of Interval to avoid sorting by size
		airColumn = 0.5 * (ac.p/(k*ac.t) + acLast.p/(k*acLast.t)) * abs(ac.z-acLast.z)
		dodList.append (odArray(dod, ac.x, PairOfFloats(acLast.z,ac.z),
		                                   PairOfFloats(acLast.p,ac.p), PairOfFloats(acLast.t,ac.t), airColumn))
		if verbose:  print(dodList[-1].info())
		else:        print(' layer # %3i  %9.1f --%6.1fkm  %10.3emolec/cm2  ---> %12i   <dod>=%7.2e  %10.4g < dod < %9.4g' %
		                   (l, cgs('!km',acLast.z), cgs('!km',ac.z), airColumn, len(dod), dod.mean(), min(dod), max(dod)))
		totalColumn += airColumn
		acLast = ac

	print (' %3i layers:                        %10.3emolec/cm2' % (nLevels-1, totalColumn))
	return dodList


####################################################################################################################################

def _ac2od_  (acFile, odFile=None, commentChar='#', zToA=0.0, zBoA=0.0, xLimits=None, mode='d',
             interpolate='linear', nanometer=False, flipUpDown=False, xFormat='%12f', yFormat='%11.5g', verbose=False):
	"""
	Read absorption coefficients and integrate over vertical path through atmosphere.

	acFile:      molecular absorption file
        xLimits:     Interval with lower and upper wavenumber [cm-1]
	verbose      boolean flag

	For the optional arguments see the documentation of the atmos1D and lbl2od modules.
	"""

	from absCo import acRead

	# read absorption coefficient file
	acList = acRead (acFile, zToA, zBoA, xLimits, commentChar)

	# integrate absorption coefficients to optical depth
	dodList = ac2dod  (acList, verbose)

	# finally save delta, cumulative or total optical depth
	if mode.lower()=='t':
		odSave (dod2tod(dodList), odFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)
	elif mode.lower()=='c' or mode.lower()=='r':
		back = mode.lower()=='r'
		odSave (dod2cod(dodList, back), odFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)
	else:
		odSave (dodList, odFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'about'},
	       {'ID': 'm', 'name': 'mode', 'type': str, 'default': 'd', 'constraint': 'mode.lower() in "cdrt"'},
	       {'ID': 'ToA', 'name': 'zToA', 'type': float, 'constraint': 'zToA>0.0'},
	       {'ID': 'BoA', 'name': 'zBoA', 'type': float, 'constraint': 'zBoA>=0.0'},
	       {'ID': 'i', 'name': 'interpolate', 'type': str, 'default': '2', 'constraint': 'interpolate in "234lLqQcC"'},
	       {'ID': 'xFormat', 'name': 'xFormat', 'type': str, 'default': '%12f'},
	       {'ID': 'yFormat', 'name': 'yFormat', 'type': str, 'default': '%11.5g'},
	       {'ID': 'r',   'name': 'flipUpDown'},
	       {'ID': 'nm',  'name': 'nanometer'},
	       {'ID': 'v',   'name': 'verbose'}
	       ]

	inFiles, options, commentChar, outFile = parse_command (opts,1)

	if 'h' in options:      raise SystemExit (__doc__ + "\n End of ac2od help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID'] in ['h','help','about'])]
	for key in boolOptions:  options[key] = key in options

	_ac2od_ (inFiles[0], outFile, commentChar, **options)
