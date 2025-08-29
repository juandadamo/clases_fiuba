#!/usr/bin/env python3

"""  xs2od

  Read atmospheric and molecular absorption cross section files,
  sum to absorption coefficients, and integrate over vertical path through atmosphere.

  usage:
  xs2od  [options]  atm_file  line_parameter_file(s)

  -h               help
  -c     char      comment character(s) used in input,output file (default '#')
  -o     string    output file for saving of optical depth (if not given: write to StdOut)
                   (if the output file's extension ends with ".nc", ".ncdf" or ".netcdf",
                    a netcdf file is generated, otherwise the file is ascii tabular or pickled)

 --scale floats    scaling factors for molecular concentrations
  -m     char      mode: 'c' ---> cumulative optical depth
                         'd' ---> difference (delta) optical depth (default)
                         'r' ---> reverse cumulative optical depth
                         't' ---> total optical depth
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


####################################################################################################################################

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.aeiou import awrite, join_words
from py4cats.art.atmos1D import atmRead
from py4cats.art.xSection import xsRead, xsInfo
from py4cats.art.oDepth import odSave, dod2tod, dod2cod
from py4cats.art.xs2ac import xs2ac
from py4cats.art.ac2od import ac2dod


####################################################################################################################################

def xs2dod (atmos, xssDict, interpolate='linear', verbose=False):
	""" Compute absorption coefficients as product cross section times molecular density, summed over all molecules.
	    Integrate absorption coefficients layer-by-layer along vertical path thru atmosphere to get delta optical depths.

	    This function is essentially a combiniation of the xs2ac and ac2dod functions.

	ARGUMENTS:
	----------
	atmos:         a structured array of atmospheric data
	xssDict:       a dictionary of cross sections, one entry per molecule
	interpolate:   default linear
	verbose:       flag, default False

	RETURNS:
	--------
	optDepth:    a list of subclassed numpy arrays with the layer/delta optical depths,
                     including z, p, T, and wavenumber grid information as attributes (similar to xsArray)
	"""

	# cross sections to absorption coefficients
	acList= xs2ac (atmos, xssDict, interpolate, verbose)

	# integrate absorption coefficients to optical depth
	dodList = ac2dod  (acList, verbose)

	return dodList


####################################################################################################################################

def _xs2od_ (atmosFile, xsFiles, outFile=None, commentChar='#',
             zToA=0.0, zBoA=0.0, mode='d', scaleFactors=None,
             interpolate='linear', nanometer=False, flipUpDown=False, xFormat='%12f', yFormat='%11.5g', verbose=False):
	"""
	Read atmospheric and molecular absorption cross section files,
	sum to absorption coefficients, and integrate over vertical path through atmosphere.

	atmosFile:    file(name) with atmospheric data to be read: altitude, pressure, temperature, concentrations
	xsFiles:      molecular absorption cross section files (one for each molecule)

	For the optional arguments see the documentation of the atmos1D and lbl2xs modules.
	"""

	# read all cross section files
	xssDict = {}
	for file in xsFiles:
		xss = xsRead (file, commentChar)
		xssDict[xss[0].molec] = xss

	species = list(xssDict.keys())

	if verbose:  xsInfo(xssDict)
	else:        print (len(xssDict), ' cross section file(s) read for ', join_words(species))

	# read atmospheric data from file,  returns a structured array
	# optionally remove low or high altitudes (NOTE:  atmos uses cgs units, i.e. altitudes in cm)
	eXtract=['z', 'p', 'T']+species
	atmos = atmRead(atmosFile, commentChar, extract=eXtract, zToA=zToA*1e5, zBoA=zBoA*1e5, scaleFactors=scaleFactors, verbose=verbose)
	print('\nAtmosphere:  %s   %i species * %i levels' % (atmosFile, len(species), len(atmos)))
	if verbose:  awrite (atmos, format=' %10.3g', comments=len(atmos.dtype)*' %10s' % tuple(atmos.dtype.names))

	dodList = xs2dod (atmos, xssDict, interpolate, verbose)

	if mode.lower()=='t':
		odSave (dod2tod(dodList), outFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)
	elif mode.lower()=='c' or mode.lower()=='r':
		back = mode.lower()=='r'
		odSave (dod2cod(dodList, back), outFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)
	else:
		odSave (dodList, outFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)


####################################################################################################################################

if __name__ == "__main__":

	from os.path import splitext
	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'about'},
	       {'ID': 'm', 'name': 'mode', 'type': str, 'default': 'd', 'constraint': 'mode.lower() in "cdrt"'},
	       {'ID': 'ToA', 'name': 'zToA', 'type': float, 'constraint': 'zToA>0.0'},
	       {'ID': 'BoA', 'name': 'zBoA', 'type': float, 'constraint': 'zBoA>=0.0'},
	       {'ID': 'scale', 'name': 'scaleFactors', 'type': np.ndarray, 'constraint': 'scaleFactors>0.0'},
	       {'ID': 'i', 'name': 'interpolate', 'type': str, 'default': '2', 'constraint': 'interpolate in "234lLqQcC"'},
	       {'ID': 'xFormat', 'name': 'xFormat', 'type': str, 'default': '%12f'},
	       {'ID': 'yFormat', 'name': 'yFormat', 'type': str, 'default': '%11.5g'},
	       {'ID': 'r',   'name': 'flipUpDown'},
	       {'ID': 'nm',  'name': 'nanometer'},
	       {'ID': 'v',   'name': 'verbose'}
	       ]

	inFiles, options, commentChar, outFile = parse_command (opts,(2,99))

	if 'h' in options:      raise SystemExit (__doc__ + "\n End of xs2od help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID'] in ['h','help','about'])]
	for key in boolOptions:  options[key] = key in options

	fileExt = [splitext(file)[1] for file in inFiles]
	if all([fileExt[0]==fx for fx in fileExt]):
		print ("WARNING:  all file extensions identical ", fileExt[0],
		     "\n          but the first file should be atmospheric data, others with cross sections")

	# unpack list of input files
	atmFile, xsFiles = inFiles[0], inFiles[1:]

	_xs2od_ (atmFile, xsFiles, outFile, commentChar, **options)
