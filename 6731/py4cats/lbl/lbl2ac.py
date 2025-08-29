#!/usr/bin/env python3

"""  lbl2ac
  computation of line-by-line absorption coefficients due to molecular absorption

  usage:
  lbl2ac  [options]  atm_file  line_parameter_file(s)

  -h               help
  -c     char      comment character(s) used in input,output file (default '#')
  -o     string    output file for saving of optical depth (if not given: write to StdOut)
                   (if the output file's extension ends with ".nc", ".ncdf" or ".netcdf",
                    a netcdf file is generated, otherwise the file is ascii tabular)

 --BoA   float     bottom-of-atmosphere altitude [km]  (compute opt.depth only for levels above)
 --ToA   float     top-of-atmosphere altitude [km]     (compute opt.depth only for levels below)
                   NOTE:  no interpolation, i.e. integration starts/stops at the next level above/below BoA/ToA
 --scale floats    multiply molecular concentrations with scaleFactors
                   (either a comma separated list of floats (no blanks) in the same order as for the line data files,
                   or just a single float to scale the profile of the molecule corresponding to the first lineFile.)
  -x     Interval  lower,upper wavenumbers (comma separated pair of floats [no blanks!],
                                            default set according to range of lines in datafiles)

  For more information see also the lbl2xs and lbl2od help.
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

# import some standard python modules

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.aeiou import commonExtension, awrite, join_words
from py4cats.aux.pairTypes import Interval
from py4cats.lbl.molecules import molecules
from py4cats.lbl.lines import read_line_file, lineArray
from py4cats.lbl.lbl2xs import lbl2xs
from py4cats.art.atmos1D import atmRead
from py4cats.art.xSection import xsSave, xsArray
from py4cats.art.xs2ac import xs2ac
from py4cats.art.absCo import acSave

molecNames   = list(molecules.keys())

####################################################################################################################################


def lbl2ac (atmos, lineListsDict,
            xLimits=None, lineShape="Voigt", sampling=5.0, nGrids=3, gridRatio=8, nWidths=25.0, lagrange=2,
            interpolate='2', xsFile=None, verbose=False):
	""" Compute cross sections for some molecule(s) and some pressure(s),temperature(s) by summation of line profiles;
	    Compute absorption coefficients as product cross section times molecular density, summed over all molecules.

	    ARGUMENTS:
	    ----------
	    atmos:         atmospheric data set, notably including zGrid=atmos['z']
	    lineListsDict  (molecular) line parameters: a dictionary of structured arrays

	    RETURNS:
	    --------
	    absCo:        a subclassed numpy array with the absorption coefficients,
                          with z, p, T, and wavenumber grid information as attributes (similar to xsArray)

	    See the lbl2xs and lbl2od functions for details about further optional arguments.
	"""
	if isinstance(atmos,(dict,lineArray)):
		raise SystemExit ("ERROR --- lbl2ac:  first argument appears to be line data instead of atmos data")

	# cross sections
	xssDict = lbl2xs (lineListsDict, atmos['p'], atmos['T'],
                          xLimits, lineShape, sampling, nGrids, gridRatio, nWidths, lagrange, verbose)
	if isinstance(xsFile,str):  xsSave(xssDict, xsFile)

	if   isinstance(xssDict,dict):
		pass
	elif isinstance(xssDict,list):
		if all([xs.molec==xssDict[0].molec for xs in xssDict]) and len(xssDict)==len(atmos):
			xssDict = {xssDict[0].molec:  xssDict}
			print('\n INFO --- lbl2ac:  lbl2xs returned a list, put into a dictionary\n')
		else:
			raise SystemExit ("ERROR --- lbl2ac:  lbl2xs returned a list, but length inconsistent with number of atmospheric levels")
	elif isinstance(xssDict,xsArray):
		xssDict = {xssDict.molec:  [xssDict]}
		print('\n INFO --- lbl2ac:  lbl2xs returned a xsArray, put into a dictionary with a one-element list\n')
	else:
		raise SystemExit ('%s %s' % ("ERROR --- lbl2ac:  lbl2xs returned incorrect type of xs data", type(xssDict)))

	# absorption coefficients
	absCoList = xs2ac (atmos, xssDict, interpolate, verbose)

	return absCoList


####################################################################################################################################

def _lbl2ac_ (atmFile, lineFiles, outFile=None, commentChar='#',
              zToA=0.0, zBoA=0.0,
              xLimits=None, lineShape='Voigt', sampling=5.0, wingExt=5.0, nGrids=2, gridRatio=0, nWidths=25.0,
              scaleFactors=None, airWidth=0.1,  interpolate='l',
              xsFile=None, verbose=False):
	"""
	Read atmospheric and molecular line parameter data files, compute cross sections,
	sum to absorption coefficients, and integrate over vertical path through atmoshere.

	atmFile:      file(name) with atmospheric data to be read: altitude, pressure, temperature, concentrations
	lineFiles:    tabular-ascii foles with core spectroscopic parameters (position, strengths, ...) for some molecules

	For the optional arguments see the documentation of the atmos1D and lbl2xs modules.
	"""

	# Read a set of line data files (hitran/geisa extracts of core parameters) and return a dictionary
	lineListsDict = read_line_file(lineFiles, xLimits, wingExt, airWidth, commentChar=commentChar, verbose=verbose)
	species = list(lineListsDict.keys())

	# read atmospheric data from file,  returns a structured array
	# optionally remove low or high altitudes (NOTE:  atmos uses cgs units, i.e. altitudes in cm)
	eXtract=['z', 'p', 'T']+species
	atmos = atmRead(atmFile, commentChar, extract=eXtract, zToA=zToA*1e5, zBoA=zBoA*1e5, scaleFactors=scaleFactors, verbose=verbose)
	print('\nAtmosphere:  %s   %i species * %i levels' % (atmFile, len(species), len(atmos)))
	awrite (atmos, format=' %10.3g', comments=len(atmos.dtype)*' %10s' % tuple(atmos.dtype.names))

	# check if all species have profiles in the atmospheric data
	for gas in species:
		if gas not in atmos.dtype.names:
			raise SystemExit ('%s %s' % ('\nERROR --- lbl2ac:  no atmospheric data for species', gas))

	# interpolation method used in multigrid algorithms: only Lagrange!
	if   interpolate in 'cC':  lagrange=4
	elif interpolate in 'qQ':  lagrange=3
	elif interpolate in 'lL':  lagrange=2
	else:                      lagrange=int(interpolate)

	absCoList = lbl2ac (atmos, lineListsDict,
	                    xLimits, lineShape, sampling, nGrids, gridRatio, nWidths, lagrange,
	                    interpolate, xsFile, verbose)

	acSave (absCoList, outFile, commentChar, interpolate)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'help'},
	       {'ID': 'about'},
	       {'ID': 'ToA', 'name': 'zToA', 'type': float, 'constraint': 'zToA>0.0'},
	       {'ID': 'BoA', 'name': 'zBoA', 'type': float, 'constraint': 'zBoA>=0.0'},
	       {'ID': 'scale', 'name': 'scaleFactors', 'type': np.ndarray, 'constraint': 'scaleFactors>0.0'},
	       {'ID': 'a', 'name': 'airWidth', 'type': float, 'constraint': 'airWidth>0.0'},
	       {'ID': 'L', 'name': 'lineShape', 'type': str, 'constraint': 'lineShape[0] in ["V","L","G"]', 'default': 'Voigt'},
	       {'ID': 'i', 'name': 'interpolate', 'type': str, 'default': 'l', 'constraint': 'interpolate in "234lLqQcCbBsS"'},
	       {'ID': 's', 'name': 'sampling', 'type': float, 'constraint': 'sampling>0.0', 'default': 5.0},
	       {'ID': 'w', 'name': 'wingExt', 'type': float, 'constraint': 'wingExt>0.0', 'default': 5.0},
	       {'ID': 'x', 'name': 'xLimits', 'type': Interval, 'constraint': 'xLimits.lower>=0.0'},
	       {'ID': 'n', 'name': 'nGrids', 'type': int, 'default': 3, 'constraint': 'nGrids>0'},
	       {'ID': 'g', 'name': 'gridRatio', 'type': int, 'default': 8, 'constraint': 'gridRatio in [4,8]'},
	       {'ID': 'W', 'name': 'nWidths', 'type': float, 'default': 25.0,'constraint': 'nWidths>2.0'},
	       {'ID': 'xs', 'name': 'xsFile', 'type': str},
	       {'ID': 'v',   'name': 'verbose'} ]

	inFiles, options, commentChar, outFile = parse_command (opts,(2,99))

	if 'h' in options:      raise SystemExit (__doc__ + "\n End of lbl2ac help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID'] in ['h','help','about'])]
	for key in boolOptions:  options[key] = key in options

	# unpack list of input files
	atmFile, lineFiles = inFiles[0], inFiles[1:]

	if commonExtension(inFiles):
		print('\nWARNING:  all input files have the same extension,',
		      '\n          first file probably NOT an atmospheric data file!!!',
		      '\n          atmospheric data file: ', atmFile,
		      '\n          line data file(s):     ', join_words(lineFiles), '\n',
		      '\n          (Trying to continue)\n')

	_lbl2ac_ (atmFile, lineFiles, outFile, commentChar, **options)
