#!/usr/bin/env python3

"""  lbl2od
  computation of line-by-line optical depth due to molecular absorption

  usage:
  lbl2od  [options]  atm_file  line_parameter_file(s)

  -h               help
  -c     char      comment character(s) used in input,output file (default '#')
  -o     string    output file for saving of optical depth (if not given: write to StdOut)
                   (if the output file's extension ends with ".nc", ".ncdf" or ".netcdf",
                    a netcdf file is generated, otherwise the file is ascii tabular or pickled)

  -m     char      mode: 'c' ---> cumulative optical depth
                         'd' ---> difference (delta) optical depth (default)
                         'r' ---> reverse cumulative optical depth
	  	         't' ---> total optical depth

 --avg   int       running average:  return mean of some points
 --BoA   float     bottom-of-atmosphere altitude [km]  (compute opt.depth only for levels above)
 --ToA   float     top-of-atmosphere altitude [km]     (compute opt.depth only for levels below)
                   NOTE:  no interpolation, i.e. integration starts/stops at the next level above/below BoA/ToA
 --scale floats    multiply molecular concentrations with scaleFactors
                   (either a comma separated list of floats (no blanks) in the same order as for the line data files,
		    or just a single float to scale the profile of the molecule corresponding to the first lineFile.)
  -x     Interval  lower,upper wavenumbers (comma separated pair of floats [no blanks!],
                                            default set according to range of lines in datafiles)
 --nm              on output write optical depth versus wavelength [nm] (default: wavenumber 1/cm)
 --xFormat string  format to be used for wavenumbers,   default '%12f'   (only for ascii tabular)
 --yFormat string  format to be used for optical depth, default '%11.5f' (only for ascii tabular)
                   (if xFormat or yFormat is an empty string, netcdf or pickled format will be used)
  -r               on output reverse layer optical depth order:  top <--> bottom of atmosphere

  For more information use
  lbl2od --help
"""

more_Help = """
                 the following set of options are essentially related to numerics etc.

  -i   char      interpolation method   [default: '2' two-point Lagrange,  choices are one of "234lqc"]
  -L   char      Lineshape: V(oigt), L(orentz), G(auss)     [default: Voigt]
  -n   int       number of grids --- selects 'multigrid mode' for nGrids=2 or 3  (default: nGrids=3 three grid)
  -s   float     sampling rate used for x-grid (default: 5.0 grid points per (mean) half width)
  -w   float     wing extension (cutoff wavenumber, default 10.0cm-1)
  -g   int       gridRatio = ratio of coarse to fine grid spacing  (only 2, 4, or 8, default 8)
  -W   float     transition from fine to coarse grid  (in units of half widths, default 25.0)
 --xs  string    file extension:  save cross sections for all p,T and molecules to ascii files with this extension
                 (the file name is generated automatically with the scheme molec_pressure_temperature.xs)


  line_parameter_file(s)
  The line parameter file(s) should contain a list of (preselected) lines
  that can be generated from HITRAN or GEISA database with extract.py
  (See the documentation header of lbl2xs.py for more details)

  atm_file
  The file containing an user's atmospheric profile data has to be in xy  format
  with columns for altitude, pressure, temperature and molecular concentrations
  (see the documentation for details and the data directory for some examples)

  NOTES:
  interpolation required for multigrid approach and to regrid cross sections to the final dense grid (typically in the top levels)
  Lagrange two-point interpolation (selected by "2", "l", or "L") is clearly the least elaborate and most robust,
  even three-point interpolation can sometimes lead to oscillations associated with some negative xs values
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

# import some standard python modules

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.aeiou import commonExtension, awrite
from py4cats.aux.pairTypes import Interval
from py4cats.lbl.lines import read_line_file, lineArray
from py4cats.lbl.lbl2xs import lbl2xs
from py4cats.art.atmos1D import atmRead
from py4cats.art.xSection import xsSave
from py4cats.art.xs2ac import xs2ac
from py4cats.art.ac2od import ac2dod
from py4cats.art.oDepth import odArray, odSave, dod2tod, dod2cod
from py4cats.art.rayleigh import rayleighx
from py4cats.art.cia import add_cia2absCo
from py4cats.art.aerosol import aerosol_od


####################################################################################################################################

def lbl2od (atmos, lineListsDict, xLimits=None,
            cia=None, rayleigh=None, aerosol=None,
            lineShape="Voigt", sampling=5.0, nGrids=3, gridRatio=8, nWidths=25.0, lagrange=2,
            interpolate=2, xsFile=None, verbose=False):
	"""
	Compute molecular cross sections, sum over molecules to absorption coefficient, and 'integrate' to layer optical depth:
	1. Compute cross sections for some molecule(s) and some pressure(s),temperature(s) by summation of line profiles;
	2. Compute absorption coefficients as product cross section times molecular density, summed over all molecules;
	3. 'Integrate' absorption coefficients along vertical path thru atmosphere to get layer/delta optical depth.
	4. Optionally add CIA (collision induced absorption) and/or Rayleigh extinction.

	ARGUMENTS:
	----------
	atmos:         atmospheric data set, notably including zGrid=atmos['z']
	lineListsDict  (molecular) line parameters: a dictionary of structured arrays
        xLimits:       Interval with lower and upper wavenumber [cm-1]
	cia:           collision induced absorption, default None;
	               either a single file (Hitran format) or list thereof (* wildcard supported!)
	rayleigh:      Rayleigh extinction, default None;
	               if True, use nicolet model;  other choices: bodhaine, bucholtz, CO2, H2
	               a positive number is interpreted as 'Rayleigh enhancement factor' (with the Nicolet model)
	aerosol:       Aerosol extinction 8.85e-30*N*v**1.3, default None;
	               a positive number is interpreted as 'Aerosol enhancement factor'
		       a list of two floats is interpreted as factor and Angstroem exponent (default 1.3)

	RETURNS:
	--------
	dodList        a list of delta (layer) optical depths (instances of odArray)
	               (for an atmosphere with nLevels=len(atmos) the list returned has nLevels-1 layers!)

	See the lbl2xs function for details about other optional arguments.
	"""

	if isinstance(atmos,(dict,lineArray)):
		raise SystemExit ("ERROR --- lbl2od:  first argument appears to be line data instead of atmos data")
	elif isinstance(atmos,np.ndarray):
		if not atmos.dtype.names:
			raise SystemExit ("ERROR --- lbl2od:  first argument apparently not a structured numpy array of atmospheric data")
	else:
		raise SystemExit ("ERROR --- lbl2od:  'incorrect' first argument, expected a structured numpy array of atmospheric data")

	if isinstance(lineListsDict,lineArray):
		print("\nINFO --- lbl2od:  got a single line list for ", lineListsDict.molec, ",  packed into dictionary!\n")
		lineListsDict = {lineListsDict.molec:  lineListsDict}

	# cross sections
	xssDict = lbl2xs (lineListsDict, atmos['p'], atmos['T'],
                          xLimits, lineShape, sampling, nGrids, gridRatio, nWidths, lagrange, verbose)
	if isinstance(xsFile,str):  xsSave(xssDict, xsFile)

	# absorption coefficients:  this list has nLevels=len(acList) elements
	acList = xs2ac (atmos, xssDict, interpolate, verbose)

	# collision induced absorption
	if cia:  acList = add_cia2absCo(acList, cia)

	# Rayleigh extinction
	if rayleigh:
		if   isinstance(rayleigh, str):          acList = rayleighx(acList, model=rayleigh)
		elif isinstance(rayleigh, (int,float)):  acList = rayleighx(acList, factor=rayleigh)
		else:                                    acList = rayleighx(acList)

	# delta optical depths:  this list has len(acList)-1 layer optical depths
	dodList = ac2dod  (acList, verbose)

	# aerosol extinction
	if aerosol:
		if   isinstance(aerosol, bool):                              optArgs = []
		elif isinstance(aerosol, (int,float)) and aerosol>0:         optArgs = [int(aerosol)]
		elif isinstance(aerosol, (list,tuple)) and len(aerosol)<=2:  optArgs = aerosol
		else:  raise ValueError ("lbl2od:  invalid argument for aerosol extinction,\n" +
		                         "            expected (list of) one or two (positive) float(s)")
		for l, dod in enumerate(dodList):
			dodList[l] = odArray(dod.base+aerosol_od(dod.grid(), dod.N, *optArgs), dod.x, dod.z, dod.p, dod.t, dod.N)

	return dodList


####################################################################################################################################

def _lbl2od_ (atmFile, lineFiles, outFile=None, commentChar='#',
              zToA=0.0, zBoA=0.0, mode='d',
              xLimits=None, lineShape='Voigt', sampling=5.0, wingExt=5.0, nGrids=2, gridRatio=0, nWidths=25.0,
              runAvg=0, scaleFactors=None, airWidth=0.1,
              interpolate='l', nanometer=False, flipUpDown=False, xFormat='%12f', yFormat='%11.5g',
              xsFile=None, verbose=False):
	"""
	Read atmospheric and molecular line parameter data files, compute cross sections,
	sum to absorption coefficients, and integrate over vertical path through atmosphere.

	atmFile:      file(name) with atmospheric data to be read: altitude, pressure, temperature, concentrations
	lineFiles:    tabular-ascii files with core spectroscopic parameters (position, strengths, ...) for some molecules

        xLimits:       Interval with lower and upper wavenumber [cm-1]
	mode:          d  differential = delta optical depth, returns nLevels-1 layer optical depths (default)
	               c  cumulative (accumulated)
		       r  reverse cumulative (accumulated)
		       t  total

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
			raise SystemExit ('%s %s' % ('\nERROR --- lbl2od:  no atmospheric data for species', gas))

	# interpolation method used in multigrid algorithms: only Lagrange!
	if isinstance(interpolate,str):
		if   interpolate in 'cC':  lagrange=4
		elif interpolate in 'qQ':  lagrange=3
		else:                      lagrange=2
	else:
		lagrange=int(interpolate)

	# Compute cross sections, sum to absorption coefficients, and integrate layers individually:  return delta opt. depth
	odList = lbl2od (atmos, lineListsDict,
                         xLimits, lineShape, sampling, nGrids, gridRatio, nWidths, lagrange,
                         interpolate, xsFile, verbose)

	if runAvg>1:
		print('WARNING --- _oDepth_:  running average imperfect, xLimits have to be adjusted slightly!')
		odList = [odArray(np.array([sum(od.base[i:i+runAvg])/runAvg for i in range(0,runAvg*len(od)/runAvg,runAvg)]),
		                  od.x, od.z, od.p, od.t, od.N) for od in odList]

	# save optical depth
	if len(xFormat.strip())==0 or len(yFormat.strip())==0:
		commentChar='';  print('odSave using pickled or netcdf format')

	if mode.lower()=='t':
		odSave (dod2tod(odList), outFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)
	elif mode.lower()=='c' or mode.lower()=='r':
		back = mode.lower()=='r'
		odSave (dod2cod(odList, back), outFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)
	else:
		odSave (odList, outFile, commentChar, nanometer, flipUpDown, interpolate, xFormat, yFormat)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'help'},
	       {'ID': 'about'},
	       {'ID': 'm', 'name': 'mode', 'type': str, 'default': 'd', 'constraint': 'mode.lower() in "cdrt"'},
	       {'ID': 'ToA', 'name': 'zToA', 'type': float, 'constraint': 'zToA>0.0'},
	       {'ID': 'BoA', 'name': 'zBoA', 'type': float, 'constraint': 'zBoA>=0.0'},
	       {'ID': 'scale', 'name': 'scaleFactors', 'type': np.ndarray, 'constraint': 'scaleFactors>0.0'},
	       {'ID': 'a', 'name': 'airWidth', 'type': float, 'constraint': 'airWidth>0.0'},
	       {'ID': 'L', 'name': 'lineShape', 'type': str, 'constraint': 'lineShape[0] in ["V","L","G"]', 'default': 'Voigt'},
	       {'ID': 'i', 'name': 'interpolate', 'type': str, 'default': '2', 'constraint': 'interpolate in "234lLqQcC"'},
	       {'ID': 's', 'name': 'sampling', 'type': float, 'constraint': 'sampling>0.0', 'default': 5.0},
	       {'ID': 'w', 'name': 'wingExt', 'type': float, 'constraint': 'wingExt>0.0', 'default': 5.0},
	       {'ID': 'x', 'name': 'xLimits', 'type': Interval, 'constraint': 'xLimits.lower>=0.0'},
	       {'ID': 'n', 'name': 'nGrids', 'type': int, 'default': 3, 'constraint': 'nGrids>0'},
	       {'ID': 'g', 'name': 'gridRatio', 'type': int, 'default': 8, 'constraint': 'gridRatio in [4,8]'},
	       {'ID': 'W', 'name': 'nWidths', 'type': float, 'default': 25.0,'constraint': 'nWidths>2.0'},
	       {'ID': 'avg', 'name': 'runAvg', 'type': int, 'default': 0, 'constraint': 'runAvg>2'},
	       {'ID': 'xs', 'name': 'xsFile', 'type': str},
	       {'ID': 'xFormat', 'name': 'xFormat', 'type': str, 'default': '%12f'},
	       {'ID': 'yFormat', 'name': 'yFormat', 'type': str, 'default': '%11.5g'},
	       {'ID': 'r',   'name': 'flipUpDown'},
	       {'ID': 'nm',  'name': 'nanometer'},
	       {'ID': 'v',   'name': 'verbose'}]

	inFiles, options, commentChar, outFile = parse_command (opts,(2,99))

	if 'h' in options:      raise SystemExit (__doc__ + "\n End of lbl2od help")
	if 'help' in options:   raise SystemExit (__doc__[:-42] + more_Help + "\n End of lbl2od help (extended)")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID'] in ['h','help','about'])]
	for key in boolOptions:  options[key] = key in options

	if commonExtension(inFiles):
		print('\n WARNING:  all input files have the same extension,',
		      '\n           first file probably NOT an atmospheric data file!!!',
		      '\n           (Trying to continue)\n')

	# unpack list of input files
	atmFile, lineFiles = inFiles[0], inFiles[1:]

	_lbl2od_ (atmFile, lineFiles, outFile, commentChar, **options)
