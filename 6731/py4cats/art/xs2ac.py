#!/usr/bin/env python3

"""  xs2ac

  Read atmospheric and molecular absorption cross section files, scale by density, and sum to absorption coefficients

  usage:
  xs2ac  [options]  atm_file  line_parameter_file(s)

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

from py4cats.aux.cgsUnits import cgs
from py4cats.aux.misc import approx
from py4cats.aux.aeiou import awrite, join_words
from py4cats.art.atmos1D import  atmRead, gases
from py4cats.art.xSection import xsArray, xsRead, xsInfo
from py4cats.art.absCo import acArray, acSave


####################################################################################################################################

def xs2ac (atmos, xssDict, interpolate='l', verbose=False):
	""" Compute absorption coefficients as product cross section times molecular density, summed over all molecules.

	ARGUMENTS:
	----------
	atmos:        a structured numpy array with atmospheric data, esp. molecular number densities
	xssDict:      a dictionary of cross section lists, one list with npT=nLevels xsArrays
	              (subclassed numpy arrays) for each molecule.
	interpolate:  string or integer to select the interpolation method (default: linear using numpy.interp)
	verbose:      boolean flag

	RETURNS:
	--------
	absCo:        a subclassed numpy array with the absorption coefficients,
                      with z, p, T, and wavenumber grid information as attributes (similar to xsArray)
	"""

	# first check 'consistency' of atmospheric and cross section data
	if isinstance(atmos,np.ndarray) and atmos.dtype.names:
		# the standard case: a structured array
		nGases  = len(gases(atmos))  # number of molecules with atmospheric data (densities)
		nLevels = len(atmos)         # number of atmospheric levels
	elif isinstance(atmos,np.void) and atmos.dtype.names:
		print(" INFO --- xs2ac:  atmos structured array for just a single level!?!")
		nGases  = len(gases(atmos))  # number of molecules with atmospheric data (densities)
		nLevels = 1                  # number of atmospheric levels
	else:
		raise SystemExit ("ERROR --- xs2ac:  expected a structured array `atmos`, but got " + str(type(xssDict)))

	# check 'consistency' of atmospheric data
	if not nLevels==atmos.size:
		print("WARNING --- xs2ac:  len(atmos)=%i,  atmos.size=%i,  atmos.shape=%s  inconsistent, trying to continue" %
		      (len(atmos), atmos.size, atmos.shape))

	if isinstance(xssDict,dict):
		molecules = list(xssDict.keys())
		nMolecules= len(xssDict)
	elif isinstance(xssDict,(list,tuple)) and all([isinstance(xs,xsArray) for xs in xssDict]):
		# fallback for a single molecule, single list of xsArray's
		firstMolec=xssDict[0].molec
		if len(xssDict)==nLevels and all([xs.molec==firstMolec for xs in xssDict]):
			nMolecules = 1
			molecules  = [firstMolec]
			xssDict    = {firstMolec: xssDict}
			print("INFO --- xs2ac:  got a list of xsArray's for ", firstMolec, ",    packed into a dictionary")
		else:
			raise SystemExit ("ERROR --- xs2ac: expected a dictionary of cross sections, \n",
			                  "                 but got a list with different entries, incorrect size, ...")
	else:
		raise SystemExit ("ERROR --- xs2ac:  expected a dictionary of cross sections, but got " + str(type(xssDict)))

	# check if there are cross sections for more molecules than gases in the atmospheric data set
	if nMolecules>nGases:
		print(' atmos: ', nGases, gases(atmos), '\n xssDict:', nMolecules, list(xssDict.keys()))
		raise SystemExit ('ERROR ---xs2ac:  more cross section molecules than gases in atmos data!')
	else:
		# check if there are atmospheric data (densities) for all cross section molecules
		for molec in molecules:
			if molec not in gases(atmos):
				raise SystemExit ('%s %s %s' % ('ERROR --- xs2ac:  cross sections for', molec, ' but no atmospheric data!'))

	# compare the number of levels (p,T pairs) in the cross section data with the number of atmospheric levels
	if all([isinstance(xss,(list,tuple)) for xss in list(xssDict.values())]):
		for molec,xss in list(xssDict.items()):
			if not len(xss)==nLevels:
				raise SystemExit ('%s %i %s %s %s %i' %
				      ('ERROR --- xs2ac: ', len(xss), '= number of levels (p,T pairs) in ', molec,
				       ' cross section differs from number of atmospheric levels', nLevels))
	elif all([isinstance(xss,xsArray) for xss in list(xssDict.values())]):
		if nLevels==1:
			for molec,xss in list(xssDict.items()):  xssDict[molec] = [xss]
		else:
			raise SystemExit ("ERROR --- xs2ac:  got a multilevel atmosphere, " +
			                  "but a dictionary of xsArray's for a single level")
	else:
		raise SystemExit ("ERROR --- xs2ac:  cross section dictionary entries neither a xsArray or list thereof")

	print('\n INFO --- xs2ac:  atmosphere with ', nLevels, ' levels with ', nGases, ' molecules ', join_words(gases(atmos)), end=' ')
	if nMolecules==nGases:  print('      consistent with cross section data\n')
	else:                   print('      but ', nGases-nMolecules, ' of', nGases, ' gases without cross sections!\n')

	# three (nMolecules*nLevels) matrices with number of cross section values, first and last wavenumber grid point
	nFreqsMatrix = np.array([[len(xs) for xs in xss] for xss in list(xssDict.values())])
	vLowMatrix   = np.array([[xs.x.lower for xs in xss] for xss in list(xssDict.values())])
	vHighMatrix  = np.array([[xs.x.upper for xs in xss] for xss in list(xssDict.values())])

	# interval (vLowMax,vHighMin) defines the wavenumber range common to all data
	vLowMin   = np.min(vLowMatrix);   vLowMax  = np.max(vLowMatrix)
	vHighMin  = np.min(vHighMatrix);  vHighMax = np.max(vHighMatrix)

	if verbose:
		print('     '+nMolecules*'%10s' % tuple(molecules))
		for l in range(nLevels):  print('%4i'%l, nMolecules*'%10i' % tuple(nFreqsMatrix[:,l]))
		print('\n spectral grid point spacing:')
		deltaV      = np.array([[xs.x.size()/(len(xs)-1) for xs in xss] for xss in list(xssDict.values())])
		for l in range(nLevels):  print('%4i'%l, nMolecules*'%10.3g' % tuple(deltaV[:,l]))
		print('\n vLow   min', vLowMin, ' max', vLowMax)
		if vLowMin<vLowMax:
			for l in range(nLevels):  print(3*'%10g' % tuple(vLowMatrix[:,l]))
		print('\n vHigh  min', vHighMin,' max', vHighMax)
		if vHighMin<vHighMax:
			for l in range(nLevels):  print(3*'%10g' % tuple(vHighMatrix[:,l]))

	if approx(vLowMin,vLowMax,1e-3) and approx(vHighMin,vHighMax,1e-3):
		print(' INFO --- xs2ac:  wavenumber limits identical for all xs:  ', vLowMin, '<= v <=', vHighMax)

		# initialize a list of absorption coefficients
		acList = []
		frmt = '\nlevel%3i %10.2emb ' + nMolecules*' %i' + ' ----> %i'

		for l in range(nLevels):
			nFreqs = [len(xssDict[mol][l]) for mol in molecules]
			nx     = max(nFreqs)

			if nLevels>1:  atmLevel = atmos[l]
			else:          atmLevel = atmos    # special case just a single levels
			if verbose:  print(frmt % tuple([l, cgs('!mb',atmLevel['p'])] + nFreqs + [nx]))

			absCo = np.zeros(nx)
			molDensityDict = {'air': atmLevel['air']}

			for molec,xss in list(xssDict.items()):  # loop over molecules
				# check p, T consistency
				if abs(xss[l].p-atmLevel['p'])/atmLevel['p']>0.01:
					raise SystemExit ('%s %s @ level # %i %12.1fK <---> %.1fK  %s' % ('ERROR --- xs2ac:',
					                  'inconsistent atmospheric and cross section pressures',
							   l, atmLevel['p'], xss[l].p, xss[l].molec))
				if abs(xss[l].t-atmLevel['T'])>0.1:
					raise SystemExit ('%s %s @ level # %i %12.1fK <---> %.1fK  %s' % ('ERROR --- xs2ac:',
					                  'inconsistent atmospheric and cross section temperatures',
							   l, atmLevel['T'], xss[l].t, xss[l].molec))

				density = atmLevel[molec]  # == atmos[molec][l]
				molDensityDict[molec] = density
				absCo  += density*xss[l].regrid(nx, interpolate)

				if verbose:  print('%5s %8.2fkm  %10g/cm**3   <ac> = %9.3g    %10.3g < ac < %9.3g' %
				                 (molec, cgs('!km',atmLevel['z']), density, np.mean(absCo), min(absCo), max(absCo)))
				if min(absCo)<0:
					XSS = xss[l].regrid(nx, interpolate)
					print ("\nWARNING --- negative absCo ",
					       molec, density, "at level", l, cgs('!km',atmLevel['z']),
					       '\n', len(xss[l]), min(xss[l]), "<xs<", max(xss[l]), " ---> regridded",
					       '\n', len(XSS),    min(XSS),    "<XS<", max(XSS), " ===> ",
					       min(absCo), "<ac<", max(absCo), " interpolation failed?!?")

			# List.append( acArray(absCo, xss[l].x, atmLevel['z'], atmLevel['p'], atmLevel['T'], join_words(molecules)) )
			acList.append( acArray(absCo, xss[l].x, atmLevel['z'], atmLevel['p'], atmLevel['T'], molDensityDict) )

			vGrid = np.linspace(xss[l].x.lower, xss[l].x.upper, nx)
			print('%5i %8.2fkm  %15i points    <ac> = %9.3g    %10.3g < ac < %9.3g  @ %10fcm-1' %
				      (l, 1e-5*atmLevel['z'], nx, np.mean(absCo), min(absCo), max(absCo), vGrid[np.argmax(absCo)]))
	else:
		raise SystemExit ('ERROR --- xs2ac:  different wavenumber grid limits (not yet implemented)!')

	return acList


####################################################################################################################################


def _xs2ac_ (atmosFile, xsFiles, outFile=None, commentChar='#',
             zToA=0.0, zBoA=0.0, scaleFactors=None,
             interpolate='linear', xFormat='%12f', yFormat='%11.5g', verbose=False):
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

	# sum molecular cross sections scaled by number density to absorption coefficient, level-by-level
	acList = xs2ac (atmos, xssDict, interpolate, verbose)

	# save data
	acSave (acList, outFile, commentChar, interpolate, xFormat, yFormat)


####################################################################################################################################

if __name__ == "__main__":

	from os.path import splitext
	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'about'},
	       {'ID': 'ToA', 'name': 'zToA', 'type': float, 'constraint': 'zToA>0.0'},
	       {'ID': 'BoA', 'name': 'zBoA', 'type': float, 'constraint': 'zBoA>=0.0'},
	       {'ID': 'scale', 'name': 'scaleFactors', 'type': np.ndarray, 'constraint': 'scaleFactors>0.0'},
	       {'ID': 'i', 'name': 'interpolate', 'type': str, 'default': '2', 'constraint': 'interpolate in "234lLqQcC"'},
	       {'ID': 'xFormat', 'name': 'xFormat', 'type': str, 'default': '%12f'},
	       {'ID': 'yFormat', 'name': 'yFormat', 'type': str, 'default': '%11.5g'},
	       {'ID': 'v',   'name': 'verbose'}
	       ]

	inFiles, options, commentChar, outFile = parse_command (opts,(2,99))

	if 'h' in options:      raise SystemExit (__doc__ + "\n End of xs2ac help")
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

	_xs2ac_ (atmFile, xsFiles, outFile, commentChar, **options)
