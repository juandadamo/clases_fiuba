#!/usr/bin/env python3

"""
  radiance2radiance
  convert radiance (intensity) vs. wavenumber to radiance vs frequency etc.

  usage:
  radiance2radiance [options] file[s]

  optional command line options without arguments:
    -h         help
    -c char    comment line character           (default: "#")
    -o file    output file                      (default: standard out)

    -a string  area  unit of input  radiance:   'cm2' (square centimeter, default) or 'm2'
    -A string  area  unit of output radiance
    -p string  power unit of input  radiance:   'erg/s' (default) or 'W' or 'mW' or 'nW'
    -P string  power unit of output radiance
    -x string  unit of x-axis on input          'cm-1' (default) or wavelength or frequency (see below)
    -X string  unit of x-axis on output

    -r         reverse sequence of x and corresponding y values (convenient for wavenumber <--> wavelength conversion)
    -C int     number of y column to be processed (default: all y columns)
    -T float   temperature [K]:  return radiance normalized to Planck radiance B(T)

  x-units currently supported: 'cm-1', 'm-1', 'Hz', 'MHz', 'GHz', 'THz', 'mue', 'nm'
  y-units currently supported: ""erg/s/([A] sr [x])  or  W/([A] sr [x])"  or  "photons/s/([A] sr [x])"
                                 (where [A] is m^2 or cm^2 and [x] is x-unit as listed above)

  For conversion of radiance to equ. Blackbody temperature use "radiance2Kelvin"
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
#####  ToDo:                                                                                             #####
#####                                                                                                    #####
#####       - read units in input file from file header                                                  #####
#####                                                                                                    #####
##############################################################################################################

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.ir import h, c
from py4cats.aux.aeiou  import readDataAndComments, awrite, cstack
from py4cats.aux.cgsUnits import wavelengthUnits # frequencyUnits also contains cm-1
from py4cats.art.planck import planck

C1 = 2.0*h*c*c         # first radiation constant (without pi)
hc = h*c

# conversion factors
frequencyFactor  = {'Hz': 1.0, 'kHz': 1.0e3, 'MHz': 1.0e6, 'GHz': 1.0e9, 'THz': 1.0e12}

####################################################################################################################################

def photons2energy (x, y, xUnit):
	""" Convert radiance given in units of photons/s/(area*sr*[x]) to erg/s/(area*sr*[x]. """
	# unit of x-axis?
	if   xUnit=='cm-1':
		for j in range(y.shape[1]):   y[:,j]  =  y[:,j] * hc * x
	elif xUnit=='Hz':
		for j in range(y.shape[1]):   y[:,j]  =  y[:,j] * h * x
	elif xUnit in list(frequencyFactor.keys()):
		for j in range(y.shape[1]):   y[:,j]  =  y[:,j] * h * frequencyFactor[xUnit]*x
	elif xUnit=='cm':
		for j in range(y.shape[1]):   y[:,j]  =  y[:,j] * hc / x
	elif xUnit in ['mue','micrometer','nm','A']:
		for j in range(y.shape[1]):   y[:,j]  =  y[:,j] * hc / (wavelengthUnits[xUnit]*x)
	else:
		raise SystemExit ('ERROR --- photons2energy: invalid/unknown X unit ' + repr(xUnit))
	return y


####################################################################################################################################

def radiance2radiance_Y (y, oldA, newA, oldP, newP):
	""" Convert radiance to a new Y unit (i.e. change power or area unit). """
	# scale radiance to convert power unit (multiplicative, does not effect x unit)
	if oldP==newP:
		if oldP not in ['W','mW','uW','nW','erg/s']:
			raise SystemExit ('ERROR: invalid/unknown power unit (only W, mW, uW, nW, or erg/s)!')
	elif newP=='erg/s':
		if   oldP=='W':     y  = 1.0e+7 * y
		elif oldP=='mW':    y  = 1.0e+4 * y
		elif oldP=='uW':    y  = 1.0e+1 * y
		elif oldP=='nW':    y  = 1.0e-2 * y
		else:               raise SystemExit ('ERROR: invalid/unknown power unit (only W, mW, uW, nW, or erg/s)!')
	elif newP=='W':
		if   oldP=='erg/s': y  = 1.e-7 * y
		elif oldP=='mW':    y  = 1.e-3 * y
		elif oldP=='uW':    y  = 1.e-6 * y
		elif oldP=='nW':    y  = 1.e-9 * y
		else:               raise SystemExit ('ERROR: invalid/unknown power unit (only W, mW, uW, nW, or erg/s)!')
	elif newP=='mW':
		if   oldP=='erg/s': y  = 1.e-4 * y
		elif oldP=='W':     y  = 1.e+3 * y
		elif oldP=='uW':    y  = 1.e-3 * y
		elif oldP=='nW':    y  = 1.e-6 * y
		else:               raise SystemExit ('ERROR: invalid/unknown power unit (only W, mW, uW, nW, or erg/s)!')
	elif newP=='uW':  # microWatt
		if   oldP=='erg/s': y  = 1.e-1 * y
		elif oldP=='W':     y  = 1.e+6 * y
		elif oldP=='mW':    y  = 1.e+3 * y
		elif oldP=='nW':    y  = 1.e-3 * y
		else:               raise SystemExit ('ERROR: invalid/unknown power unit (only W, mW, uW, or erg/s)!')
	elif newP=='nW':
		if   oldP=='erg/s': y  = 1.e+2 * y
		elif oldP=='W':     y  = 1.e+9 * y
		elif oldP=='mW':    y  = 1.e+6 * y
		elif oldP=='uW':    y  = 1.e+3 * y
		else:               raise SystemExit ('ERROR: invalid/unknown power unit (only W, mW, uW, or erg/s)!')
	else:
		raise SystemExit (oldP + ' --> ' + newP + '\nERROR: invalid/unknown power unit conversion!')

	# scale radiance to convert area unit (multiplicative, does not effect x unit)
	if oldA==newA:
		if oldA not in ['m2', 'm**2', 'm^2', 'cm2', 'cm**2', 'cm^2']:
			raise SystemExit ('ERROR: invalid/unknown area unit!')
	elif newA in ['m2', 'm**2', 'm^2']:
		if oldA in ['cm2', 'cm**2', 'cm^2']:  y *= 1.e+4
		else:                                 raise SystemExit ('ERROR: invalid/unknown area unit conversion to m^2!')
	elif newA in ['cm2', 'cm**2', 'cm^2']:
		if oldA in ['m2', 'm**2', 'm^2']:    y *= 1.e-4
		else:                                raise SystemExit ('ERROR: invalid/unknown area unit conversion to cm^2!')
	else:
		raise SystemExit ('ERROR: invalid/unknown area unit conversion!')

	yUnit = newP + ' / (' + newA + ' sr [x])'
	print('radiance2radiance_Y ---> ', yUnit, end=' ')
	return y, yUnit


####################################################################################################################################

def radiance2radiance_X (x, y, oldX, newX, yUnit):
	""" Convert radiance to a new X unit. """
	# conversion due to change of abscissa unit
	if   oldX==newX:
		print()
	elif oldX=='cm-1':
		if newX.endswith('Hz'):
			scaleY = frequencyFactor[newX]/c
			x = x / scaleY
			y = y * scaleY
		elif newX in ['1/m', 'm-1']:
			x = x * 100.
			y = y / 100.
		elif newX in ['mue','micro',',micrometer','um','nm']:
			scaleY = 1/wavelengthUnits[newX]
			# intensity:  [power]/([area]*sr*cm-1)  ---> [power]/([area]*sr*cm)
			y = np.transpose(y.T * (x*x))       # vector and matrix, for j in range(y.shape[1]): y[:,j] = y[:,j] * (x*x)
			# intensity:  [power]/([area]*sr*cm)   --->  [power]/([area]*sr*mue)  or  [power]/([area]*sr*nm)
			y = y/scaleY
			# wavenumber:  [cm-1]   --->  wavelength [mue] or [nm]
			x = scaleY / x
	elif oldX.endswith('Hz'):
		if newX=='cm-1':
			scaleY = c/frequencyFactor[oldX]
			x = x / scaleY
			y = y * scaleY
		elif newX.endswith('Hz'):
			factor = frequencyFactor[newX]/frequencyFactor[oldX]
			x = x / factor
			y = y * factor
		elif newX in ['cm','mue','nm','A']:
			# conversion to radiance vs Hz
			x = x*frequencyFactor[oldX]  # freq[Hz]
			y = y/frequencyFactor[oldX]
			# I_dlambda = f^2 I_df / c  --> radiance vs lambda[cm]
			#xx = x**2/c
			y = np.transpose(y.T * (x*x))      # vector and matrix, for j in range(y.shape[1]): y[:,j] = y[:,j] * xx
			x = c/x  # lambda [cm]
			x = x/wavelengthUnits[newX]
			y = y*wavelengthUnits[newX]
		else:
			raise SystemExit ('ERROR: sorry, not yet implemented, frequency to ' + newX)
	elif oldX in ['mue','micro',',micrometer']:
		if newX=='cm-1':
			# conversion to radiance vs cm
			x = x*wavelengthUnits[oldX]  # lambda[cm]
			y = y/wavelengthUnits[oldX]
			# I_dnu = lambda^2 I_dlambda
			y = np.transpose(y.T * (x*x))      # vector and matrix, for j in range(y.shape[1]): y[:,j] = y[:,j] * xx
			x = 1/x  # nu [cm-1]
		elif newX=='nm':
			x = 1.e+3 * x
			y = 1.e-3 * y
		elif newX in ['mue','micro',',micrometer']:
			newX='mue'
		else:
			raise SystemExit ('ERROR: sorry, not yet implemented, wavelength to frequency')
	elif oldX=='nm':
		if newX=='cm-1':
			# conversion to radiance vs cm
			x = x*wavelengthUnits[oldX]  # lambda[cm]
			y = y/wavelengthUnits[oldX]
			# I_dnu = lambda^2 I_dlambda
			y = np.transpose(y.T * (x*x))      # vector and matrix, for j in range(y.shape[1]): y[:,j] = y[:,j] * xx
			x = 1/x
		elif newX[:3] in ['mue','mic']:
			newX='mue'
			x = 1.e-3 * x
			y = 1.e+3 * y
		else:
			raise SystemExit ('ERROR: sorry, not yet implemented, wavelength to frequency')
	else:
		raise SystemExit ('ERROR: invalid/unknown X unit conversion ' + oldX + ' --> ' + newX)

	# adjust radiance unit string
	ix = yUnit.find('[x]')
	if ix>0:
		yUnit = yUnit[:ix] + newX + yUnit[ix+2:]
	else:
		ix = yUnit.find(oldX)
		head = yUnit[:ix];  tail = yUnit[ix+len(newX):]
		yUnit = head + newX + tail
	print('radiance2radiance_X ---> ', yUnit)
	return x, y, yUnit


####################################################################################################################################

def normalized_radiance (x, y, temp, xUnit,  yUnit):
	""" Convert Radiance (in power / (area*sr*[X])) to normalized radiance by division of Planck function at given temperature. """
	# compute planck function vs wavenumber, wavelength, or frequency (as given by xUnit) in cgs units of erg/s/(cm2 sr [x])
	B = planck (x, temp, xUnit)
	print('Planck:', yUnit, temp)
	# correct for power unit unit not in cgs
	if   yUnit[:1]=='W':    B = 1.0e-7 * B
	elif yUnit[:2]=='mW':   B = 1.0e-4 * B
	elif yUnit[:2]=='nW':   B = 1.0e+2 * B
	# correct square meter area unit
	if yUnit.find('(m')>0:  B = 1.e4*B
	# now divide radiance by Planck
	for j in range(y.shape[1]):  y[:,j] = y[:,j] / B
	yInfo = 'normalized radiance:  I / B(' + repr(temp) + 'K)'

	return x,y, yInfo


####################################################################################################################################

def radiance2radiance (xGrid, yValues, oldX='cm-1', newX='cm-1', oldA='cm2', newA='cm2', oldP='erg/s', newP='erg/s', verbose=False):
	""" Convert radiance, e.g. wavenumber <--> frequency <--> wavelength  or  area unit  or power unit.

	ARGUMENTS:
	----------
	xGrid          wavenumber, frequency, or wavelength abscissa
	yValues        radiance values defined on the xGrid
	oldX, newX     old and new unit of xGrid, default wavenumber [cm-1]
	oldA, newA     old and new area unit of radiance, default [cm**2]
	oldP, newP     old and new power unit of radiance, default [erg/s]

	verbose        additionally return two strings with x and y infos
	               default:  only new xGrid and yValues arrays
		       (NOTE:    when oldX!=newX, the xGrid will change, too)
	RETURNS:
	--------
	xGrid, yValues:  two arrays with the new abscissa and radiances
	                 !!! optionally return x and y infos, too !!!
	"""

	# convert photon number to power
	if oldP[:6]=='photon':
		yValues = photons2energy (xGrid, yValues, oldX)
		oldP = 'erg/s'
		print('  photons  --->  ', oldP, end=' ')

	yValues, yUnit        = radiance2radiance_Y (       yValues, oldA, newA, oldP, newP)
	xGrid, yValues, yUnit = radiance2radiance_X (xGrid, yValues, oldX, newX, yUnit)

	if   newX in ['cm-1','m-1']:    xInfo = 'wavenumber: ' + newX
	elif newX[-2:]=='Hz':           xInfo = 'frequency:  ' + newX
	elif newX[-2:] in ['mue','nm']: xInfo = 'wavelength: ' + newX
	else:                           xInfo = newX

	yInfo = 'radiance:   ' + yUnit

	if verbose:  return xGrid, yValues, xInfo, yInfo
	else:        return xGrid, yValues


####################################################################################################################################

def _radiance2radiance_ (iFile, oFile,  oldX='cm-1', newX='cm-1', oldA='cm2', newA='cm2', oldP='erg/s', newP='erg/s',
                         temperature=0, flip=0, commentChar='#'):

	if nColumn>0:
		# read y from a single column only (assume x in very first column)
		data, comments = readDataAndComments (iFile, commentChar, usecols=[0,nColumn])
		xGrid = data[:,0];  yValues = np.atleast_2d(data[:,1]).T;  del data
	else:
		# read entire file, including header
		data, comments = readDataAndComments (iFile, commentChar)
		xGrid = data[:,0];  yValues = data[:,1:];  del data

	xGrid, yValues, xInfo, yInfo = radiance2radiance (xGrid, yValues, oldX, newX, oldA, newA, oldP, newP, True)

	if flip:
		xGrid   = xGrid[::-1]
		yValues = yValues[::-1]

	if temperature>0.0:
		if not ('A' in options or 'P' in options):
			newRadUnit = '%s / (%s.%s.%s)' % (newP, newA, 'sr', newX)
			xGrid, yValues, yInfo = normalized_radiance (xGrid, yValues, temperature, newX, newRadUnit)
		else:
			raise SystemExit ('new power and/or area units for radiance irrelevant for normalized radiance!')

	# infos for file header:  old file header and  name, units of x, y appended
	Comments = [comment.strip() for comment in comments]
	Comments = Comments + ['---> ' + info.strip() for info in [xInfo,yInfo]]

	# save converted radiance
	awrite (cstack(xGrid,yValues), oFile, " %12f %12.5e", Comments)


####################################################################################################################################

if __name__ == "__main__":
	from py4cats.aux.command_parser import parse_command, standardOptions, multiple_outFiles

	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
               {'ID': 'about'},
               {'ID': 'x', 'name': 'oldX', 'type': str, 'default': 'cm-1'},
               {'ID': 'X', 'name': 'newX', 'type': str, 'default': 'cm-1'},
               {'ID': 'p', 'name': 'oldP', 'type': str, 'default': 'erg/s'},
               {'ID': 'P', 'name': 'newP', 'type': str, 'default': 'erg/s'},
               {'ID': 'a', 'name': 'oldA', 'type': str, 'default': 'cm2'},
               {'ID': 'A', 'name': 'newA', 'type': str, 'default': 'cm2'},
               {'ID': 'C', 'name': 'nColumn', 'type': int, 'default': None},
               {'ID': 'r', 'name': 'reverse'},
               {'ID': 'T', 'name': 'temperature',  'type': float, 'default': 0.0}
               ]

	files, options, commentChar, outFile =  parse_command (opts,(1,99))
	outFiles = multiple_outFiles (files, outFile)

	if 'h' in options:      raise SystemExit (__doc__ + "\n End of radiance2radiance help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	if options['oldX']==options['newX'] and options['oldP']==options['newP'] and options['oldA']==options['newA']:
		raise SystemExit ('no new units specified, no conversion!')

	options['flip'] = 'reverse' in options
	if options['flip']: del options['reverse']
	temperature = options.pop('temperature')
	nColumn     = options.pop('nColumn')

	# now loop over files
	for iFile,oFile in zip(files,outFiles):
		_radiance2radiance_ (iFile, oFile,  **options)
