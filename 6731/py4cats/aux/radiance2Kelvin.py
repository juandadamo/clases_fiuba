#!/usr/bin/env python3

"""
  radiance2Kelvin
  convert radiance (intensity) to equivalent blackbody temperatures according to Planck

  usage:
    radiance2Kelvin [options] file[s]

  options:
    -c      char     comment character(s) used in files (default '#', several characters allowed)
    -C      int      number of y column to be processed (default: all y columns)
    -h               help
    -o      file     output (print) file (default: standard out)
    -P               plot:  quicklook using matplotlib
    -x      string   unit of x-axis (wavenumber, frequency or wavelength): 'cm-1' (default), 'm-1', 'Hz', 'MHz', 'GHz', 'mue'
    -y      string   unit of y-axis (radiance): 'erg/s' for erg/s/(cm^2 sr cm^-1) (default) or 'W' for W(cm^2 sr cm^-1)
   -xFormat string   format to be used for x values (default:'% 12.6f')
   -yFormat string   format to be used for y values (default:'% 10.4f')
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
##########    ToDo:                                                                                                       ##########
##########                                                                                                                ##########
##########         - use radiance2radiance for yUnit conversion                                                           ##########
##########         - frequency in MHz, GHz, ...                                                                           ##########
##########                                                                                                                ##########
####################################################################################################################################

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.ir import h, c, k, C2, pi
from py4cats.aux.aeiou import readDataAndComments, parse_comments, cstack, awrite

hcc2 = 2.0*h*c*c  # first radiation constant without pi:    3.742E-05 / pi = 1.1910428303938702e-05 erg cm**2 / s
xUnitsValid = ['cm-1', 'Hz', 'kHz', 'THz', 'GHz', 'MHz', 'mue', 'nm']

####################################################################################################################################

def ergs_to_Kelvin (x,y):
	""" Convert radiance in erg/s/(cm^2 sr cm^-1) to BlackBody temperature via inverse Planck. """
	#print("# radiance [erg/s/(cm^2 sr cm^-1)] vs wavenumber [cm-1]!")
	if isinstance(x,(int,float)) and isinstance(y,float):
		T = C2*x / np.log((hcc2*x**3/y)+1.0)
	else:
		if len(y.shape)==1:
			T = C2*x / np.log((hcc2*x**3/y)+1.0)
		else:
			T = np.zeros(y.shape,np.float)
			for j in range(y.shape[1]):  T[:,j] = C2*x / np.log((hcc2*x**3/y[:,j])+1.0)
	return T


####################################################################################################################################

def ergs_to_Kelvin_frequency (x,y, xUnit='Hz'):
	""" Convert radiance in erg/s/(cm^2 sr Hz) to BlackBody temperature via inverse Planck. """
	print("# radiance vs frequency (" + xUnit + ")!")
	print('\nWARNING:\nWARNING:  discrepancies between Hz and MHz, GHz results!?!\nWARNING')
	if xUnit=='Hz':
		H=h; C=c; c1f = 2.0*h/(c*c)
	elif xUnit=='MHz':
		H=1e6*h; C=1e-6*c; c1f = 2.0*H/(C*C)  # H=6.62e-21 erg*MHz, C=2.998e4 cm*MHz
	elif xUnit=='GHz':
		H=1e9*h; C=1e-9*c; c1f = 2.0*H/(C*C)  # H=6.62e-18 erg*GHz, C=29.98 cm*GHz
	else:
		raise SystemExit ('ERROR --- ergs_to_Kelvin_frequency: invalid/unknown frequency unit!  '+repr(xUnit))

	if len(y.shape)==1:
		T = h*x / (k*np.log((c1f*x**3 / y)+1.0))
	else:
		T = np.zeros(y.shape,np.float)
		for j in range(y.shape[1]):  T[:,j] = h*x / (k*np.log((c1f*x**3 / y[:,j])+1.0))

	return T


####################################################################################################################################

def ergs_to_Kelvin_wavelength (x,y, xUnit):
	""" Convert radiance in erg/s/(cm^2 sr micrometer) to BlackBody temperature via inverse Planck. """
	print("# radiance vs wavelength!")
	# note: first convert wavelength in micrometers to wavelength in cm to have cgs units consistently
	if xUnit=='A':
		wl = 1e-8 * x
		y  = 1e8  * y
	elif xUnit=='nm':
		wl = 1e-7 * x
		y  = 1e7  * y
	elif xUnit in ['mue', 'micrometer','mu m']:
		wl = 1e-4 * x
		y  = 1e4  * y

	if len(y.shape)==1:
		T = C2 / (wl*np.log(hcc2/(wl**5*y) + 1.0))
	else:
		T = np.zeros(y.shape,np.float)
		for j in range(y.shape[1]): T[:,j] = C2 / (wl*np.log(hcc2/(wl**5*y[:,j]) + 1.0))

	return T


####################################################################################################################################

def parse_radiance_file_header (comments, commentChar='#'):
	""" Read the header of the radiance file and try to find x and y units. """
	cDict = parse_comments (comments, ['wavenumber','frequency','wavelength','radiance', 'x', 'y'])
	if   'wavenumber' in cDict:   xUnit=cDict['wavenumber']
	elif 'wavelength' in cDict:   xUnit=cDict['wavelength']
	elif 'frequency'  in cDict:   xUnit=cDict['frequency']
	elif 'x' in cDict:            xUnit=None;  print('WARNING: no parser for "x" file header record yet, sorry!')
	else:                         xUnit=None

	if 'radiance'  in cDict:   yUnit=cDict['radiance']
	elif 'y' in cDict:         yUnit=None; print('WARNING: no parser for "y" file header record yet, sorry!')
	else:                      yUnit=None

	if not (xUnit and yUnit):
		# nothing found, try again
		for line in comments:
			entries = line.split()
			if len(entries)>1 and entries[0] in xUnitsValid:
				if entries[0] in entries[1]: xUnit=entries[0]; yUnit=entries[1];  break
		print(commentChar, 'guessing units:', xUnit, yUnit, '(no infos found in data file!)\n')

	return xUnit, yUnit


##############################################################################################################################################

def radiance2Kelvin (xGrid, yValues, xUnit='cm-1', yUnit='erg/s/(cm2 sr cm-1)', flux=False):
	""" Convert radiance spectra to equivalent Black Body Temperature by inversion of Planck's Law.

	    ARGUMENTS:
	    ----------
	    xGrid     wavenumber|frequency|wavelength
	    yValues   radiance value(s)
	    xUnit     physical unit of the xGrid data  [default cm-1]
	    yUnit     physical unit of the radiances   [default erg/s/(cm2 sr cm-1)]
	    flux      flag, assume yValues is flux, so divide by pi

	    RETURNS:
	    --------
	    temperature(s)  [Kelvin]
	"""

	if flux:
		yValues = yValues/pi;  print('--> assuming flux spectrum, dividing by pi <--')

	# better use radiance2radiance to convert to cgs units erg/s/(cm2.sr.cm-1)
	if yUnit.startswith('W'):
		yValues = 1.0e+7 * yValues
		print("# radiance Watt converted to erg/s")
	elif yUnit.startswith('nW'):
		yValues = 1.0e-2 * yValues
		print("# radiance nWatt converted to erg/s")

	if   'cm2' in yUnit or 'cm**2' in yUnit or 'cm^2' in yUnit:
		pass
	elif  'm2' in yUnit or  'm**2' in yUnit or  'm^2' in yUnit:
		yValues = 1.0e-4 * yValues
		print("# radiance per square meter converted to cm^2")

	# now convert to equ blackbody temperatures
	if xUnit in ['cm-1','1/cm']:
		T = ergs_to_Kelvin (xGrid,yValues)
	elif xUnit in ['m-1','1/m']:
		T = ergs_to_Kelvin (xGrid/100.,yValues*100.)
	elif xUnit in ['mue','micrometer','mu m', 'nm', 'A']:
		T = ergs_to_Kelvin_wavelength (xGrid,yValues, xUnit)
	elif xUnit in ['Hz','MHz','GHz']:
		T = ergs_to_Kelvin_frequency (xGrid,yValues, xUnit)
	else:
		raise SystemExit ('ERROR:  invalid/unknown unit for x-axis (frequency, wavenumber or wavelength only)!')

	return T


####################################################################################################################################

def _radiance2Kelvin_ (radFile, tmpFile, nColumn=-1, xUnit='cm-1', yUnit='erg/s/cm2/sr/cm-1',
                       xFormat=' %12.6f', yFormat=' %10.4g', commentChar='#', flux=False, plot=False):

	if nColumn>0:
		# read y from a single column only (assume x in very first column)
		xy, comments = readDataAndComments (radFile, commentChar, usecols=[0,nColumn])
		xGrid = xy[:,0];  yValues = xy[:,1];  del xy
	else:
		# read entire file, including header
		xy, comments = readDataAndComments  (radFile, commentChar)
		xGrid = xy[:,0];  yValues = xy[:,1:];  del xy

	# try to find x and y units from file header
	unitX, unitY = parse_radiance_file_header (comments)
	if not unitX:  unitX=xUnit
	if not unitY:  unitY=yUnit

	# now use inverse Planck
	T = radiance2Kelvin (xGrid, yValues, unitX, unitY, flux)

	# and save (or plot) data
	awrite (cstack(xGrid,T), tmpFile, xFormat+yFormat, comments=['%s %s' % (commentChar,cmt) for cmt in comments])

	if plot:
		from pylab import plot, show
		plot (xGrid,T); show()

####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions, multiple_outFiles
	# parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
               {'ID': 'about'},
               {'ID': 'x', 'name': 'xUnit', 'type': str, 'default': 'cm-1'},
               {'ID': 'y', 'name': 'yUnit', 'type': str, 'default': 'erg/s/cm2/sr/cm-1'},
               {'ID': 'xFormat', 'type': str, 'default': ' %12.6f'},
               {'ID': 'yFormat', 'type': str, 'default': ' %10.4f'},
               {'ID': 'C', 'name': 'nColumn', 'type': int, 'default': -1},
	       {'ID': 'f', 'name': 'flux'},
	       {'ID': 'P', 'name': 'plot'}
               ]

	files, options, commentChar, outFile = parse_command (opts,(1,99))
	outFiles = multiple_outFiles (files, outFile)

	if 'h' in options:      raise SystemExit (__doc__ +"\n End of radiance2Kelvin help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	options['plot'] = 'plot' in options
	options['flux'] = 'flux' in options

	for iFile,oFile in zip(files,outFiles):
		_radiance2Kelvin_ (iFile, oFile, **options)
