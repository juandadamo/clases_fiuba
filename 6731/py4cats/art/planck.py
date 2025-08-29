#!/usr/bin/env python3

"""
  planck.py

  usage:
  planck  [options]  xGrid  temperature(s)

  compute Planck's black body function as a function of wavenumber (or frequency, wavelength) for given temperature(s).

  xGrid:  [xStart,]xStop[,xStep]
          specification of wavenumber/frequency/wavelength grid
          comma separated (no blanks!) list of one, two, or three floats similar to Python's range function:
	  If xStart argument is omitted, it defaults to 0.0; If xStep argument is omitted, it defaults to 1.0

	  Alternatively a file can be specified here, the xGrid is then read from the file's first column

  temperatures:
          comma separated (no blanks!) list of values (Kelvin)

  options:
  -c char      comment character to be used in output (default: '#')
  -h           display help
  -m           SI area units  (default:  cgs area units [cm^2])
  -o file      output file (default: standard output)
  -r           Rayleigh-Jeans approximation
  -W           power in SI power units (default:  power in cgs units [erg/s])
  -X string    x-axis unit: wavenumber [cm-1] (default), frequency (Hz, kHz, MHz, GHz) or wavelength (cm, mue, nm, A)

  UNITS:
  default is to return Planck function values in cgs units, i.e. [erg/s / (cm^2 sr [x])]
  Using the -W and -m switches you can convert the power unit in the nominator
  or the area unit in the denominator to Watt and square meters, respectively.

"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

#### standard modules:
import sys, os

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

if __name__ == "__main__": sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.ir import c,h,k,C2
from py4cats.aux.cgsUnits import wavelengthUnits, frequencyUnits
from py4cats.aux.aeiou import awrite, join_words

C1 = 2.0*h*c*c         # first radiation constant (without pi): 1.191042e-05 erg cm**2 / s

####################################################################################################################################
#####  ToDo:                                                                                                                   #####
#####       - automatic xGrid:  better use displacement law !!!                                                                #####
####################################################################################################################################

def Planck_Wavenumber (v, temp):
	""" For a given wavenumber [1/cm] array and temperature return corresponding Planck function values. """
	B = C1*v**3 / (np.exp(C2*v/temp)-1.0)
	return B

def dPlanck_Wavenumber_dT (v, temp):
	""" For a given wavenumber [1/cm] array and temperature return derivative of Planck function wrt temperature. """
	exb  = np.exp(C2*v/temp)
	dBdT = C1*h*c*v**4 * exb / (k*temp**2 * (exb-1.0)**2)
	return dBdT

####################################################################################################################################

def Planck_Frequency (f, temp):
	""" For a given frequency [Hz] array and temperature return corresponding Planck function values. """
	hkT = h/(k*temp)
	# C1, C2 in cgs units, hence need frequency in "Hz", too !?!?!
	B   = (2.0*h/c**2) * f**3 / (np.exp(hkT*f)-1.0)
	return B

####################################################################################################################################

def Planck_Wavelength (l, temp):
	""" For a given wavelength [cm] array and temperature return corresponding Planck function values. """
	# C1, C2 in cgs units, hence need wavelength in "cm", too !!!
	B = C1 / (l**5 * (np.exp(C2/(l*temp))-1.0))
	return B


####################################################################################################################################
####################################################################################################################################

def RayleighJeans_Wavenumber (v, temp):
	""" For a given wavenumber [1/cm] array and temperature return corresponding Rayleigh-Jeans approximations. """
	B = 2.0*c*k*temp * v**2
	return B

####################################################################################################################################

def RayleighJeans_Frequency (f, temp):
	""" For a given frequency [Hz] array and temperature return corresponding Rayleigh-Jeans approximations. """
	B   = (2.0*k*temp/c**2) * f**2
	return B

####################################################################################################################################

def RayleighJeans_Wavelength(l, temp):
	""" For a given wavelength [cm] array and temperature return corresponding Rayleigh-Jeans approximations. """
	# C1, C2 in cgs units, hence need wavelength in "cm", too !?!?!
	B = (2.0*c*k*temp) / l**4
	maxExp = max(C2/(l*temp))
	if maxExp > 0.1:
		print('WARNING:  bad arguments for RayleighJeans_Approximation', maxExp)
	return B

####################################################################################################################################

def RayleighJeans (x, temp, xUnit='cm-1'):
	""" Call appropriate Rayleigh-Jeans routine (with x in cgs units, i.e. v[1/cm], f[Hz] or lambda[cm]) ."""
	if xUnit=='cm-1':
		y = RayleighJeans_Wavenumber (x, temp)
	elif xUnit in list(frequencyUnits.keys()):
		factor = frequencyUnits[xUnit]
		# compute B for frequency in Hz
		y = RayleighJeans_Frequency (factor*x, temp)
		# rescale Planck if abscissa unit is not Hz
		y = y*factor
	elif xUnit in list(wavelengthUnits.keys()):
		factor = wavelengthUnits[xUnit]
		#compute B for wavelength in cm
		y = RayleighJeans_Wavelength (factor*x, temp)
		# rescale Planck if abscissa unit is not cm
		y = y*factor
	else:
		raise SystemExit ('ERROR --- RayleighJeans:  unknown unit for x-axis, \nuse cm-1, Hz')

	return y


####################################################################################################################################
####################################################################################################################################

def planck (x, T, xUnit='cm-1'):
	""" Compute Planck blackbody function for wavenumber v[1/cm] (default) or frequency f or wavelength array and temperature T.
	    Returns B(x,T) in [power]/([area]*sr*[x])
	    default power erg/s
	            area  cm**2
	"""
	if xUnit=='cm-1':
		y = Planck_Wavenumber (x, T)
	elif xUnit in list(frequencyUnits.keys()):
		factor = frequencyUnits[xUnit]
		# compute B for frequency in Hz
		y = Planck_Frequency (factor*x, T)
		# rescale Planck if abscissa unit is not Hz
		y = y*factor
	elif xUnit in list(wavelengthUnits.keys()):
		factor = wavelengthUnits[xUnit]
		# compute B for wavelength in cm
		y   = Planck_Wavelength (factor*x, T)
		# rescale Planck if abscissa unit is not cm
		y = y*factor
	else:
		xUnits = ['cm-1'] + list(frequencyUnits.keys()) + list(wavelengthUnits.keys())
		raise SystemExit ('unknown unit for x-axis, use\n' + join_words(xUnits))

	return y

####################################################################################################################################

def _planck_ (xGridSpec, tempSpec, commentChar='#', rayleighJeans=False, xUnit='cm-1', areaSI=False, powerSI=False):
	""" Read/parse x grid specification (wavenumber|freqquency|wavelength) and temperature
	    and compute Planck's black body function.

	    Optionally use Rayleigh-Jeans approximation
	               convert to SI power (Watt) or area (m**2) units. """

	# get wavenumber/frequency/wavelength grid
	if os.path.isfile(xGridSpec):
		xGrid = np.loadtxt(xGridSpec, comments=commentChar, usecols=[0])
	else:
		xGrid = np.arange(*list(map(float,xGridSpec.split(','))))

	# get temperature(s)
	tempSpec = list(map(float,tempSpec.split(',')))
	if len(tempSpec)==1:
		temperatures = np.array(tempSpec)
	elif len(tempSpec)==2:
		if tempSpec[1]-tempSpec[0]<=100.: tempSpec.append(20.)
		else:                             tempSpec.append(100.)
		tempSpec[1]  = tempSpec[1]+tempSpec[2]
		temperatures = np.arange(*tempSpec)
	elif len(tempSpec)==3 and tempSpec[2]<0.5*(tempSpec[1]-tempSpec[0]):
		tempSpec[1] = tempSpec[1]+tempSpec[2]
		temperatures = np.arange(*tempSpec)
	else:
		temperatures = np.array(tempSpec)

	# initialize BB array
	yValues = np.zeros([len(xGrid),len(temperatures)],np.float)

	if not rayleighJeans:
		title = 'Planck black body function'
		for l,T in enumerate(temperatures):  yValues[:,l] = planck (xGrid, T, xUnit)
	else:
		title = 'Planck black body function --- Rayleigh-Jeans approximation'
		for l,T in enumerate(temperatures):  yValues[:,l] = RayleighJeans (xGrid, T, xUnit)

	# conversion to non cgs units
	if powerSI:
		yUnit   = 'W';     yValues = 1.0e-7 * yValues
	else:
		yUnit   = 'erg/s'
	if areaSI:
		yUnit += '/(m**2.sr.'+xUnit+')';   yValues = 1.0e4 * yValues
	else:
		yUnit += '/(cm**2.sr.'+xUnit+')'

	# save Planck function values in tabular ascii file
	comments = [title]
	comments.append ('temperature [K]' + len(temperatures)*' %12.2f' % tuple(temperatures))
	comments.append ('%9s    %s' % (xUnit, yUnit))

	awrite ((xGrid,yValues), outFile, ' %10f %12e', comments, commentChar=commentChar)


####################################################################################################################################

if __name__ == "__main__":
	from py4cats.aux.command_parser import parse_command, standardOptions

	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
               dict(ID='r', name='rayleighJeans'),
               dict(ID='m', name='areaSI'),
               dict(ID='W', name='powerSI'),
	       dict(ID='X', name='xUnit', default='cm-1')]

	# parse command line input
	xGrid_and_temp, options, commentChar, outFile = parse_command (opts,2)

	# check user input and conflicting options:
	if 'h' in options:   raise SystemExit (__doc__+ "\n End of planck help")
	for opt in ['rayleighJeans', 'areaSI', 'powerSI']:  options[opt] = opt in options

	xGrid, temperatures = xGrid_and_temp

	_planck_ (xGrid, temperatures, commentChar, **options)
