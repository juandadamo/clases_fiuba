#!/usr/bin/env python3
"""
atmos1D

Read atmospheric data file(s) in xy tabular ascii format:
extract profiles (columns), convert to cgs units, optionally truncate top or bottom levels, save in xy format.
If several files are given, the profiles from the second, third, ... file are interpolated to the grid of the first file.

Usage:

  atmos1D [options] file(s)

  -h              help
  -c    char      comment character(s) used in input,output file (default '#')
  -o    string    output file for saving of atmospheric data
  -i    string    interpolation method (default: numpy's linear interp)
  -r              replace duplicate profile from preceeding file(s) with new data
                  (default:  ignore duplicate profile(s) from second, third, ... file)
  -w    string    the leading word identifying the header row specifying the columns  (default: "what")
  -u    string    the leading word identifying the header row specifying the units    (default: "units")
  -x    string    eXtract profiles from file(s) (a string with comma seperated entries (default: "z,p,T,H2O"))
 --BoA   float    bottom-of-atmosphere altitude in km
 --ToA   float    top-of-atmosphere altitude in km
 --scale floats   multiply molecular concentrations with scaleFactors
                  (either a comma separated list of floats (no blanks) in the same order as for the line data files,
                  or just a single float to scale the profile of the molecule corresponding to the first lineFile.)
  -z     floats   regrid to (evtl. equidistant) altitude grid
"""

_LICENSE_ = """\n
This file is part of the Py4CAtS package.

Authors:
Franz Schreier
DLR Oberpfaffenhofen
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
####################################################################################################################################

import sys
import re
from string import ascii_uppercase

if __name__ == "__main__":
	import os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

try:
	from scipy.interpolate import pchip_interpolate
except ImportError as msg:
	print(str(msg) + '\nWARNING -- atmos1D: "from scipy.interpolate import pchip_interpolate failed", use default numpy.interp')
else:
	pass # print(' from scipy.interpolate import pchip_interpolate')

from py4cats.aux.aeiou import grep_from_header, awrite, join_words
from py4cats.aux.struc_array import loadStrucArray, strucArrayAddField, strucArrayDeleteFields
from py4cats.aux.ir import k as kBoltzmann
from py4cats.aux.cgsUnits import unitConversion, cgs, mixingRatioUnits, lengthUnits, pressureUnits

from py4cats.lbl.molecules import molecules
molecNames   = list(molecules.keys())
#molecNamesLo = [mol.lower() for mol in molecNames]

simpleNames={'height':'z', 'heights':'z', 'altitude':'z', 'altitudes':'z',
             'Height':'z', 'Heights':'z', 'Altitude':'z', 'Altitudes':'z',
             'press':'p', 'pressure':'p', 'pressures':'p',
             'Press':'p', 'Pressure':'p', 'Pressures':'p',
             'temp': 'T', 'temperature': 'T', 'temperatures': 'T',
             'Temp': 'T', 'Temperature': 'T', 'Temperatures': 'T',
             'dens':'air', 'Density':'air', 'density':'air',
             'HGT': 'z',  'PRE': 'p',  'TEM': 'T'}  # the last three 3-letter names are used in the MIPAS-RFM files
goodNames = list(simpleNames.keys()) + ['z', 'p', 'T', 'air'] + molecNames


####################################################################################################################################

def atmTruncate (atmos, zToA=None, zBoA=None, verbose=False):
	""" Truncate atmosphere at top or bottom, i.e. delete levels for altitudes above zToA or below zBoA. """

	if not (zToA or zBoA):
		print('WARNING --- atmTruncate:  neither zToA nor zBoA given, nothing to do, returning data as is!')

	if zToA:
		if zToA<250.:
			zToA = cgs('km', zToA)
			print(' WARNING --- atmTruncate:  zToA very small, assuming kilometer units')
		nz = sum([1 for z in atmos['z'] if z<=zToA])
		if verbose:
			print(' atmos1D.atmTruncate --- ToA', cgs('!km', zToA), 'km,  skip ', len(atmos['z'])-nz,' levels at top')
		atmos = atmos[:nz]
	if zBoA:
		if zBoA<250.:
			zBoA = cgs('km', zBoA)
			print(' WARNING --- atmTruncate:  zBoA very small, assuming kilometer units')
		nz = len(atmos['z']) - sum([1 for z in atmos['z'] if z>=zBoA])
		if verbose:
			print(' atmos1D.atmTruncate --- BoA', cgs('!km',zBoA), 'km,  skip ', nz,' levels at bottom')
		atmos = atmos[nz:]

	if verbose:
		print(' atmos1D.atmTruncate --- returning ', len(atmos.dtype.names), 'profiles at ', len(atmos), ' levels')
	return atmos


####################################################################################################################################

def vcd (atmos, what=None, zMin=None, zMax=None):
	""" Vertical column densities [molec/cm**2] for air and individual molecules.
	    If a gas (or "air") is specified, return only this particular vcd.

	    Returns integral_zMin^zMax density dz

	    ARGUMENTS:
	    what   default None, i.e. return all VCDs
	    zMin   lower integral limit (default BoA)
	    zMax   upper integral limit (default ToA)

	    Notes:
	    * small zMin, zMax values (z<250) are interpreted as km and automatically converted to cm;
	    * the integral(s) start/stop at the given altitude levels, i.e. zMin or zMax is "rounded" to the next grid point.  """
	# check altitudes
	if zMin and isinstance(zMin,(int,float)):
		if zMin<250.0:  zMin=1e5*zMin  # apparently altitude in km
		lMin = np.argmin(abs(atmos['z']-zMin))
	else:
		lMin = 0
	if zMax and isinstance(zMax,(int,float)):
		if zMax<250.0:  zMax=1e5*zMax  # apparently altitude in km
		lMax = np.argmin(abs(atmos['z']-zMax))+1
	else:
		lMax = len(atmos)

	# altitude steps for trapezoid rule
	deltaZ = atmos['z'][lMin+1:lMax] - atmos['z'][lMin:lMax-1]

	if what:
		if what not in gases(atmos)+['air']:  raise SystemExit ('ERROR --- atmos.vcd:  unknown gas '+what)
		return  0.5*sum(deltaZ*(atmos[what][lMin+1:lMax] + atmos[what][lMin:lMax-1]))
	else:
		columns = np.zeros(len(atmos.dtype.names))
		for m,gas in enumerate(atmos.dtype.names):
			if gas not in ['z', 'p', 'T']:
				columns[m] = 0.5*sum(deltaZ*(atmos[gas][lMin+1:lMax] + atmos[gas][lMin:lMax-1]))
		return np.extract(columns>0, columns)


def cmr (atmos, what=None, zMin=None, zMax=None, verbose=False):
	""" Column mixing ratio:  quotient of molecular vertical column densities and air vertical column density. """
	if what:
		column = vcd(atmos, what, zMin, zMax)
		vcdAir = vcd(atmos, 'air', zMin, zMax)
		ratio  = column/vcdAir
		if verbose:
			print ('vcd [molec/cm**2]   %s  %.3g    air  %.3g   ===>  ratio [ppm]  %.3g' %
			       (what, column, vcdAir, 1e6*ratio))
		return ratio
	else:
		# first integrate number densities to columns
		columns = vcd(atmos, zMin=zMin, zMax=zMax)
		# ... and divide by air column
		ratios = columns/vcd(atmos, 'air', zMin, zMax)
		if verbose:
			print (36*' ', len(columns)*'%12s' % tuple([name for name in atmos.dtype.names if name not in ['z','p','T']]))
			print (' Vertical Column Densities [molec/cm**2]\n', 36*' ', len(columns)*' %11.4g' % tuple(columns))
			print (' CMR = Column Mixing Ratio\n',36*' ', len(columns)*' %11.4g' % tuple(ratios))
		return ratios


####################################################################################################################################

def pT (atmos):
	""" Pressure and temperature:
	    given a numpy structured array of atmospheric data (z, p, T, gases)
	    return a numpy array with shape (nLevels,2). """
	ptArray = np.array([atmos[name] for name in 'p T'.split()]).T
	return ptArray


def zpT (atmos):
	""" Altitude, pressure and temperature:
	    given a numpy structured array of atmospheric data (z, p, T, gases)
	    return a numpy array with shape (nLevels,3). """
	zptArray = np.array([atmos[name] for name in 'z p T'.split()]).T
	return zptArray


def densities (atmos, what=None):
	""" Densities (1/cm**3) of molecules in atmospheric data, return a numpy array with shape (nLevels,nGases).
	    If a gas (or "air") is specified, return only this particular density. """
	if what:
		if what not in gases(atmos)+['air']:  raise SystemExit ('ERROR --- atmos.densities:  unknown gas '+what)
		return atmos[what]
	else:
                return np.array([atmos[name] for name in atmos.dtype.names if name not in ['z','p','T','air']]).T


def vmr (atmos, what=None):
	""" Volume mixing ratios of molecules in atmospheric data, return a numpy array with shape (nLevels,nGases).
	    If a gas is specified, return only this particular VMR. """
	if what:
		if what not in gases(atmos):  raise SystemExit ('ERROR --- atmos.vmr:  unknown gas '+what)
		return atmos[what]/atmos['air']
	else:
                return np.array([atmos[name]/atmos['air'] for name in atmos.dtype.names if name not in ['z','p','T','air']]).T


def dry_vmr (atmos, what=None):
	""" Volume mixing ratios of molecules in atmospheric data with respect to dry air,
	    return a numpy array with shape (nLevels,nGases).
	    If a gas is specified, return only this particular VMR. """
	if 'H2O' not in gases(atmos):  raise SystemExit ('ERROR --- atmos.dry_vmr:  no H2O in data')
	if what:
		if what not in gases(atmos):  raise SystemExit ('ERROR --- atmos.dry_vmr:  unknown gas '+what)
		return atmos[what]/(atmos['air']-atmos['H2O'])
	else:
                return np.array([atmos[name]/(atmos['air']-atmos['H2O']) for name in atmos.dtype.names
		                                                         if name not in ['z','p','T','air','H2O']]).T


def gases (atmos, what=None):
	""" Return names of molecules in atmospheric data.
	    If a gas is specified, only check if it is present. """
	if what:  return what in atmos.dtype.names
	else:     return [name for name in atmos.dtype.names if name not in ['z','p','T','air','density']]


def scaleHeight (atmos):
	""" Return the scale height [cm], i.e. the reciprocal of the slope of log(pressure) vs. altitude. """
	nLevels = len(atmos)
	zGrid   = atmos['z']
	pValues = np.log(atmos['p'])
	# for simplicity use the normal equation solution
	# (S. Brandt: Datenanalyse (B.I. Wissenschaftsverlag, 1981): Eq. (12.1.7))
	slope = (nLevels*sum(zGrid*pValues)-sum(zGrid)*sum(pValues)) / (nLevels*sum(zGrid*zGrid) - sum(zGrid)**2)
	return -1.0/slope


def scaleDensities (data, scaleFactors, gas=None, verbose=False):
	""" Scale molecular densities (either a single gas or all).
	    If a single factor is given and no gas is specified, scale the very first.
        """
	if verbose:  awrite (data, format=' %10.3g', comments=len(data.dtype)*' %-10s' % tuple(data.dtype.names))
	# get molecular names
	atmGases = gases(data);  nGases=len(atmGases)

	if   isinstance(scaleFactors,np.ndarray):              pass
	elif isinstance(scaleFactors,(int,float,list,tuple)):  scaleFactors = np.atleast_1d(scaleFactors)
	else:   raise SystemExit (' ERROR ---- atmos1D.scaleDensities:  invalid type of scaleFactors')

	newData = data.copy()  # initialize a true copy

	if len(scaleFactors)==1:
		if gas:
			if gas not in gases(data):  raise SystemExit ('ERROR --- atmos.scaleDensities:  unknown gas '+gas)
			print('\n scaleFactor for ', gas, ' density:', scaleFactors[0])
			newData[gas] = scaleFactors[0]*newData[gas]  # scale this gas only
		else:
			print('\n scaleFactor for ', atmGases[0], ' density:', scaleFactors[0])
			newData[atmGases[0]] = scaleFactors[0]*newData[atmGases[0]]  # scale first gas only
	elif len(scaleFactors)==nGases:
		print('\n scaleFactors for molecular densities:', atmGases, scaleFactors)
		for m, name in enumerate(atmGases):  newData[name] = scaleFactors[m]*newData[name]
	else:
		raise SystemExit (' ERROR ---- atmos1D.scaleDensities:  inconsistent length of scaleFactors and density profiles')

	if verbose:  awrite (newData, format=' %10.3g', comments=len(newData.dtype)*' %-10s' % tuple(newData.dtype.names))

	return newData


####################################################################################################################################

def atmPlot (data, what='T', vertical='km', vmr="", **kwArgs):
	""" Plot atmospheric profile(s) vs. altitude (or pressure).

	    ARGUMENTS:
	    ----------
	    data:      either a numpy structured array of atmospheric data (z, p, T, gases)
	               OR a list thereof
	    what:      the data field to plot, default T (=temperature)
	    vertical:  unit for vertical axis, default 'km' for altitude
	               any of the altitude or pressure units defined in cgsUnits supported
	    vmr:       plot volume mixing ratio instead of number density (default number density)
	               (ignored for non-gas profiles and when vmr is not a valid mixingRatioUnit)
	    kwArgs     passed directly to plot / semilogy and can be used to set colors, line styles and markers etc.
	               ignored (cannot be used) in recursive calls with lists or dictionaries of atmos data.
	"""
	try:
		from matplotlib.pyplot import plot, semilogy, xlabel, ylabel, ylim
	except ImportError as msg:
		raise SystemExit (str(msg) + '\nERROR --- atmos1D.atmPlot:  matplotlib not available, no quicklook!')

	if isinstance(vmr,(bool,int)) and vmr:  vmr='ppV'

	if isinstance(data,(list,tuple)):
		showVMR = vmr in list(mixingRatioUnits.keys())
		if kwArgs:  print ("WARNING --- atmPlot:  got a list of atmospheres, ignoring kwArgs!")
		for dd in data:
			atmPlot (dd, what, vertical, vmr)
	elif isinstance(data,dict):
		showVMR = vmr in list(mixingRatioUnits.keys())
		if kwArgs:  print ("WARNING --- atmPlot:  got a dictionary of atmospheres, ignoring kwArgs!")
		for key,dd in data.items():
			atmPlot (dd, what, vertical, vmr, label=key)
	elif isinstance(data,np.ndarray) and data.dtype.names:
		showVMR = vmr in list(mixingRatioUnits.keys()) and what in gases(data)
		if what not in data.dtype.names:
			raise SystemExit ("ERROR --- atmPlot:  atmospheric data do not include " + what
			                  + "\n            " + join_words(data.dtype.names))

		if showVMR:  values = cgs('!'+vmr,data[what]/data['air'])
		else:        values = data[what]

		if   vertical in lengthUnits:
			plot (values, unitConversion(data['z'], 'length',   new=vertical), **kwArgs)
		elif vertical in pressureUnits:
			semilogy (values, unitConversion(data['p'], 'pressure', new=vertical), **kwArgs)
		else:
                        raise SystemExit ('%s\nz-units:  %s\np-units:  %s' %
                                          ('ERROR --- atmPlot:  incorrect unit for vertical axis, only length or pressure',
                                          join_words(lengthUnits.keys()), join_words(pressureUnits.keys())))
	else:
		raise SystemExit ("ERROR --- atmPlot:  incorrect datatype, expected numpy structured array of atmospheric data or list thereof")

	if   what=='T':  xlabel (r'Temperature $T \rm\,[K]$')
	elif what=='p':  xlabel (r'Pressure $p \rm\,[g/cm/s^2]$')
	elif showVMR:    xlabel ('%s [%s]' % (what, vmr))
	else:            xlabel (what + ' [cm$^{-3}]$')

	if   vertical in lengthUnits:
		ylabel (r'Altitude $z$ ['+vertical+']')
	else:
		ylabel (r'Pressure $p$ ['+vertical+']')
		pMin, pMax = ylim()
		if pMin<pMax:  ylim(pMax,pMin)  # flip axis up/down


####################################################################################################################################

def atmInfo (data):
	""" Print some basic information about the atmospheric data. """

	if isinstance(data,(list,tuple)):
		for dd in data:  atmInfo (dd)
	elif isinstance(data,np.ndarray) and data.dtype.names:
		nMol = len(gases(data))
		print('%3i %s %6.2f ... %6.2f %s' %
		      (len(data), ' levels in ', unitConversion(data['z'][0],  'z', new='km'),
						  unitConversion(data['z'][-1], 'z', new='km'),'km'))
		print('%3i %s ' % (nMol, ' molecules'), '       air', nMol*' %10s' % tuple(gases(data)))
		frmt = (nMol+1)*' %10.3g'+'\n'
		print(' VCD [molec/cm**2]' + frmt % tuple(vcd(data)))
	else:
		raise SystemExit ("ERROR --- atmInfo:  incorrect datatype, expected numpy structured array of atmospheric data or list thereof")


####################################################################################################################################

def atmSave (atmos, outFile=None, units=None):
	""" Write atmospheric data to ascii tabular file. """

	columnNames = len(atmos.dtype)*'%11s ' % tuple(atmos.dtype.names)
	columnIDs   = ['what:   ' + columnNames.strip()]

	if isinstance(units,(list,tuple)) and len(units)==len(atmos.dtype):
		columnIDs.append('units:' + len(units)*'%11s ' % tuple(units))
	elif isinstance(units,str):
		columnIDs.append('units:' + units)
	elif units:
		units=''
		for what in atmos.dtype.names:
			if   what=='z':  units +='%11s ' % 'cm'
			elif what=='p':  units +='%11s ' % 'g/cm/s**2'
			elif what=='T':  units +='%11s ' % 'K'
			elif what in molecNames+['air']:  units +='%11s ' % '1/cm**3'
			else: units +='%11s ' % '???'
		columnIDs.append('units:  ' + units.strip())

	awrite (atmos, outFile, format='%11.2f %11.5g', comments=columnIDs)

	if not outFile and 'air' in atmos.dtype.names:
		columns = vcd(atmos)
		cmr     = columns/vcd(atmos,what='air')
		frmt    = len(cmr)*' %11.4g'+'\n'
		sys.stdout.write ('\n# VCD = Vert. Column Density [1/cm^3]' + frmt % tuple(columns))
		sys.stdout.write (  '# CMR = Column Mixing Ratio          ' + frmt % tuple(cmr))


def atmSave_namelist (atmos, outFile=None, vmr=False, zpT=False):
	""" Write atmospheric data to namelist formatted file (used by GARLIC). """

	nLevels = len(atmos)

	if outFile:  out = open(outFile,'w')
	else:        out = sys.stdout

	frmt = " %s = %s            unit='%s'\n            %s = " + nLevels*"%10.2f" + " /\n\n"
	out.write ( frmt % tuple(["&ALTITUDE  nz", nLevels, 'km','z'] + list(cgs('!km',atmos['z'])) ) )
	out.write ( frmt % tuple(["&PROFILE   what", "'temperature'", 'K', 'prof'] + list(atmos['T']) ) )

	frmt = " %s = %s            unit='%s'\n            %s = " + nLevels*"%10.3g" + " /\n\n"
	out.write ( frmt % tuple(["&PROFILE   what", "'pressure'", 'mb', 'prof'] + list(cgs('!mb',atmos['p'])) ) )

	if not zpT:
		for gas in gases(atmos):
			if vmr:  out.write ( frmt %
			                     tuple(["&PROFILE   what", repr(gas), 'ppV', 'prof'] + list(atmos[gas]/atmos['air']) ) )
			else:    out.write ( frmt % tuple(["&PROFILE   what", repr(gas), '1/cm**3', 'prof'] + list(atmos[gas]) ) )

	if outFile:  out.close()


####################################################################################################################################

def atmRegrid (data, zNew, interpolate='linear'):
	""" Interpolate atmospheric data (given as structured array) to a new altitude grid.

	    ARGUMENTS:
	    ----------
	    data          original atmospheric data (structured array)
	    zNew          the new altitude grid (list or numpy array)
	    interpolate   select a specific method,  default "" for linear Lagrange with numpy.interp
			  0 | z   zero order spline
	                  1 | s   first order spline
			  2 | q   second order spline
			  3 | c   third order spline
			  p | n   previous or next, i.e. use the left/right point
			  h       Piecewise cubic Hermite (scipy.pchip)
                          (see scipy.interpolate.interp1d)

	    HINT:
	    you can use the parseGridSpec function of the grid module to generate a new altitude grid,
	    e.g.:  zNew=parseGridSpec('0[2]16[4]60');  atmNew=atmRegrid(atmOld,zNew)

	    NOTE:
	    All atmospheric data incl. the altitude grid are in cgs units, hence z[cm].
	    However, for convenience zNew can be given in km:
	    if all new altitude values are 'small' (z<500), they are silently assumed to be kilometer and scaled to cm.
	"""

	if isinstance(interpolate,int):  # replace integer spec by string, fall-back to numpy's linear
		interpolate = {0: 'zero', 1: 'slinear', 2: 'quadratic', 3: 'cubic'}.get(interpolate,'linear')

	# import requested interpolation method first, switch to default numpy linear interpolation in case of problems
	if isinstance(interpolate,str) and len(interpolate)>0 and interpolate[0].lower() in 'zsqcnp0123':
		try:                 from scipy.interpolate import interp1d
		except ImportError:  interpolate=' '; print(' import scipy.interpolate.interp1d failed, using numpy.interp')
		else:
			interpolate = {'0': 'zero', '1': 'slinear', '2': 'quadratic', '3': 'cubic',
			               'z': 'zero', 's': 'slinear', 'q': 'quadratic', 'c': 'cubic',
				       'n': 'next', 'p': 'previous'}[interpolate[0].lower()]
			print (' from scipy.interpolate import interp1d;  kind=', repr(interpolate))
	elif interpolate.startswith('h') or interpolate.startswith('H'):
		if not globals().get('pchip_interpolate'):
			interpolate = 'linear'; print ("atmRegrid using numpy.interp, pchip_interpolate not available")
	else:
		interpolate = 'linear'
		print(' atmRegrid with default numpy.interp linear interpolation')

	# check altitude units (need cgs)
	if max(zNew)<500.:
		zGrid = cgs('km') * zNew
		print(' WARNING --- atmos1D.atmRegrid:  all altitudes<500, assuming km, converting to cm')
	else:
		zGrid = zNew

	# check monotonicity and 'outliers'
	if not all(np.ediff1d(zGrid)):
		raise SystemExit ('ERROR --- atmos1D.atmRegrid:  new altitude grid is not monotonically increasing!')
	if not (zGrid[0]>=data['z'][0] and zGrid[-1]<=data['z'][-1]):
		print(' WARNING --- atmos1D.atmRegrid:  new altitude grid exceeding old grid, interpolation might fail to extrapolate')
		print('                             ', data['z'][0], '<= zOld <=', data['z'][-1])
		print('                             ', zGrid[0],      '<= zGrid <=', zGrid[-1])

    	# initialize new profile matrix
	dataNew = np.empty(len(zGrid), dtype={'names': data.dtype.names, 'formats': len(data.dtype.names)*[np.float]})
	dataNew['z'] = zGrid

    	# loop over all original profiles

	if interpolate.startswith('h') or interpolate.startswith('H'):
		if 'p'   in data.dtype.names:
			dataNew['p'] = np.exp(pchip_interpolate (data['z'],np.log(data['p']), zGrid))
		if 'T'   in data.dtype.names:
			dataNew['T'] = pchip_interpolate (data['z'],data['T'], zGrid)
		if 'air' in data.dtype.names:
			dataNew['air'] = np.exp(pchip_interpolate (data['z'],np.log(data['air']), zGrid))
		for what in gases(data):
			dataNew[what] = pchip_interpolate (data['z'],data[what], zGrid)
	elif interpolate[0] in 'zsqcnp':
		if 'p'   in data.dtype.names:
			try:   dataNew['p'] = np.exp(interp1d(data['z'],np.log(data['p']),interpolate)(zGrid))
			except ValueError as errMsg:
				raise SystemExit ('ERROR --- atmos1D.atmRegrid:   p interpolation failed\n' + str(errMsg))
		if 'T'   in data.dtype.names:
			try:   dataNew['T'] = interp1d(data['z'],data['T'],interpolate)(zGrid)
			except ValueError as errMsg:
				raise SystemExit ('ERROR --- atmos1D.atmRegrid:   T interpolation failed\n' + str(errMsg))
		if 'air' in data.dtype.names:
			try:   dataNew['air'] = np.exp(interp1d(data['z'],np.log(data['air']),interpolate)(zGrid))
			except ValueError as errMsg:
				raise SystemExit ('ERROR --- atmos1D.atmRegrid:   air interpolation failed\n' + str(errMsg))
		for what in gases(data):
			try:                       dataNew[what] = interp1d(data['z'],data[what],interpolate)(zGrid)
			except ValueError as errMsg: raise SystemExit ('ERROR --- atmos1D.atmRegrid:   interpolation failed\n' + str(errMsg))
	else:
		if 'p' in data.dtype.names:
			dataNew['p']   = np.exp(np.interp(zGrid, data['z'], np.log(data['p'])));   print(' interpolating log(p)')
		if 'T' in data.dtype.names:
			dataNew['T'] = np.interp (zGrid, data['z'], data['T'])
		if 'air' in data.dtype.names:
			dataNew['air'] = np.exp(np.interp(zGrid, data['z'], np.log(data['air']))); print(' interpolating log(air)')
		for what in gases(data):
			dataNew[what] = np.interp (zGrid, data['z'], data[what])
	return dataNew


####################################################################################################################################

def extract_profiles (data, units, extract='main'):
	""" Extract some profiles from the atmospheric data (given as structured array),
	    i.e., remove all other profiles.

	    Default is to return (extract) altitude, pressure, temperature and the 'main' gases H2O, CO2, O3, ...
	"""
	print ("extract_profiles:", extract, len(data.dtype.names), data.dtype.names)
	# first check type
	if   isinstance(extract,str):           extract = re.split(r'[\s,]+',extract)  # extract.split(',')
	elif isinstance(extract,tuple):         extract = list(extract)
	elif isinstance(extract,list):          pass
	else:                                   raise SystemExit ('ERROR --- extractProfiles:  need string or list for "extract"')

	if extract[0].lower()=='main':
		extract='z,p,T,H2O,CO2,O3,N2O,CH4,CO,O2'.split(',')

	# check if all requests can be provided
	for name in extract:
		if name not in data.dtype.names:
			raise SystemExit ('%s %s %s' % ('ERROR --- atmos1D.extractProfiles:  ', name, ' not in original dataset'))

	# check if mandatory fields are requested (or only molecules?)
	if 'T' not in extract:  extract.insert(0,'T')
	if 'p' not in extract:  extract.insert(0,'p')
	if 'z' not in extract:  extract.insert(0,'z')

	# now extract data and units
	xData  = data[extract]
	xUnits = [units[data.dtype.names.index(name)] for name in extract]

	return xData, xUnits


####################################################################################################################################

def _atmosData2cgs (data, units, verbose=False):
	""" Convert atmospheric data to cgs units, check consistency p <--> T <--> air, and transform VMR to densities. """

	names = data.dtype.names
	if not len(names)==len(units):
		print (len(names), 'names: ', names)
		print (len(units), 'units: ', units)
		print ("WARNING --- atmosData2cgs:  inconsistent length of data fields and units list")

	# cgs units for p, T, air already !!!
	for m,name in enumerate(names):
		if   name=='z':
			data['z'] *= cgs(units[m]);     units[m]='cm'
		elif name=='p':
			data['p'] *= cgs(units[m]);     units[m]='g/(cm s^2)'
		elif name=='T':
			data['T'] *= unitConversion(1.0, 'temperature', units[m]);  units[m]='K'
		else:
			pass

	# pressure <---> temperature <---> air number density
	if   int('p' in names) + int('T' in names) + int('air' in names) == 3:
		print(' Atmos1d: got p, T, air', end=' ')
		deltaT = data['T'] - data['p']/(kBoltzmann*data['air'])
		if max(abs(deltaT))>1.0:
			if verbose:
				print ('\n%12s %12s %8s %8s' % ('p', 'n', 'T', 'T-p/kn'))
				awrite ([data['p'], data['air'], data['T'], data['T']-data['p']/(kBoltzmann*data['air'])],
				        format='%12.5g %12.5g %8.2f %8.3f')
				raise SystemExit ('ERROR --- atmos1D:  inconsistent data  max|T-p/kn|>1K')
	elif 'p' in names and   'T' in names:
		print(' Atmos1d: got p & T (computed air density from p/kT)', end=' ')
		data = strucArrayAddField (data, data['p']/(kBoltzmann*data['T']),   'air')
		units.append('1/cm**3')
	elif 'T' in names and 'air' in names:
		print(' Atmos1d: got T & air density (computed p from nkT)', end=' ')
		data = strucArrayAddField (data, data['air']*(kBoltzmann*data['T']), 'p')
		units.append('g/cm/s**2')
	elif 'p' in names and 'air' in names:
		print(' Atmos1d: got p & air density (computed T from p/nk)', end=' ')
		data = strucArrayAddField (data, data['p']/(kBoltzmann*data['air']), 'T')
		units.append('K')
	else:
		raise SystemExit ('ERROR --- atmos1D:  need at least two of three profiles: p, T, air')

	# check molecular concentrations and convert to cgs densities consistently
	nMolecules = 0
	for m,name in enumerate(names):
		if name not in ['z', 'p', 'T'] and name[0] in ascii_uppercase:
			nMolecules += 1
			if units[m] in mixingRatioUnits:
				data[name] = data['air'] * unitConversion(data[name], 'mixingRatio', units[m])
			else:
				data[name] = unitConversion(data[name], 'density', units[m])
			units[m] = '1/cm**3'
	print('  and', nMolecules, 'molecules on', len(data), 'levels')

	return data, units


####################################################################################################################################

def atmMerge (atmData1, atmData2, replace=False, interpolate='linear', verbose=False):
	""" Combine two atmospheric datasets (given as numpy structured array), interpolate second one if necessary.

	ARGUMENTS:
	----------
	atmData1, atmData2:   the two atmospheres to merge
	replace:              when a particular profile is given in both datasets,
	                      use the first and ignore the second (default)
			      OR replace the first by the second
	interpolate:          the interpolation method used for regridding (see the atmRegrid doc)
	                      NOTE:  the final grid is given by the grid of atmData1
	verbose:

	RETURNS:              a unified atmospheric dataset (as a numpy structured array)
	"""

	nLevels1, nLevels2 = len(atmData1), len(atmData2)
	oldNames = atmData1.dtype.names  # save the original list of entries
	newNames = atmData2.dtype.names

	if verbose:
		print ("\n first  atmosphere: ", nLevels1, " levels and ", len(oldNames), " profiles:", join_words(oldNames))
		print (  " second atmosphere: ", nLevels2, " levels and ", len(newNames), " profiles:", join_words(newNames))

	# check altitude grids and interpolate if necessary
	if nLevels1==nLevels2 and max(abs(atmData1['z']-atmData2['z']))<100:  # test if grid levels are identical with 1m tolerance
		if verbose:  print(' altitude grids identical',  nLevels1, ' levels with ', atmData1['z'][0], '<=z<=', atmData1['z'][-1])
	else:
		print('%s (%i %s <= %.1fkm) %s  %i %s <= %.1fkm' % (' atmMerge:   regridding second atmosphere ',
		       nLevels2, 'levels', cgs('!km',max(atmData2['z'])), ' to first atmosphere grid with',
		       nLevels1, 'levels', cgs('!km',max(atmData1['z']))))
		if verbose:  print('    ', cgs('!km',atmData2['z']), '\n -->', cgs('!km',atmData1['z']))
		atmData2 = atmRegrid(atmData2, atmData1['z'], interpolate)

	# check if the new data set provides mixing ratios or densities:  simply assume vmr<1 and density>1 for all data
	gases2   = gases(atmData2)
	maxValue = max([atmData2[gas].max() for gas in gases2])
	allVMR   = len(gases2)+1==len(atmData2.dtype.names) and 'z' in atmData2.dtype.names and maxValue<1.0
	if allVMR:
		print(" INFO --- atmMerge:  second dataset apparently only VMR\n         ",
		      len(gases2), "gas profile(s) with max. data value =", maxValue,
		      "\n          convert to densities by multiplication with air from first data set")

	# copy/move profiles from second dataset into first
	for what in atmData2.dtype.names:
		if what=='z':  continue
		if what in atmData1.dtype.names:
			if replace:
				if allVMR:  atmData1[what] = atmData2[what]*atmData1['air']
				else:       atmData1[what] = atmData2[what]
				if verbose:  print(" atmMerge:  replaced ", what)
		else:
			if allVMR:  atmData1 = strucArrayAddField (atmData1, atmData2[what]*atmData1['air'], what)
			else:       atmData1 = strucArrayAddField (atmData1, atmData2[what], what)
			if verbose:  print(" atmMerge:  added ", what)

	print ("INFO:    returning combined data with", len(atmData1.dtype.names)-1, "profiles, added",
	      join_words([name for name in atmData1.dtype.names if name not in oldNames]))
	return atmData1


####################################################################################################################################

def vmrRead (vmrFile, commentChar='#', extract=None, zToA=0.0, zBoA=0.0, what='what', units='units', scaleFactors=None,
	     verbose=False):
	""" Read volume mixing ratio (VMR) data file, convert to cgs units, optionally truncate top/bottom or scale concentrations;
	    and return a structured numpy array.

	    See the atmRead function for a full explanation of all arguments.

	    The main purpose of this function is to read trace gases concentrations from a file without pressure/temperature data;
	    then you can combine these VMRs with p, T, and the main gas number densities using the `atmMerge` function.
	"""

	# read the data and return a structured array with ID's taken from the file header
	if isinstance(what,str) and not what.endswith(':'):  what+=':'  # the keyword for the column IDs
	vmrData  = loadStrucArray (vmrFile, key2names=what, changeNames=simpleNames, commentChar=commentChar)

	# read units from file header
	vmrUnits = grep_from_header(vmrFile, units).split()

	for m,name in enumerate(vmrData.dtype.names):
		if   name=='z' and vmrUnits[m] in lengthUnits:
			vmrData['z'] *= cgs(vmrUnits[m])
			if verbose:  print (" altitude scaled by", cgs(vmrUnits[m]))
		elif name[0] in ascii_uppercase and vmrUnits[m] in mixingRatioUnits:
			vmrData[name] *= cgs(vmrUnits[m])
			if verbose:  print (" VMR ", name, vmrUnits[m], " scaled by", cgs(vmrUnits[m]))
		else:
			print("WARNING --- vmrRead:  column #",
			      m, " is neither an altitude nor a volume mixing ratio:", name, vmrUnits[m])

	if 'z' not in vmrData.dtype.names:  raise SystemExit ("ERROR --- vmrRead:  no zGrid / altitudes found in vmrFile!")

	# remove some gases
	if extract:
		if   isinstance(extract, str):  extract = extract.split()
		delete = [name for name in gases(vmrData) if not (name=='z' or name in extract)]
		vmrData = strucArrayDeleteFields (vmrData, delete)

	# delete bottom and top levels
	if zToA or zBoA:  vmrData = atmTruncate (vmrData, zToA=zToA, zBoA=zBoA, verbose=verbose)

	# scale molecular densities
	if scaleFactors:  vmrData = scaleDensities (vmrData, scaleFactors, verbose=verbose)

	if verbose:
		for name in vmrData.dtype.names:  print('%10g < %10s < %10g' % (min(vmrData[name]), name, max(vmrData[name])))

	return vmrData


####################################################################################################################################

def atmRead (atmFile, commentChar='#', extract=None, zToA=0.0, zBoA=0.0, what='what', units='units', scaleFactors=None,
	     returnUnits=False, verbose=False):
	""" Read atmospheric data file, convert to cgs units, optionally truncate top/bottom or scale concentrations;
	    and return a structured numpy array.

	    Parameters:
	    -----------
	    atmFile        string   file(name) with data to be read
	    extract        strings  list of entries (essentially gases) to be returned (default None ==> read all)
	    zToA           float    top    of atmosphere altitude [km] (default: 0.0)
	    zBoA           float    bottom of atmosphere altitude [km] (default: 0.0)
	    what           string   first 'word' in file header line identifying column names (default: "what")
	    units          string   first 'word' in file header line identifying units        (default: "units")
	    scaleFactors   floats   change abundances, i.e. multiply molecular concentrations (default: 1.0 for all gases)
	    returnUnits    flag     optionally also return units (default: False)

	    Note: all data are returned in cgs units
	          z[cm],  p[dyn/cm**2],  T[K],  n[molec/cm**3)

            If you want to combine data from different files (e.g. 'standard' p&T and 'exotic' trace gas concentrations)
	    then read the files separately using this `atmRead` (or vmrRead) function and then use the `atmMerge` function

	    atmRead expects that pressure, temperature, and air density (at least two of these three!) are given;
	    If you want to read a file with only molecular VMRs or densities, use `vmrRead`.

            Atmospheric data file(s):  see the data/atmos directory for examples.
	"""
	# read atmospheric data file assuming columns are specified in "what:" row of header section
	if not what.endswith(':'):  what+=':'
	data  = loadStrucArray (atmFile, key2names=what, changeNames=simpleNames, commentChar=commentChar, verbose=verbose)
	names = data.dtype.names
	if verbose:  print(' atmRead ', len(names), 'columns (data fields) in ', repr(atmFile), join_words(names,', '))

	# read units from file header
	units = grep_from_header(atmFile, units)
	if isinstance(units,str):  units = units.split()
	else:  raise SystemExit (' ERROR ---- atmRead:  could not find a header line starting with "units:" declaring the units')

	if verbose: print('         ', len(units), ' units', join_words(units,'  '))
	if not len(names)==len(units):
		raise SystemExit (' ERROR ---- atmRead:  number of data columns and units inconsistent!')

	# extract required columns (and delete other columns)
	if isinstance(extract,(list,tuple)) or (isinstance(extract,str) and extract not in ['*', 'all']):
		if verbose:  print (' atmRead', len(names), ' atmospheric profiles:', names,' extract', type(extract), extract)
		data, units =  extract_profiles (data, units, extract)
		if verbose:  print (' --->', data.dtype.names)
	else:
		for name in data.dtype.names:
			if name not in goodNames:
				n = data.dtype.names.index(name)
				data = strucArrayDeleteFields(data,name);  popUnit = units.pop(n)
				if verbose:  print (" WARNING:  removed column #", n, name, "[", popUnit,"]")

	# convert atmospheric data to cgs units and transform VMR to densities
	# check p <--> T <--> air  (and append one of them if not read from file)
	data, units = _atmosData2cgs (data, units, verbose)

	# check z grid
	deltaZ = np.ediff1d(data['z'])
	if   all(deltaZ>0):
		if verbose:  print (' z Grid monotone increasing with %.3f <= dz <= %.2fkm' %
		                    (cgs('!km',min(deltaZ)),cgs('!km',max(deltaZ))))
	elif all(deltaZ<0):
		for name in data.dtype.names: data[name] = data[name][::-1]
		if verbose:  print (' z Grid monotone decreasing, flipped up/down')
	else:   raise SystemExit ('ERROR --- atmRead:  zGrid neither increasing nor decreasing monotonically')

	# delete bottom and top levels
	if zToA or zBoA:  data = atmTruncate (data, zToA=zToA, zBoA=zBoA, verbose=verbose)

	# scale molecular densities
	if scaleFactors:  data = scaleDensities (data, scaleFactors, verbose=verbose)

	# optionally return units (all cgs!)
	if returnUnits:  return data, units
	else:            return data


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions
	np.set_printoptions(linewidth=120)

	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       dict(ID='about'),
	       dict(ID='BoA',   name='zBoA',     type=float, constraint='zBoA>=0.0'),
	       dict(ID='ToA',   name='zToA',     type=float, constraint='zToA>0.0'),
	       dict(ID='scale', name='scaleFactors', type=np.ndarray, constraint='scaleFactors>0.0'),
	       dict(ID='i',     name='interpolate', type=str, default='', constraint='interpolate.lower() in "l0123zsqcpnh"'),
	       dict(ID='u',     name='units',    type=str, default='units'),
	       dict(ID='w',     name='what',     type=str, default='what'),
	       dict(ID='x',     name='extract',  type=str),  # z,p,T,H2O,CO2,O3,N2O,CO,CH4'),
	       dict(ID='r',     name='replace'),
	       dict(ID='v',     name='verbose') ]

	atmFiles, options, commentChar, outFile = parse_command (opts,[1,99])

	if 'h'     in options:   raise SystemExit (__doc__ + "\n end of atmos1D help")
	if 'about' in options:   raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID'] in ('h', 'about'))]
	for key in boolOptions:  options[key] = key in options
	interpolate = options.pop('interpolate', 2)
	replace     = options.pop('replace', False)

	if len(atmFiles)==1:
		options['returnUnits']=True
		atmData, atmUnits = atmRead (atmFiles[0], commentChar, **options)

		atmSave (atmData, outFile, atmUnits)
	else:
		if options['extract']:  print ("WARNING:  extract of subset of profiles not yet working for multiple files!")
		atmData = atmRead (atmFiles[0], commentChar, **options)
		for aFile in atmFiles[1:]:
			nextAtmos = vmrRead (aFile, commentChar, **options)
			atmData = atmMerge (atmData, nextAtmos, replace, interpolate, verbose=options['verbose'])
		awrite (atmData, outFile, format='%12.0f %11.4g', comments=len(atmData.dtype)*'%11s ' % tuple(atmData.dtype.names))
