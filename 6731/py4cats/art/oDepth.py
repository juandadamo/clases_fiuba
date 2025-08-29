#!/usr/bin/env python3

""" oDepth

  Read and convert / plot / write molecular optical depth (e.g., to reformat or plot).

  usage:
  oDepth [options] od_file(s)

  command line options:
    -c  char(s)   comment character in input file(s) (default #)
    -f  string    format for output file [default: 'xy']
    -h            help
    -m  char      "a|A"   accumulate delta optical depth to cumulative optical depth
                          from bottom --> top OR top --> bottom
                  "d|D"   convert cumulative optical depth to delta optical depth
                  "s|S"   sum delta optical depth to total path optical depth       (default!)
 	          "r|R"   revert accumulated optical depths
                  "t"     convert optical depths to transmissions = exp(-od)
                  "T"     sum delta optical depth to total and convert to transmission = exp(-od)
                  "1"     locate (approximately) the altitude/distance where od=1 (for uplooking view)
   --BoA  float   bottom-of-atmosphere altitude [km]  (read opt.depth only for levels above)
   --ToA  float   top-of-atmosphere altitude [km]     (read opt.depth only for levels below)
                  NOTE:  no interpolation, i.e. integration starts/stops at the next level above/below BoA/ToA
    -i   string   interpolation method for spectral domain (optical depth vs wavenumber)
    -n            convert x axis from wavenumber to wavelength[nanometer]
                  NOTE: if the input x is already wavelength[nanometer], then the new x is converted back to wavenumber!
    -o  file      output file (default: standard output)
    -p            matplotlib for quicklook of input optical depth(s)
    -P            matplotlib for quicklook of output optical depth(s) or transmission/weighting function
    -r            on output reverse layer optical depth order:  top <--> bottom of atmosphere
    -x  Interval  lower,upper wavenumbers/wavelengths (comma separated pair of floats, no blanks!)
   --xFormat string  format to be used for wavenumbers,   default '%12f'   (only for ascii tabular)
   --yFormat string  format to be used for optical depth, default '%11.5f' (only for ascii tabular)
    -z  float     zenith angle:  scale optical depth by 1/cos(zenithAngle)   (0dg=uplooking, 180dg=downlooking)

  optical depth files:
  *   xy formatted ascii file with wavenumbers in column 1 and optical depth(s) (for some layers) in the following column(s).
  *   pickled optical depth
  *   [ netcdf formatted optical depth file ]

  NOTES:
  *  If no output file is specified, only a summary 'statistics' is given !!!

  WARNING:
  oDepth does NOT know the type of optical depth (delta or accumulated ...) given in the input file!

  CAUTION:    mode='t' for oDepth -> transmission is inconsistent with mode='t' for total/sum oDepth
              in the lbl2od and ac2od modules
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

import os
import sys
from string import punctuation
from math import ceil
from pickle import dump, load

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

try:                 from matplotlib.pyplot import plot, semilogy, legend, xlabel, ylabel, title, show, figure, rc
except ImportError:  print ('WARNING --- oDepth:  matplotlib not available, no quicklook!')
else:                rc('figure.subplot', left=0.1, right=0.88)

if __name__ == "__main__":  sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats import __version__
from py4cats.aux.ir import c as cLight
from py4cats.aux.cgsUnits import cgs, lengthUnits, pressureUnits, frequencyUnits, wavelengthUnits
from py4cats.aux.pairTypes import Interval, PairOfFloats
from py4cats.aux.aeiou import parse_comments, readDataAndComments, open_outFile, minmaxmean, awrite, cstack, join_words, read_first_line
from py4cats.aux.struc_array import dict2strucArray
from py4cats.aux.misc  import regrid, monotone
from py4cats.aux.convolution     import convolveBox, convolveTriangle, convolveGauss
from py4cats.aux.moreFun import cosdg
from py4cats.art.atmos1D  import simpleNames

####################################################################################################################################
####################################################################################################################################

class odArray (np.ndarray):
	""" A subclassed numpy array of optical depths with xLimits, z, p, T, ... attributes added.
	    The z, p, t attributes here contain the corresponding height, pressure, temperature intervals!

	    Furthermore, some convenience functions are implemented:
	    *  info:     print the attributes and the minimum and maximum od values
	    *  dx:       return wavenumber grid point spacing
	    *  grid:     return a numpy array with the uniform wavenumber grid
	    *  regrid:   return an odArray with the od data interpolated to a new grid (same xLimits!)
	    #  truncate: return an odArray with the wavenumber range (xLimits) truncated
	    *  __eq__:   the equality test accepts 0.1% differences of pressure and all od values
	"""
	# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

	def __new__(cls, input_array, xLimits=None, z=None, p=None, t=None, N=None):
		# Input array is an already formed ndarray instance
		# First cast to be our class type
		obj = np.asarray(input_array).view(cls)
		# add the new attributes to the created instance
		obj.x     = xLimits
		obj.z     = z
		obj.p     = p
		obj.t     = t   # cannot use capital "T" because this means 'transpose'
		obj.N     = N   # N = integral n(z) dz = vcd =vertical column density
		return obj  # Finally, we must return the newly created object:

	def __array_finalize__(self, obj):
		# see InfoArray.__array_finalize__ for comments
		if obj is None: return
		self.x     = getattr(obj, 'x', None)
		self.z     = getattr(obj, 'z', None)
		self.p     = getattr(obj, 'p', None)
		self.t     = getattr(obj, 't', None)
		self.N     = getattr(obj, 'N', None)

	def __str__ (self):
		return '# (z=%s,   p=%s,   T=%sK)  %i \n %s' % (self.z, self.p, self.t, len(self), self.__repr__())

	def info (self):
		""" Return basic information (z, p, T, wavenumber and oDepth range). """
		return '%6.1f --%5.1fkm %12.3e -- %9.3emb %8.1f -- %5.1fK  %9.3e/cm**3 %12i points in  %10f -- %10f cm-1  with  %10.4g < od < %9.4g' % \
                       (cgs('!km',self.z.left), cgs('!km',self.z.right),
		        cgs('!mb',self.p.left), cgs('!mb',self.p.right),  self.t.left, self.t.right, self.N,
		        len(self), self.x.lower, self.x.upper, min(self), max(self))

	def dx (self):
		""" Return wavenumber grid point spacing. """
		return  self.x.size()/(len(self)-1)

	def grid (self):
		""" Setup a uniform, equidistant wavenumber grid. """
		return  self.x.grid(len(self))  # calls the grid method of the Interval class (in pairTypes.py)

	def regrid (self, new, method='l', yOnly=False):
		""" Interpolate optical depth to (usually denser) uniform, equidistant wavenumber grid. """
		if   isinstance(new,odArray):  yNew = regrid(self.base,len(new),method)
		elif isinstance(new,int):      yNew = regrid(self.base,new,method)
		else:                          raise SystemExit ("ERROR --- odArray.regrid:  expected an integer or odArray")
		if yOnly:  return  yNew
		else:      return  odArray (yNew, self.x, self.z, self.p, self.t, self.N)

	def truncate (self,xLimits):
		""" Return an optical depth in a truncated (smaller) wavenumber interval. """
		if isinstance(xLimits,(tuple,list)) and len(xLimits)==2:  xLimits=Interval(*xLimits)
		dx = self.dx()
		iLow  = max(int((xLimits.lower-self.x.lower)/dx), 0)
		iHigh = min(int(ceil((xLimits.upper-self.x.lower)/dx)), len(self)-1)
		xLow  = self.x.lower + iLow*dx
		xHigh = self.x.lower + iHigh*dx
		return  odArray (self.base[iLow:iHigh+1], Interval(xLow,xHigh), self.z, self.p, self.t, self.N)

	def convolve (self, hwhm=1.0, srf='Gauss'):
		""" Return optical depth convolved with a spectral response function of half width @ half maximum:
		    More precisely:  convolve absorption=1-transmission=1-exp(-od) and return -log(1-<abs>)
		"""
		if not isinstance(hwhm,(int,float)):
			raise ValueError ("ERROR --- odArray.convolve:  expected a float, first argument is `hwhm`")
		if self.x.size()<10*hwhm:
			raise ValueError ("ERROR --- odArray.convolve:  hwhm too large!")
		if srf.upper().startswith('G'):
			wGrid, od = convolveGauss (self.grid(), self.base, hwhm, 'o')
		elif srf.upper().startswith('T'):
			wGrid, od  =  convolveTriangle(self.grid(), self.base, hwhm, 'o')
		else:
			wGrid, od  =  convolveBox(self.grid(), self.base, hwhm,'o')
		return  odArray (od, Interval(wGrid[0],wGrid[-1]), self.z, self.p, self.t, self.N)

	def __add__ (self,other):
		""" Add two layer optical depths spectra and also 'combine' the layer bounds (z, p, T). """
		if isinstance(other,odArray):
			# check layer bounds and 'combine'
			if   self.z.right==other.z.left and self.p.right==other.p.left:
				zz = PairOfFloats(self.z.left,other.z.right)
				pp = PairOfFloats(self.p.left,other.p.right)
				tt = PairOfFloats(self.t.left,other.t.right)
				N  = self.N + other.N
			elif self.z.left==other.z.right and self.p.left==other.p.right:
				zz = PairOfFloats(other.z.left,self.z.right)
				pp = PairOfFloats(other.p.left,self.p.right)
				tt = PairOfFloats(other.t.left,self.t.right)
				N  = self.N + other.N
			elif self.z.left==other.z.left and self.z.right==other.z.right and \
			     self.p.left==other.p.left and self.p.right==other.p.right and \
			     self.t.left==other.t.left and self.t.right==other.t.right:
				zz = self.z
				pp = self.p
				tt = self.t
				N  = self.N  # + other.N    ??? todo, check, ...  currently this is airColumn p/kT !!!
			else:
				raise SystemExit ("ERROR --- odArray.__add__:   altitude and pressure intervals not neighboring")
			# check identity of wavenumber intervals
			if not self.x.approx(other.x):
				raise SystemExit ("ERROR --- odArray.__add__:   wavenumbers interval not (approx) equal")
			# interpolate coarser spectrum to fine grid and add both spectra
			if   len(self)>len(other): yy = self.base + other.regrid(len(self), yOnly=True)
			elif len(self)<len(other): yy = self.regrid(len(other), yOnly=True) + other.base
			else:                      yy = self.base+other.base
			return  odArray (yy, self.x, zz, pp, tt, N)
		else:
			raise SystemExit ("ERROR --- odArray.__add__:   other is not an odArray instance")

	def __sub__ (self,other):
		""" Subtract two layer optical depths spectra and also 'reduce' the layer bounds (z, p, T). """
		if isinstance(other,odArray):
			if   self.z.left==other.z.left and other.z.right<self.z.right:
				zz = PairOfFloats(other.z.right,self.z.right)
				pp = PairOfFloats(other.p.right,self.p.right)
				tt = PairOfFloats(other.t.right,self.t.right)
			elif self.z.right==other.z.right and other.z.left<other.z.right:
				zz = PairOfFloats(other.z.left,self.z.left)
				pp = PairOfFloats(other.p.left,self.p.left)
				tt = PairOfFloats(other.t.left,self.t.left)
			else:
				print (self.info())
				print (other.info())
				raise SystemExit ("ERROR --- odArray.__sub__:   altitude and pressure intervals not neighboring")
			# check identity of wavenumber intervals
			if not self.x.approx(other.x):
				raise SystemExit ("ERROR --- odArray.__sub__:   wavenumbers interval not (approx) equal")
			if   len(self)>len(other): yy = self.base - other.regrid(len(self), yOnly=True)
			elif len(self)<len(other): yy = self.regrid(len(other), yOnly=True) - other.base
			else:                      yy = self.base-other.base
			return  odArray (yy, self.x, zz, pp, tt, self.N-other.N)
		else:
			raise SystemExit ("ERROR --- odArray.__sub__:   other is not an odArray instance")

	def __mul__ (self, other):
		""" Multiply optical depth spectrum, e.g. to 'mimick' a slant path
		    (NOTE: attributes are copied without any change). """
		if isinstance(other,(int,float)):
			return  odArray (other*self.base, self.x, self.z, self.p, self.t, self.N)
		elif isinstance(self,(int,float)) and isinstance(other,odArray):
			return  odArray (self*other.base, other.x, other.z, other.p, other.t, other.N)

	def __rmul__ (self, other):
		if isinstance(other,(int,float)):
			return  odArray (other*self.base, self.x, self.z, self.p, self.t, self.N)
		elif isinstance(self,(int,float)) and isinstance(other,odArray):
			return  odArray (self*other.base, other.x, other.z, other.p, other.t, other.N)

	def __eq__(self, other):
		""" Compare optical depth including its attributes.
		    (For p and od relative differences < 0.1% are seen as 'equal') """
		return self.x==other.x \
			   and self.z.min()==other.z.min() and self.z.max()==other.z.max() \
			   and self.t.min()==other.t.min() and self.t.max()==other.t.max() \
			   and abs(self.p.min()-other.p.min())<0.001*self.p.min() \
			   and abs(self.p.max()-other.p.max())<0.001*self.p.max() \
			   and abs(self.N-other.N)<0.001*self.N \
			   and np.allclose(self.base,other.base,atol=0.0,rtol=0.001)  #and all(deltaOD<0.001)


####################################################################################################################################
####################################################################################################################################

def odInfo (optDepth):
	""" Print information (min, max, mean, ...) for one or several optical depth(s). """
	if isinstance(optDepth,(list,tuple)) and all([isinstance(od,odArray) for od in optDepth]):
		for lyr, od in enumerate(optDepth):  print('%3i %s' % (lyr, od.info()))
	elif isinstance(optDepth,odArray):
		print(optDepth.info())
	else:
		raise TypeError ("ERROR --- oDepth.odInfo:  unknown/invalid data type, expected an odArray or a list thereof!")


####################################################################################################################################

def odConvolve (optDepth, hwhm=1.0, srf='Gauss'):
	""" Return optical depth(s) convolved with a spectral response function of half width @ half maximum:
	    More precisely:  convolve absorption=1-transmission=1-exp(-od) and return -log(1-<abs>). """
	if isinstance(optDepth,(list,tuple)):
		return [od.convolve(hwhm, srf) for od in optDepth]
	elif isinstance(optDepth,odArray):
		return optDepth.convolve(hwhm, srf)
	else:
		raise TypeError ("ERROR --- oDepth.odConvolve:  unknown/invalid data type, expected an odArray or a list thereof!")


####################################################################################################################################

def odPlot (optDepth, xUnit='1/cm', trans=False, tag='p', **kwArgs):
	""" Quicklook plot of optical depth vs wavenumber.

	    ARGUMENTS:
	    ----------
	    optDepth:    differential, cumulative or total optical depth(s)
	                 either an odArray instance or a list thereof
	    xUnit:       default cm-1, other choices frequencies (Hz, kHz, MHz, GHz, THz) or wavelength (um, mue, nm)
	    trans:       flag, default False; if True plot transmission=exp(-od)
	    tag:         select z|p|t for display in legend labels (default 'p')
	    kwArgs       passed directly to semilogy and can be used to set colors, line styles and markers etc.
	                 ignored (cannot be used) in recursive calls with lists or dictionaries of optical depths.
	"""

	if isinstance(xUnit, odArray):
		raise ValueError ("odPlot: second argument `xUnit` is an optial depth array\n" +
		                  "            (to plot a list of odArray's put these in brackets)")

	if isinstance(optDepth,(list,tuple)):
		if kwArgs:  print ("WARNING --- odPlot:  got a list of optical depths, ignoring kwArgs!")
		for od in optDepth:
			odPlot (od, xUnit, trans, tag)
		legend (fontsize='small')
	elif isinstance(optDepth,odArray):
		od = optDepth  # just a shortcut
		if 'label' in kwArgs:
			labelText=kwArgs.pop('label')
		else:  # extract the altitude, pressure or temperature interval for the legend
			if tag.lower()=='z':
				zz = cgs('!km',od.z);  labelText ='%8.1f - %.1fkm' % (zz.left, zz.right)
			elif tag.lower()=='t':
				labelText ='%8.1f - %.1fK' % (od.t.left, od.t.right)
			elif tag.lower()=='N':
				labelText ='%10.3gmolec/cm**3' % od.N
			else:
				pp = cgs('!mb',od.p);  labelText ='%10.3g - %.3gmb' % (pp.left, pp.right)
		# plot optical depth (or transmission) vs. wavenumber (or wavelength)
		if   xUnit in ['Hz', 'kHz', 'MHz', 'GHz', 'THz']:
			if trans:  plot (cLight/frequencyUnits[xUnit]*od.grid(), np.exp(-od.base), label=labelText, **kwArgs)
			else:      semilogy (cLight/frequencyUnits[xUnit]*od.grid(), od.base, label=labelText, **kwArgs)
		elif xUnit in wavelengthUnits.keys():
			if trans:  plot (1.0/(wavelengthUnits[xUnit]*od.grid()), np.exp(-od.base), label=labelText, **kwArgs)
			else:      semilogy (1.0/(wavelengthUnits[xUnit]*od.grid()), od.base, label=labelText, **kwArgs)
		else:
			if trans:  plot (od.grid(), np.exp(-od.base), label=labelText, **kwArgs)
			else:      semilogy (od.grid(), od.base, label=labelText, **kwArgs)
	else:
		raise TypeError ("ERROR --- oDepth.odPlot:  expected an odArray or a list thereof!")

	if   xUnit in ['Hz', 'kHz', 'MHz', 'GHz', 'THz']:   xlabel (r'frequency $\nu$ [%s]' % xUnit)
	elif xUnit in wavelengthUnits.keys():               xlabel (r'wavelength $\lambda \rm\,[\mu m]$')
	else:                                               xlabel (r'wavenumber $\nu \rm\,[cm^{-1}]$')
	if trans:  ylabel (r'transmission  $\mathcal{T}(\nu)$')
	else:      ylabel (r'optical depth  $\tau(\nu)$')


####################################################################################################################################

def od_list2matrix (odList, interpol='l'):
	""" Convert a list of (delta) optical depths to a matrix and also return the wavenumber grid. """

	if not isinstance(odList,(list,tuple)):
		raise TypeError ("ERROR --- od_list2matrix:  expected a list of odArray's, but got %s" % type(odList))
	if not all([isinstance(od,odArray) for od in odList]):
		raise TypeError ("ERROR --- od_list2matrix:  got a list of %i elements, but not all are odArray's" % len(odList))
	if not all([od.x==odList[0].x for od in odList[1:]]):
		raise ValueError ("ERROR --- od_list2matrix:  wavenumber intervals do not agree!")

	# what is the largest array with the densest grid
	nMax = max([len(od) for od in odList])
	# interpolate to this grid
	odMatrix  = np.array([od.regrid(nMax, interpol, yOnly=True) for od in odList]).T
	# use the last item of the list comprehension above and the Interval grid method
	vGrid = odList[-1].x.grid(nMax)
	# return an array of wavenumbers along with the matrix of optical depths
	return vGrid, odMatrix


def oDepth_zpT (odList):
	""" Extract altitude, pressure, and temperature levels from list of odArrays and check for consistency.

	    RETURN:   three numpy arrays

	    NOTE:     the altitudes (pressures) returned are always monotonically increasing (decreasing)!!!
	              if the optical depths are non-contiguous, an ERROR is raised.
	"""
	# initial check not required, already done by od_list2matrix
	# if not (isinstance(odList,(list,tuple)) and all([isinstance(od,odArray) for od in odList])):
	#	raise TypeError ("ERROR --- oDepth_zpT:  expected a list of odArray's, but ...")

	# initialize with bounds of first layer
	zGrid = [odList[0].z.left, odList[0].z.right]
	pGrid = [odList[0].p.left, odList[0].p.right]
	tData = [odList[0].t.left, odList[0].t.right]
	# append top (or bottom) altitude etc of further layers
	for od in odList[1:]:
		if   od.z.left==zGrid[-1] and  od.p.left==pGrid[-1] and  od.t.left==tData[-1]:
			zGrid.append(od.z.right)
			pGrid.append(od.p.right)
			tData.append(od.t.right)
		elif od.z.right==zGrid[0] and od.p.right==pGrid[0] and od.t.right==tData[0]:
			zGrid.insert(0,od.z.left)
			pGrid.insert(0,od.p.left)
			tData.insert(0,od.t.left)
		else:
			errMsg = "the %i layers collected so far in %s km and %s mb\nare NOT connected to the current layer %s km and %s mb" % \
			         (len(zGrid)-1,
				  cgs('!km',Interval(zGrid[0],zGrid[-1])),
				  cgs('!mb',Interval(pGrid[0],pGrid[-1])),
			          cgs('!km',Interval(od.z.left,od.z.right)),
				  cgs('!mb',Interval(od.p.left,od.p.right)))
			raise ValueError ("ERROR --- oDepth_zpT:  altitude or pressure grid non-contiguous\n" + errMsg)
	return np.array(zGrid), np.array(pGrid), np.array(tData)


def oDepth_altitudes (odList, lengthUnit=None):
	""" Extract altitude levels and check for consistency and monotonicity (see oDepth_zpT).
	    Optionally return altitude array converted into appropriate length unit (default 'cm'). """
	# initialize with altitude bounds of first layer
	zGrid = [odList[0].z.left, odList[0].z.right]
	# append top altitude of further layers
	for od in odList[1:]:
		if   od.z.left  == zGrid[-1]:  zGrid.append(od.z.right)
		elif od.z.right == zGrid[0]:   zGrid.insert(0,od.z.left)
		else:                          raise ValueError ('ERROR --- oDepth_altitudes:  altitude grid non-contiguous')

	if lengthUnit in lengthUnits:  return cgs('!'+lengthUnit, np.array(zGrid))
	elif lengthUnit:               raise SystemExit ("ERROR --- oDepth_altitudes:  invalid/unknown length unit!")
	else:                          return np.array(zGrid)


def oDepth_pressures (odList, pressureUnit=None):
	""" Extract pressure levels and check for consistency and monotonicity (see oDepth_zpT).
	    Optionally return pressure array converted into appropriate pressure unit (default 'g/cm/s**2)'). """
	# initialize with pressure bounds of first layer
	pGrid = [odList[0].p.left, odList[0].p.right]
	# append top pressure of further layers
	for od in odList[1:]:
		if   od.p.left  == pGrid[-1]:  pGrid.append(od.p.right)
		elif od.p.right == pGrid[0]:   pGrid.insert(0,od.p.left)
		else:                          raise ValueError ('ERROR --- oDepth_pressures:  pressure grid non-contiguous')

	if pressureUnit in pressureUnits:  return cgs('!'+pressureUnit, np.array(pGrid))
	elif pressureUnit:                 raise SystemExit ("ERROR --- oDepth_pressures:  invalid/unknown pressure unit!")
	else:                              return np.array(pGrid)


def oDepth_temperatures (odList):
	""" Extract temperatures and check for consistency (see oDepth_zpT). """
	# initialize with temperature bounds of first layer
	temperatures = [odList[0].t.left, odList[0].t.right]
	# append top altitude of further layers
	for od in odList[1:]:
		if   od.t.left  == temperatures[-1]:  temperatures.append(od.t.right)
		elif od.t.right == temperatures[0]:   temperatures.insert(0,od.t.left)
		else:                                 raise SystemExit ('ERROR --- oDepth_temperatures:  data non-contiguous')
	return np.array(temperatures)


####################################################################################################################################
####################################################################################################################################

def odSave (optDepth, odFile=None, commentChar=None, nanometer=False, transmission=False, flipUpDown=False, interpol='l',
            xFormat='%12f', yFormat='%11.5g'):
	""" Write optical depth to pickled or ascii (tabular) output file.

	ARGUMENTS:
	----------
	optDepth       an odArray instance with an optical depth spectrum and some attributes (zz, pp, tt, ...)
	               OR a list thereof
	odFile:        file where data are to be stored (if not given, write to stdout)
	               if file extension is 'nc' | 'ncdf' | 'netcdf' use netcdf format
	commentChar:   if none (default), save data in numpy pickled file,
                       otherwise ascii-tabular (wavenumber in first column, oDepth data interpolated to common, densest grid)

		       !!! the following options are only relevant/supported for ascii output !!!

	nanometer      flag:  save wavelength (instead of default wavenumber)
	transmission   flag:  save exp(-od)   (instead of default optical depth)
	flipUpDown     flag:  reorder od list and save last to first optical depths
	interpol       interpolation method, default 'l' for linear interpolation with numpy.interp
	                                     2 | 3 | 4  for self-made Lagrange interpolation
	xFormat:       output format for wavenumbers, default '%12f'
	yFormat:       output format for optical depth, default '%10.4g'


	RETURNS:       nothing

	NOTE:          if you want ascii tabular output WITHOUT interpolation,
	               save data in individual files, i.e. call odSave in a loop over all layers
	"""

	if not (isinstance(optDepth,odArray) or (isinstance(optDepth,(tuple,list)) and all([isinstance(od,odArray) for od in optDepth]))):
		raise TypeError ("ERROR --- odSave: input data NOT an odArray or list thereof")

	if isinstance(odFile,str) and os.path.splitext(odFile)[1].lower() in ('.nc', '.ncdf', '.netcdf'):
		_odSave_netcdf (optDepth, odFile, nanometer)
	elif isinstance(commentChar,str) and commentChar in punctuation and not commentChar.isspace():
		# NOTE:  '' in punctuation ---> True
		_odSave_xy (optDepth, odFile, nanometer, transmission, flipUpDown, interpol, xFormat, yFormat, commentChar)
	else:
		_odSave_pickled (optDepth, odFile)


####################################################################################################################################

def _odSave_pickled (optDepth, odFile=None):
	""" Write optical depth(s) (incl. its attributes) to output (file) using Python's pickle module. """
	if odFile:  out = open (odFile, 'wb')
	else:       raise SystemExit('ERROR --- oDepth.odSave_pickled:  no optical depth pickling for standard out!')

	# write the shell command
	sysArgv = join_words(sys.argv)
	if 'ipython' in sysArgv or 'ipykernel' in sysArgv or 'jupyter' in sysArgv:
		dump('%s (version %s): %s @ %s' % ('ipy4cats', __version__, os.getenv('USER'),os.getenv('HOST')), out)
	else:
		dump(join_words([os.path.basename(sys.argv[0])] + sys.argv[1:]), out)

	# save optical depth 'spectra' along with attributes as a standard dictionary
	# "unpack" PairOfFloat and Interval instances into simply tuples to avoid py4CAtS data structures in the pickled file
	if isinstance(optDepth,(list,tuple)):
		for od in optDepth:
			dump ({'x': od.x.limits(), 'z': od.z.list(), 'p': od.p.list(), 't': od.t.list(), 'N': od.N, 'y': od.base},
			out)
	elif isinstance(optDepth,odArray):
		dump ({'z': optDepth.z.list(),
		       'p': optDepth.p.list(),
		       't': optDepth.t.list(),
		       'N': optDepth.N,
		       'x': optDepth.x.limits(),
		       'y': optDepth.base}, out)
	out.close()


####################################################################################################################################

def _odSave_xy (optDepth, odFile=None, nanometer=False, transmission=False, flipUpDown=False, interpol='l',
                xFormat='%12f', yFormat='%11.5g', commentChar='#'):
	""" Save optical depth (vs wavenumber or wavelength) in ascii tabular format. """
	out = open_outFile (odFile, commentChar)
	xyFormat = xFormat+yFormat

	if isinstance(optDepth,(list,tuple)) and all([isinstance(od,odArray) for od in optDepth]):
		nLevels = len(optDepth)+1
		# simply extract levels without check for consistency
		zzz = cgs('!km',[optDepth[0].z.left] + [od.z.right for od in optDepth])
		ppp = cgs('!mb',[optDepth[0].p.left] + [od.p.right for od in optDepth])
		ttt =           [optDepth[0].t.left] + [od.t.right for od in optDepth]

		# check for monotonicity
		if not (all(np.ediff1d(zzz)>0) and all(np.ediff1d(ppp)<0)):
			print('WARNING --- odSave:  altitude (pressure) grid is not monotonically increasing (decreasing)!!!')

		if flipUpDown:
			comments = ['altitude [km]:  ' + nLevels*' %10.1f' % tuple(np.flipud(zzz)),
			            'temperature [K]:' + nLevels*' %10.2f' % tuple(np.flipud(ttt)),
			            'pressure [mb]:  ' + nLevels*' %10.4g' % tuple(np.flipud(ppp))]
		else:
			comments = ['altitude [km]:  ' + nLevels*' %10.1f' % tuple(zzz),
			            'temperature [K]:' + nLevels*' %10.2f' % tuple(ttt),
			            'pressure [mb]:  ' + nLevels*' %10.4g' % tuple(ppp)]

		# interpolate all data to common, densest grid
		vGrid, odMatrix = od_list2matrix (optDepth, interpol)

		# reverse sequence of layers
		if flipUpDown:  odMatrix = np.fliplr(odMatrix)

		if nanometer and transmission:
			comments += ['%12s --- %10s' % ('wavelength','transmission'),  '%10s' % 'nm']
			awrite (np.flipud(cstack(1e7/vGrid, np.exp(odMatrix))), odFile, xyFormat, comments, commentChar=commentChar)
		elif nanometer:
			comments += ['%12s --- %10s' % ('wavelength','optical depth'),  '%10s' % 'nm']
			awrite (np.flipud(cstack(1e7/vGrid, odMatrix)), odFile, xyFormat, comments, commentChar=commentChar)
		elif transmission:
			comments += ['%12s --- %10s' % ('wavenumber','transmission'),  '%10s' % '1/cm']
			awrite (np.flipud(cstack(1e7/vGrid, odMatrix)), odFile, xyFormat, comments, commentChar=commentChar)
		else:
			comments += ['%12s --- %10s' % ('wavenumber','optical depth'),  '%10s' % '1/cm']
			awrite ((vGrid, odMatrix), odFile, xyFormat, comments, commentChar=commentChar)
	else:
		comments = ['altitude [km]:  ' + 2*' %10.1f' % tuple(cgs('!km',optDepth.z).list()),
		            'temperature [K]:' + 2*' %10.2f' %           tuple(optDepth.t.list()),
		            'pressure [mb]:  ' + 2*' %10.4g' % tuple(cgs('!mb',optDepth.p).list())]
		if nanometer:
			vGrid = optDepth.grid()
			if transmission:
				comments += ['%12s --- %10s' % ('wavelength','transmission'),  '%10s' % 'nm']
				awrite (np.flipud(cstack(1e7/vGrid, np.exp(-optDepth.base))), odFile,
				        xyFormat, comments,commentChar=commentChar)
			else:
				comments += ['%12s --- %10s' % ('wavelength','optical depth'),  '%10s' % 'nm']
				awrite (np.flipud(cstack(1e7/vGrid, optDepth)), odFile, xyFormat, comments,commentChar=commentChar)
		else:
			if transmission:
				comments += ['%12s --- %10s' % ('wavenumber','transmission'),  '%10s' % '1/cm']
				awrite ((optDepth.grid(), np.exp(-optDepth.base)), odFile,
				        xyFormat, comments, commentChar=commentChar)
			else:
				comments += ['%12s --- %10s' % ('wavenumber','optical depth'),  '%10s' % '1/cm']
				awrite ((optDepth.grid(), optDepth.base), odFile, xyFormat, comments, commentChar=commentChar)

	if odFile:  out.close()


####################################################################################################################################

def _odSave_netcdf (odList, odFile=None, nanometer=False):
	""" Save optical depth (vs wavenumber or wavelength) in netcdf format. """

	# see also scipy.io.netcdf !?!
	try:
		# according to PyPI this is a "standalone" version of Scientific.IO.NetCDF built for NumPy
		from pynetcdf import NetCDFFile
	except ImportError:
		print('INFO:  import pynetcdf failed, trying Scientific.IO.NetCDF')
		try:                 from Scientific.IO.NetCDF import NetCDFFile
		except ImportError:  raise SystemExit('import "Scientific.IO.NetCDF " failed, cannot find module!')
		else:                print('INFO:  import Scientific.IO.NetCDF')
	else:   print('INFO:  import pynetcdf')

	# Open netcdf file for writing
	ncf = NetCDFFile(odFile,'w')

	# interpolate all data to common, densest grid
	vGrid, odMatrix = od_list2matrix (odList)

	# Create Dimensions
	ncf.createDimension('nlev',odMatrix.shape[1]+1)  # fgs: changed, nLevels=len(zGrid)
	ncf.createDimension('nlyr',odMatrix.shape[1])    #               nLayers=number of intervals of zGrid
	if nanometer:  ncf.createDimension('nwvl',odMatrix.shape[0])
	else:          ncf.createDimension('nwvn',odMatrix.shape[0])
	ncf.createDimension('none',1)

	# Create Variables:
	z      = ncf.createVariable('z', 'd', ('nlev',))
	if nanometer:
		tau    = ncf.createVariable('tau', 'd', ('nlyr', 'nwvl',))
		wvlMin = ncf.createVariable('wvlmin', 'd', ('none',))
		wvlMax = ncf.createVariable('wvlmax', 'd', ('none',))
		wvl    = ncf.createVariable('wvl', 'd', ('nwvl',))
	else:
		tau    = ncf.createVariable('tau', 'd', ('nlyr', 'nwvn',))
		wvnMin = ncf.createVariable('wvnmin', 'd', ('none',))
		wvnMax = ncf.createVariable('wvnmax', 'd', ('none',))
		wvn    = ncf.createVariable('wvn', 'd', ('nwvn',))

	# Assign data
	zGrid = [odList[0].z.left] + [od.z.right for od in odList]           # returns altitudes in cm units
	z.assignValue(cgs('!km',zGrid))                           # see the setattr statement below

	if nanometer:
		depthOpt = np.flipud(odMatrix)
		tau.assignValue(depthOpt.T)
		vGrid = 1e7/vGrid  # convert wavenumber -> wavelength (note: simply overwrite variable!)
		wvl.assignValue(vGrid)
		wvlMin.assignValue(vGrid[0])
		wvlMax.assignValue(vGrid[-1])
	else:
		tau.assignValue(odMatrix.T)
		wvn.assignValue(vGrid)
		wvnMin.assignValue(vGrid[0])
		wvnMax.assignValue(vGrid[-1])

	# Set units as attributes
	setattr(z, 'units', 'km')
	setattr(tau, 'units', '-')
	if nanometer:
		setattr(wvlMin, 'units', 'nm')
		setattr(wvlMax, 'units','nm')
		setattr(wvl, 'units', 'nm')
	else:
		setattr(wvnMin, 'units', '1/cm')
		setattr(wvnMax, 'units','1/cm')
		setattr(wvn, 'units', '1/cm')

	# close netcdf file
	ncf.close()


####################################################################################################################################
####################################################################################################################################

def odRead (odFile, zToA=0.0, zBoA=0.0, xLimits=None, commentChar='#', verbose=False):
	""" Read an ascii tabular OR pickled OR netcdf file of optical depth(s)
	    and return data along with atmosphere attributes.

	    ARGUMENTS:
	    ----------
	    odFile:       data file with optical depths
	    zToA, zBoA:   ignore layers above top-of-atmosphere and/or below bottom-of-atmosphere altitudes
	    xLimits:      wavenumber interval (default None, i.e. read all)
	    commentChar:  default '#'
	    verbose:      boolean flag, default False

	    RETURNS:
	    --------
	    odList:       an odArray instance with an optical depth spectrum and some attributes (zz, pp, tt, ...)
	                  OR a list thereof
	"""

	if os.path.splitext(odFile)[1].lower().startswith('.nc'):
		raise SystemExit ('ERROR --- py4cats.oDepth:  reading netcdf not yet implemented')
	else:
		# try to determine filetype automatically from first nonblank character in file
		firstLine = read_first_line(odFile)
		if firstLine.startswith(commentChar):  odList = _odRead_xy (odFile, commentChar, verbose)
		else:                                  odList = _odRead_pickled (odFile)

	# remove top and bottom layers
	if zToA:
		if zToA<200:
			print('INFO --- odRead:  zToA =', zToA, 'very small, probably km', end=' --> ')
			zToA = cgs('km',zToA);  print('converted to ', zToA, 'cm')
		odList = [od for od in odList if od.z.max()<=zToA]

	if zBoA:
		if zBoA<200:
			print('INFO --- odRead:  zBoA =', zBoA, 'very small, probably km', end=' -->')
			zBoA = cgs('km',zBoA);  print('converted to ', zBoA, 'cm')
		odList = [od for od in odList if od.z.min()>zBoA]

	# remove left and right of spectral window of interest
	if xLimits:
		return [od.truncate(xLimits) for od in odList]

	if verbose:
		print ("\n list of optical depths with %i elements (layers):" % len(odList))
		odInfo(odList)

	return odList


####################################################################################################################################

def _odRead_xy (odFile, commentChar='#', verbose=False):
	""" Read an optical depth file formatted as ascii tabular. """

	# read optical depth including some attributes, nb. temperature
	xyData, commentLines = readDataAndComments (odFile,commentChar)
	vGrid    = xyData[:,0]
	optDepth = xyData[:,1:]

	# try to  parse comment header and extract some infos
	goodNames   = list(simpleNames.keys())+['z', 'p', 'T']   # molecules.keys()
	odAttributes =  parse_comments (commentLines, goodNames, commentChar=commentChar)

	if verbose:
		print('\nlen(vGrid) =', len(vGrid), '    shape(od) =', optDepth.shape, '\n')
		from pprint import pprint
		print('read_oDepth_xy: ', odFile, '--> attributes')
		pprint(odAttributes,width=120);  print()

	if odAttributes:
		# try to setup atmosphere structure from file header
		atmStrucArray = dict2strucArray(odAttributes, simpleNames)
		if verbose:
			awrite (atmStrucArray, format='%12.4g',
			        comments=len(atmStrucArray.dtype.names)*'%12s' % tuple(atmStrucArray.dtype.names))

		if optDepth.shape[1]+1==len(atmStrucArray):
			odList = []
			atmLast = atmStrucArray[0]
			for l, atm in enumerate(atmStrucArray[1:]):
				odList.append(odArray(optDepth[:,l], Interval(vGrid[0],vGrid[-1]),
				                      PairOfFloats(atmLast['z'],atm['z']),
				                      PairOfFloats(atmLast['p'],atm['p']),
				                      PairOfFloats(atmLast['T'],atm['T'])))
				atmLast = atm
		elif optDepth.shape[1]==1:
			# either total optical depth or just a single layer
			odList = odArray(optDepth[:,0], Interval(vGrid[0],vGrid[-1]),
				                        PairOfFloats(atmStrucArray['z'][0],atmStrucArray['z'][-1]),
				                        PairOfFloats(atmStrucArray['p'][0],atmStrucArray['p'][-1]),
				                        PairOfFloats(atmStrucArray['T'][0],atmStrucArray['T'][-1]))

	return odList


####################################################################################################################################

def _odRead_pickled (odFile):
	""" Read optical depth(s) (incl. attributes) from pickled output file.

	ARGUMENTS and RETURNS:  see the odRead doc
	"""

	try:                    pf = open(odFile,'rb')
	except IOError as msg:  raise SystemExit ('%s %s %s\n %s' % ('ERROR:  opening pickled optical depth file ',
	                                                           repr(odFile), ' failed (check existance!?!)', msg))

	# initialize list of absorption coefficients to be returned
	odList = []

	# NOTE:  the sequence of loads has to corrrespond to the sequence of dumps!
	info  = load(pf); print(odFile, info)

	while 1:
		try:
			odDict = load(pf)
			# wavenumber interval
			if   isinstance(odDict['x'],Interval): pass
			elif isinstance(odDict['x'],(list,tuple)) and len(odDict['x'])==2:  odDict['x']=Interval(*odDict['x'])
			else:  raise SystemExit ("ERROR --- odRead:  incorrect (wavenumber) info %s, not an Interval" % odDict['x'])
			# altitudes
			if   isinstance(odDict['z'],PairOfFloats): pass
			elif isinstance(odDict['z'],(list,tuple)) and len(odDict['z'])==2:  odDict['z']=PairOfFloats(*odDict['z'])
			else:  raise SystemExit ("ERROR --- odRead:  incorrect altitudes info %s, no two floats" % odDict['z'])
			# pressures
			if   isinstance(odDict['p'],PairOfFloats): pass
			elif isinstance(odDict['p'],(list,tuple)) and len(odDict['p'])==2:  odDict['p']=PairOfFloats(*odDict['p'])
			else:  raise SystemExit ("ERROR --- odRead:  incorrect pressures info %s, no two floats" % odDict['p'])
			# temperatures
			if   isinstance(odDict['t'],PairOfFloats): pass
			elif isinstance(odDict['t'],(list,tuple)) and len(odDict['t'])==2:  odDict['t']=PairOfFloats(*odDict['t'])
			else:  raise SystemExit ("ERROR --- odRead:  incorrect temperatures info %s, no two floats" % odDict['p'])
			# (vertical) column density
			if 'N' in odDict and isinstance(odDict['N'],(int,float)):  pass
			else:                                                      odDict['N'] = 0.0
			odList.append(odArray(odDict['y'], odDict['x'], odDict['z'], odDict['p'], odDict['t'], odDict['N']))
		except EOFError:
			zGrid = oDepth_altitudes(odList)
			print ('INFO --- odRead_pickled:  EOF reached, got %i optical depth(s) in %.2f -- %.1fkm' %
			       (len(odList), cgs('!km', zGrid[0]), cgs('!km',zGrid[-1])))
			pf.close();  break
	return odList


####################################################################################################################################
####################################################################################################################################

def oDepthOne (dodList, nadir=False, interpol='l', extrapolate=False, verbose=False):
	""" For each wavenumber scan (cumulative) optical depth as a function of altitude distance
	    and return distance from observer to OD=1.0

	    ARGUMENTS:
	    ----------
	    dodList:      delta / layer optical depth, a list of odArray instances
	    nadir:        flag, default uplooking (zenith), alternative downlooking
	    interpol:     string or number (2|3|4) indicating interpolation method
	    extrapolate:  flag; default False, i.e. return 'inf'
	    verbose:      flag; default False

	    RETURNS:
	    --------
	    distances     [cm]

	    NOTES:
	    ------
	    * Distance are returned in cgs units, i.e. centimeters
	      If you want to see the "OD=1 altitude spectrum"  (ie. the altitudes where OD=1), then evaluate
	      dodList[-1].z.max()-oDepthOne(dodList,1)      for a nadir view, or
	      dodList[0].z.min() +oDepthOne(dodList)        for a zenith view
	    * Cumulative optical depths are computed starting with the very first (or last) layer
	      If the observer is at an altitude somewhere in between, give only the subset of relevant odArray's.
	      However, oDepthOne does not consider an observer between the levels (layer bounds).
	    * Extrapolation:  if the delta optical depths of the last layers are tiny,
	                      the results might be 'nonsense'
	"""
	# interpolate all data to common, densest grid
	vGrid, dodMatrix = od_list2matrix (dodList, interpol)

	# extract altitude and pressure levels and check for consistency and monotonicity
	zGrid, pGrid, tData = oDepth_zpT(dodList)
	if monotone(zGrid)*monotone(pGrid)>=0:
		print('WARNING --- oDepthOne:  altitude (pressure) grid is not monotonically increasing (decreasing)!!!')

	# assume input is delta opt depth, from bottom to top
	if nadir:
		zDelta = zGrid[-1]-np.flipud(zGrid[:-1]);  cod = np.cumsum(np.fliplr(dodMatrix),1)
		if verbose:  print('oDepthOne nadir:  zDelta[km]=', cgs('!km',zDelta))
	else:
		zDelta = zGrid[1:]-zGrid[0];               cod = np.cumsum(dodMatrix,1)
		if verbose:  print('oDepthOne zenith:  zDelta[km]=', cgs('!km',zDelta))

	# locate next grid point
	lOne = np.array([cod[i,:].searchsorted(1.0) for i in range(cod.shape[0])])

	# estimate distance by inverse interpolation (linear)
	zOne = np.zeros(cod.shape[0])
	for i,l in enumerate(lOne):
		if    lOne[i]==0:                        # OD=1.0 somewhere in the very first layer next to observer
			zOne[i] =  zDelta[0] / cod[i,0]  # zDelta[0] is thickness of first layer
		elif  lOne[i]==dodMatrix.shape[1]:    # OD=1.0 somewhere beyond the very last layer:  extrapolation might produce nonsense
			if extrapolate:  zOne[i] = zDelta[-1] + (1.0-cod[i,-1]) / (cod[i,-1]-cod[i,-2]) * (zDelta[-1]-zDelta[-2])
			else:            zOne[i] = np.inf
		else:
			zOne[i] = zDelta[l-1] + (1.0-cod[i,l-1]) / (cod[i,l]-cod[i,l-1]) * (zDelta[l]-zDelta[l-1])

	return zOne


####################################################################################################################################

def cod2dod (codList):
	""" Subtract consecutive cumulated optical depths to delta (layer) optical depths and return a list of odArray's. """
	# NOTE:  backwards cumulated ooptical depths generated with dod2cod(dodList,1) cannot easily transformed to dod
	#        because all are given on the finest wavenumber grid
	dodList = [codList[0]]
	for l,od in enumerate(codList[1:]):
		dod = od-codList[l]
		dodList.append(dod)
	return dodList


def dod2cod (dodList, back=False):
	""" Accumulate all delta (layer) optical depths to cumulative optical depths and return a list of odArray's.

	    back=False    start accumulating with the very first layer (default)
	                  ===> the first element in the list returned is the bottom layer optical depth:  codList[0]=dodList[0]
	                       the second element corresponds to the first two layers:  codList[1]=dodList[0]+dodList[1]
	                       the last element in the list returned should be the total optical depth
	    back=True     start accumulating with the very last layer
	                  ===> the first element in the list returned should be the total optical depth
	                       the last element in the list returned is the top layer optical depth
	"""
	if back:  dodList=flipod(dodList)
	codList = [dodList[0]]
	for od in dodList[1:]:  codList.append(od+codList[-1])
	if back:  return flipod(codList)
	else:     return codList


def dod2tod (dodList, interpol='l'):
	""" Sum all delta (layer) optical depths to the total optical depth.
	    Returns an odArray with the spectrum on the densest grid (typically corresponding to the highest layer)
	"""
	# interpolate all optical depth to common, densest grid
	vGrid, odMatrix = od_list2matrix (dodList, interpol)
	# add all columns
	totalColumn = np.sum([od.N for od in dodList])
	return odArray (odMatrix.sum(1), Interval(vGrid[0],vGrid[-1]),
					 PairOfFloats(dodList[0].z.left,dodList[-1].z.right),
					 PairOfFloats(dodList[0].p.left,dodList[-1].p.right),
					 PairOfFloats(dodList[0].t.left,dodList[-1].t.right),
					 totalColumn)


def flipod (dodList):
	""" Flip a list of (delta) optical depths (odArray instances) up-down.
	    (In contrast to the 'simple' dodList[::-1] this function also swaps the z,p,t attributes) """
	return [odArray(od.base, od.x, od.z.swap(), od.p.swap(), od.t.swap(), od.N) for od in dodList[::-1]]


####################################################################################################################################
####################################################################################################################################

def oDepth (optDepth, mode='', interpol='l'):
	""" Manipulate optical depth:  convert cumulative <--> delta <--> sum (total).

	    ARGUMENTS:
	    ----------
	    optDepth:    differential (or cumulative) optical depth
	    mode:        character triggering the transformation
                         "a|A"   accumulate delta optical depth to cumulative optical depth
                                 from bottom --> top OR top --> bottom
                         "d|D"   convert cumulative optical depth to delta optical depth
                         "s|S"   sum delta optical depth to total path optical depth
 	                 "r|R"   revert accumulated optical depths
                         "t"     convert optical depth to transmission = exp(-od)
                         "T"     sum delta optical depth to total and convert to transmission = exp(-od)
                         "1"     locate (approximately) the altitude/distance where od=1 (for uplooking view)

	    NOTE:
	    -----
	    This functions assumes that 'optDepth' is the delta|layer optical depth (the default output of lbl2od)
	    Only if you request mode='d' or 'r', then the input is assumed to be cumulative optical depth

	    CAUTION:    mode='t' for oDepth -> transmission is inconsistent with mode='t' for total/sum oDepth
                        in the lbl2od, xs2od, ac2od functions
	"""

	if  not mode:
		print("WARNING --- oDepth:  no mode specified, doing & returning nothing!!!")

	# check input optical depth, should be either an odArray or (usually) a list of odArray instances
	if isinstance(optDepth,list) and all([isinstance(od,odArray) for od in optDepth]):
		if mode in 'ac':  # accumulate / cumulative
			return dod2cod(optDepth)
		elif mode in 'AC':  # reverse accumulate
			codList = [optDepth[-1]]
			for od in optDepth[-2::-1]:  codList.append(od+codList[-1])
			return codList
		elif mode in 'dD':  # from cumulative optical depth back to delta optical depth
			dodList = [optDepth[0]]
			for l,od in enumerate(optDepth[1:]):
				dodList.append(od-optDepth[l])
			return dodList
		elif mode in 'rR':  #  revert accumulated optical depths
			return dod2cod(flipod(optDepth))
		elif mode in [1,'1']:  # from the cumulative optical depth estimate altitude/distance with od=1
			return oDepthOne (optDepth, interpol)
		elif mode in 'sS':  # sum delta optical depth to total optical depth
			return dod2tod(optDepth, interpol)
		elif mode=='t':     # convert (layer) optical depth to transmission
			return [odArray (np.exp(-od.base), od.x, od.z, od.p, od.t, od.N) for od in optDepth]
		elif mode=='T':     # sum (layer) optical depths to total optical depth and convert to transmission
			return np.exp(-dod2tod(optDepth, interpol).base)
			#?????  raise SystemExit ("ERROR --- oDepth:  exp(-sum(od)) currently unimplemented, sorry") ?????
		else:
			raise SystemExit ("ERROR --- oDepth:  invalid/unknown mode " + repr(mode) + " for an odArray list")
	elif isinstance(optDepth,odArray):
		if mode in 'tT':     # convert (total?) optical depth to transmission
			return odArray (np.exp(-optDepth), optDepth.x, optDepth.z, optDepth.p, optDepth.t, optDepth.N)
		else:
			raise SystemExit ("ERROR --- oDepth:  invalid/unknown mode " + repr(mode) + " for a single odArray")
	else:
		raise SystemExit ('ERROR --- oDepth:  expected an odArray or list thereof')


####################################################################################################################################

def _oDepth_ (odFile, outFile=None, commentChar='#', zToA=0.0, zBoA=0.0, xLimits=None, mode='', zenithAngle=0., interpol=2,
              plotIn=False, plotOut=False, xFormat='%12f', yFormat='%11g', nanometer=False, flipUpDown=False, verbose=False):

	# read data from file
	optDepth = odRead (odFile, zToA, zBoA, xLimits, commentChar, verbose)

	odInfo(optDepth)

	# optionally plot original optical depths
	if plotIn:
		if plotOut: figure('oDepth_'+odFile)
		odPlot (optDepth);  title(odFile);  show()

	# scale optical depth (assuming strict vertical) to account for a slant path
	if zenithAngle>0.0:
		mue  = 1.0/cosdg(zenithAngle)
		optDepth *= mue
		print('\noptical Depth scaled by 1/cos(angle) =', mue, ' for zenith angle', zenithAngle, 'dg\n')
		if plotOut:
			if plotIn: figure('oDepth_%.1fdg' % zenithAngle)
			odPlot (optDepth);   title('%s  zenithAngle %.1fdg' % (odFile, zenithAngle))

	if   mode.lower() in 'acdrst1':
		# do something with the optical depth
		optDepth = oDepth (optDepth, mode)

		# prepare print and plot
		info = {'a': 'accumulated',  'c': 'cumulative',   'd': 'delta',  'r': 'reverse',
		        's': 'sum (total)',  't': 'transmission', '1': 'oDepth=1.0'}
		print('\n =======>', info[mode.lower()], len(optDepth))
		if plotIn and plotOut: figure('oDepth '+info[mode])

		if isinstance(optDepth,(odArray,list,tuple)):
			if plotOut:  odPlot (optDepth);  title ('%s --- %s' % (odFile, info[mode.lower()]))
		elif isinstance(optDepth,np.ndarray):
			minmaxmean(optDepth)
			if plotOut:  plot (optDepth)
		else:
			pass
	elif mode.strip():
		raise SystemExit ('ERROR --- _oDepth_:  invalid mode '+repr(mode))
	else:
		raise SystemExit ("WARNING --- _oDepth_:  no mode specified, terminating")

	if plotIn or plotOut:  show()

	if outFile:
		if len(xFormat.strip())==0 or len(yFormat.strip())==0:  commentChar=''
		if isinstance(optDepth,(list,tuple,odArray)):
			odSave (optDepth, outFile, commentChar, nanometer, flipUpDown, interpol, xFormat, yFormat)
		elif isinstance(optDepth,np.ndarray):
			awrite (optDepth, outFile, commentChar=commentChar)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions, multiple_outFiles
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
               dict(ID='m', name='mode', default=' ', type=str,  constraint='mode.lower() in "adrst1"'),
	       dict(ID='x', name='xLimits', type=Interval, constraint='0.0<=xLimits.lower<xLimits.upper'),
	       dict(ID='z', name='zenithAngle', type=float, constraint='0.0<=zenithAngle<=180. and not zenithAngle==90.'),
               dict(ID='BoA', name='zBoA', type=float, constraint='zBoA>0.0'),
               dict(ID='ToA', name='zToA', type=float, constraint='zToA>0.0'),
               dict(ID='i', name='interpol', type=str, default='2',
	                    constraint='len(interpol)==1 and interpol.lower() in "0234lqcbhks"'),
               dict(ID='n', name='nanometer'),
               dict(ID='r', name='flipUpDown'),
               dict(ID='v', name='verbose'),
               dict(ID='xFormat', type=str, default='%12f'),
               dict(ID='yFormat', type=str, default='%11.5g'),
               dict(ID='p', name='plotIn'),
               dict(ID='P', name='plotOut')]

	Files, options, commentChar, outFile = parse_command (opts,(1,99))

	if 'h' in options:  raise SystemExit (__doc__ + "\n End of oDepth help")

	outFiles    = multiple_outFiles (Files, outFile)

	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID']=='h')]
	for key in boolOptions:  options[key] = key in options

	for iFile,oFile in zip(Files,outFiles):
		_oDepth_ (iFile, oFile, commentChar, **options)
