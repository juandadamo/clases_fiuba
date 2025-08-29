"""  wgtFct

  functions to read, write or plot weighting functions (e.g., to reformat, truncate, or interpolate).
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
import os, sys
from math import ceil
from string import punctuation


try:   import numpy as np
except ImportError as msg:  raise SystemExit(str(msg) + '\nimport numeric python failed!')

try:
	from matplotlib.pyplot import plot, legend, xlabel, ylabel, annotate, grid, contourf, colorbar
except ImportError as msg:
	print (str(msg) + '\nWARNING --- wgtFct:  matplotlib not available, no quicklook!')
else:
	pass  # print 'from matplotlib.pyplot import plot, ...'


from ..aux.aeiou import awrite, join_words, grep_from_header, read_first_line
from ..aux.pairTypes import Interval
from ..aux.misc import regrid
from ..aux.moreFun import quadratic_polynomial
from ..aux.cgsUnits import cgs
from ..aux.convolution     import convolveBox, convolveTriangle, convolveGauss


####################################################################################################################################
####################################################################################################################################

class wfArray (np.ndarray):
	""" A subclassed numpy array of weighting functions with xLimits, sGrid, ... attributes added.

	    Furthermore, some convenience functions are implemented:
	    *  dx:       return wavenumber grid point spacing
	    *  grid:     return a numpy array with the uniform wavenumber grid
	    *  regrid:   return a wfArray with the wf data interpolated to a new grid (same xLimits!)
	    #  truncate: return a wfArray with the wavenumber range (xLimits) truncated
	"""
	# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

	def __new__(cls, input_array, xLimits=None, sGrid=None, zObs=None, angle=None):
		# Input array is an already formed ndarray instance
		# First cast to be our class type
		obj = np.asarray(input_array).view(cls)
		# add the new attributes to the created instance
		obj.x      = xLimits
		obj.s      = sGrid
		obj.z      = zObs
		obj.angle = angle
		return obj  # Finally, we must return the newly created object:

	def __array_finalize__(self, obj):
		# see InfoArray.__array_finalize__ for comments
		if obj is None: return
		self.x      = getattr(obj, 'x', None)
		self.s      = getattr(obj, 's', None)
		self.z      = getattr(obj, 'z', None)
		self.angle = getattr(obj, 'angle', None)

	def __str__ (self):
		return '# (zObs=%s,   angle=%s)  %i \n %s' % (self.z, self.angle, len(self), self.__repr__())

	def info (self):
		""" Return basic information (z, p, T, wavenumber and radiance range). """
		infoList = ['%.3fkm %12g <= wf[:,%2.2i] <= %12g' %
		            (cgs('!km',self.s[j]), min(self.base[:,j]), j, max(self.base[:,j])) for j in range(self.shape[1])]
		return "\n".join(infoList)

	def dx (self):
		""" Return wavenumber grid point spacing. """
		return  self.x.size()/(len(self)-1)

	def grid (self):
		""" Setup a uniform, equidistant wavenumber grid of len(self). """
		return  self.x.grid(len(self))  # calls the grid method of the Interval class (in pairTypes.py)

	def regrid (self, new, method='l', yOnly=False):
		""" Interpolate weighting functions to (usually denser) uniform, equidistant wavenumber grid. """
		yNew = regrid(self.base,new,method)
		if yOnly:  return  yNew
		else:      return  wfArray (yNew, self.x, self.s, self.z, self.angle)

	def truncate (self, xLimits):
		""" Return weighting functions in a truncated (smaller) wavenumber interval. """
		if isinstance(xLimits,(tuple,list)) and len(xLimits)==2:  xLimits=Interval(*xLimits)
		dx = self.dx()
		iLow  = max(int((xLimits.lower-self.x.lower)/dx), 0)
		iHigh = min(int(ceil((xLimits.upper-self.x.lower)/dx)), len(self)-1)
		xLow  = self.x.lower + iLow*dx
		xHigh = self.x.lower + iHigh*dx
		return  wfArray (self.base[iLow:iHigh+1,:], Interval(xLow,xHigh), self.s, self.z, self.angle)

	def zTruncate (self, zToA):
		""" Return weighting functions in a truncated (smaller) altitude interval, i.e. top removed. """
		if zToA<250.:
			zToA = cgs('km', zToA)
			print(' WARNING --- wgtFct.zTruncate:  zToA very small, assuming kilometer units')
		if self.angle>90.0:  # nadir downlooking
			zGrid = self.z-self.s
			lToA  = np.argmin(abs(zGrid-zToA))
			wgtFct = self.base[:,lToA:]
			sGrid  = self.s[lToA:]+zToA-self.z
		else:
			zGrid = self.s
			raise SystemExit ("ERROR --- wfArray.zTruncate: downlooking not yet implemented!")
		return  wfArray (wgtFct, self.x, sGrid, zToA, self.angle)

	def convolve (self, hwhm=1.0, srf='Gauss'):
		""" Return weighting function convolved with a spectral response function of half width @ half maximum """
		if not isinstance(hwhm,(int,float)):
			raise ValueError ("ERROR --- wfArray.convolve:  expected a float, first argument is `hwhm`")
		if self.x.size()<10*hwhm:
			raise ValueError ("ERROR --- wfArray.convolve:  hwhm too large!")
		if srf.upper().startswith('G'):
			wGrid, wf = convolveGauss (self.grid(), self.base, hwhm)
		elif srf.upper().startswith('T'):
			wGrid, wf  =  convolveTriangle(self.grid(), self.base, hwhm)
		else:
			wGrid, wf  =  convolveBox(self.grid(), self.base, hwhm)
		return  wfArray (wf, Interval(wGrid[0],wGrid[-1]), self.s, self.z, self.angle)


####################################################################################################################################

def wfPeakHeight (wgtFct, verbose=False):
	""" Locate the maximum altitude of the weighting function.

	    NOTE:   altitudes are returned in centimeters !!!
	"""

	vGrid = wgtFct.grid()
	if wgtFct.angle>90.0:  zGrid = wgtFct.z-wgtFct.s
	else:                  zGrid = wgtFct.s

	zMax = np.empty_like(vGrid)
	for i,v in enumerate(vGrid):
		lz = np.argmax(wgtFct[i,:])
		zMax[i] = zGrid[lz]
		if lz>0 and lz<len(zGrid)-1:
			if verbose:  print (i, v, lz, zGrid[lz-1:lz+2],wgtFct.base[i,lz-1:lz+2], end='')
			a,b,c = quadratic_polynomial (zGrid[lz-1:lz+2],wgtFct[i,lz-1:lz+2])
			zMax[i] = -b / (2*a)
			if verbose:  print ('--->', zMax[i])
	return zMax


####################################################################################################################################

def wfPlot (wgtFct, wavenumber=None, wLevels=None, labels='xyz', verbose=False):
	""" Plot weighting functions at specific wavenumber(s) (if given) or try a contourf plot.

	    Arguments:
	    ----------
	    wgtFct:      a subclassed 2D np.array with len(vGrid) rows and len(sGrid) columns;
	                 attributes:  wavenumber interval, sGrid (distance [cm] to observer), and angle
	    wavenumber:  wavenumber(s) of interest
	                 if a single integer n is given, plot for n uniformly spaced wavenumber
			 if float(s) are given, pick the wgtFct(s) next to these wavenumber(s)
	                 if unspecified, try a color contour plot
	                 -------------------------> ? Interval then zoom contour ? <-------------------------
	    wLevels:     number of contour levels (default: choose automatically)
	    labels:      default 'xyz' to show/print x-, y-, and z-labels for contour plot
	    verbose:     flag, if True annotate plot

	    NOTE:
	    this function plots weighting function vs altitude, not distance!
	"""

	if isinstance(wgtFct,(list,tuple)):
		if isinstance(wavenumber,(int,float,list,tuple,np.ndarray)):
			for wf in wgtFct:  wfPlot(wf, wavenumber)
			return
		else:
			raise SystemExit ("ERROR --- wfPlot:  no contour plot for a list of weighting functions")
	elif isinstance(wgtFct,wfArray):
		pass
	else:
		raise SystemExit ("ERROR --- wfPlot:  invalid type for first argument wgtFct:", type(wgtFct))

	vGrid = wgtFct.grid()
	dv    = wgtFct.dx()
	zObs  = cgs('!km', wgtFct.z)
	sGrid = cgs('!km', wgtFct.s)

	if not wgtFct.shape[1]==len(sGrid):
		raise SystemExit ("ERROR --- wfPlot: inconsistent size of weighting function matrix and sGrid\n" +
		                  '%i columns vs. %i path steps <= %f.1' % (wgtFct.shape[1], len(sGrid), sGrid[-1]))

	if wgtFct.angle>90.0:  zGrid = zObs-sGrid
	else:                  zGrid = sGrid

	if   isinstance(wavenumber,float):  wavenumber = np.array([wavenumber])
	elif isinstance(wavenumber,int):    wavenumber = np.linspace(vGrid[0],vGrid[-1],wavenumber)
	else:                               pass

	# scale weighting function 1/cm --> 1/km
	wgtFctK = cgs('km', wgtFct.base)

	if isinstance(wavenumber,(list,tuple,np.ndarray)):
		for v in wavenumber:
			if v in wgtFct.x:
				iv = int((v-vGrid[0])/dv)
				try:
					lz = np.argmax(wgtFctK[iv,:])
					if verbose:
						plot (wgtFctK[iv,:], zGrid)
						if 0<lz<len(zGrid):
							aText = r'$\leftarrow$ %f$\rm\, cm^{-1}$ @ %6.2fkm' % (v, zGrid[lz])
							annotate (aText, (wgtFctK[iv,lz],zGrid[lz]), verticalalignment='center')
						else:
							aText = r'%s $\rm cm^{-1}$' %v
							annotate (aText, (wgtFctK[iv,lz],zGrid[lz]),
							          verticalalignment='left', rotation=60.)
					else:
						plot (wgtFctK[iv,:], zGrid, label='%f %.1f' % (v,zGrid[lz]))
				except ValueError as msg:
					errMsg = '%s %s\n%s%s' % ('ERROR --- wfPlot:  ',
					      'inconsistent length of zGrid arrays and number of weighting function matrix columns',
					      20*'_',msg)
					raise SystemExit (errMsg)
			else:
				print('WARNING --- wfPlot:  wavenumber ', v, ' outside interval!')
		xlabel (r'Weighting function $\partial \mathcal{T} / \partial z \rm\, [1/km]$')
		ylabel (r'Altitude $z \rm\,[km]$');  grid(True)
		if not verbose:  legend()
	else:
		if   isinstance(wLevels,int) and wLevels>1:  contourf (vGrid, zGrid, wgtFctK.T, wLevels)
		elif isinstance(wLevels,(np.ndarray,list)):  contourf (vGrid, zGrid, wgtFctK.T, wLevels)
		else:                                        contourf (vGrid, zGrid, wgtFctK.T)
		if 'x' in labels:  xlabel (r'Wavenumber $\nu \rm\,[cm^{-1}]$')
		if 'y' in labels:  ylabel (r'Altitude $z \rm\,[km]$')
		if 'z' in labels:  colorbar(label=r'$\partial \mathcal{T} / \partial z \rm\, [1/km]$', fraction=0.05, pad=0.02)
		else:              colorbar(fraction=0.05, pad=0.02)


####################################################################################################################################

def wfSave (wgtFct, outFile=None, atmos=None, transposeWF=False, commentChar=None):
	""" Write weighting functions vs. wavenumber or distance to file.

	    Arguments:
	    ----------
	    wgtFct:       a subclassed 2D np.array with len(vGrid) rows and len(sGrid) columns;
	                  attributes:  wavenumber interval, sGrid (distance [cm] to observer), and angle
	    outFile:      if unspecified, write to standard output
	    commentChar:  if none (default), save data in numpy pickled file, otherwise xy-ascii-tabular
	    atmos:        optional: save atmospheric data in file header (xy tabular file only)
	    transposeWF:  save the transposed weighting function (xy tabular file only)
	                  default: wavenumber as very first column, further columns for altitude levels

	    NOTE:
	    this function saves weighting function vs distance, not altitude!
	"""
	if isinstance(commentChar,str) and len(commentChar)>0 and commentChar[0] in punctuation:
		_wfSave_xy (wgtFct, outFile, atmos, transposeWF, commentChar)
	else:
		_wfSave_pickled (wgtFct, outFile)


####################################################################################################################################

def _wfSave_pickled (wgtFct, outFile):
	""" Write weighting functions vs. wavenumber or distance to pickled file. """

	if outFile:  out = open (outFile, 'wb')
	else:        raise SystemExit ('\n ERROR --- wfSave:  no weighting function pickling for standard out!')

	import pickle
	if 'ipython' in join_words(sys.argv) or 'ipykernel' in join_words(sys.argv):
		pickle.dump('ipy4cats', out)
	else:
		pickle.dump(join_words([os.path.basename(sys.argv[0])] + sys.argv[1:]), out)

	pickle.dump ({'x':  wgtFct.x, 's': wgtFct.s, 'z': wgtFct.z, 'angle': wgtFct.angle, 'wf':  wgtFct.base}, out)
	# pickle.dump (wgtFct, out)
	# for j in range(wgtFct.base.shape[1]):  pickle.dump(wgtFct.base[:,j], out)

	out.close()


####################################################################################################################################

def _wfSave_xy (wgtFct, outFile=None, atmos=None, transposeWF=False, commentChar='#'):
	""" Write weighting functions vs. wavenumber and distance to ascii tabular file. """

	from atmos1D import gases

	if isinstance(atmos,np.ndarray) and len(atmos.dtype.names)>4:
		nz       = len(atmos)
		species  = gases(atmos)
		comments = ['altitude   [km]:' + nz*'%10g'   % tuple(cgs('!km',atmos['z'])),
		            'pressure   [mb]:' + nz*'%10g'   % tuple(cgs('!mb',atmos['p'])),
		            'temperature [K]:' + nz*'%10.2f' % tuple(atmos['T']),
		            'gases:          ' + len(species)*' %s' % tuple(species)]
	else:
		comments = []

	vGrid = wgtFct.grid()
	sGrid = wgtFct.s

	comments.append('angle [dg]: %.2f' % wgtFct.angle)
	comments.append('zObs [km]:  %.2f' % cgs('!km',wgtFct.z))

	if transposeWF:
		if len(vGrid)<1000:
			comments.append('vGrid indices:' + len(vGrid)*' %12i' % tuple(np.arange(len(vGrid))+1))
			comments.append('vGrid [cm-1]: ' + len(vGrid)*' %12f' % tuple(vGrid))
		comments.append('sGrid [km] vs. weightingFunctions [1/cm]')
		awrite ((cgs('!km',sGrid), wgtFct.T), outFile, '%12f %12g', commentChar=commentChar, comments=comments)
	else:
		comments.append('sGrid [km]:' + len(sGrid)*' %12.2f' % tuple(cgs('!km',sGrid)))
		comments.append('vGrid [1/cm] vs. weightingFunctions [1/cm]')
		awrite ((vGrid, wgtFct), outFile, '%12f %12g', commentChar=commentChar, comments=comments)


####################################################################################################################################
####################################################################################################################################

def wfRead (wfFile, xLimits=None, commentChar='#'):
	""" Read weighting functions vs. wavenumber and distance from file.

	    wfFile:       the ascii tabular or pickled data file
	    xLimits:      wavenumber interval to return a subset of the data;  default None, i.e. read all

	    RETURNS:
	    --------
	    wgtFct        an wfArray instance with a 2D matrix of weighting functions (wavenumber vs distance)
	                  and some attributes
	"""

	if not os.path.isfile(wfFile):  raise SystemExit ('ERROR --- wfRead:  wfFile "' + wfFile + '"not found')

	# try to determine filetype automatically from first nonblank character in file
	firstLine = read_first_line (wfFile)

	if firstLine.startswith(commentChar):
		data   = np.loadtxt(wfFile)
		sGrid  = grep_from_header(wfFile,'sGrid')
		angle  = grep_from_header(wfFile,'angle')[0][0]
		zObs   = grep_from_header(wfFile,'zObs')[0][0]*1e5
		wgtFct = wfArray (data[:,1:], Interval(data[0,0],data[-1,0]), sGrid, zObs, angle)
	else:
		from pickle import load
		pf = open(wfFile,'rb')
		info  = load(pf);  print(wfFile, info)
		wfDict = load(pf)
		pf.close()

		wgtFct = wfArray (wfDict['wf'], wfDict['x'], wfDict['s'], wfDict['z'], wfDict['angle'])
		print ("wfRead -", wfFile, '--->', wgtFct.shape, 'nFreqs*nSteps in ', wgtFct.x, ' and ',
		       cgs('!km',wgtFct.s[0]),cgs('!km',wgtFct.s[-1]), 'km  with ', wgtFct.angle, 'dg')

	if xLimits:  raise SystemExit ("ERROR --- wfRead:  xLimits not yet supported")

	return wgtFct
