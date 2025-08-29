"""  radInt

  Definition of a subclassed numpy array for radiance/intensity along with attributes
  Functions to read, write, convolve, or plot radiation intensity (e.g., to reformat, truncate, or interpolate).
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
import os, sys
from math import ceil
from string import punctuation
from pickle import dump, load, UnpicklingError

try:   import numpy as np
except ImportError as msg:  raise SystemExit(str(msg) + '\nimport numeric python failed!')

from numpy.random import randn

try:
	from matplotlib.pyplot import plot, legend, xlabel, ylabel
except ImportError:
	print ('WARNING --- radInt:  matplotlib not available, no quicklook!')


from ..aux.aeiou import awrite, cstack, grep_from_header, join_words, read_first_line
from ..aux.pairTypes import Interval
from ..aux.cgsUnits import cgs
from ..aux.euGrid import is_uniform
from ..aux.misc import regrid
from ..aux.convolution     import convolveBox, convolveTriangle, convolveGauss
from ..aux.radiance2radiance import radiance2radiance
from ..aux.radiance2Kelvin import radiance2Kelvin, ergs_to_Kelvin

####################################################################################################################################
####################################################################################################################################

class riArray (np.ndarray):
	""" A subclassed numpy array of radiance intensity with xLimits, z, ... attributes added.

	    Furthermore, some convenience functions are implemented:
	    *  dx:       return wavenumber grid point spacing
	    *  grid:     return a numpy array with the uniform wavenumber grid
	    *  info:     return basic information (z, p, T, wavenumber and radiance range)
	    *  regrid:   return a riArray with the ri data interpolated to a new grid (same xLimits!)
	    #  convolve: return a riArray of radiance convolved with a spectral response function
	    #  truncate: return a riArray with the wavenumber range (xLimits) truncated
	    #  kelvin:   return a numpy array of equivalent brightness temperatures
	"""
	# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

	def __new__(cls, input_array, xLimits=None, tBack=None, zenith=None, z=None, p=None, t=None, srf=""):
		# Input array is an already formed ndarray instance
		# First cast to be our class type
		obj = np.asarray(input_array).view(cls)
		# add the new attributes to the created instance
		obj.x      = xLimits
		obj.tBack  = tBack
		obj.zenith = zenith
		obj.z      = z    # altitude interval (bottom and top)
		obj.p      = p    # pressure interval (top and bottom)
		obj.t      = t    # temperature interval (min, max;  cannot use capital "T" because this means 'transpose')
		obj.srf    = srf  # spectral response function
		return obj  # Finally, we must return the newly created object:

	def __array_finalize__(self, obj):
		# see InfoArray.__array_finalize__ for comments
		if obj is None: return
		self.x      = getattr(obj, 'x', None)
		self.tBack  = getattr(obj, 'tBack', None)
		self.zenith = getattr(obj, 'zenith', None)
		self.z      = getattr(obj, 'z', None)
		self.p      = getattr(obj, 'p', None)
		self.t      = getattr(obj, 't', None)
		self.srf    = getattr(obj, 'srf', "")

	def __str__ (self):
		return '# %i grid points for x=%s  (backTemp=%.1fK,  zenith=%sdg)\n %s' % \
		       (len(self), self.x, self.tBack, self.zenith, self.__repr__())

	def info (self):
		""" Return basic information (z, p, T, wavenumber and radiance range). """
		return '%6.1f -- %.1fkm %12.3e -- %9.3emb %8.1f -- %5.1fK  %10i points in  %10f -- %10f cm-1  with  %10.4g < ri < %10.6g  %s' % \
                       (cgs('!km',self.z.lower), cgs('!km',self.z.upper),
		        cgs('!mb',self.p.lower), cgs('!mb',self.p.upper),  self.t.lower, self.t.upper,
		        len(self), self.x.lower, self.x.upper, min(self), max(self), self.srf)

	def dx (self):
		""" Return wavenumber grid point spacing. """
		return  self.x.size()/(len(self)-1)

	def grid (self):
		""" Setup a uniform, equidistant wavenumber grid of len(self). """
		return  self.x.grid(len(self))  # calls the grid method of the Interval class (in pairTypes.py)

	def regrid (self, new, method='l', yOnly=False):
		""" Interpolate radiance (intensity) to (usually denser) uniform, equidistant wavenumber grid. """
		yNew = regrid(self.base,new,method)
		if yOnly:  return  yNew
		else:      return  riArray (yNew, self.x, self.tBack, self.zenith, self.z, self.p, self.t, self.srf)

	def truncate (self, xLimits):
		""" Return a radiance / intensity in a truncated (smaller) wavenumber interval. """
		if isinstance(xLimits,(tuple,list)) and len(xLimits)==2:  xLimits=Interval(*xLimits)
		dx = self.dx()
		iLow  = max(int((xLimits.lower-self.x.lower)/dx), 0)
		iHigh = min(int(ceil((xLimits.upper-self.x.lower)/dx)), len(self)-1)
		xLow  = self.x.lower + iLow*dx
		xHigh = self.x.lower + iHigh*dx
		return  riArray (self.base[iLow:iHigh+1], Interval(xLow,xHigh),
		                 self.tBack, self.zenith, self.z, self.p, self.t, self.srf)

	def convolve (self, hwhm=1.0, srf='Gauss'):
		""" Return a radiance / intensity convolved with a spectral response function of half width @ half maximum. """
		if not isinstance(hwhm,(int,float)):
			raise ValueError ("ERROR --- riArray.convolve:  expected a float, first argument is `hwhm`")
		if self.x.size()<10*hwhm:
			raise ValueError ("ERROR --- riArray.convolve:  hwhm too large!")
		if srf.upper().startswith('G'):
			srfInfo = '%s %s' % ('Gauss:  ', hwhm)
			wGrid, rad = convolveGauss (self.grid(), self.base, hwhm)
		elif srf.upper().startswith('T'):
			srfInfo = '%s %s' % ('Triangle:  ', hwhm)
			wGrid, rad  =  convolveTriangle(self.grid(), self.base, hwhm)
		else:
			srfInfo = '%s %s' % ('Box:  ', hwhm)
			wGrid, rad  =  convolveBox(self.grid(), self.base, hwhm)
		return  riArray (rad, Interval(wGrid[0],wGrid[-1]), self.tBack, self.zenith, self.z, self.p, self.t, srfInfo)

	def __eq__(self, other):
		""" Compare radiance/intensity including its attributes.
		    (For p and od relative differences < 0.1% are seen as 'equal') """
		return self.x==other.x \
			   and self.z.lower==other.z.lower and self.z.upper==other.z.upper \
			   and self.t.lower==other.t.lower and self.t.upper==other.t.upper \
			   and abs(self.p.lower-other.p.lower)<0.001*self.p.lower \
			   and abs(self.p.upper-other.p.upper)<0.001*self.p.upper \
			   and abs(self.tBack-other.tBack)<0.1 \
			   and abs(self.zenith-other.zenith)<0.1 \
			   and self.srf==other.srf \
			   and np.allclose(self.base,other.base,atol=0.0,rtol=0.001)

	def kelvin (self, grid=False):
		""" Convert radiance (vs. wavenumber) in erg/s/(cm^2 sr cm^-1) to BlackBody temperature via inverse Planck.

		    When the wavenumber grid is returned too (default no, i.e. only temperature as numpy array)
		    then the returned tuple can be easily plotted, e.g.
		    plot (*radiance.kelvin(1))
		    (Note:  riPlot can do this anyway)
		"""
		if grid:
			return self.grid(), ergs_to_Kelvin (self.grid(),self.base)
		else:
			return ergs_to_Kelvin (self.grid(),self.base)

	def noise (self, snr=10, radOnly=False):
		""" Add Gaussian noise of signal-to-noise ratio `snr` to radiance. """
		radiance = self.base * (1.0 + 1.0/snr*randn(len(self)))     # gaussian noise
		if radOnly:  return radiance
		else:        return riArray (radiance, self.x, self.tBack, self.zenith, self.z, self.p, self.t, self.srf)


####################################################################################################################################
####################################################################################################################################

def riInfo (riData):
	""" Print information (min, max, mean, ...) for one or several radiance(s). """
	if isinstance(riData,(list,tuple)) and all([isinstance(ri,riArray) for ri in riData]):
		for l, ri in enumerate(riData):  print('%3i %s' % (l, ri.info()))
	elif isinstance(riData,dict) and all([isinstance(ri,riArray) for ri in riData.values()]):
		for key, ri in riData.items():  print('%s %s' % (key, ri.info()))
	elif isinstance(riData,riArray):
		print(riData.info())
	else:
		raise ValueError ("radInt.riInfo:  unknown/invalid data type, expected an riArray or a list or dict thereof!")


####################################################################################################################################

def riConvolve (radi, hwhm=1.0, srf='Gauss'):
	""" Convolve radiance(s) with spectral response function.

	    radi:          radiance intensity spectrum (along with attributes) or a list or dictionary thereof
	    hwhm:          half width at half maximum, default 1.0
	    srf:           spectral respionse function, default 'Box', other choices 'traiangle', 'Gauss', ...
	"""

	if isinstance(radi,(list,tuple)):
		# recursively call the plot function
		return [ri.convolve(hwhm, srf) for ri in radi]
	elif isinstance(radi,dict):
		# recursively call the plot function
		return {key:  ri.convolve(hwhm,srf) for key,ri in radi.items()}
	elif isinstance(radi,riArray):
		return radi.convolve(hwhm, srf)
	else:
		raise SystemExit ("ERROR --- radInt.riConvolve:  expected a riArray or a list/dictionary thereof!")


####################################################################################################################################

def riPlot (radi, mue=False, kelvin=False, **kwArgs):
	""" Quicklook plot radiance intensity vs. wavenumber.

 	    ARGUMENTS:
 	    ----------

	    radi:          radiance intensity spectrum (along with attributes) or a list or dictionary thereof
	    mue:           flag:  plot radiance vs. wavelength [mue] (default False)
	    kelvin:        flag:  plot equ. brightness temperature vs. wavenumber (default False)
	    kwArgs:        passed directly to semilogy and can be used to set colors, line styles and markers etc.
	                   ignored (cannot be used) in recursive calls with lists or dictionaries of radiances.
	"""
	if isinstance(mue, riArray):
		raise ValueError ("riPlot: second argument `mue` is an radiance array\n" +
		                  "            (to plot a list of riArray's put these in brackets)")

	if isinstance(radi,(list,tuple)):
		# recursively call the plot function
		if kwArgs:  print ("WARNING --- riPlot:  got a list of radiances, ignoring kwArgs!")
		for ri in radi:
			riPlot (ri, mue, kelvin)
	elif isinstance(radi,dict):
		# recursively call the plot function
		if kwArgs:  print ("WARNING --- riPlot:  got a dictionary of radiances, ignoring kwArgs!")
		for key,ri in radi.items():
			riPlot (ri, mue, kelvin, label=key)
	elif isinstance(radi,riArray):
		if 'label' in kwArgs:
			labelText=kwArgs.pop('label')
		else:
			labelText = '%.1fdg' % radi.zenith
			if isinstance(radi.tBack,(int,float)) and radi.tBack>0:  labelText += ' %.1fK' % radi.tBack
			if isinstance(radi.srf,str) and radi.srf:                labelText += '  %s  ' % radi.srf
			labelText += '  %.4g < I < %.4g' % (min(radi.base),max(radi.base))
		if mue:
			if kelvin:  print("WARNING --- riPlot:  brightness temperature vs. wavelength not implemented")
			lambdaGrid, radiance = radiance2radiance(radi.grid(), radi.base, newX='mue')
			plot (lambdaGrid, radiance, label=labelText, **kwArgs)
			xlabel (r'Wavelength ~ $\lambda \rm\,[\mu m]$')
			ylabel (r'Radiance ~ $I(\lambda) \rm\, [erg/s/(cm^2\:sr\:\mu m)]$')
		else:
			if kelvin:
				vGrid = radi.grid()
				plot (vGrid, radiance2Kelvin(vGrid, radi.base), label=labelText, **kwArgs)
				ylabel (r'Equ. Temperature ~ $T_{\rm B}(\nu) \rm\, [K]$')
			else:
				plot (radi.grid(), radi.base, label=labelText, **kwArgs)
				ylabel (r'Radiance ~ $I(\nu) \rm\;  [erg/s/(cm^2\:sr\:cm^{-1})]$')
			xlabel (r'Wavenumber $\quad \nu \rm\,[cm^{-1}]$')
	else:
		raise SystemExit ("ERROR --- radInt.riPlot:  expected an riArray or a list/dictionary thereof!")
	legend (fontsize='small')


####################################################################################################################################
####################################################################################################################################

def riRead (riFile, commentChar='#'):
	""" Read radiance intensity vs. wavenumber along with some attributes from file. """

	if not os.path.isfile(riFile):  raise SystemExit ('ERROR --- riRead:  riFile "' + riFile + '"not found')

	# try to determine filetype automatically from first nonblank character in file
	firstLine = read_first_line (riFile)

	if firstLine.startswith(commentChar):
		radiance = _riRead_xy (riFile, commentChar)
	else:
		radiance = _riRead_pickled (riFile)

	return  radiance


####################################################################################################################################

def _riRead_pickled (riFile):
	""" Read radiance intensity vs. wavenumber along with some attributes from pickle file. """

	try:                    pf = open(riFile,'rb')
	except IOError as msg:  raise SystemExit ('%s %s %s\n %s' % ('ERROR:  opening pickled radiance file ',
	                                                           repr(riFile), ' failed (check existance!?!)', msg))

	# NOTE:  the sequence of loads has to corrrespond to the sequence of dumps!
	info  = load(pf); print(riFile, info)

	# initialize list of radiances to be returned
	riData = []

	while 1:
		try:
			riDict = load(pf) #  print (type(riDict), riDict)
			riData.append(riDict)
		except EOFError as msg:
			print ('EOF: ', msg);  pf.close();  break
		except IOError as msg:
			print ('IO-Error: ', msg);  pf.close();  break
		except UnpicklingError as msg:
			print ('Unpickling-Error: ', msg, '\ntrying to continue');  pf.close();  break

	if len(riData)>1:
		# check if all entries have a key indicating a dictionary of radiances
		if all(['key' in ri for ri in riData]):
			print('INFO --- riRead_pickled:  got', len(riData), 'radiances, returning a dictionary')
			return {ri['key']:  riArray(ri['y'], ri['x'], ri.get('tBack'), ri.get('angle'),
			                            ri.get('altitude', ri.get('z')),
			                            ri.get('pressure', ri.get('p')),
			                            ri.get('temperature', ri.get('t')),
			                            ri.get('srf')) for ri in riData}
		else:
			print('INFO --- riRead_pickled:  got', len(riData), 'radiances, returning a list')
			return [riArray(ri['y'], ri['x'], ri.get('tBack'), ri.get('angle'),
			                ri.get('altitude',ri.get('z')),
			                ri.get('pressure',ri.get('p')),
			                ri.get('temperature',ri.get('t')),
			                ri.get('srf')) for ri in riData]
	else:
		print('INFO --- riRead_pickled:  got a single radiance/intensity, returning an riArray')
		ri = riData[0]
		return riArray(ri['y'], ri['x'], ri.get('tBack'), ri.get('angle'),
		               ri.get('altitude', ri.get('z')),
		               ri.get('pressure', ri.get('p')),
		               ri.get('temperature', ri.get('t')),
		               ri.get('srf'))


####################################################################################################################################

def _riRead_xy (riFile, commentChar='#'):
	""" Read radiance intensity vs. wavenumber along with some attributes from ascii tabular file.

	    NOTE:  multi-column data (i.e. several radiances vs. wavenumber) are not yet implemented!
	"""

	vGrid, riValues = np.loadtxt(riFile, usecols=(0,1), unpack=1, comments=commentChar)

	if is_uniform(vGrid,0.002, 1):  vLimits = Interval(vGrid[0],vGrid[-1])
	else:                           raise SystemExit ("ERROR --- riRead_xy:  wavenumber grid is not equidistant/uniform")

	angle     = grep_from_header(riFile,'angle')[0][0]
	backTemp  = grep_from_header(riFile,'back temperature')[0][0]
	try:
		zInfo     = grep_from_header(riFile,'zInterval');  zInterval = Interval(*cgs(zInfo[1],zInfo[0]))
	except Exception as msg:
		zInterval = Interval(0,0);  print (str(msg) + "\nWARNING --- riRead:  failed to read zInterval")
	try:
		pInfo     = grep_from_header(riFile,'pInterval');  pInterval = Interval(*cgs(pInfo[1],pInfo[0]))
	except Exception as msg:
		pInterval = Interval(0,0);  print (str(msg) + "\nWARNING --- riRead:  failed to read pInterval")
	try:
		tInfo     = grep_from_header(riFile,'tInterval');  tInterval = Interval(*tInfo[0])
	except Exception as msg:
		tInterval = Interval(0,0);  print (str(msg) + "\nWARNING --- riRead:  failed to read tInterval")
	srfInfo = grep_from_header(riFile,'specResponseFct')
	if not srfInfo:  srfInfo=''

	return  riArray (riValues, vLimits, backTemp, angle, zInterval, pInterval, tInterval, srfInfo)


####################################################################################################################################
####################################################################################################################################

def riSave (radi, outFile=None, info=None, commentChar=None, xFormat='%10f', yFormat='%10.5g'):
	""" Write radiance intensity vs. wavenumber along with some attributes to file.

 	    ARGUMENTS:
 	    ----------

	    radi:          radiance intensity spectrum (along with attributes) or a list thereof
	    outFile:       destination file, stdout if unspecified
	    info:          information string to be included in file header (default None)
	    commentChar    character to be used for comment lines in file header
	                   default None, i.e. pickle radiance(s)
	    xFormat        format to be used for wavenumbers, default '%10f'
	    yFormat        format to be used for radiance, default '%10.5g'
	                   the format specifiers are only used for the ascii tabular output;
			   if given, ascii output is generated even if commentChar is not set (uses '#')
	"""

	if isinstance(commentChar,str) and len(commentChar)>0 and commentChar[0] in punctuation:
		_riSave_xy (radi, outFile, info, commentChar, xFormat, yFormat)
	elif xFormat!='%10f' or yFormat!='%10.5g':
		_riSave_xy (radi, outFile, info, '#', xFormat, yFormat)
	else:
		_riSave_pickled (radi, outFile, info)


####################################################################################################################################

def _riSave_xy (radi, outFile=None, info=None, commentChar='#', xFormat='%10f', yFormat='%10.5g'):
	""" Write radiance intensity vs. wavenumber along with some attributes to ascii file.  """

	# atmospheric data for the file header
	if isinstance (info, str):  comments = [info]
	else:                       comments = []

	comments.append('wavenumber [cm-1]')
	comments.append('radiance [erg/s/(cm2 sr cm-1)]')

	if isinstance(radi, (list,tuple)) and all([isinstance(ri,riArray) for ri in radi]):
		# check if all wavenumber grids are identical
		if all([rad.x==radi[0].x and len(rad)==len(radi[0]) for rad in radi[1:]]):
			comments.insert(-2,'angle [dg]:     ' + len(radi)*' %10.1f ' % tuple([rad.zenith for rad in radi]))
			comments.insert(-2,'back temperature [K]:' + len(radi)*' %10.1f ' % tuple([rad.tBack for rad in radi]))
			awrite ((radi[0].grid(),cstack(radi)), outFile,
			        comments=comments, format=xFormat+' '+yFormat, commentChar=commentChar)
		else:
			for rad in radi:  print(rad.x, len(rad))
			raise SystemExit ("ERROR --- radInt.riSave:  radiance spectra do have different wavenumber grids!")
	elif isinstance(radi,riArray):
		# save further attributes if given
		if isinstance (radi.tBack, (int,float)):   comments.insert(-2,'back temperature [K]: %10.2f' % radi.tBack)
		if isinstance (radi.zenith, (int,float)):  comments.insert(-2,'angle [dg]:     %10.2f' % radi.zenith)
		if isinstance (radi.z, Interval):
			comments.insert(-2,'zInterval [km]: %10.2f %10.2f' % (cgs('!km',radi.z.lower),cgs('!km',radi.z.upper)))
		if isinstance (radi.p, Interval):
			comments.insert(-2,'pInterval [mb]: %10.2g %10.2g' % (cgs('!mb',radi.p.lower),cgs('!mb',radi.p.upper)))
		if isinstance (radi.t, Interval):  comments.insert(-2,'tInterval  [K]: %10.2f %10.2f' % (radi.t.lower,radi.t.upper))
		if isinstance (radi.srf, str) and len(radi.srf)>0:  comments.insert(-2,'specResponseFct:  %s' % radi.srf)
		# now write ascii tabular file
		awrite ((radi.grid(),radi), outFile, comments=comments, format=xFormat+' '+yFormat, commentChar=commentChar)
	else:
		raise SystemExit ("ERROR --- radInt.riSave:  expected a riArray or a list thereof!")


####################################################################################################################################

def _riSave_pickled (radi, outFile, info=None):
	""" Write radiance / intensity spectrum/spectra to file using pickle format. """
	if outFile:  out = open (outFile, 'wb')
	else:        raise SystemExit ('\n ERROR --- riSave:  no radiance/intensity pickling for standard out!')

	# save a header
	if 'ipython' in join_words(sys.argv) or 'ipykernel' in join_words(sys.argv):
		if isinstance (info, str):  dump('ipy4cats:  ' + info, out)
		else:                       dump('ipy4cats', out)
	else:
		dump(join_words([os.path.basename(sys.argv[0])] + sys.argv[1:]), out)

	if isinstance(radi, (list,tuple)) and all([isinstance(ri,riArray) for ri in radi]):
		for ri in radi:
			dump ({'angle': ri.zenith, 'tBack': ri.tBack, 'altitude': ri.z,
			       'pressure': ri.p, 'temperature': ri.t, 'srf': ri.srf, 'x': ri.x,  'y': ri.base}, out)
	elif isinstance(radi, dict) and all([isinstance(ri,riArray) for ri in radi.values()]):
		for key,ri in radi.items():
			dump ({'key': key,
			       'angle': ri.zenith, 'tBack': ri.tBack, 'altitude': ri.z,
			       'pressure': ri.p, 'temperature': ri.t, 'srf': ri.srf, 'x': ri.x,  'y': ri.base}, out)
	elif isinstance(radi,riArray):
		dump ({'angle': radi.zenith, 'tBack': radi.tBack, 'altitude': radi.z,
		       'pressure': radi.p, 'temperature': radi.t, 'srf': radi.srf, 'x': radi.x,  'y': radi.base}, out)
	else:
		raise SystemExit ("ERROR --- radInt.riSave:  expected an riArray or a list or dict thereof!")
	out.close()
