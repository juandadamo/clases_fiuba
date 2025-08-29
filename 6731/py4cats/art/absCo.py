#!/usr/bin/env python3

""" absCo

  Read and write / plot (molecular) absorption coefficients (e.g., to reformat or interpolate).

  usage:
  absCo [options] ac_file(s)

  command line options:
   -c   char(s)  comment character in input file(s) (default #)
   -f   string   format for output file ['a' | 't' | 'xy' for ascii tabular (default), 'h' for hitran format, otherwise pickle]
   -h            help
   -i   string   interpolation method for spectral domain (absorption coefficient vs wavenumber)
                 "2", "3", "4" for Lagrange interpolation, "s" for spline
                 default: '3' three-point Lagrange
		 '0' in combination with 'xy' tabular output format generates individual files for each p, T, molecule
   -p            matplotlib for quicklook of absorption coefficients
  --xFormat string  format to be used for wavenumbers,   default '%12f'   (only for ascii tabular)
  --yFormat string  format to be used for optical depth, default '%11.5f' (only for ascii tabular)
   -o   file     output file (default: standard output)

  Absorption coefficients files:
  *   xy formatted ascii file with wavenumbers in column 1 and absorption coefficient(s) (for some p,T) in the following column(s).
  *   hitran formatted file similar to cross sections
  *   pickled file (default output of lbl2ac)

  NOTE:
  *  If no output file is specified, only a summary 'statistics' is given !!!
  *  xy tabular output format:
     Absorption coefficients of all p,T pairs will be interpolated to a common wavenumber grid and saved as 'matrix';
     To write an individual file for each p/T, suppress interpolation with '-i0' option.
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

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

# prepare plotting
try:                        from matplotlib.pyplot import semilogy, legend, xlabel, ylabel, show
except ImportError as msg:  print (str(msg) + '\nWARNING --- absCo:  matplotlib not available, no quicklook!')
else:                       pass  # print 'matplotlib imported and setup'

if __name__ == "__main__":  sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats import __version__
from py4cats.aux.ir import c as cLight
from py4cats.aux.cgsUnits import cgs, frequencyUnits, wavelengthUnits
from py4cats.aux.aeiou import awrite, readFileHeader, parse_comments, join_words, read_first_line
from py4cats.aux.euGrid import is_uniform
from py4cats.aux.pairTypes import Interval
from py4cats.aux.misc  import regrid
from py4cats.aux.struc_array import dict2strucArray
from py4cats.lbl.molecules import molecules
from py4cats.art.atmos1D import  simpleNames


####################################################################################################################################
####################################################################################################################################

class acArray (np.ndarray):
	""" A subclassed numpy array of absorption coefficients with x, z, p, T, ... attributes added.

	    Furthermore, some convenience functions are implemented:
	    *  info:     print the attributes and the minimum and maximum ac values
	    *  dx:       return wavenumber grid point spacing
	    *  grid:     return a numpy array with the uniform wavenumber grid
	    *  regrid:   return an acArray with the ac data interpolated to a new grid (same xLimits!)
	    #  truncate: return an acArray with the wavenumber range (xLimits) truncated
	    *  __eq__:   the equality tests accepts 0.1% differences of pressure and all ac values

	"""
	# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

	def __new__(cls, input_array, xLimits=None, z=None, p=None, t=None, molDensityDict=None):
		# Input array is an already formed ndarray instance
		# First cast to be our class type
		obj = np.asarray(input_array).view(cls)
		# add the new attributes to the created instance
		obj.x     = xLimits
		obj.z     = z
		obj.p     = p
		obj.t     = t   # cannot use capital "T" because this means 'transpose'
		obj.molec = molDensityDict
		return obj  # Finally, we must return the newly created object:

	def __array_finalize__(self, obj):
		# see InfoArray.__array_finalize__ for comments
		if obj is None: return
		self.x     = getattr(obj, 'x', None)
		self.z     = getattr(obj, 'z', None)
		self.p     = getattr(obj, 'p', None)
		self.t     = getattr(obj, 't', None)
		self.molec = getattr(obj, 'molec', None)

	def __str__ (self):
		#if len(self)<np.get_printoptions().get('threshold',100):
		return '# %s  (T=%5.1fK, p=%.3e) %i \n %s' % (self.molec, self.t, self.p, len(self), self.__repr__())

	def info (self):
		""" Return basic information (z, p, T, wavenumber and absCo range). """
		print('%s %7.1fkm %10.3emb %6.1fK  %15i wavenumbers in  %10f ... %10f cm-1  with  %8.2g < ac < %8.2g' %
                       (acMolecInfo(self.molec), cgs('!km',self.z), cgs('!mb',self.p), self.t,
		        len(self), self.x.lower, self.x.upper, min(self), max(self)))

	def dx (self):
		""" Return wavenumber grid point spacing. """
		return  self.x.size()/(len(self)-1)

	def grid (self):
		""" Setup a uniform, equidistant wavenumber grid. """
		return  self.x.grid(len(self))  # calls the grid method of the Interval class (in pairTypes.py)

	def regrid (self, new, method='l'):
		""" Interpolate cross section to (usually denser) uniform, equidistant wavenumber grid. """
		return acArray (regrid(self.base,new,method), self.x, self.z, self.p, self.t, self.molec)

	def truncate (self,xLimits):
		""" Return an absorption coefficient in a truncated (smaller) wavenumber interval. """
		if isinstance(xLimits,(tuple,list)) and len(xLimits)==2:  xLimits=Interval(*xLimits)
		dx = self.dx()
		iLow  = max(int((xLimits.lower-self.x.lower)/dx), 0)
		iHigh = min(int(ceil((xLimits.upper-self.x.lower)/dx)), len(self)-1)
		xLow  = self.x.lower + iLow*dx
		xHigh = self.x.lower + iHigh*dx
		return  acArray (self.base[iLow:iHigh+1], Interval(xLow,xHigh), self.z, self.p, self.t, self.molec)

	def __eq__(self, other):
		""" Compare absorption coefficients including their attributes.
		    (For p and ac relative differences < 0.1% are seen as 'equal') """
		return self.x==other.x \
		   and self.z==other.z \
		   and self.t==other.t \
		   and abs(self.p-other.p)<0.001*self.p \
		   and np.allclose(self.base,other.base,atol=0.0,rtol=0.001)


####################################################################################################################################
####################################################################################################################################

def acInfo (absCo):
	""" Print min, max, mean information for one or several absorption coefficient(s). """
	if isinstance(absCo,(list,tuple)):
		for ac in absCo:  ac.info()
	elif isinstance(absCo,acArray):
		absCo.info()
	else:
		raise SystemExit ('ERROR --- acInfo:  unknown/invalid data type, expected an acArray or a list thereof')


def acMolecInfo (acMolec):
	""" Return a 'pretty-print' version of the dictionary of absorbing molecules saved as acArray attribute. """
	if isinstance(acMolec, dict):
		if   len(acMolec)>9:  info = "%i %s" % (len(acMolec), 'absorbing molecules')
		elif len(acMolec)<4:  info = join_words(['%s %.3g ' % (mol, den) for mol,den in acMolec.items()])
		else:                 info = join_words(acMolec.keys())
	else:
		info = acMolec
	return info


####################################################################################################################################

def acPlot (absCo, xUnit='1/cm', tag='', **kwArgs):
	""" Plot one or several absorption coefficients.

	    ARGUMENTS:
	    ----------
	    absCo:       absorption coefficient
	                 either an acArray instance or a list thereof
	    xUnit:       default cm-1, other choices frequencies (Hz, kHz, MHz, GHz, THz) or wavelength (um, mue, nm)
	    tag:         select z|p|t for display in legend labels (default '' for p, T, and molecule list)
	    kwArgs:      passed directly to semilogy and can be used to set colors, line styles and markers etc.
	                 ignored (cannot be used) in recursive calls with lists of absorption coefficients.
	"""

	if isinstance(absCo,(list,tuple)):
		if kwArgs:  print ("WARNING --- acPlot:  got a list of absorption coefficients, ignoring kwArgs!")
		for ac in absCo:  acPlot (ac, xUnit, tag)
	elif isinstance(absCo,acArray):
		ac  = absCo  # just a shortcut
		if 'label' in kwArgs:
			labelText=kwArgs.pop('label')
		else:
			if   tag.lower()=='z':  labelText = '%.1fkm' % cgs('!km', ac.z)
			elif tag.lower()=='p':  labelText = '%10.3gmb' % cgs('!mb', ac.p)
			elif tag.lower()=='t':  labelText = '%8.3fK' % ac.t
			elif tag.lower()=='m':  labelText = acMolecInfo(ac.molec)
			else:                   labelText = '%10.3gmb  %7.2fK  %s' % (cgs('!mb',ac.p), ac.t, acMolecInfo(ac.molec))

		if   xUnit in ['Hz', 'kHz', 'MHz', 'GHz', 'THz']:
			semilogy (cLight/frequencyUnits[xUnit]*ac.grid(), ac.base, label=labelText, **kwArgs)
		elif xUnit in wavelengthUnits.keys():
			semilogy (1.0/(wavelengthUnits[xUnit]*ac.grid()), ac.base, label=labelText, **kwArgs)
		else:
			semilogy (ac.grid(), ac.base, label=labelText, **kwArgs)
	else:
		raise ValueError ('ERROR --- acPlot:  unknown/invalid data type, expected an acArray nor a list thereof')

	if   xUnit in ['Hz', 'kHz', 'MHz', 'GHz', 'THz']:   xlabel (r'frequency $\nu$ [%s]' % xUnit)
	elif xUnit in wavelengthUnits.keys():               xlabel (r'wavelength $\lambda \rm\,[\mu m]$')
	else:                                               xlabel (r'wavenumber $\nu \rm\,[cm^{-1}]$')
	ylabel (r'absorption coefficient  $\alpha \rm\,[1/cm]$')
	legend(fontsize='small')


####################################################################################################################################

def ac_list2matrix (acList, interpol='l'):
	""" Convert a list of absorption coefficients (acArray's) to a matrix and also return the wavenumber grid. """

	if not isinstance(acList,(list,tuple)):
		raise SystemExit ("ERROR --- ac_list2matrix:  expected a list of acArray's, but got %s" % type(acList))
	if not all([isinstance(ac,acArray) for ac in acList]):
		raise SystemExit ("ERROR --- ac_list2matrix:  got a list of %i elements, but not all are acArray's" % len(acList))

	# what is the largest array with the densest grid
	nMax = max([len(ac) for ac in acList])
	# interpolate to this grid
	acMatrix  = np.array([ac.regrid(nMax, interpol) for ac in acList]).T
	# use the last item of the list comprehension above and the Interval grid method
	vGrid = acList[-1].x.grid(nMax)

	return vGrid, acMatrix


####################################################################################################################################
####################################################################################################################################

def acRead (acFile, zToA=0.0, zBoA=0.0, xLimits=None, commentChar='#'):
	""" Read absorption coefficients vs. wavenumber from file, ideally return atmosphere data, too.

	    ARGUMENTS:
	    ----------
	    acFile:       the ascii tabular data file
	    zToA, zBoA:   ignore levels above top-of-atmosphere and/or below bottom-of-atmosphere altitudes
	    xLimits:      wavenumber interval to return a subset of the data;  default None, i.e. read all
	    commentChar:  default '#'

	    RETURNS:
	    --------
	    absCo:        an acArray instance with an absorption coefficient spectrum and some attributes (z, p, T, ...)
	                  OR a list thereof
	"""

	if not os.path.isfile(acFile):  raise SystemExit ('ERROR --- acRead:  acFile "' + acFile + '"not found')

	# try to determine filetype automatically from first nonblank character in file
	firstLine = read_first_line (acFile)

	if firstLine.startswith(commentChar):
		absCo = acRead_xy (acFile, commentChar)
	elif firstLine.split()[0] in list(molecules.keys()):
		absCo = acRead_hitran (acFile)
	else:
		absCo = acRead_pickled (acFile)
	#else: raise SystemExit ('ERROR --- acRead:  unknown/unsupported file type')

	# remove top and bottom layers
	if zToA:
		if zToA<200:
			print('INFO --- acRead:  zToA =', zToA, 'very small probably km, converted to cm', end=' ')
			zToA = cgs('km',zToA);  print(zToA)
		absCo = [ac for ac in absCo if ac.z<zToA]

	if zBoA:
		if zBoA<200:
			print('INFO --- acRead:  zBoA =', zBoA, 'very small probably km, converted to cm', end=' ')
			zBoA = cgs('km',zBoA);  print(zBoA)
		absCo = [ac for ac in absCo if ac.z>zBoA]

	if xLimits:
		return [ac.truncate(xLimits) for ac in absCo]
	else:
		return absCo


####################################################################################################################################

def acRead_hitran (acFile):
	""" Read absorption coefficients vs. wavenumber from hitran formatted file.

	    ARGUMENTS and RETURNS:  see the acRead doc
	"""

	try:    hf = open(acFile)
	except: raise SystemExit ('ERROR --- acRead_hitran:  opening hitran absorption coefficient file ' +
	                          repr(acFile) + ' failed (check existance!?!)')

	acList = []
	while 1:
		try:
			# read the header line of attributes
			listOfAttributes = hf.readline().split()
			lLoA = len(listOfAttributes)
			print(lLoA, listOfAttributes)
			if lLoA==0:
				print(len(acList), 'absorption coefficient(s) read from hitran type file', repr(acFile))
				hf.close(); break
			molec = listOfAttributes[:lLoA-6].join()
			xLow, xHigh, nx, pressure, temperature, height = listOfAttributes[lLoA-6:]
			print(len(acList), repr(molec),  xLow, xHigh, nx, pressure, temperature, height)
			# read the record with an array of absorption coefficient values
			data       = hf.readline()
			acValues   = np.array(list(map(float,data.split())))
			if len(acValues)==int(nx):
				acList.append(acArray(acValues, Interval(float(xLow),float(xHigh)),
				              cgs('km',float(height)), cgs('mb',float(pressure)), float(temperature), molec))
			else:
				raise SystemExit ('%s %s %s %i' %
                                     ('ERROR --- acRead_hitran:  inconsistent number of ac values given in header line', nx,
                                      ' and actual number of values', len(acValues)))
		except ValueError as msg:
			raise SystemExit ('%s\n%s\n%s' %
			          ('ERROR --- acRead_hitran:  could not parse attributes record', repr(listOfAttributes), str(msg)))
		except EOFError:
			print(len(acList), 'absorption coefficient(s) read from Hitran file')
			hf.close();  break
	return acList


####################################################################################################################################

def acRead_pickled (acFile):
	""" Read absorption coefficients (incl. attributes) from pickled output file.

	    ARGUMENTS and RETURNS:  see the acRead doc
	"""

	try:                    pf = open(acFile,'rb')
	except IOError as msg:  raise SystemExit ('%s %s %s\n %s' % ('ERROR:  opening pickled absorption coefficient file ',
	                                                         repr(acFile), ' failed (check existance!?!)', msg))

	from pickle import load

	# initialize list of absorption coefficients to be returned
	acList = []

	# NOTE:  the sequence of loads has to corrrespond to the sequence of dumps!
	info  = load(pf); print(acFile, info)

	while 1:
		try:
			acDict = load(pf)
			acList.append(acArray(acDict['y'], acDict['x'], acDict['altitude'],
			                      acDict['pressure'], acDict['temperature'], acDict['molecules']))
		except EOFError:
			pf.close(); print(len(acList)); break
	return acList


####################################################################################################################################

def acRead_xy (acFile, commentChar='#'):
	""" Read absorption coefficients vs. wavenumber from ascii tabular output file.

	    ARGUMENTS and RETURNS:  see the acRead doc
	"""

	# read entire absorption coefficient file (incl. commented header lines)
	try:
		xyData   = np.loadtxt (acFile,comments=commentChar)
		commentLines = readFileHeader(acFile,commentChar)
	except:
		raise SystemExit ('ERROR --- acRead:  reading absorption coefficient file ' + repr(acFile)
		                  + ' failed (check format etc!?!)')

	# parse comment header and extract some infos about the atmosphere
	goodNames    = list(simpleNames.keys()) + ['z', 'p', 'T', 'molecules'] + list(molecules.keys())
	acAttributes = parse_comments (commentLines, goodNames, commentChar=commentChar)
	molecList    = acAttributes.pop('molecules', None)

	vGrid, absCo = xyData[:,0], xyData[:,1:]
	if is_uniform(vGrid,0.0025, 1):
		vLimits = Interval(vGrid[0],vGrid[-1])
	else:
		raise SystemExit ("ERROR --- acRead_xy:  wavenumber grid is not equidistant/uniform")

	if len(acAttributes)>0:
		atmStrucArray = dict2strucArray(acAttributes, changeNames=simpleNames)
		if absCo.shape[1]==len(atmStrucArray):
			acList = []
			for l, atm in enumerate(atmStrucArray):
				acList.append( acArray(absCo[:,l], vLimits, atm['z'], atm['p'], atm['T'], molecList) )
		return acList
	else:
		print("WARNING --- acRead:  could not find any information about atmosphere (z,p,T) in file header")
		print("                     returning list of absorption coefficients only!")
		return [acArray(absCo[:,l], vLimits) for l in range(absCo.shape[1])]


####################################################################################################################################
####################################################################################################################################

def acSave (absCo, outFile=None, commentChar=None, interpol='l', xFormat='%12.6f', yFormat='%11.5g'):
	""" Write absorption coefficients to ascii (tabular or hitran) or pickled output file.

	ARGUMENTS:
	----------
	absCo:         an acArray instance with an absorption coefficient spectrum and some attributes (p, T, ...)
	               OR a list thereof
	outFile:       file where data are to be stored (if not given, write to stdout)
	commentChar:   if none (default), save data in numpy pickled file,
                       if "H",  save data in Hitran (ascii) format,
                       otherwise ascii-tabular (wavenumber in first column, absCo data interpolated to common, densest grid)
	interpol       interpolation method, default 'l' for linear interpolation with numpy.interp
	                                     2 | 3 | 4  for self-made Lagrange interpolation
	xFormat:       format to be used for wavenumber, default '%12f' (only for ascii and hitran output)
	yFormat:       format to be used for absorption coefficient, default '%11.5g' (only for ascii and hitran output)


	RETURNS:       nothing

	NOTE:          if you want ascii tabular output WITHOUT interpolation,
	               save data in individual files, i.e. call acSave in a loop over all levels
	"""

	if isinstance(absCo,acArray):
		absCo = [absCo]
	elif isinstance(absCo,(list,tuple)):
		if not all([isinstance(ac,acArray) for ac in absCo]):
			raise SystemExit ("ERROR - acSave:  Some of the data in the list are not acArray's!")
	else:
		raise SystemExit ("ERROR - acSave:  got neither a single acArray nor a list/tuple thereof!")

	if isinstance(commentChar,str) and len(commentChar)>0:
		if commentChar.lower().startswith('h'):  _acSave_hitran (absCo, outFile, yFormat)
		elif commentChar[0] in punctuation:      _acSave_xy (absCo, outFile, commentChar, interpol, xFormat, yFormat)
		else:                                    raise ValueError ("ERROR --- acSave:  invalid comment character")
	else:
		_acSave_pickled (absCo, outFile)

####################################################################################################################################

def _acSave_hitran (absCoList, outFile, yFormat='%11.5g'):
	""" Write list of absorption coefficients to Hitran formatted ascii output file. """
	if outFile: out = open (outFile, 'w')
	else:       out = sys.stdout

	for ac in absCoList:
		# NOTE:  if you add (or remove) some of the attributes, adjust the acRead_hitran function
		out.write ('%s   %f %f   %i   %g %8.3f   %8.2f\n' %
		           (ac.molec, ac.x.lower, ac.x.upper, len(ac), cgs('!mb',ac.p), ac.t, cgs('!km',ac.z)))
		# all data in a single long line
		out.write (str(len(ac)*yFormat+'\n') % tuple(ac))
	if outFile: out.close()


####################################################################################################################################

def _acSave_xy (absCoList, outFile, commentChar='#', interpol='l', xFormat='%12.6f', yFormat='%11.5g'):
	""" Write list of absorption coefficients to tabular ascii output file. """
	# retrieve altitudes, pressure, and temperatures
	zzz = np.array([ac.z for ac in absCoList])
	ppp = np.array([ac.p for ac in absCoList])
	ttt = np.array([ac.t for ac in absCoList])

	# prepare the file header
	comments = ['altitude   [km]:' + len(absCoList)*'%12.2f' % tuple(cgs('!km',zzz)),
	            'pressure   [mb]:' + len(absCoList)*'%12g'   % tuple(cgs('!mb',ppp)),
	            'temperature [K]:' + len(absCoList)*'%12.2f' % tuple(ttt),
	            '']
	if all([absCoList[0].molec==ac.molec for ac in absCoList]):  # save molecules only if consistent
		comments.insert(0,'molecules:         ' + absCoList[0].molec)

	# convert list of acArray's to matrix and also return wavenumber grid
	vGrid, acMatrix = ac_list2matrix (absCoList, interpol)
	# and save data plus header info to file
	comments.append ('%s  %i wavenumbers [cm-1] vs %i levels' %
	                 ('Absorption coefficients [1/cm]', len(vGrid), len(absCoList)))
	awrite ((vGrid, acMatrix), outFile, xFormat+yFormat, comments, commentChar=commentChar)


####################################################################################################################################

def _acSave_pickled (absCoList, outFile):
	""" Write list of absorption coefficients to pickle formatted output file. """
	if outFile: out = open (outFile, 'wb')
	else:       raise SystemExit ('\n ERROR --- acSave:  no absorption coefficient pickling for standard out!')

	import pickle
	sysArgv = join_words(sys.argv)
	if 'ipython' in sysArgv or 'ipykernel' in sysArgv or 'jupyter' in sysArgv:
		pickle.dump('%s (version %s): %s @ %s' % ('ipy4cats', __version__, os.getenv('USER'),os.getenv('HOST')), out)
	else:
		pickle.dump(join_words([os.path.basename(sys.argv[0])] + sys.argv[1:]), out)

	for ac in absCoList:
		pickle.dump ({'molecules': ac.molec, 'x': ac.x, 'altitude': ac.z,
				  'pressure': ac.p, 'temperature': ac.t,  'y': ac.base}, out)
	out.close()


####################################################################################################################################
####################################################################################################################################

def _absCo_ (acFile, outFile, commentChar, zToA=0.0, zBoA=0.0, xLimits=None, interpolate='2',
             acFormat='', xFormat='%12.6f', yFormat='%11g', plot=False, verbose=False):

	acList = acRead (acFile, zToA, zBoA, xLimits, commentChar)

	if len(acList)<1:  raise SystemExit ("WARNING --- absCo:  no absorption coefficients found")

	if verbose:  acInfo(acList)

	if plot:     acPlot(acList)

	if outFile:
		if   acFormat.lower().strip() in ['a', 'xy', 't']:  pass
		elif acFormat.lower().strip()=='h':                 xFormat='';  commentChar='h'
		else:                                             commentChar=xFormat=yFormat=''
		acSave(acList, outFile, commentChar, interpolate, xFormat, yFormat)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions, multiple_outFiles
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       dict(ID='x', name='xLimits', type=Interval, constraint='0.0<=xLimits.lower<xLimits.upper'),
               dict(ID='BoA', name='zBoA', type=float, constraint='zBoA>0.0'),
               dict(ID='ToA', name='zToA', type=float, constraint='zToA>0.0'),
               dict(ID='f', name='acFormat', default='', constraint='acFormat in ["a","h","t","xy"]'),
               dict(ID='i', name='interpolate', type=str, default='2',
	                    constraint='len(interpolate)==1 and interpolate.lower() in "0234lqcbhks"'),
               dict(ID='p', name='plot'),
               dict(ID='xFormat', type=str, default='%12.6f'),
               dict(ID='yFormat', type=str, default='%11.5g'),
               dict(ID='v', name='verbose')]

	acFiles, options, commentChar, outFile = parse_command (opts,(1,9))

	if 'h' in options:  raise SystemExit (__doc__ + "End of absCo help")

	outFiles    = multiple_outFiles (acFiles, outFile)
	options['plot']    = 'plot' in options
	options['verbose'] = 'verbose' in options

	for inFile, outFile in zip(acFiles,outFiles):
		_absCo_ (inFile, outFile, commentChar, **options)

	if options['plot']:  show()
