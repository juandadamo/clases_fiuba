#!/usr/bin/env python3

""" xSection

  Read and write / plot molecular cross sections (e.g., to reformat or interpolate).

  usage:
  xSection [options] xs_file(s)

  command line options:
   -c   char(s)  comment character in input file(s) (default #)
   -C   ints     sequence of integers: cross section (levels/layers) to extract
                 (in the xy ascii file 0 corresponds to the first column=wavenumber)
   -f   string   format for output file ['a' | 'xy' for ascii (default), 'h' for hitran format]
   -h            help
   -i   string   interpolation method for spectral domain (cross section vs wavenumber)
                 "2", "3", "4" for Lagrange interpolation, "s" for spline
                 default: '3' three-point Lagrange
                 '0' in combination with 'xy' tabular output format generates individual files for each p, T, molecule
  --plot         matplotlib for quicklook of cross sections
   -o   file     output file (default: standard output)

  Cross Section Files:
  *   xy formatted ascii file with wavenumbers in column 1 and cross section(s) (for some p,T) in the following column(s).
  *   Hitran formatted cross section file
  *   pickled cross section file (default output of lbl2xs)

  NOTE:
  *  If no output file is specified, only a summary 'statistics' is given !!!
  *  xy tabular output format:
     For each molecule cross sections of all p,T pairs will be interpolated to a common wavenumber grid and saved as 'matrix';
     To write an individual file for each p/T, suppress interpolation with '-i0' option.
  *  When reading files,  xSection tries to determine the format from the first line (record)
  *  When reading Hitran xs files, the header record is sometimes incorrectly formatted (missing blanks), and xSection might fail.
     (Reading files with xs data distributed over several records is not yet implemented!).
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
from glob import glob
from math import ceil
from string import punctuation

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

# prepare plotting
try:                 from matplotlib.pyplot import figure, semilogy, legend, xlabel, ylabel, show
except ImportError:  print ('WARNING --- xSection:  matplotlib not available, no quicklook!')
else:                pass  # print 'matplotlib imported and setup'

if __name__ == "__main__":
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats import __version__
from py4cats.aux.ir import c as cLight
from py4cats.aux.cgsUnits import cgs, frequencyUnits, wavelengthUnits
from py4cats.aux.pairTypes import Interval
from py4cats.aux.aeiou import parse_comments, readDataAndComments, awrite, cstack, join_words, read_first_line
from py4cats.aux.misc  import regrid
from py4cats.lbl.molecules import molecules


####################################################################################################################################
####################################################################################################################################

class xsArray (np.ndarray):
	""" A subclassed numpy array of cross sections with x, p, T, ... attributes added.

	    Furthermore, some convenience functions are implemented:
	    *  info:     print the attributes and the minimum and maximum xs values
	    *  dx:       return wavenumber grid point spacing
	    *  grid:     return a numpy array with the uniform wavenumber grid
	    *  regrid:   return new xsArray with the xs data interpolated to a new grid (same xLimits!)
	    *  __eq__:   the equality tests accepts 0.1% differences of pressure and all xs values

	"""
	# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

	def __new__(cls, input_array, xLimits=None, p=None, t=None, molec=None, lineShape=None):
		# Input array is an already formed ndarray instance
		# First cast to be our class type
		obj = np.asarray(input_array).view(cls)
		# add the new attributes to the created instance
		obj.x     = xLimits
		obj.p     = p
		obj.t     = t   # cannot use capital "T" because this means 'transpose'
		obj.molec = molec
		obj.lineShape = lineShape
		# Finally, we must return the newly created object:
		return obj

	def __array_finalize__(self, obj):
		# see InfoArray.__array_finalize__ for comments
		if obj is None: return
		self.x     = getattr(obj, 'x', None)
		self.p     = getattr(obj, 'p', None)
		self.t     = getattr(obj, 't', None)
		self.molec = getattr(obj, 'molec', None)
		self.lineShape = getattr(obj, 'lineShape', None)

	def __str__ (self):
		#if len(self)<np.get_printoptions().get('threshold',100):
		return '# %s  (T=%5.1fK, p=%.3e) %i \n %s' % (self.molec, self.t, self.p, len(self), self.__repr__())

	def info (self):
		print('%-9s  %9i wavenumbers in  %10f ... %10f cm-1  with  %8.2g < xs < %10.4g   (T=%5.1fK, p=%.3e)' %
                       (self.molec, len(self), self.x.lower, self.x.upper, min(self), max(self), self.t, self.p))

	def dx (self):
		""" Return wavenumber grid point spacing. """
		return  self.x.size()/(len(self)-1)

	def grid (self):
		""" Setup a uniform, equidistant wavenumber grid. """
		return  self.x.grid(len(self))  # calls the grid method of the Interval class (in pairTypes.py)

	def regrid (self, newLen, method='l'):
		""" Interpolate cross section to (usually denser) uniform, equidistant wavenumber grid. """
		return xsArray (regrid(self.base,newLen,method), self.x, self.p, self.t, self.molec, self.lineShape)

	def truncate (self,xLimits):
		""" Return a cross section in a truncated (smaller) wavenumber interval. """
		if isinstance(xLimits,(tuple,list)) and len(xLimits)==2:  xLimits=Interval(*xLimits)
		dx = self.dx()
		iLow  = max(int((xLimits.lower-self.x.lower)/dx), 0)
		iHigh = min(int(ceil((xLimits.upper-self.x.lower)/dx)), len(self)-1)
		xLow  = self.x.lower + iLow*dx
		xHigh = self.x.lower + iHigh*dx
		return  xsArray (self.base[iLow:iHigh+1], Interval(xLow,xHigh), self.p, self.t, self.molec, self.lineShape)

	def __eq__(self, other):
		""" Compare cross sections including their attributes.
		    (For p and xs relative differences < 0.1% are seen as 'equal') """
		if len(self)==len(other):
			deltaXS = abs(self.base-other.base)/self.base
			return self.molec==other.molec \
			   and self.x==other.x \
			   and self.t==other.t \
			   and abs(self.p-other.p)<0.001*self.p \
			   and all(deltaXS<0.001)  # np.allclose(self.base,otherbase,atol=0.0,rtol=0.001)
		else:
			return False


####################################################################################################################################
####################################################################################################################################

def xsSave (data, outFile=None, commentChar=None, interpolate=None):
	""" Save (write) cross section(s) to pickled or ascii (tabular or Hitran) file(s).

	    data:           a single xs, a list of xs of a single molecule for different p,T
	                    or a dictionary of cross section(s) for some molecules (and some p,T)
	    outFile:        if unspecified and data is a single xs, the name is generated using molecular name, p, and T
	    commentChar:    if none (default), save data in numpy pickled file,
                            if "H" or "h",  save data in Hitran (ascii) format,
                            otherwise (if punctuation character) ascii-tabular
	    interpolate:    if (False or None) and tabular-ascii format, save data in individual files
	                    2, 3, 4  uses the 'self_made' lagrange_regularGrid functions
	                    l        uses numpy.interp linear interpolation
	                    q, c     uses scipy.interp1d for spline interpolation of second or third order

	    WARNING:        output filename determination is a bit tricky and might not work perfectly for recursive calls.
	"""

	if isinstance(data,dict) and all([molec in list(molecules.keys()) for molec in list(data.keys())]):
		# recursively call this function molecule-by-molecule
		for molec,xss in list(data.items()):
			if   isinstance(outFile,str):
				if '*' in outFile:
					iw=outFile.index('*');  oFile = outFile[:iw]+molec+outFile[iw+1:]
				elif outFile.startswith('.'):   oFile = molec+outFile[1:]
				elif molec not in outFile:      oFile = molec+'.'+outFile
				else:                           oFile = outFile
			else:   oFile = None
			xsSave (xss, oFile, commentChar, interpolate)
	elif isinstance(data,(list,tuple)):
		if isinstance(commentChar,str):
			if commentChar.lower().startswith('h'):
				# save all cross sections in Hitran format
				xsSave_hitran (data, outFile)
			else:
				# save all cross sections in a tabular ascii file (assume all data are for the same molecule!)
				if interpolate:
					nMax  = max([len(xs) for xs in data])              # size of largest array with densest grid
					XSS   = [xs.regrid(nMax, interpolate) for xs in data]  # interpolate to this grid
					comments = [' %-16s %s' % ('molecule:', data[0].molec),
						    ' %-16s %s' % ('lineShape:', data[0].lineShape),
						    ' %-16s %s' % ('interpolate', interpolate),
						    ' pressure [mb]:  ' + len(XSS)*' %10g' % tuple([cgs('!mb',xs.p) for xs in data]),
						    ' temperature [K]:' + len(XSS)*' %10g' % tuple([xs.t for xs in data]),
						    ' %-12s %15s' % ('wavenumber', 'cross section')]
					awrite (cstack(XSS[0].grid(), cstack([xs for xs in XSS])), outFile,
					        '%10f %-10g', comments, commentChar=commentChar)
				else:
					# recursive call level-by-level to save data individually in tabular files: no interpolation
					for l,xs in enumerate(data):
						if isinstance(outFile,str):
							if xs.molec in outFile:
								oFile = outFile.replace(xs.molec,
									'%s_%.4gmb_%.1fK' % (xs.molec, cgs('!mb',xs.p), xs.t))
							elif '*' in outFile:
								oFile = outFile.replace('*', '%.4gmb_%.1fK' % (cgs('!mb',xs.p), xs.t))
							else:
								oFile = '%s_%i' % (outFile, l)
						else:   oFile = outFile
						xsSave (xs, oFile, commentChar, interpolate)
		else:
			# save all cross sections in numpy pickled file
			xsSave_pickled (data, outFile)
	elif isinstance(data,xsArray):
                # a single cross section array for a particular p, T, and molecule
		if isinstance(commentChar,str):
			if commentChar.lower().startswith('h'):
				xsSave_hitran ([data], outFile)  # Hitran format
			elif commentChar in punctuation:
				pmb = cgs('!mb',data.p)
				if isinstance(outFile,str):  oFile = outFile
				else:                        oFile = '%s_%.4gmb_%.1fK.xs' % (data.molec, pmb, data.t)
				comments = [' %-16s %s' % ('molecule:',  data.molec),
					    ' %-16s %s' % ('lineShape:', data.lineShape),
					    ' pressure [mb]:   %10g' % pmb,
					    ' temperature [K]: %10.2f' % data.t]
				awrite (cstack(data.grid(), data.base), oFile, '%10f %10g', comments, commentChar=commentChar)
			else:
				raise SystemExit ('ERROR --- xsSave: invalid commentChar "' + commentChar +
				                  '\n        expected a punctuation character, e.g. ' + punctuation)
		else:
			xsSave_pickled ([data], outFile)  # numpy pickled file

	else:
		raise SystemExit ('ERROR --- xsSave:  unknown/invalid type of cross section data ' + repr(type(data)))


####################################################################################################################################

def xsSave_pickled (crossSections, outFile=None):
	""" Write cross sections (typically a list of xs instances for some p,T) to output (file) using Python's pickle module. """
	import pickle
	if outFile: out = open (outFile, 'wb')
	else:       raise SystemExit('\n ERROR --- xsSave_pickled:  no cross section pickling for standard out!' +
	                              '\n                            (choose different format and/or specify output file(s))')
	# NOTE:  the sequence of loads has to corrrespond to the sequence of dumps!
	sysArgv = join_words(sys.argv)
	if 'ipython' in sysArgv or 'ipykernel' in sysArgv or 'jupyter' in sysArgv:
		pickle.dump('%s (version %s): %s @ %s' % ('ipy4cats', __version__, os.getenv('USER'),os.getenv('HOST')), out)
	else:                                    pickle.dump(join_words([os.path.basename(sys.argv[0])] + sys.argv[1:]), out)

	#pickle.dump([len(xs) for xs in crossSections], out) # do we really need this ???
	for xs in crossSections:
		# pack everything in a dictionary and dump to file
		#ickle.dump({'molecule': xs.molec, 'x': xs.x, 'pressure': xs.p,  'temperature': xs.t,  'y': xs.base}, out)
		pickle.dump({'molecule': xs.molec,  'lineShape': xs.lineShape,  'pressure': xs.p,  'temperature': xs.t,
		             'x': xs.x,  'y': xs.base}, out)
	out.close()


####################################################################################################################################

def xsSave_hitran (crossSections, outFile=None, format='%10.4g'):
	""" Write cross sections (typically a list of xs instances for some p,T) to output (file) using the Hitran format. """
	if outFile: out = open (outFile, 'w')
	else:       out = sys.stdout
	for xs in crossSections:
		out.write ('%s   %f %f   %i   %g %8.3f   %g\n' %
	          	(xs.molec, xs.x.lower, xs.x.upper, len(xs), cgs('!mb',xs.p), xs.t, max(xs)))
		# all data in a single long line
		out.write (str(len(xs)*format+'\n') % tuple(xs))
	if outFile: out.close()


####################################################################################################################################
####################################################################################################################################

def xsRead (xsFile, commentChar='#', verbose=0):
	""" Read cross section data from file(s) and return a list of xsArray's.

	    If xsFile is a list of files or a string including a wildcard (e.g. '*.xs'),
	    xsRead is called recursively for each file and a dictionary of cross section (lists) is returned.

	    xsArray:  a subclassed numpy array with the cross section 'spectrum'
	              and attributes p, t, molec, lineShape added. """

	if isinstance(xsFile,str) and ('*' in xsFile or '?' in xsFile):
		xsFile = glob(xsFile)
		print ('xsRead --- reading xs from', len(xsFile), 'files', xsFile)

	if isinstance(xsFile,(list,tuple)):
		# recursively call this function and read files (probably one file per molecule)
		xssList = [xsRead(file,commentChar) for file in xsFile]
		# check data returned and try to make a dictionary, with one entry per molecule
		xssDict = {}
		for xss,file in zip(xssList,xsFile):
			if isinstance(xss,list):
				if verbose:  print ("file", file, " with", len(xss), "cross sections")
				gases = [xs.molec for xs in xss]
				gas0  = gases[0]
				if all([gas0==gas for gas in gases]):
					if gas0 not in list(xssDict.keys()):
						xssDict[gas0] = xss
					else:
						raise SystemExit ("ERROR --- xsRead:  double xs data for molecule " + gas0 +
						                  " --- again in file " + file)
				else:
					raise SystemExit ("ERROR --- xsRead: inconsistent xs data set read from file " + file +
					                  "\n            expected cross sections for one molecule, but found\n" +
							  join_words(gases))
				if gas0 not in list(molecules.keys()):
					print ("WARNING --- xsRead:  molecule", gas0, " not found in list of 'known' molecules")
			else:
				raise SystemExit ("ERROR --- xsRead:  expected a list of cross sections read from file" + file +
				                  "\n             but found" + type(xss))
		# final check:  compare length of data (number of p,T levels)
		lenXS = [len(xss) for xss in xssDict.values()]
		if not all([lxs==lenXS[0] for lxs in lenXS]):
			if verbose:
				for gas in xssDict.keys():  print (gas, len(xssDict[gas]))
			raise SystemExit ("ERROR --- xsRead:  different number of xsArrays read from files")
		return xssDict

	if not (isinstance(xsFile,str) and os.path.isfile(xsFile)):
		raise SystemExit ('ERROR --- xsRead:  xsFile "' + xsFile + '"not found or name invalid')

	# try to determine filetype automatically from first nonblank character in file
	firstLine = read_first_line(xsFile)

	if firstLine.startswith(commentChar):
		crossSections = xsRead_xy (xsFile, commentChar)
	elif firstLine.split()[0] in list(molecules.keys()):
		crossSections = xsRead_hitran (xsFile)
	else:
		crossSections = xsRead_pickled (xsFile)

	if verbose:  xsInfo(crossSections)

	if len(crossSections)==1:
		crossSections = crossSections[0]
		print ("\n xsRead --- INFO: a single xs read from file ", xsFile,
		       "\n                  returning a single xsArray (instead of a list)")

	return crossSections


####################################################################################################################################

def xsRead_pickled (xsFile):
	""" Read one cross section file (python pickle format; one molecule, several p,T pairs). """
	try:    f = open(xsFile,'rb')
	except: raise SystemExit('ERROR:  opening pickled cross section file ' + repr(xsFile) + ' failed (check existance!?!)')
	# initialize list of cross sections to be returned
	xsList = []
	# NOTE:  the sequence of loads has to corrrespond to the sequence of dumps!
	import pickle
	info  = pickle.load(f);  print(xsFile, info)
	while 1:
		try:
			xsDict = pickle.load(f)
			xsList.append(xsArray(xsDict['y'], xsDict['x'], xsDict['pressure'], xsDict['temperature'],
			                      xsDict['molecule'], xsDict.get('lineShape',"")))
		except EOFError:
			print(len(xsList), ' cross section(s) read from pickled file', xsFile)
			f.close()
			break
	return xsList


####################################################################################################################################

def xsRead_hitran (file):
	""" Read one cross section file (Hitran format; one molecule, several p,T pairs). """
	try:    hf = open(file)
	except: raise SystemExit('ERROR:  opening hitran cross section file ' + repr(file) + ' failed (check existance!?!)')

	xsList = []
	while 1:
		try:
			attributes = hf.readline().strip()
			if len(attributes)==0:
				print(len(xsList), 'cross section(s) read from hitran file', repr(file))
				hf.close(); break
			molec, xLow, xHigh, nx, pressure, temperature, info = attributes.split(maxsplit=6)
			data       = hf.readline()
			xsValues   = np.array(list(map(float,data.split())))
			if len(xsValues)==int(nx):
				xsList.append(xsArray(xsValues, Interval(float(xLow),float(xHigh)), float(pressure), float(temperature), molec))
			else:
				raise SystemExit ('%s %s %s %i' %
				      ('ERROR --- xsRead_hitran:  inconsistent number of xs values given in header line', nx,
				      ' and actual number of values', len(xsValues)))
		except ValueError as msg:
			raise SystemExit ('%s\n%s\n%s' % ('ERROR --- xsRead_hitran:  could not parse attributes record', repr(attributes), str(msg)))
		except EOFError:
			print(len(xsList), 'cross section(s) read from Hitran file')
			hf.close(); break
	return xsList


####################################################################################################################################

def xsRead_xy (xsFile, commentChar='#'):
	""" Read one cross section file (xy ascii format; typically one molecule, several p,T pairs). """
	# initialize list of cross sections to be returned
	xsList = []
	# read entire cross section file (incl. commented header lines)
	try:
		xyData, comments = readDataAndComments(xsFile,commentChar)
	except:
		raise SystemExit ('ERROR --- xsRead_xy:  reading cross section file failed (check format etc!?!)\n' + repr(xsFile))

	# parse comment header and extract some infos
	need = ('pressure', 'temperature', 'molecule')
	headerDict = parse_comments (comments, need, commentChar=commentChar)

	npT = xyData.shape[1]-1  # the very first column is wavenumber
	xLimits = Interval(xyData[0,0], xyData[-1,0])

	for l in range(npT):
		xsList.append(xsArray(xyData[:,l+1], xLimits, headerDict['pressure'][l], headerDict['temperature'][l], headerDict['molecule']))

	print(len(xsList), ' cross section(s) read from tabular file', xsFile)

	return xsList


####################################################################################################################################
####################################################################################################################################

def xsInfo (xsData):
	""" Print information on cross sections. """

	if isinstance(xsData,dict):
		print ("Dictionary of cross sections with", len(xsData), " elements:")
		for key, xs in xsData.items():
			print ('\n ', key)
			xsInfo(xs)
	elif isinstance(xsData,(list,tuple)):
		for xs in xsData:  xs.info()
	elif isinstance(xsData,xsArray):
		xsData.info()
	else:
		raise SystemExit ("ERROR --- xsInfo:  unknown/invalid data type,  expected an xsArray or a list/dictionary thereof")


####################################################################################################################################

def xsPlot (xsData, xUnit='1/cm', info='ptm', **kwArgs):
	""" Plot one or several cross sections.

	    xsData can be a single xsArray (for one molecule and p,T),
	    a list of xsArray's (for one molecule and some levels),
	    or a dictionary of (lists of) xsArray's (for some molecules, p, T).

	    OPTIONAL ARGUMENTS:
	    xUnit    default cm-1, other choices frequencies (Hz, kHz, MHz, GHz, THz) or wavelength (um, mue, nm)
	    info     default 'ptm'  ==> show pressure, temperature, molecule as legend label;
	             other choices: 'f' for line shape function,  'n' for array length;
	             Ignored when label is set explicitely (in kwArgs).

	    kwArgs is passed directly to semilogy and can be used to set colors, line styles and markers etc.
	    kwArgs is ignored (cannot be used) in recursive calls with lists or dictionaries of cross sections.
	"""
	if isinstance(xsData,dict):
		if kwArgs:  print ("WARNING --- xsPlot:  got a dictionary of cross sections, ignoring kwArgs!")
		if all([key in list(molecules.keys()) for key in list(xsData.keys())]):
			for molec,xss in list(xsData.items()):
				if isinstance(xss,list):  figure(molec)
				xsPlot(xss, xUnit, info)
		elif 'x' in xsData and 'y' in xsData:
			xs  = xsData
			xGrid  = xs['x'].grid(len(xs['y']))
			semilogy (xGrid, xs['y'], label='%10.3gmb  %7.2fK' % (cgs('!mb',xs['p']), xs['T']))
		else:
			raise ValueError ('ERROR --- xsPlot:  got a dictionary, but entries are not xsArray or a list thereof!')
	elif isinstance(xsData,(list,tuple)):
		if kwArgs:  print ("WARNING --- xsPlot:  got a list of cross sections, ignoring kwArgs!")
		for xs in xsData:  xsPlot(xs, xUnit, info)
	elif isinstance(xsData,xsArray):
		xs  = xsData
		infoText = "";  info=info.lower()
		if 'm' in info:  infoText +=  '%s' % xs.molec
		if 'p' in info:  infoText += ' %10.3gmb' % cgs('!mb',xs.p)
		if 't' in info:  infoText += ' %7.2fK' % xs.t
		if 'f' in info:  infoText += ' %s' % xs.lineShape
		if 'n' in info:  infoText += ' %i' % len(xs)
		if 'label' in kwArgs:  infoText=kwArgs.pop('label')
		if   xUnit in ['Hz', 'kHz', 'MHz', 'GHz', 'THz']:
			semilogy (cLight/frequencyUnits[xUnit]*xs.grid(), xs.base, label=infoText, **kwArgs)
		elif xUnit in wavelengthUnits.keys():
			semilogy (1.0/(wavelengthUnits[xUnit]*xs.grid()), xs.base, label=infoText, **kwArgs)
		else:
			semilogy (xs.grid(),     xs.base, label=infoText, **kwArgs)
	else:
		raise ValueError ("ERROR --- xsPlot:  unknown/invalid data type,  expected an xsArray or a list/dictionary thereof")

	if   xUnit in ['Hz', 'kHz', 'MHz', 'GHz', 'THz']:  xlabel (r'frequency $\nu$ [%s]' % xUnit)
	elif xUnit in wavelengthUnits.keys():              xlabel (r'wavelength $\lambda$ [%s]' % xUnit)
	else:                                              xlabel (r'wavenumber $\nu \rm\,[cm^{-1}]$')
	ylabel (r'cross section $k(\nu) \rm\,[cm^2/molec]$')
	legend(fontsize='small')


####################################################################################################################################
####################################################################################################################################

def _xSection_ (iFile, oFile=None, commentChar='#', format=None, interpolate='3', columns=[], plot=False, verbose=False):

	# read cross sections from file
	crossSections = xsRead (iFile, commentChar)

	# remove unwanted cross sections
	if columns:
		columns = [int(C)-1 for C in columns.split(",")]
		newCrossSections = [xs for l,xs in enumerate(crossSections) if l in columns]
		crossSections = newCrossSections

	# print summary
	if verbose:
		for xs in crossSections:
			nv     = len(xs) - 1  # number of intervals !!!
			deltaX = xs.x.size() / nv
			iMax   = np.argmax(xs)
			print('%-10s %10i wavenumbers:  dv = %-10.3g  %s  %10gmb %8.2fK %12.3g < xs < %10.3g  @ %i %10f   avg %10.3g' %
			     (xs.molec, nv+1, deltaX, xs.x, cgs('!mb',xs.p), xs.t,
			     min(xs), max(xs), iMax, xs.x.lower+iMax*deltaX, np.mean(xs)))

	if plot:
		figure()
		xsPlot (crossSections)

	# save cross sections
	if oFile:
		if   format in ".pP":  commentChar=None  # save pickled
		elif format in "hH":   commentChar="H"   # save in Hitran format
		xsSave (crossSections, outFile, commentChar, interpolate)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions, multiple_outFiles
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
               {'ID': 'f', 'name': 'format', 'default': 'xy',  'constraint': 'format in [" ","a","h","t","xy"]'},
	       {'ID': 'C', 'name': 'columns', 'type': str,
	        'constraint': 'all([digit.strip().isdigit() for digit in split(columns,",")])'},
               {'ID': 'i', 'name': 'interpolate', 'type': str,
	        'default': '3', 'constraint': 'len(interpolate)==1 and interpolate.lower() in "0234lqcbhks"'},
               {'ID': 'plot'},
               {'ID': 'v', 'name': 'verbose'}]

	Files, options, commentChar, outFile = parse_command (opts,(1,99))

	if 'h' in options:  raise SystemExit (__doc__ + "End of xSection help")

	outFiles    = multiple_outFiles (Files, outFile)
	options['plot']    = 'plot' in options
	options['verbose'] = 'verbose' in options

	for iFile,oFile in zip(Files,outFiles):
		_xSection_ (iFile, oFile, commentChar, **options)

	if options['plot']:  show()
