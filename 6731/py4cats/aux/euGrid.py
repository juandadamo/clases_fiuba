#!/usr/bin/env python3

"""
euGrid

Read command argument and try to generate a numpy array defining an equidistant/uniform grid.

Usage:

  euGrid [options] gridSpec

  gridSpec can be
  *  an integer
  *  a plain ascii file (grid will be read from first column (# 0, default))
  *  a numpy expression, e.g. "arange(10)" or "linspace(10.,20.)"
  *  a string in the format 'start[step1]stop1[step2]stop' etc to set up an piecewise equidistant/uniform grid.
  NOTE:  except for the first two cases enclose gridSpec in quotes to prevent any 'misinterpretation' by the unix shell

Options:

  -h               help
  -c     char      comment character(s) used in input,output file (default '#')
  -o     string    output file for saving of grid
  -C     int       number of column to read from file (numpy convention;  default: 0 ---> first column)
  -r               reverse (flip) grid points up <--> down
 --scale float     multiplication factor for grid  (default:  1)
 --shift float     additive constant to shift the grid  (default:  0)
                   NOTE:  scaling is done before shifting !
"""

_LICENSE_ = """\n
This file is part of the Py4CAtS package.

Authors:
Franz Schreier
DLR Oberpfaffenhofen
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
##############################################################################################################

from os.path import isfile

try:
	import numpy as np
except ImportError as msg:
	raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')
else:
	arange   = np.arange
	linspace = np.linspace
	logspace = np.logspace


####################################################################################################################################

def parseGridSpec (gridSpec):
	""" Set up a piecewise equidistant/uniform grid (altitude) specified in format 'start[step1]stop1[step2]stop' or similar.

            Example:  "0[1]10[2]20[2.5]30[5]50" returns an array with 24 grid points.
        """
	# get indices of left and right brackets
	lp = [];  rp = []
	for i,gs in enumerate(gridSpec):
		if   gs=='[':  lp.append(i)
		elif gs==']':  rp.append(i)
		else:          pass
	if len(lp)==len(rp):
		gridStart = [];  gridStop = [];  gridStep = []
		for i in range(len(lp)):
			if i>0:  start=rp[i-1]+1
			else:    start=0
			if i<len(lp)-1: stop=lp[i+1]
			else:           stop=len(gridSpec)

			try:
				gridStart.append(float(gridSpec[start:lp[i]]))
			except ValueError:
				raise SystemExit ('ERROR:  cannot parse grid start specification\nstring not a number!')
			try:
				gridStep.append(float(gridSpec[lp[i]+1:rp[i]]))
			except ValueError:
				raise SystemExit ('ERROR:  cannot parse grid step specification\nstring not a number!')
			try:
				gridStop.append(float(gridSpec[rp[i]+1:stop]))
			except ValueError:
				raise SystemExit ('ERROR:  cannot parse grid stop specification\nstring not a number!')
			if i==0:
				if gridStop[0]<=gridStart[0]: print('incorrect grid specification:  Stop < Start'); raise SystemExit
				newGrid = np.arange(gridStart[0], gridStop[0]+gridStep[0], gridStep[0])
			else:
				if gridStop[i]<=gridStart[i]: print('incorrect grid specification:  Stop < Start'); raise SystemExit
				newGrid = np.concatenate((newGrid[:-1],np.arange(gridStart[i], gridStop[i]+gridStep[i], gridStep[i])))
	else:
		raise SystemExit ('ERROR --- parseGridSpec:  cannot parse grid specification string\n' +
		     '                                      number of opening and closing braces differs!\n' +
		     '                                      Use format start[step]stop')
	# set up new altitude grid
	return newGrid


####################################################################################################################################

def is_uniform (xGrid, eps=0.001, verbose=0):
	""" Compute grid point spacing and return a flag True/False if xGrid is uniform/equidistant or not.
	    If verbose, return a string if uniform, otherwise print info and return an empty string. """
	delta   = np.ediff1d(xGrid)
	uniform = abs(max(delta)-min(delta))<eps*min(delta)
	if verbose:
		if uniform:  return ' equidistant (%g)' % np.mean(delta)
		else:
			print (' nonequidistant grid (%g < dx < %g)' % (min(delta), max(delta)));  return ''
	else:
		return uniform


####################################################################################################################################

def _isFloat (strng):
	""" Check if strng is a float number. """
	try:                float(strng)
	except ValueError:  return False
	else:               return True


####################################################################################################################################

def euGrid (gridSpec, commentChar='#', column=0, scale=None, shift=None, reverse=False, verbose=False):
	""" Parse "gridSpec" and return a numpy array defining a (monotone) grid.

  	gridSpec can be
  	*  an integer
  	*  a plain ascii file (grid will be read from first (default) column)
  	*  a numpy expression, e.g. 'arange(10)' or 'linspace(10.,20.)'
  	*  a string in the format 'start[step1]stop1[step2]stop' etc to set up a piecewise equidistant/uniform grid
           Example: '0[1]10[2.5]20[5]50'   --->   returns a grid with 21 points.

        Further arguments:
        ------------------
        commentChar:   The character used to indicate the start of comments in the data file
        column:        number of column to read from file (default 0 = very first column)
        scale:         a scaling factor (default None)
        shift:         a constant to be added (default None)
                       NOTE:  the grid is first scaled, then shifted
        reverse:       flag to flip "up/down"
        verbose:       flag (default False)
	"""
	print(type(gridSpec), gridSpec, type(column), column)
	if isinstance(gridSpec,(int,float)):
		gridArray = np.arange(gridSpec)
	elif isfile(gridSpec):
		gridArray = np.loadtxt(gridSpec, comments=commentChar, usecols=(column,))
	elif gridSpec.isdigit():
		gridArray = np.arange(int(gridSpec))
	elif _isFloat(gridSpec):
		gridStop = float(gridSpec)
		if abs(gridStop-int(gridStop)):  raise SystemExit("ERROR --- euGrid:  no idea how to form a grid given a noninteger float")
		elif gridStop<0.0:               gridArray = np.arange(0.,gridStop,-1.0)
		else:                            gridArray = np.arange(gridStop)
	elif 'arange' in gridSpec or 'linspace' in gridSpec or 'logspace' in gridSpec:
		try:
			gridArray = eval(gridSpec)
		except TypeError as errMsg:
			raise SystemExit ("ERROR --- euGrid:  failed to evaluate the grid specification\n" + str(errMsg))
	elif gridSpec.count('[')==gridSpec.count(']') > 0:
		gridArray = parseGridSpec (gridSpec)
	else:
		raise SystemExit ('WARNING --- euGrid:  no idea about how to interpret "'+gridSpec+'" and generate a grid array')

	if isinstance(scale,(int,float)):  gridArray *= scale
	if isinstance(shift,(int,float)):  gridArray += shift

	if reverse:  gridArray = np.flipud(gridArray)

	if   np.alltrue(np.ediff1d(gridArray)>0):
		if verbose:  print("INFO: --- euGrid:  gridArray increasing monotonically")
	elif np.alltrue(np.ediff1d(gridArray)<0):
		if not reverse:  print("INFO: --- euGrid:  gridArray decreasing monotonically")
	else:
		print("WARNING --- euGrid:  not monotone!")

	return gridArray


####################################################################################################################################

if __name__ == "__main__":

	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
	from py4cats.aux.command_parser import parse_command, standardOptions
	from py4cats.aux.aeiou import awrite

	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       dict(ID='about'),
	       dict(ID='C',   name='column',    type=int, constraint='column>=0'),
	       dict(ID='scale',   type=float, constraint='abs(scale)>0'),
	       dict(ID='shift',   type=float),
	       dict(ID='r',   name='reverse'),
	       dict(ID='v',   name='verbose')
	       ]

	#ridSpec, options, commentChar, outFile = parse_command (opts,[1,3])
	gridSpec, options, commentChar, outFile = parse_command (opts,1)

	if 'h'     in options:  raise SystemExit (__doc__ + "\n end of euGrid help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	# translate some options to boolean flags
	boolOptions = [opt.get('name',opt['ID']) for opt in opts if not ('type' in opt or opt['ID'] in ('h', 'about'))]
	for key in boolOptions:  options[key] = key in options

	gridArray = euGrid (gridSpec[0], commentChar, **options)
	if outFile:  awrite (gridArray, outFile)
	else:        print(gridArray)
