#!/usr/bin/env python3

""" lbl2xs
  computation of line-by-line molecular absorption cross sections

  usage:
  lbl2xs [options] line_parameter_file(s)

  -h            help
  -c  char      comment character(s) used in input,output file (default '#')
  -o  string    output file for saving of cross sections (if not given: write to StdOut)
                (in case of a list of input files / molecules, use a star as wildcard:
		 this will be replaced by the molecule name)

  -p  float(s)  pressure(s) (in mb,  default: p_ref of linefile, usually 1013.25mb=1atm)
  -T  float(s)  Temperature(s) (in K, default: T_ref of linefile, usually 296K)
 --pT file      with list of pressures in mb (first column) and temperatures in K (second column)
 --cpT 2ints    a pair of (non-negative) integers indicating the pressure and temperature columns in the pT file
                (default 0,1 for the very first and second column)

  -n  int       number of grids --- selects 'multigrid mode' for nGrids=2 or 3 (default) (nGrids=1 brute force)
  -g  int       gridRatio = ratio of coarse to fine grid spacing (only 2, 4, or 8, default 8)
  -W  float     transition from fine to coarse grid (in units of half widths, default 25.0)

  -a  float     air broadening half width, only used when not given in line list file
  -f  string    format for output file:  'a'='t'='xy' tabular ascii  OR  'h' hitran  OR  "."="p" pickled (default)
  -i  string    interpolation method   [default: '3' three-point Lagrange,  choices are one of "234lqcs"]
                (only required for multigrid approach or when cross sections for several p,T pairs are saved in xy format)
  -L  char      Lineshape: V(oigt), L(orentz), G(auss)     [default: Voigt]
  -s  float     sampling rate used for x-grid (default: 5.0 grid points per (mean) half width)
  -w  float     wing extension (cutoff wavenumber, default 10.0cm-1)
                (Currently no impact on the lbl computation,
		it only expands the interval set by the -x option when reading the line data files)
  -x  Interval  lower,upper wavenumbers (comma separated pair of floats [no blanks!],
                                        default set according to range of lines in datafile)

  If several line parameter files are given (usually for several molecules)
  AND if an output file (extension) has been specified (-o option)
  a cross section file will be generated for each line file:
  * if all line files have the same extension,
    the cross section files will have the old basename with the extension as specified by the -o option
  * otherwise the input file name will be augmented by the string specified as -o option

  For more information use
  lbl2xs --help
"""

_more_Help = \
"""
  The line parameter file should contain a header section indicating the molecule, pressure, temperature,
  and type of columns followed by a list of (preselected) lines in the format
  # molecule: XYZ
  # temperature: 296 K
  # pressure:    1013.25 mb
  # format: vSEan
  position1  strength1  energy1  airWidth1  tDep1
  position2  strength2  energy2  airWidth2  tDep2
  position3  strength3  energy3  airWidth3  tDep3
  .........  .........  .......  .........  .....

  This file can be generated from HITRAN, SAO, or GEISA database with extract.py
  The line list should contain at least 2 columns with line positions (in cm-1) and line strengths.
  If lower state energy is missing, cross sections can be calculated only at the reference temperature of the line list.
  If air broadened half width is missing, it can be set to a (constant) default value with the -a option.
  If temperature dependence of the air broadened half width is missing, the value in the molecular data file is used.
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
#    time import clock as _clock     # deprecated, invalid with latest python versions
from time import perf_counter as _clock

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.aux.aeiou import join_words, awrite
from py4cats.aux.cgsUnits import cgs, unitConversion, pressureUnits
from py4cats.aux.pairTypes import Interval, PairOfInts
from py4cats.aux.lagrange_interpolation import lagrange2_interpolate2, lagrange2_interpolate4, lagrange2_interpolate8, \
				               lagrange3_interpolate2, lagrange3_interpolate4, lagrange3_interpolate8, \
				               lagrange4_interpolate2, lagrange4_interpolate4, lagrange4_interpolate8
from py4cats.lbl.lines import lineArray, read_line_file, xVoigt_parameters, meanLineWidths
from py4cats.lbl.lineshapes import Lorentz, LorentzMix, Gauss, vanVleckWeisskopf, vanVleckHuber, \
		       Voigt, VoigtMix, SpeedVoigt, SpeedVoigtMix, Voigt_Kuntz_Humlicek1, voigtWidth, \
		       Rautian, SpeedRautian, RautianMix, SpeedRautianMix
from py4cats.lbl.molecules import get_molec_data, molecules
from py4cats.art.xSection import xsArray, xsSave

_logFile = None

####################################################################################################################################

# interpolation functions used for multigrid lbl
interpolationDict = {(2,2): lagrange2_interpolate2,
                     (3,2): lagrange3_interpolate2,
                     (4,2): lagrange4_interpolate2,
                     (2,4): lagrange2_interpolate4,
                     (3,4): lagrange3_interpolate4,
                     (4,4): lagrange4_interpolate4,
                     (2,8): lagrange2_interpolate8,
                     (3,8): lagrange3_interpolate8,
                     (4,8): lagrange4_interpolate8}

####################################################################################################################################

def lbl_3grids (positions, strengths, gammaL, gammaD,
                uLimits,  sampling=5.0, gridRatio=8, nWidths=25.0, lagrange=2, verbose=False):
	""" Compute lbl cross sections using three grids: fine grid near line center, medium grid, and coarse grid everywhere.
	    (xs to be computed on monochromatic uniform grid in interval (uLow,uHigh) with spacing defined by half widths. """

	# check optional spectral range, reset if not given (here just for safety, usually done by calling routine lbl2xs)
	if uLimits:
		uLow  = uLimits.lower;  uHigh = uLimits.upper
	else:
		uLow  = positions[0];  uHigh = positions[-1]
		print('INFO --- lbl_3grids:  uLow, uHigh:', uLow, uHigh)

	# Lagrange interpolation method
	interpolation = interpolationDict [(lagrange, gridRatio)]  # print ' lbl_3grids:  ', interpolation.__doc__

	# average line width for Lorentz, Gaussian, and Voigt line shapes
	meanGV = np.mean(voigtWidth(gammaL, gammaD))

	nGrids=3
	c2fhalf = int(gridRatio/2)
	ipl = lagrange

	# fine monochromatic wavenumber grid 'u' with u[0]=uLow and u[-1]=uHigh
	uGrid, xsFine = init_xSection (uLow, uHigh, meanGV, sampling, gridRatio, nGrids, verbose)
	nu    = len(uGrid)-1     # number of grid intervals
	du    = (uHigh-uLow)/nu  # grid spacing

	# medium resolution grid 'v' with v[1]=u[0] and v[-2]=u[-1]
	dv    = gridRatio*du
	recdv = 1.0/dv
	vGrid = np.arange(uLow-c2fhalf*dv,uHigh+(c2fhalf+0.9)*dv,dv)
	nv  = len(vGrid)-1  # number of grid intervals = index of last element

	# coarse resolution grid 'w' with w[1]=v[0] and w[-2]=v[-1]
	dw    = gridRatio*dv
	recdw = 1.0/dw
	wGrid = np.arange(vGrid[0]-c2fhalf*dw,vGrid[-1]+(c2fhalf+0.9)*dw,dw)
	nw  = len(wGrid)-1  # number of grid intervals = index of last element
	#if verbose:
	#	print('\n lbl_3grids: ', '  uGrid', len(uGrid), du, '  vGrid', len(vGrid), dv, '  w', len(wGrid), dw)
	#	print(' u:', 5*'%12.6f' % tuple(uGrid[:5]), ' ...', 5*'%12.6f' % tuple(uGrid[-5:]))
	#	print(' v:', 5*'%12.6f' % tuple(vGrid[:5]), ' ...', 5*'%12.6f' % tuple(vGrid[-5:]))
	#	print(' w:', 5*'%12.6f' % tuple(wGrid[:5]), ' ...', 5*'%12.6f' % tuple(wGrid[-5:]))

	if len(wGrid)<ipl:
		raise SystemExit ('ERROR --- lbl_3grids:  coarse grid W has less grid points than required for interpolation!')

	# and some frequently used wavenumber grid points
	v0      = vGrid[0]
	vEnd    = vGrid[-1]
	w0      = wGrid[0]
	wEnd    = wGrid[-1]
	vLftMin = vGrid[-ipl]
	vRgtMax = vGrid[ipl-1]
	wLftMin = wGrid[-ipl]
	wRgtMax = wGrid[ipl-1]

	# allocate coarse, medium, and fine cross section arrays
	xsCrude   = np.zeros_like(wGrid)
	xsMedium  = np.zeros_like(vGrid)
	vgtMedium = np.zeros_like(vGrid)
	vgtFine   = np.zeros_like(uGrid)

	# sum over lines
	for l in range(len(positions)):
		# compute cross section on entire coarse grid
		vgtCoarse = Voigt_Kuntz_Humlicek1 (wGrid, positions[l], strengths[l], gammaL[l], gammaD[l])
		#gtCoarse = Lorentz               (wGrid, positions[l], strengths[l], gammaL[l])  # almost 10% speedup
		xsCrude   = xsCrude + vgtCoarse

		# set limits for medium resolution region
		centerExtension = nWidths*max(gammaL[l],gammaD[l])
		vLeft  = positions[l]-gridRatio*centerExtension
		vRight = positions[l]+gridRatio*centerExtension
		## need at least 2, 3, or 4 coarse grid points (according to degree of interpolation)
		if vRight<wRgtMax or vLeft>wLftMin:  continue   # line outside spectral interval: next line

		kLeft  = max(int((vLeft-w0)*recdw),c2fhalf)             # left and right coarse grid indices
		kRight = min(nw-int((wEnd-vRight)*recdw),nw-c2fhalf)
		if kLeft>kRight:  continue

		jLeft  = int(gridRatio*(kLeft-c2fhalf))                           # left and right medium grid indices
		jRight = int(gridRatio*(kRight-c2fhalf)+1)                        # add 1 because of pythons range end

		# compute cross section on medium grid in extended center region
		vgtMedium[jLeft:jRight] = Voigt (vGrid[jLeft:jRight], positions[l], strengths[l], gammaL[l], gammaD[l])
		# interpolate coarse grid cross section near extended line center
		li = interpolation (vgtCoarse[kLeft-1:kRight+2])
		# accumulate medium (near extended line center only) cross section
		xsMedium[jLeft:jRight] = xsMedium[jLeft:jRight] + vgtMedium[jLeft:jRight] - li[gridRatio:-gridRatio]

		# set limits for fine resolution region (for Voigt better use voigt width!!!)
		uLeft  = positions[l]-centerExtension
		uRight = positions[l]+centerExtension

		if uRight<vRgtMax or uLeft>vLftMin:  continue   # line outside spectral interval: next line

		iLeft  = max(int((uLeft -v0)*recdv),c2fhalf)           # left and right medium grid indices
		iRight = min(nv-int((vEnd-uRight)*recdv),nv-c2fhalf)
		if iLeft>iRight:  continue

		jLeft  = int(gridRatio*(iLeft-c2fhalf))                           # left and right fine grid indices
		jRight = int(gridRatio*(iRight-c2fhalf) +1)

		# compute cross section on fine grid in center region
		vgtFine[jLeft:jRight] = Voigt (uGrid[jLeft:jRight], positions[l], strengths[l], gammaL[l], gammaD[l])
		# interpolate medium grid cross section near line center
		li = interpolation (vgtMedium[iLeft-1:iRight+2])
		# accumulate fine (near line center only) and medium (everywhere) cross section
		xsFine[jLeft:jRight] = xsFine[jLeft:jRight] + vgtFine[jLeft:jRight] - li[gridRatio:-gridRatio]

	# finally interpolate coarse cross section to medium grid and add to medium
	IXSc  = interpolation(xsCrude)
	XScm  = xsMedium + IXSc[c2fhalf*gridRatio:-c2fhalf*gridRatio]

	# and interpolate coarse cross section to fine grid and add to fine
	IXSm   = interpolation(XScm)
	XS    = xsFine + IXSm[c2fhalf*gridRatio:-c2fhalf*gridRatio]

	return uGrid, XS

####################################################################################################################################

def lbl_2grids (positions, strengths, gammaL, gammaD,
                vLimits,  sampling=5.0, gridRatio=8, nWidths=25.0, lagrange=2, verbose=False):
	""" Compute lbl cross sections using two grids: fine grid near line center, coarse grid everywhere. """

	# check optional spectral range, reset if not given (here just for safety, usually done by calling routine lbl2xs)
	if vLimits:
		vLow  = vLimits.lower;  vHigh = vLimits.upper
	else:
		vLow  = positions[0];  vHigh = positions[-1]
		print('INFO --- lbl_2grids:  vLow, vHigh:', vLow, vHigh)

	# Lagrange interpolation method
	interpolation = interpolationDict [(lagrange, gridRatio)];  print(' lbl_2grids:  ', interpolation.__doc__)

	# average line width for Lorentz, Gaussian, and Voigt line shapes
	#eanGL, meanGD, meanGV  =  meanLineWidths (gammaL, gammaD)
	meanGV = np.mean(voigtWidth(gammaL, gammaD))

	# setup fine and coarse wavenumber grids
	nGrids = 2
	vGrid, xsFine = init_xSection (vLow, vHigh, meanGV, sampling, gridRatio, nGrids, verbose)
	nv     = len(vGrid)    # number of points of fine grid

	dw    = gridRatio*(vHigh-vLow)/(nv-1)       # coarse resolution grid spacing
	recdw = 1.0/dw
	wGrid = np.arange(vLow-dw,vHigh+1.9*dw,dw)  # coarse grid
	nW    = len(wGrid)    # number of points of coarse grid
	iMax  = nW-1          # number of intervals of coarse grid
	if verbose: print('%s %8i%s %8f %8f %8f %8f %s %8f %8f %s %g%s' % ('coarse w  grid: ', len(wGrid)-1, '+1 points: ',
	                   wGrid[0], wGrid[1], wGrid[2], wGrid[3], ' ... ', wGrid[-2], wGrid[-1], '  (delta ', dw,')'))

	# allocate coarse and fine cross section arrays and initialize
	vgtFine   = np.zeros_like(vGrid)  # cross section on fine grid for a single line
	xsCoarse  = np.zeros_like(wGrid)  # cross section on coarse grid: sum over all lines
	vgtCoarse = np.zeros_like(wGrid)  # cross section on coarse grid for a single line

	# some frequently used variables
	wBegin  = wGrid[0]
	wEnd    = wGrid[-1]
	wLftMin = wGrid[-lagrange]
	wRgtMax = wGrid[lagrange-1]

	# sum over all lines
	for l in range(len(positions)):
		#gtCoarse = Voigt_Kuntz_Humlicek1 (wGrid, positions[l], strengths[l], gammaL[l], gammaD[l])
		vgtCoarse = Voigt                 (wGrid, positions[l], strengths[l], gammaL[l], gammaD[l])

		xsCoarse  = xsCoarse + vgtCoarse

		# set limits for line center region
		centerExtension = nWidths*max(gammaL[l],gammaD[l])
		vLeft  = positions[l]-centerExtension
		vRight = positions[l]+centerExtension
		if vRight<wRgtMax or vLeft>wLftMin:  continue   # next line

		iLeft  = max(int((vLeft-wBegin)*recdw),1)          # left and right coarse grid indices
		iRight = min(iMax-int((wEnd-vRight)*recdw),iMax-1)
		jLeft  = gridRatio*(iLeft-1)                             # left and right fine grid indices
		jRight = gridRatio*(iRight-1) + 1                        # add 1 because of pythons range end
		# compute cross section on fine grid in center region
		vgtFine[jLeft:jRight] = Voigt (vGrid[jLeft:jRight], positions[l], strengths[l], gammaL[l], gammaD[l])
		# interpolate coarse grid cross section near line center
		vgtCoarseInt = interpolation (vgtCoarse[iLeft-1:iRight+2])  # add 1 because of pythons range end
		# accumulate fine (near line center only) and coarse (over entire line domain) cross section
		xsFine[jLeft:jRight] = xsFine[jLeft:jRight] + vgtFine[jLeft:jRight] - vgtCoarseInt[gridRatio:-gridRatio]

	# finally interpolate coarse cross section to fine grid and add to fine
	xsCoarseInt = interpolation (xsCoarse)
	XS    = xsFine + xsCoarseInt[gridRatio:-gridRatio]

	return vGrid, XS


####################################################################################################################################

def lbl_brute (voigtLines, vLimits, lineShape='Voigt', sampling=5.0, temperature=None, verbose=False):
	""" Compute lbl cross sections the usual way (brute force, no cutoff in wings).

	    voigtLines    effective line parameters adjusted to p,T
	                  !!! for lineshapes "beyond Voigt" additional parameters are also included here !!!
	    vLimits       wavenumber limits (default: first and last line)
	    lineShape     default "voigt"  (NOTE:  lower case!)
	    sampling      number of grid points per mean half width (default 5.0)
	    temperature   to be used for vanVleck-Huber line shape
	    verbose
	"""

	#print ("lbl_brute:", len(voigtLines), lineShape)
	# next check for nonzero extra (Dicke, speed-dependence, mix) parameters
	if 'speed' in lineShape and np.count_nonzero(voigtLines['AA'])<=0:
		print ("WARNING --- lbl_brute:  speed-dependent Voigt or Rautian, but speed parameters all zero")
		lineShape = lineShape.replace('speed','')
	if 'raut' in lineShape and np.count_nonzero(voigtLines['N'])<=0:
		print ("WARNING --- lbl_brute:  Rautian, but mix parameters all zero")
		lineShape = lineShape.replace('rautian','voigt')
	if 'mix' in lineShape and np.count_nonzero(voigtLines['m'])<=0:
		lineShape = lineShape.replace('mix','')
		print ("\n WARNING --- lbl_brute:  line mix Voigt or Rautian, but mix parameters all zero",
		       "\n                         using 'unmix' lineShape ", lineShape)

	if len(voigtLines)<5 and verbose:  awrite (voigtLines, lineShape.strip().replace(' ','_'), '%12.6f %12g')

	# check optional spectral range, reset if not given (here just for safety, usually done by calling routine lbl2xs)
	if vLimits:
		vLow  = vLimits.lower;  vHigh = vLimits.upper
	else:
		vLow  = voigtLines['v'][0]; vHigh = voigtLines['v'][-1]
		print('INFO --- lbl_brute:  vLow, vHigh:', vLow, vHigh)

	# average line width for Lorentz, Gaussian, and Voigt line shapes
	meanGL, meanGD, meanGV  =  meanLineWidths (voigtLines['L'], voigtLines['D'])

	if   lineShape.startswith('vvh') or 'huber' in lineShape:
		# define uniform wavenumber grid, allocate cross section array and initialize
		vGrid, xSec = init_xSection (vLow, vHigh, meanGL, sampling)
		# sum over all lines
		for line in voigtLines:  xSec += vanVleckHuber (vGrid, line['v'], line['S'], line['L'], temperature)
	elif   lineShape.startswith('vvw') or 'weisskopf' in lineShape:
		# define uniform wavenumber grid, allocate cross section array and initialize
		vGrid, xSec = init_xSection (vLow, vHigh, meanGL, sampling)
		# sum over all lines
		for line in voigtLines:  xSec += vanVleckWeisskopf (vGrid, line['v'], line['S'], line['L'])
	elif lineShape.startswith('l') or lineShape.strip()=='lorentz':
		# define uniform wavenumber grid, allocate cross section array and initialize
		vGrid, xSec = init_xSection (vLow, vHigh, meanGL, sampling)
		# sum over all lines
		if 'mix' in lineShape.lower():
			for line in voigtLines: xSec += LorentzMix (vGrid, line['v'], line['S'], line['L'], line['m'])
		else:
			for line in voigtLines: xSec += Lorentz (vGrid, line['v'], line['S'], line['L'])
	elif lineShape.startswith('g') or lineShape.strip()=='gauss':
		# define uniform wavenumber grid, allocate cross section array and initialize
		vGrid, xSec = init_xSection (vLow, vHigh, meanGD, sampling)
		# sum over all lines
		for line in voigtLines:  xSec += Gauss (vGrid, line['v'], line['S'], line['D'])
	elif lineShape.startswith('v') or 'voigt' in lineShape:
		# define uniform wavenumber grid, allocate cross section array and initialize
		vGrid, xSec = init_xSection (vLow, vHigh, meanGV, sampling)
		# sum over all lines
		if 'speed' in lineShape and 'mix' in lineShape:
			for line in voigtLines:
				if line['AA']>0:  xSec += SpeedVoigtMix (vGrid, line['v'], line['S'], line['L'], line['D'], line['AA'], line['m'])
				else:             xSec +=      VoigtMix (vGrid, line['v'], line['S'], line['L'], line['D'], line['m'])
		elif 'speed' in lineShape:
			for line in voigtLines:
				if line['AA']>0:  xSec += SpeedVoigt (vGrid, line['v'], line['S'], line['L'], line['D'], line['AA'])
				else:             xSec +=      Voigt (vGrid, line['v'], line['S'], line['L'], line['D'])
		elif 'mix' in lineShape:
			for line in voigtLines:   xSec += VoigtMix (vGrid, line['v'], line['S'], line['L'], line['D'], line['m'])
		else:
			for line in voigtLines:   xSec += Voigt (vGrid, line['v'], line['S'], line['L'], line['D'])
	elif lineShape.startswith('r') or 'rautian' in lineShape:
		# define uniform wavenumber grid, allocate cross section array and initialize
		vGrid, xSec = init_xSection (vLow, vHigh, meanGV, sampling)
		# sum over all lines
		if 'speed' in lineShape and 'mix' in lineShape:
			for line in voigtLines:
				if line['AA']>0:  xSec += SpeedRautianMix (vGrid, line['v'], line['S'], line['L'], line['D'], line['AA'], line['N'], line['m'])
				else:             xSec +=      RautianMix (vGrid, line['v'], line['S'], line['L'], line['D'], line['N'], line['m'])
		elif 'speed' in lineShape:
			for line in voigtLines:
				if line['AA']>0:  xSec += SpeedRautian (vGrid, line['v'], line['S'], line['L'], line['D'], line['AA'], line['N'])
				else:             xSec +=      Rautian (vGrid, line['v'], line['S'], line['L'], line['D'], line['N'])
		elif 'mix' in lineShape:
			for line in voigtLines:  xSec += RautianMix (vGrid, line['v'], line['S'], line['L'], line['D'], line['N'], line['m'])
		else:
			for line in voigtLines:  xSec += Rautian (vGrid, line['v'], line['S'], line['L'], line['D'], line['N'])
	elif 'hartmann-tran' in lineShape:
		from ht import HartmannTran, HartmannTranMix, myHartmannTranMix
		# define uniform wavenumber grid, allocate cross section array and initialize
		vGrid, xSec = init_xSection (vLow, vHigh, meanGV, sampling)
		# sum over all lines
		if 'mix' in lineShape:
			for line in voigtLines:  xSec += myHartmannTranMix (vGrid, line['v'], line['S'], line['L'], line['D'], line['AA'], line['N'], yMix=line['m'])
		else:
			for line in voigtLines:  xSec += HartmannTran (vGrid, line['v'], line['S'], line['L'], line['D'], line['AA'], line['N'])
	else:
		raise SystemExit ('ERROR --- lbl_brute:  unknown lineShape ' + repr(lineShape))

	#if verbose:
		#print ('%s %s  %i lines %12g %s %12g' %
		       #('lbl_brute --- ', lineShape, len(voigtLines), min(voigtLines['S']), '<=S<=', max(voigtLines['S'])))
		#if 'speed' in lineShape: print ('speed: %12g %s %12g' % (min(voigtLines['AA']), '<=AA<=', max(voigtLines['AA'])))
		#if 'mix'   in lineShape: print ('mix:   %12g %s %12g' % (min(voigtLines['m']),  '<=m<= ', max(voigtLines['m'])))
		#if 'rau'   in lineShape: print ('Dicke: %12g %s %12g' % (min(voigtLines['N']),  '<=N<= ', max(voigtLines['N'])))

	return vGrid, xSec


####################################################################################################################################

def init_xSection (vLow, vHigh, gammaMean, sampling=5.0,  gridRatio=1, nGrids=1, verbose=False):
	""" Set up equidistant wavenumber grid appropriate to typical line width and allocate corresponding cross section array. """
	# grid point spacing
	dv = gammaMean/sampling
	# number of grid point intervals
	nv = int(round( (vHigh-vLow)/dv ))

	if nGrids<1:  raise SystemExit('ERROR --- init_xSection:  nonpositive number of grids!')
	if gridRatio not in [1,2,4,8]:  raise SystemExit('ERROR --- init_xSection:  invalid grid ratio!')

	# adjust number of grid point intervals to an integer multiple of gridRatio^(nGrids-1)
	mm = gridRatio**(nGrids-1)
	nv = mm*max(int(nv/mm),1)  # make sure that there are at least mm grid points

	# adjust spacing to provide an integer number of intervals
	dv   = (vHigh-vLow)/nv
	# set up array of grid points
	vGrid  = np.arange(vLow,vHigh+0.9*dv,dv)

	# finally allocate cross section array and initialize to zero
	xSec = np.zeros_like(vGrid)

	if verbose: print('%s %8i%s %8f %8f %8f %s %8f %8f %s %g%s %i' %
	                  (' wavenumber grid: ', nv, '+1 points: ', vGrid[0], vGrid[1], vGrid[2],
	                  ' ... ', vGrid[-2], vGrid[-1], '  (delta ', dv,')',mm))
	return vGrid, xSec


####################################################################################################################################

def lbl2xs (lineData, pressure=None, temperature=None, xLimits=None,
            lineShape="Voigt", sampling=5.0, nGrids=3, gridRatio=8, nWidths=25.0, lagrange=2, verbose=False):
	""" Compute cross sections for some molecule(s) and some pressure(s),temperature(s) by summation of line profiles.

            The returned data depends on the type of the incoming lineData, pressure, and temperature:
            * lineData dictionary:  a dictionary of (list of) cross section dictionaries, molecule-by-molecule
            * lineData list:        a list       of (list of) cross section dictionaries, molecule-by-molecule
            * lineData (structured) numpy array:
              If there are several pressure(s) or temperature(s), a list of cross section dictionaries is returned;
              For a single p,T a 'plain' cross section dictionary is returned.

            For each mol,p,T a subclassed numpy array "xsArray" is returned with the cross section 'spectrum'
	    (defined on an equidistant wavenumber grid)
            along with attributes such as molecule, p, T, and wavenumber interval.

            ARGUMENTS:
            ----------

            lineData:     a structured numpy array of line core parameters for a single molecule
                          or a dictionary/list thereof (for several molecules)
            pressure:     a float or a list/array of floats with p[dyn/cm**2]
                          (default:  line parameter database pressure)
                          If a list is given, you can specify the pressure unit in the first or last entry.
            temperature:  a float or a list/array of floats with T[K]
                          (default:  line parameter database temperature)
            xLimits:      Interval with lower and upper wavenumber [cm-1] (default: first and last line)
            lineShape:    "Voigt" default, alternatives are:
	                  "Lorentz", "Gauss", "Rautian", "speed Voigt/Rautian", "vanVleck Huber/Weisskopf"
			  optionally with line mixing
			  short aliases:  SDV, SDVM, SDR, SDRM
	    sampling:     sampling rate used for x-grid (default: 5.0 grid points per (mean) half width)
	    nGrids:       number of grids used for speedup: 1 bruteforce; 2 coarse and fine; 3 coarse, medium, and fine
	    gridRatio:    ratio of grid point intervals in the finer to coarser grid (2, 4, or default: 8)
	    nWidths:      defines the limits of the fine grid (default: 25.0)
	    lagrange:     Lagrange 2 (default), 3, or 4 point interpolation for medium and coarse grid cross sections
            """

	# check lineShape specification to simplify further queries
	lineShape = lineShape.lower()
	if lineShape.startswith('sd') or 'speed' in lineShape:
		if   'v' in lineShape and 'm' in lineShape:  lineShape='speed voigt mix'
		elif 'v' in lineShape:                       lineShape='speed voigt'
		elif 'r' in lineShape and 'm' in lineShape:  lineShape='speed rautian mix'
		elif 'r' in lineShape:                       lineShape='speed rautian'
		else:  raise SystemExit ("ERROR --- lbl_brute:  speed-dependent lineShape, but neither Voigt nor Rautian!?!")
	if lineShape.startswith('ht') or lineShape.startswith('ha') or 'tran' in lineShape:
		if   'mix' in lineShape:  lineShape='hartmann-tran mix'
		else:                     lineShape='hartmann-tran'
	if not lineShape==lineShape.lower():  print ('lbl2xs --- INFO:  lineShape="%s"' % lineShape)

	if isinstance(xLimits,(tuple,list)):  xLimits=Interval(*xLimits)

	if isinstance(lineData,dict):
		# some initial checks
		if not all([isinstance(data,lineArray) for data in list(lineData.values())]):
			print('\nline data dictionary with ', len(lineData), 'items:\n',
			      [(mol,type(lineData[mol])) for mol in list(lineData.keys())])
			raise SystemExit ("ERROR --- lbl2xs:  expected a dictionary of lineArray's")
		if not all([mol in list(molecules.keys()) for mol in list(lineData.keys())]):
			print('WARNING --- lbl2xs: some of the keys of the line data dictionary not found in the list of molecules!')

		# spectral range of line data
		vLimits = Interval(min([lineData[mol]['v'][0]  for mol in list(lineData.keys())]),
                                   max([lineData[mol]['v'][-1] for mol in list(lineData.keys())]))
		if xLimits:
			if not xLimits.overlap(vLimits):
				for mol in list(lineData.keys()):
					print ('%10s %12.3f - %10.3f' % (mol, lineData[mol]['v'][0], lineData[mol]['v'][-1]))
				raise SystemExit('%s %s' % ('ERROR --- lbl2xs:  no lines in requested spectral range ', xLimits))
		else:
			# use the largest interval of line positions (some gases can have lines in a subinterval only)
			xLimits=vLimits

		# now compute cross sections molecule-by-molecule and return a dictionary
		crossSectionDict = {}
		for molec, lines in list(lineData.items()):
			crossSectionDict[molec] = lbl2xs (lines, pressure, temperature, xLimits, lineShape,
	                                                  sampling, nGrids, gridRatio, nWidths, lagrange, verbose)
		return crossSectionDict
	elif isinstance(lineData,(list,tuple)):
		if not xLimits:
			xLimits = Interval(min([lines['v'][0]  for lines in lineData]),
                                           max([lines['v'][-1] for lines in lineData]))
		print(' lbl2xs:', len(lineData), 'list(s) of linedata', [lines.molec for lines in lineData], xLimits)
		# compute cross sections molecule-by-molecule and return a list
		for lines in lineData:
		        # compute cross sections molecule-by-molecule and return a list
			crossSectionList = [lbl2xs (lines, pressure, temperature, xLimits,
	                                            lineShape, sampling, nGrids, gridRatio, nWidths, lagrange, verbose)
						    for lines in lineData]
		return crossSectionList
	elif isinstance(lineData,lineArray):
		molData    = get_molec_data (lineData.molec)
		# check if line data cover the requested spectral range
		if len(lineData)==0:
			raise SystemExit ('ERROR --- lbl2xs:  empty lineArray for ' + repr(lineData.molec) +
			                  '\n            Hint:  delete this molecule from the dictionary of line lists')
		if xLimits and not xLimits.overlap(Interval(lineData['v'][0], lineData['v'][-1])):
			raise SystemExit ('ERROR --- lbl2xs:  no lines in requested spectral range!  ' + repr(lineData.molec) +
			                  '\n            Hint:  delete this molecule from the dictionary of line lists')

		if isinstance(pressure,(list,tuple)):  # for convenience accept non-cgs units as first or last entry
			if   isinstance(pressure[0],str)  and pressure[0] in pressureUnits:
				pressure = unitConversion(np.array(pressure[1:]), 'p', pressure[0])
			elif isinstance(pressure[-1],str) and pressure[-1] in pressureUnits:
				pressure = unitConversion(np.array(pressure[:-1]), 'p', pressure[-1])

		if isinstance(pressure,(list,tuple,np.ndarray)) and isinstance(temperature,(list,tuple,np.ndarray)):
			tStartPT  = _clock()
			print('%-10s %-8s %6.2famu %i %30s %10.4g %10s %.2f  ---> %i p,T pairs ' %
			      ('\n lbl2xs:',  lineData.molec, molData['mass'],
			       len(lineData), 'lines @ reference p [g/cm/s**2]', lineData.p, 'T [K]', lineData.t, len(pressure)))
			if len(pressure)==len(temperature):
				crossSectionList = [lbl2xs (lineData, p, T, xLimits,
	                                                    lineShape, sampling, nGrids, gridRatio, nWidths, lagrange, verbose)
						            for p,T in zip(pressure,temperature)]
				tStopPT = _clock()
				#print(' cross sections %-8s %i lines and %i p/T pairs:  %22.2fsec' %
				#	(molData['molecule'], len(lineData), len(pressure), tStopPT-tStartPT))
				print(' cross sections %-8s %-20s %3i p/T pairs:  %14.2fsec' %
					(molData['molecule'], lineShape, len(pressure), tStopPT-tStartPT))
				return crossSectionList
			else:
				raise SystemExit ('%s %i <--> %i' %
				                  ('ERROR --- lbl2xs:  inconsistent length of pressure and temperature arrays: ',
						   len(pressure), len(temperature)))
		elif isinstance(pressure,(list,tuple,np.ndarray)):
			print('%-10s %-8s %6.2famu  %30s %10g %10s %.2f' %
			      ('\n lbl2xs:',  lineData.molec, molData['mass'], 'reference p [g/cm/s**2]', lineData.p, 'T [K]', lineData.t))
			crossSectionList = [lbl2xs (lineData, p, temperature,
	                                            xLimits, lineShape, sampling, nGrids, gridRatio, nWidths, lagrange, verbose)
						    for p in pressure]
			return crossSectionList
		elif isinstance(temperature,(list,tuple,np.ndarray)):
			print('%-10s %-8s %6.2famu  %30s %10g %10s %.2f' %
			      ('\n lbl2xs:',  lineData.molec, molData['mass'], 'reference p [g/cm/s**2]', lineData.p, 'T [K]', lineData.t))
			crossSectionList = [lbl2xs (lineData, pressure, T,
	                                            xLimits, lineShape, sampling, nGrids, gridRatio, nWidths, lagrange, verbose)
						    for T in temperature]
			return crossSectionList
		else:
			# the ultimate destination, if lbl2xs is called recursively
			# with lineArray's for several molecules and/or several p,T
			tStart  = _clock()
			molData = get_molec_data (lineData.molec)
			if not isinstance(pressure,(int,float)):         pressure = lineData.p
			if not isinstance(temperature,(int,float)):   temperature = lineData.t
			if verbose: print('%-10s%-8s %6.2famu  %20s %10g %s %g\n%50s %10s %s %s' %
                                (' lbl2xs:',  molData['molecule'], molData['mass'],
		                 'pressure [g/cm/s**2]', lineData.p, '--->', pressure,
		                 'temperature      [K]', lineData.t, ' --->',  temperature))
			if pressure<=0 or temperature<=0.0:
				raise SystemExit ("ERROR --- lbl2xs:  non-positive pressure %g or temperature %f" %
				                  (pressure, temperature))

		        # adjust line parameters (strengths and widths) to p, T
			voigtLines = xVoigt_parameters (lineData, molData, pressure, temperature, lineShape, verbose)

			if xLimits:
				xLimits = Interval(float(xLimits.lower),float(xLimits.upper))  # need floats later on for division
			else:
				xLimits = Interval(lineData['v'][0], lineData['v'][-1])        # print 'xLimits', xLimits

			if nGrids==3 and lineShape=='voigt':
				v, xs  =  lbl_3grids (voigtLines['v'], voigtLines['S'], voigtLines['L'], voigtLines['D'],
				                      xLimits, sampling, gridRatio, nWidths, lagrange, verbose)
			elif nGrids==2 and lineShape=='voigt':
				v, xs  =  lbl_2grids (voigtLines['v'], voigtLines['S'],voigtLines['L'], voigtLines['D'],
				                      xLimits, sampling, gridRatio, nWidths, lagrange, verbose)
			else:
				v, xs  =  lbl_brute (voigtLines,  xLimits, lineShape, sampling, temperature, verbose)
			tStop = _clock(); deltaTime = tStop-tStart

			print (' cross section  %-8s %5i lines  %8gmb  %5.1fK  %10i' %
			       (molData['molecule'], len(lineData), cgs('!mb',pressure), temperature, len(xs)), end=' ')

			if _logFile:
				_logFile.write ('%10g %9.3f %9.2g\n' %
				                (cgs('!mb',pressure), 1e9*deltaTime/(len(xs)*len(voigtLines)),
				                 sqrtLn2*np.mean(voigtLines['L']/voigtLines['D'])))

			if len(xs)>0:  print(' %8.2fsec %8.2fns:  %8g < xs < %8g' %
			                     (deltaTime, 1.e9*deltaTime/(len(xs)*len(lineData)), min(xs), max(xs)))
			else:          print('WARNING:  no cross section !?!\n')

			#lblInfo = '%i %s %i %i %i' % (len(xs), lineShape, nGrids, gridRatio, nWidths)
			lblInfo = lineShape
			return xsArray(xs, xLimits, pressure, temperature, lineData.molec, lblInfo)

	else:
		raise SystemExit ('%s %s %s' % ('ERROR --- lbl2xs: invalid type ', type(lineData),
		                                'of lineData, need structured numpy array or dictionary/list thereof'))


####################################################################################################################################

def _lbl2xs_ (lineFiles, outFile='', commentChar='#', pressures=None, temperatures=None, ptFile=None, pTcolumns=(0,1),
              lineShape='Voigt',
              xLimits=None, wingExt=1.0, airWidth=0.1, sampling=5.0, interpolate='2', nGrids=3, gridRatio=8, nWidths=25.0,
	      format=None, verbose=False):
	""" Read a set of line data files (extracts),  compute absorption cross sections for some p,T,  and save spectra. """

	#commonExt = commonExtension(lineFiles)
	if not outFile:  outFile='xSec'

	# pressure(s) and temperature(s)
	if ptFile and (pressures or temperatures):
		raise SystemExit ('ERROR --- _lbl2xs_: ' +
		                  'EITHER give p,T file OR pressure(s) and temperature(s) as command line option')
	elif ptFile:
		try:
			pressures, temperatures = np.loadtxt(ptFile, unpack=1, usecols=pTcolumns.list(), comments=commentChar)
		except Exception as errMsg:
			raise SystemExit ('ERROR --- _lbl2xs_:  reading ' + ptFile + ' failed!\n' + repr(errMsg))
		else:
			if min(pressures)>0:
				pressures      = unitConversion(pressures, 'p', 'mb')
			else:
				pInfo = '(%.2g <= p <= %.2g)' % (min(pressures), max(pressures))
				raise SystemExit ('ERROR --- _lbl2xs_:  minimum pressure non-positive!  ' + pInfo +
				                 '\n                     possibly you have read altitudes instead ?!?')
			if min(temperatures)<=10 or max(temperatures)>2000:           # increase for hot Jupiters
				tInfo = '(%.2g <= T <= %.2g)' % (min(temperatures), max(temperatures))
				raise SystemExit ('ERROR --- _lbl2xs_:  min temperature x-small or max temperature x-large!' +
				                 '\n                     ' + tInfo +
				                 '\n                     possibly you have read the wrong column ?!?')
	else:
		if   isinstance(pressures,(int,float,np.ndarray)):  pressures = unitConversion(pressures, 'p', 'mb')
		elif isinstance(pressures,(list,tuple)):            pressures = unitConversion(np.ndarray(pressures), 'p', 'mb')

	# interpolation method for xs multigrid algorithms: only Lagrange!
	if   interpolate in 'cC':  lagrange=4  # cubic
	elif interpolate in 'qQ':  lagrange=3  # quadratic
	elif interpolate in 'lL':  lagrange=2  # linear
	elif interpolate in '234': lagrange=int(interpolate)
	elif isinstance(interpolate,int) and 2<=interpolate<=4:  pass
	else:                      raise SystemExit ("ERROR --- _lbl2xs_:  invalid interpolation method")

	# get the line data, essentially some structured arrays for each molecule and the reference p,T
	dictOfLineLists = read_line_file (lineFiles, xLimits, wingExt, airWidth, commentChar=commentChar,  verbose=verbose)
	print('\n %i %s %i %s\n' % (sum([len(dictOfLineLists[mol]) for mol in list(dictOfLineLists.keys())]),
                                    'lines for ', len(dictOfLineLists), 'molecule(s)'))

	# compute cross sections for line list(s) and p,T pair(s)
	crossSections = lbl2xs (dictOfLineLists, pressures, temperatures,
	                        xLimits, lineShape, sampling, nGrids, gridRatio, nWidths, lagrange, verbose)

	if   format in ".pP":  commentChar=None # save pickled
	elif format in "hH":   commentChar="H"  # save in Hitran format
	xsSave (crossSections, outFile, commentChar, interpolate)


####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.command_parser import parse_command, standardOptions

        # parse the command, return (ideally) some file(s) and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'about'},
	       {'ID': 'help'},
               {'ID': 'a', 'name': 'airWidth', 'type': float, 'constraint': 'airWidth>0.0'},
               {'ID': 'L', 'name': 'lineShape', 'type': str, 'constraint': 'lineShape[0] in "VLG"', 'default': 'Voigt'},
               {'ID': 'f', 'name': 'format', 'type': str,    'constraint': 'format.lower() in [".","a","h","p","t","xy"]',
	                   'default': '.'},
               {'ID': 'i', 'name': 'interpolate', 'type': str, 'default': '2', 'constraint': 'interpolate in "234lLqQcC"'},
               {'ID': 'p', 'name': 'pressures', 'type': np.ndarray, 'constraint': 'pressures>0.0'},
               {'ID': 'T', 'name': 'temperatures', 'type': np.ndarray, 'constraint': 'temperatures>0.0'},
               {'ID': 's', 'name': 'sampling', 'type': float, 'constraint': 'sampling>0.0', 'default': 5.0},
               {'ID': 'w', 'name': 'wingExt', 'type': float, 'constraint': 'wingExt>0.0', 'default': 5.0},
	       {'ID': 'x', 'name': 'xLimits', 'type': Interval, 'constraint': 'xLimits.lower>=0.0'},
               {'ID': 'n', 'name': 'nGrids', 'type': int, 'default': 3, 'constraint': 'nGrids>0'},
               {'ID': 'g', 'name': 'gridRatio', 'type': int, 'default': 8, 'constraint': 'gridRatio in [1,2,4,8]'},
               {'ID': 'W', 'name': 'nWidths', 'type': float, 'default': 25.0,'constraint': 'nWidths>2.0'},
               {'ID': 'pT', 'name': 'ptFile', 'type': str},
               {'ID': 'cpT', 'name': 'pTcolumns', 'type': PairOfInts, 'default': PairOfInts(0,1),
	                      'constraint': 'pTcolumns.min()>=0'},
               {'ID': 'log', 'name': 'logFile', 'type': str},
               {'ID': 'v',  'name': 'verbose'}
               ]

	lineFiles, options, commentChar, outFile = parse_command (opts,(1,99))

	if 'h' in options:      raise SystemExit (__doc__ + "\n End of lbl2xs help")
	if 'help' in options:   raise SystemExit (__doc__[:-44] + _more_Help + "\n End of lbl2xs help (extended)")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	if 'logFile' in options:
		_logFile = open(options['logFile'],'w')
		_logFile.write (commentChar + ' ' + join_words(sys.argv) + '\n' + commentChar + '\n')
		_logFile.write ('# %8s %9s %9s\n' % ('p [mb]', 't [ns]', 'y'))
		del options['logFile']
		sqrtLn2 = np.sqrt(np.log(2.))

	options['verbose'] = 'verbose' in options

	# loop over line parameter files each with data for one molecule
	_lbl2xs_ (lineFiles, outFile, commentChar, **options)
