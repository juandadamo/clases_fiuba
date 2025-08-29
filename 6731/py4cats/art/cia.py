""" Modules for Collision Induced Absorption (and CKD continua?) """

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

import os
from glob import glob

try:   import numpy as np
except ImportError as msg:  raise SystemExit(str(msg) + '\nimport numeric python failed!')

# prepare plotting
try:                 from matplotlib.pyplot import semilogy, legend, xlabel, ylabel
except ImportError:  print ('WARNING --- xSection:  matplotlib not available, no quicklook!')
else:                pass  # print 'matplotlib imported and setup'

from py4cats.aux.aeiou import grep
from py4cats.aux.euGrid import is_uniform
#from py4cats.aux.misc import approx, float_in_list, monotone
from py4cats.aux.pairTypes import Interval
from py4cats.art.absCo import acArray


####################################################################################################################################

def read_hitran_cia (ciaFile, xLimits=None, xsMin=0.0, verbose=False):
	""" Read a Hitran CIA cross section file, pack "spectrum" into a dictionary, and return a list of data.
	    Optionally truncate w.r.t. wavenumbers and/or small (default: negative) values.
	"""

	if isinstance(xLimits,(tuple,list)):  xLimits=Interval(*xLimits)
	comments = grep(ciaFile, '^ *[A-Z]')
	data     = np.loadtxt(ciaFile, usecols=(0,1), comments=comments[0][0])  # some files have 3 columns occasionally
	xsList = []
	iLow = 0
	for block in comments:
		header = block.split()
		pair   = header[0].split('-')
		if pair[1]=='Air':  pair[1]='air'   # use lower case because py4cats atmos1D structure array is lower case
		vMin, vMax, nData, temp   = float(header[1]), float(header[2]), int(header[3]), float(header[4])
		vGrid, xsValues = data[iLow:iLow+nData,0], data[iLow:iLow+nData,1]
		mask = xsValues>xsMin
		if isinstance(xLimits, Interval):  mask = np.logical_and (xLimits.member(vGrid), mask)
		if verbose and not is_uniform(vGrid):  print('WARNING - read_hitran_cia:  wavenumber grid is not uniform!')
		if sum(mask)>0:
			xsList.append({'pair': pair, 'vMin': vMin,  'vMax': vMax, 'temperature': temp,
		                       'vGrid': vGrid[mask], 'yData': xsValues[mask]})
			if verbose:  print (ciaFile, pair, temp, vMin, vMax, min(xsValues[mask]), max(xsValues[mask]))
		else:
			if verbose:  print ("WARNING --- read_hitran_cia:  CIA in %s for %f -- %f ignored" %
			                    (os.path.split(ciaFile)[1], vMin, vMax))
		iLow += nData
	return xsList


####################################################################################################################################

def ciaPlot (xsData, **kwArgs):
	""" Plot CIA cross section(s) vs. wavenumbers. """
	if isinstance(xsData,(list,tuple)):
		for xs in xsData:  ciaPlot(xs)
	elif isinstance(xsData, dict):
		if 'label' in kwArgs:  labelText=kwArgs.pop('label')
		else:                  labelText='%s-%s %.1fK' % (xsData['pair'][0], xsData['pair'][1], xsData['temperature'])
		semilogy (xsData['vGrid'], xsData['yData'], label=labelText, **kwArgs)
	else:
		raise ValueError ("ERROR --- cia:  invalid data type for ciaPlot, expected an dictionary")
	legend()
	xlabel(r"Wavenumber $\nu$ [cm$^{-1}$]")
	ylabel("CIA cross section")


####################################################################################################################################

def add_cia2absCo (absCo, ciaData, xsMin=0.0, verbose=False):
	"""
	Add collision indiced absorption 'cross section(s)' to molecular (line-by-line) absorption cross section.

	ARGUMENTS:
	absCo     absorption coefficients, either a single acArray or a list thereof
	ciaData   collision induced absorption data, either a single file or a list thereof (* wildcard supported!)
	xsMin     minimum value of cia data to accept, default 0.0 (i.e. ignore negative values)
	"""

	# a single CIA file or a list?
	if isinstance(ciaData, str):
		if '*' in ciaData:
			for cf in glob(ciaData):
				absCo = add_cia2absCo (absCo, cf, xsMin, verbose)  # call recursively
			return absCo
		else:
			cDir, cFile = os.path.split(ciaData)
	elif isinstance(ciaData, (list, tuple)):
		for cf in ciaData:
			absCo = add_cia2absCo (absCo, cf, xsMin, verbose)  # call recursively
		return absCo
	else:
		raise ValueError ("ERROR --- cia:  ciaData neither a single file nor a list thereof")

	# check input absorption coefficient(s), wavenumbers
	if isinstance(absCo, (list,tuple)):
		# check consistency of wavenumber intervals
		vLimits = absCo[0].x
		if not all([absCo[0].x==ac.x for ac in absCo[1:]]):
			raise ValueError ("ERROR --- cia:  list of absorption coefficients with inconsistent wavenumbers!")
	elif isinstance(absCo, acArray):
		vLimits = absCo.x
		absCo = [absCo]
	else:
		raise ValueError ("ERROR --- add_cia2absCo:  neither a single absorption coefficient (acArray) nor a list thereof")

	# read all CIA cross sections
	cxsList   = read_hitran_cia(ciaData, vLimits, xsMin, verbose)
	if len(cxsList)>0:
		cxsTemp = [cxs['temperature'] for cxs in cxsList]
		print ("INFO --- add_cia2ac:  %i cross sections found in %s with %.1f <T< %1.fK" %
		       (len(cxsList), ciaData, min(cxsTemp), max(cxsTemp)))
	else:
		print ("WARNING --- add_cia2abs:  no cia cross sections in %s for wavenumber %s" % (cFile, vLimits))
		return absCo

	# loop thru atmosphere
	acList = []
	for l,ac in enumerate(absCo):
		if verbose:  ac.info()
		# locate the next/best CIA cross sections w.r.t. temperature
		nxt = np.argmin(np.array([abs(cxs['temperature']-ac.t) for cxs in cxsList]))
		thisMolec, otherMolec = cxsList[nxt]['pair']
		# special cases
		if thisMolec=='N2' and otherMolec=='N2' and not 'N2' in ac.molec.keys():  # Assume Earth!
			ciaAC = (0.78*ac.molec['air'])**2 * cxsList[nxt]['yData']
		elif thisMolec=='N2' and not 'N2' in ac.molec.keys():  # Assume Earth!
			ciaAC = 0.78*ac.molec['air']*ac.molec[otherMolec] * cxsList[nxt]['yData']
		elif otherMolec=='N2' and not 'N2' in ac.molec.keys():  # Assume Earth!
			ciaAC = 0.78*ac.molec['air']*ac.molec[thisMolec] * cxsList[nxt]['yData']
		else:
			ciaAC = ac.molec[thisMolec]*ac.molec[otherMolec] * cxsList[nxt]['yData']
		# interpolate to monochromatic grid and add to lbl absorption coefficient
		acx   = ac + np.interp(ac.grid(), cxsList[nxt]['vGrid'], ciaAC)
		acList.append( acArray(acx, ac.x, ac.z, ac.p, ac.t, ac.molec) )
		if verbose:
			print ('adding cia #%i %.1fK ===>' % (nxt,  cxsList[nxt]['temperature']));  absCo[l].info();  print ()

	# test not yet perfect, on entry there could be a list with just one element!
	if len(acList)>1:  return acList
	else:              return acList[0]
