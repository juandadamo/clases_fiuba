""" Convolve spectrum/spectra with a spectral response function of fixed resolving power R.
    R = nu/delta_nu = lambda/delta_lambda = constant
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

import numpy as np

from .. aux.misc import trapez
from .. aux.srf import Gauss

####################################################################################################################################

def convolve_grating (vGrid, yValues, resolve=1000., what='i', vExt=5.0):
	""" Convolve spectra with a Gaussian spectral response function of fixed resolving power.

	    ARGUMENTS:
	    ----------
	    vGrid:     the wavenumber grid of the monochromatic (high resolution) spectra
	    yValues:   a rank-1 spectrum or rank-2 array (matrix) of spectra
	    resolve:   resolving power  R = nu/dNu = lambda/dLambda
	    vExt:      left and right wing cutoff for Gauss response function in units of width
	    what:      type of spectra:  'i' or 'r' radiance (default)
	                                 't'        transmission
					 'o'        optical depth (see note)
	    RETURNS:
	    --------
	    the spectrum/spectra convolved with a Gaussian response function

	    NOTE:
	    -----
	    * In case of transmission, the convolution uses absorption = 1-transmission
	    * In case of optical depth, this is transformed to 1-transmission = 1-exp(-od)
	      before the convolution and transformed back afterwards
	    * In contrast to the (old) gaussedSpec etc routines, the new grid is set automatically
	"""

	if   what in 'tT':  data = 1.0-yValues           # transform to absorption
	elif what in 'oO':  data = 1.0-np.exp(-yValues)  # transform to 1-transmission
	else:               data = yValues

	deltaV = np.ediff1d(vGrid)
	if max(deltaV)-min(deltaV)>0.01*deltaV[0]:
		raise SystemExit ("ERROR --- convolve_grating:  monochromatic wavenumber grid not equidistant")
	else:
		deltaV   = deltaV.mean()  # simply replace, array not needed anymore
		recDelta = 1.0/deltaV

	widthStart = vGrid[0]/resolve
	widthStop  = vGrid[-1]/resolve
	wGridStart = vGrid[0]  + widthStart*vExt
	wGridStop  = vGrid[-1] - widthStop*vExt
	deltaW     = widthStart/5.0
	print('convolve_grating:', widthStart, widthStop, wGridStart, wGridStop, deltaW)
	wGrid = np.arange(wGridStart, wGridStop+deltaW, deltaW)
	print('wGrid:', wGrid)

	if len(yValues.shape)==1:
		# allocate convolved spectrum
		yConvolved = np.zeros_like(wGrid)
		# and loop over new grid points
		for i,w in enumerate(wGrid):
			width = w/resolve
			# evaluate response function on dense grid with spacing as monochromatic grid
			sGrid = np.arange(-vExt*width,vExt*width+deltaV, deltaV)
			srf = Gauss(sGrid, width)
			# set first and last fine grid point
			iLow  = int(recDelta*(w+sGrid[0]-vGrid[0]))
			iHigh = int(recDelta*(w+sGrid[-1]-vGrid[0]))
			print(w, w+sGrid[0], w+sGrid[-1], iLow, iHigh, len(data[iLow:iHigh+1]), len(srf))
			# multiply monochromatic spectrum with Gauss and integrate
			yConvolved[i] = trapez (vGrid[iLow:iHigh+1], data[iLow:iHigh+1]*srf)
	elif len(yValues.shape)==2:
		# allocate convolved spectra
		yConvolved = np.zeros([len(wGrid),2])
		raise SystemExit ('ERROR --- convolve_grating:  2D spectra not yet supported')
	else:
		raise SystemExit ('ERROR --- convolve_grating:  ' + \
		                  'unknown shape/type of spectral data, need rank 1 (spectrum) or rank 2 array (spectra)')

	if   what in 'tT':  yConvolved = 1.0-yConvolved           # transform back to transmission
	elif what in 'oO':  yConvolved = -np.log(1.0-yConvolved)  # transform (1-Trans) back to optical depth
	else:               pass

	return yConvolved
