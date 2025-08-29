""" Convolution of monochromatic spectrum/spectra with Box, Triangle, Gaussian spectral response functions. """

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

from .. aux.misc import trapez
from .. aux.srf import Gauss


####################################################################################################################################
##### numpy/scipy convolve vs. 'handmade' multiplication and summation:                                                        #####
#####                                                                                                                          #####
##### numpy.convolve possibly faster for large arrays                                                                          #####
##### scipy.signal.convolve  automatically chooses fft or the direct method based on an estimation of which is faster.         #####
##### scipy.signal.oaconvolve generally much faster than convolve for large arrays (n > ~500),                                 #####
#####                         and generally much faster than fftconvolve when one array is much larger than the other          #####
#####                                                                                                                          #####
##### BUT:  all these numpy/scipy convolutions return convolved spectrum at monochromatic grid !!!                             #####
####################################################################################################################################

def convolveBox (vGrid, yValues, hwhm=1.0, what='i', sample=1.0, wGrid=None):
	""" Convolve spectra with a Box (rectangular) spectral response function.

	    ARGUMENTS:
	    ----------
	    vGrid:     the wavenumber grid of the monochromatic (high resolution) spectra
	    yValues:   a rank-1 spectrum or rank-2 array (matrix) of spectra
	    hwhm:      half width of box
	    what:      type of spectra:  'i' or 'r' radiance or eff.height (default)
	                                 't'        transmission
					 'o'        optical depth (see note)
	    sample:    number of new grid points per hwhm (default 1.0)
	    wGrid:     new wavenumber grid (default None ==> set automatically;  when given, do NOT return it again)

	    RETURNS:
	    --------
	    wGrid:      new wavenumber grid  (only if not given as input, i.e. set automatically!)
	    ySmoothed:  the spectrum/spectra convolved with a Gaussian with len(wGrid) spectral points.

	    NOTE:
	    -----
	    * In case of transmission, the convolution uses absorption = 1-transmission
	    * In case of optical depth, this is transformed to absorption = 1 - transmission = 1-exp(-od)
	      before the convolution and transformed back afterwards
	"""

	if not (isinstance(hwhm,(int,float)) and hwhm>0.0):
		raise SystemExit ('ERROR --- convolveBox:  hwhm is not a positive float!')

	if   isinstance(wGrid,(int,float)):  # return convolved spectrum for a single wavenumber
		if not vGrid[0]+hwhm<=wGrid<=vGrid[-1]-hwhm:
			raise SystemExit ("ERROR --- convolveBox:  new wavenumber point outside vGrid")
		wGrid=np.array([float(wGrid)])
		sample=0
	elif isinstance(wGrid,(np.ndarray,list,tuple)):
		if isinstance(wGrid,(list,tuple)):  wGrid = np.array(wGrid)
		if not vGrid[0]+hwhm<min(wGrid)<vGrid[-1]-hwhm \
		   and vGrid[0]+hwhm<max(wGrid)<vGrid[-1]-hwhm:
			raise SystemExit ("ERROR --- convolveBox:  new wavenumber points outside vGrid")
		sample=0
	else:
		if sample<=0.0:
			raise SystemExit ("ERROR --- convolveBox:  sample non-positive!")
		wLow   = vGrid[0] +hwhm
		wHigh  = vGrid[-1]-hwhm
		deltaW = hwhm/sample
		wGrid  = np.linspace(wLow, wHigh, int((wHigh-wLow)/deltaW)+1)

	if   what in 'tT':  yData = 1.0-yValues           # transform to absorption
	elif what in 'oO':  yData = 1.0-np.exp(-yValues)  # transform to 1-transmission
	else:               yData = yValues

	recFullWidth = 0.5/hwhm

	if len(yData.shape)==1:
		# allocate smoothed spectra
		yBoxed = np.zeros(len(wGrid))
		# and loop over Gaussians
		for i,ww in enumerate(wGrid):
			vLeft  = ww-hwhm
			vRight = ww+hwhm
			yBoxed[i] = trapez (vGrid, yData, vLeft, vRight) * recFullWidth
	elif len(yData.shape)==2:
		yBoxed = np.zeros((len(wGrid),yData.shape[1]))
		# and loop over Gaussians
		for i,ww in enumerate(wGrid):
			vLeft  = ww-hwhm
			vRight = ww+hwhm
			for j in range(yData.shape[1]):
				yBoxed[i,j] = trapez (vGrid, yData[:,j], vLeft, vRight) * recFullWidth
	else:
		raise SystemExit ('ERROR --- convolveBox:  unknown shape/type of spectral data,\n',
		                  '                        need rank 1 (spectrum) or rank 2 array (spectra)')

	if   what in 'tT':  yBoxed = 1.0-yBoxed           # transform back to transmission
	elif what in 'oO':  yBoxed = -np.log(1.0-yBoxed)  # transform (1-Trans) back to optical depth
	else:               pass

	if sample>0:  return wGrid, yBoxed
	else:         return        yBoxed


####################################################################################################################################

def convolveTriangle (vGrid, yValues, hwhm=1.0, what='i', sample=2.0, wGrid=None):
	""" Convolve spectra with a triangular spectral response function.

	    ARGUMENTS:
	    ----------
	    vGrid:     the wavenumber grid of the monochromatic (high resolution) spectra
	    yValues:   a rank-1 spectrum or rank-2 array (matrix) of spectra
	    hwhm:      half width of triangle
	    what:      type of spectra:  'i' or 'r' radiance or eff.height (default)
	                                 't'        transmission
					 'o'        optical depth (see note)
	    sample:    number of new grid points per hwhm (default 2.0)
	    wGrid:     new wavenumber grid (default None ==> set automatically;  when given, do NOT return it again)

	    RETURNS:
	    --------
	    wGrid:      new wavenumber grid  (only if not given as input, i.e. set automatically!)
	    ySmoothed:  the spectrum/spectra convolved with a Gaussian with len(wGrid) spectral points.

	    NOTE:
	    -----
	    * In case of transmission, the convolution uses absorption = 1-transmission
	    * In case of optical depth, this is transformed to absorption = 1 - transmission = 1-exp(-od)
	      before the convolution and transformed back afterwards
	"""

	if not (isinstance(hwhm,(int,float)) and hwhm>0.0):
		raise SystemExit ('ERROR --- convolveTriangle:  hwhm is not a positive float!')

	if   isinstance(wGrid,(int,float)):  # return convolved spectrum for a single wavenumber
		if not vGrid[0]+hwhm<=wGrid<=vGrid[-1]-hwhm:
			raise SystemExit ("ERROR --- convolveTriangle:  new wavenumber point outside vGrid")
		wGrid=np.array([float(wGrid)])
		sample=0
	elif isinstance(wGrid,(np.ndarray,list,tuple)):
		if isinstance(wGrid,(list,tuple)):  wGrid = np.array(wGrid)
		if not vGrid[0]+2*hwhm<min(wGrid)<vGrid[-1]-2*hwhm \
		   and vGrid[0]+2*hwhm<max(wGrid)<vGrid[-1]-2*hwhm:
			raise SystemExit ("ERROR --- convolveTriangle:  new wavenumber points outside vGrid")
		sample=0
	else:
		if sample<=0.0:
			raise SystemExit ("ERROR --- convolveTriangle:  sample non-positive!")
		wLow   = vGrid[0] +2*hwhm
		wHigh  = vGrid[-1]-2*hwhm
		deltaW = hwhm/sample
		wGrid  = np.linspace(wLow, wHigh, int((wHigh-wLow)/deltaW)+1)
		print ('convolveTriangle:',  vGrid[0], vGrid[-1], hwhm, ' ---> ', wLow, wHigh, deltaW, wGrid)

	if   what in 'tT':  yData = 1.0-yValues           # transform to absorption
	elif what in 'oO':  yData = 1.0-np.exp(-yValues)  # transform to 1-transmission
	else:               yData = yValues

	recFullWidth = 1./(2*hwhm)

	if len(yData.shape)==1:
		# allocate smoothed spectra
		yTriangled = np.zeros(len(wGrid))
		# and loop over Gaussians
		for i,ww in enumerate(wGrid):
			vLeft  = ww-2*hwhm
			vRight = ww+2*hwhm
			mask      = np.logical_and(vGrid>=vLeft, vGrid<=vRight)
			triangle  = np.where(mask, 1-0.5*abs(vGrid-ww)/hwhm, 0.0) * recFullWidth
			yTriangled[i] = trapez (vGrid, triangle*yData, vLeft, vRight)
	elif len(yData.shape)==2:
		yTriangled = np.zeros((len(wGrid),yData.shape[1]))
		# and loop over Gaussians
		for i,ww in enumerate(wGrid):
			vLeft  = ww-2*hwhm
			vRight = ww+2*hwhm
			mask      = np.logical_and(vGrid>=vLeft, vGrid<=vRight)
			triangle  = np.where(mask, 1-0.5*abs(vGrid-ww)/hwhm, 0.0) / (2.*hwhm)
			for j in range(yData.shape[1]):
				yTriangled[i,j] = trapez (vGrid, triangle*yData[:,j], vLeft, vRight)
	else:
		raise SystemExit ('ERROR --- convolveTriangle:  unknown shape/type of spectral data,\n',
		                  '                             need rank 1 (spectrum) or rank 2 array (spectra)')

	if   what in 'tT':  yTriangled = 1.0-yTriangled           # transform back to transmission
	elif what in 'oO':  yTriangled = -np.log(1.0-yTriangled)  # transform (1-Trans) back to optical depth
	else:               pass

	if sample>0:  return wGrid, yTriangled
	else:         return        yTriangled


####################################################################################################################################

def convolveGauss (vGrid, yValues, hwhm=1.0, what='i', nWidths=5.0, sample=4.0, wGrid=None):
	""" Convolve spectra with a Gaussian spectral response function.

	    ARGUMENTS:
	    ----------
	    vGrid:     the wavenumber grid of the monochromatic (high resolution) spectra
	    yValues:   a rank-1 spectrum or rank-2 array (matrix) of spectra
	    hwhm:      half width @ half maximum
	    what:      type of spectra:  'i' or 'r' radiance or eff.height (default)
	                                 't'        transmission
					 'o'        optical depth (see note)
	    nWidths:   left and right wing cutoff for Gauss response function in units of HWHM
	    sample:    number of new grid points per hwhm (default 4.0)
	    wGrid:     new wavenumber grid (default None ==> set automatically;  when given, do NOT return it again)

	    RETURNS:
	    --------
	    wGrid:      new wavenumber grid  (only if not given as input, i.e. set automatically!)
	    ySmoothed:  the spectrum/spectra convolved with a Gaussian with len(wGrid) spectral points.

	    NOTE:
	    -----
	    * In case of transmission, the convolution uses absorption = 1-transmission
	    * In case of optical depth, this is transformed to absorption = 1 - transmission = 1-exp(-od)
	      before the convolution and transformed back afterwards
	"""

	if not (isinstance(hwhm,(int,float)) and hwhm>0.0):
		raise SystemExit ('ERROR --- convolveGauss:  hwhm is not a positive float!')

	if   isinstance(wGrid,(int,float)):  # return convolved spectrum for a single wavenumber
		if not vGrid[0]+nWidths*hwhm<wGrid<vGrid[-1]-nWidths*hwhm:
			raise SystemExit ("ERROR --- convolveGauss:  new wavenumber points outside vGrid")
		wGrid=np.array(float(wGrid))
		sample=0
	elif isinstance(wGrid,(np.ndarray,list,tuple)):
		if isinstance(wGrid,(list,tuple)):  wGrid = np.array(wGrid)
		if not vGrid[0]+nWidths*hwhm<min(wGrid)<vGrid[-1]-nWidths*hwhm \
		   and vGrid[0]+nWidths*hwhm<max(wGrid)<vGrid[-1]-nWidths*hwhm:
			raise SystemExit ("ERROR --- convolveGauss:  new wavenumber points outside vGrid")
		sample=0
	else:
		if sample<=0.0 or nWidths<=0.0:
			raise SystemExit ("ERROR --- convolveGauss:  sample and/or nWidths non-positive!")
		wLow   = vGrid[0] +nWidths*hwhm+0.5*hwhm  # slightly shrink for security
		wHigh  = vGrid[-1]-nWidths*hwhm-0.5*hwhm
		deltaW = hwhm/sample
		wGrid  = np.linspace(wLow, wHigh, int((wHigh-wLow)/deltaW)+1)

	if   what in 'tT':  data = 1.0-yValues           # transform to absorption
	elif what in 'oO':  data = 1.0-np.exp(-yValues)  # transform to 1-transmission
	else:               data = yValues

	deltaV = np.ediff1d(vGrid)
	if max(deltaV)-min(deltaV)>0.01*deltaV[0]:
		raise SystemExit ("ERROR --- convolveGauss:  monochromatic wavenumber grid not equidistant")
	else:
		deltaV = deltaV.mean()  # simply replace, array not needed anymore
		recDelta = 1.0/deltaV

	# evaluate response function on dense grid with spacing as monochromatic grid
	nRight = int(nWidths*hwhm*recDelta)
	rGrid = vGrid[:nRight+1]-vGrid[0]                      # right side grid
	sGrid = np.concatenate((-np.flipud(rGrid[1:]),rGrid))  # symmetrically around 0
	srf   = Gauss(sGrid, hwhm)

	if len(yValues.shape)==1:
		# allocate smoothed spectra
		yGaussed = np.zeros(len(wGrid))
		# and loop over Gaussians
		for i,ww in enumerate(wGrid):
			iMid  = int(recDelta*(ww-vGrid[0]))  # index of center point in fine grid
			# multiply monochromatic spectrum with Gauss and integrate
			if iMid-nRight<0 or iMid+nRight+1>len(vGrid):
				print ("WARNING:", i, ww, iMid, vGrid[iMid], iMid-nRight, iMid+nRight+1)
			yGaussed[i] = np.trapz (data[iMid-nRight:iMid+nRight+1]*srf, dx=deltaV)
	elif len(yValues.shape)==2:
		yGaussed = np.zeros((len(wGrid),yValues.shape[1]))
		# and loop over Gaussians
		for i,ww in enumerate(wGrid):
			iMid  = int(recDelta*(ww-vGrid[0]))  # index of center point in fine grid
			for j in range(yValues.shape[1]):
				yGaussed[i,j] = np.trapz (data[iMid-nRight:iMid+nRight+1,j]*srf, dx=deltaV)
	else:
		raise SystemExit ('ERROR --- convolveGauss:  unknown shape/type of spectral data,\n',
		                  '                          need rank 1 (spectrum) or rank 2 array (spectra)')

	if   what in 'tT':  yGaussed = 1.0-yGaussed           # transform back to transmission
	elif what in 'oO':  yGaussed = -np.log(1.0-yGaussed)  # transform (1-Trans) back to optical depth
	else:               pass

	if sample>0:  return wGrid, yGaussed
	else:         return        yGaussed
