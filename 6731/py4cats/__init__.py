"""
Py4CAtS --- Python for Computational ATmospheric Spectroscopy

A collection of Python scripts for (molecular) line-by-line absorption
(cross sections, optical depths, ...) and atmospheric radiative transfer

https://doi.org/10.3390/atmos10050262    --->   Atmosphere 10(5), 262, 2019
https://atmos.eoc.dlr.de/tools/Py4CAtS/
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file  ./license.txt in the main directory.                                       ##########
####################################################################################################################################

print (__doc__)

__version__ = 'aug21'

#try:               print ('__path__', __path__, '\n__file__', __file__)
#except NameError:  print ('__path__ and/or __file__ undefined')
#else:              print ('py4cats  __version__', __version__, __file__)

from . import aux, lbl, art

##########  auxiliary  modules   ##########

from . aux.aeiou import *
from . aux.ir import *
from . aux.misc import approx, monotone, regrid, runningAverage, trapez, xTruncate, show_lambda

from . aux.cgsUnits  import cgs, lambda2nu, nu2lambda
from . aux.euGrid import euGrid, parseGridSpec
from . aux.moreFun import sindg, cosdg
from . aux.pairTypes import Interval, PairOfInts, PairOfFloats
from . aux.struc_array import *

from . aux.racef import *
from . aux.srf import srf, fts
from . aux.convolution import convolveBox, convolveTriangle, convolveGauss
from . aux.radiance2Kelvin import radiance2Kelvin
from . aux.radiance2radiance import radiance2radiance

########## line-by-line modules ##########

from . lbl.lbl2xs import lbl2xs
from . lbl.lbl2ac import lbl2ac
from . lbl.lbl2od import lbl2od

from . lbl.higstract import higstract,  save_lines_core, save_lines_orig
from . lbl.hitran import extract_hitran
from . lbl.geisa import extract_geisa

from . lbl.lines import atlas, llInfo, read_line_file, delete_traceGasLines, xVoigt_parameters, meanLineWidths

##########  subclassed numpy arrays   ##########

from . art.absCo import acArray, acPlot, acInfo, acSave, acRead
from . art.oDepth import *
from . art.radInt import riArray, riSave, riPlot, riRead, riInfo, riConvolve
from . art.wgtFct import wfPlot, wfSave, wfRead, wfPeakHeight
from . art.xSection import xsArray, xsRead, xsSave, xsPlot, xsInfo

##########  atmos radiative transfer  ##########

from . art.atmos1D import *
from . art.xs2ac import xs2ac
from . art.xs2od import xs2dod
from . art.ac2od import ac2dod
from . art.od2ri import dod2ri
from . art.ac2wf import *
from . art.limb import *
from . art.planck import planck
from . art.rayleigh import *
from . art.aerosol import aerosol_od
