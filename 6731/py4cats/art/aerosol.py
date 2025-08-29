"""
Aerosol Extinction

F. Yan et al., High-resolution transmission spectrum of the Earth's atmosphere, Eq. (3), IJAB 2015
deLeeuw et al. Chap. 6 "Retrieval of Aerosol Properties" in Burrows et al. "Remote Sensing of Tropospheric Composition from Space"

see also

Kaltenegger&Traub, Eq.(2), ApJ 2009  --->  Allen: Astrophysical Quantities (1976)
B. Toon and JB Pollack, J. Appl. Met. 15,225 (1976)
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

def aerosol_od (vGrid, airColumn, factor=1.0, ex=1.3):
	""" Aerosol optical depth --- Angstroem relation
	    tau = 8.85e-33 * N * v**1.3   with N=air column density [molec/cm-2]

	    vGrid:      wavenumber [cm**-1]
	    airColumn:  air column
	    factor:     default 1.0, can be used to scale (increase/decrease) the default amplitude 8.85e-33
	    ex:         Angstrom exponent, default 1.3
	"""

	if isinstance(factor,(int,float)) and factor>0.0:
		amplitude = factor*8.85e-33
	else:
		raise ValueError ("aerosol_od:  expected a positive float or integer for scaling the aerosol amplitude 8.85e-33")

	tau = amplitude * airColumn * vGrid**ex

	return tau
