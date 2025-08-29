#!/usr/bin/env python3

""" geisa

    usage:
    geisa [options] line_parameter_database

    command line options without arguments:
        -h  help

    optional command line options with string argument:
	-i  isotope number
	-m  molecule number
        -o  output file (default: standard output)

    optional command line options with float argument:
	-S  minimum line strength to accept
	-x  lower and upper end of wavenumber range (comma separated pair without blanks)

    Note: at least wavenumber range or molecule has to be selected!
"""

####################################################################################################################################
##########     LICENSE issues:                                                                                            ##########
##########                       This file is part of the Py4CAtS package.                                                ##########
##########                       Copyright 2002 - 2021; Franz Schreier;  DLR-IMF Oberpfaffenhofen                         ##########
##########                       Py4CAtS is distributed under the terms of the GNU General Public License;                ##########
##########                       see the file ../license.txt in the parent directory.                                     ##########
####################################################################################################################################

gIASI2003_fieldLength = list(map(int,'12 11 6 10 9 9 9 9 4 3 3 1 2 1 10 5 8 3 6 6 10 11 6 4 8 6 5 4 4 8 8 4 4'.split()))  # sum 209
geisa2003_fieldLength = list(map(int,'12 11 6 10 9 9 9 9 4 3 3 3 2 1 10 5 8 3 6 6 10 11 6 4 8 6 5 4 4 8 8 4 4'.split()))  # sum 211
geisa2008_fieldLength = list(map(int,'12 11 6 10 25 25 15 15 4 3 3 3 2 1 10 7 9 6 10 11 6 4 9 6 7 4 4 8 8 4 4'.split()))  # sum 252

# for GEISA format issues see also http://eodg.atm.ox.ac.uk/RFM/geihit.html

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py4cats.lbl.hitran import bisect_first_line
from py4cats.aux.pairTypes import Interval, PairOfInts

####################################################################################################################################

def extract_geisa (gFile, xLimits=None, molNr=0, isoCode='', strMin=0.0):
	""" Read Geisa formatted database.

	    gFile:      filename of GEISA data base
	    xLimits:    lower and upper wavenumber bounds
	    molNr:      molecular ID number as used by GEISA
	                NOTE: identical to HITRAN numbers only for the first twelve molecules!
	    isoCode:    three digits identification code (approx. similar to HITRAN)
	    strMin:     minimal line strength to accept (reject weaker lines)
	"""
	try:
		geisa  = open (gFile, 'rb')  # open in binary mode, otherwise backward-seek in bisection fails!
	except IOError:
		raise SystemExit ('ERROR --- extract_geisa:  opening GEISA data file "' + gFile + '" failed!')
	# wavenumber interval to be searched
	if isinstance(xLimits,Interval):  xLow, xHigh = xLimits.limits()
	else:                             xLow, xHigh = 0.0, 99999.9

	# initialize time and search first useful line
	if molNr<=0:
		if isoCode or strMin>0:
			raise SystemExit ('ERROR --- extract_geisa:  no isotope or linestrength selection without molecule specification!')
		lines = extract_range (geisa, xLow, xHigh)
	elif not isoCode and strMin<=0.0:
		lines = extract_Mol (geisa, xLow, xHigh, molNr)
	elif not isoCode and strMin>0.0:
		lines = extract_MolStr (geisa, xLow, xHigh, molNr, strMin)
	elif strMin<=0.0:
		lines = extract_MolIso (geisa, xLow, xHigh, molNr, isoCode)
	else:
		lines = extract_MolIsoStr (geisa, xLow, xHigh, molNr, isoCode, strMin)

	# close file and return
	geisa.close()
	return lines


####################################################################################################################################

def get_geisa_fields (fileName, what=None):
	""" Use filename to determine position (first and last indices) of wavenumber, moleculeID, isotopeID, etc.
	    If 'what' is given, return indices only for this field (e.g. moleculeID, currently no selfWidth) """

	# NOTE:  GEISA has two 'fields' for the isotope identification
	#        the G-field has three digits, e.g. '161' and '162' for 'main' H2O and HDO
	#        the L-field is a single digit for the isotope abundance number as used by Hitran, e.g. 4 for HDO

	beginFreq = 0
	if '97' in fileName:
		endFreq = 10
		beginStr, endStr = 10, 20
		beginAir, endAir = 20, 25
		beginEne, endEne = 25, 35
		beginIso, endIso = 75, 79
		beginMol, endMol = 79, 82
		beginTEx, endTEx = 72, 75
		beginSlf, endSlf = 98,103
		beginShft,endShft= -1, -1  # undefined
	elif '2003' in fileName:
		endFreq=12
		beginStr, endStr = 12, 23
		beginAir, endAir = 23, 29
		beginEne, endEne = 29, 39
		beginTEx, endTEx = 75, 79
		beginIso, endIso = 79, 82
		beginMol, endMol = 82, 85
		beginSlf, endSlf =101,106
		beginShft,endShft= -1, -1  # undefined
	elif '08' in fileName or '09' in fileName or '11' in fileName or '15' in fileName or '19' in fileName or '20' in fileName:
		endFreq=12                 # A
		beginStr, endStr = 12, 23  # B
		beginAir, endAir = 23, 29  # C
		beginEne, endEne = 29, 39  # D
		beginTEx, endTEx =119,123  # F
		beginIso, endIso =123,126  # G  # geisa coding, 3 digits, e.g. "162" for HDO
		#eginIN,  endIN  =134,135  # L  # geisa isotope number according to abundance (similar to Hitran)
		beginMol, endMol =126,129  # I
		beginSlf, endSlf =145,152  # N
		beginShft,endShft=152,161  # O
	elif 'iasi' in fileName.lower():
		endFreq=12
		beginStr, endStr = 12, 23
		beginAir, endAir = 23, 29
		beginEne, endEne = 29, 39
		beginTEx, endTEx = 75, 79
		beginIso, endIso = 79, 82
		beginMol, endMol = 82, 85
		beginSlf, endSlf = 99,105
		beginShft,endShft= -1, -1  # undefined
	else:
		raise SystemExit ("ERROR --- get_geisa_fields:  "+repr(fileName)+'\nGEISA database --- unknown version!?!' +
		                                 '\n(Cannot find release year and/or "IASI" in filename to identify proper format)')

	fields= {'iw': beginFreq, 'lw': endFreq, 'iM': beginMol, 'lM': endMol, 'iI': beginIso, 'lI': endIso,
	         'iS': beginStr,  'lS': endStr,  'iE': beginEne, 'lE': endEne, 'iD': beginShft, 'lD': endShft,
	         'iA': beginAir,  'lA': endAir, 'isw': beginSlf, 'lsw': endSlf, 'iT': beginTEx, 'lT': endTEx}

	if isinstance(what,str) and what[0] in 'wMISEAT':  return PairOfInts(fields['i'+what[0]],fields['l'+what[0]])
	else:                                              return fields

####################################################################################################################################

def extract_range (geisa, xLow, xHigh):
	""" Read all lines up to a upper wavenumber limit from Geisa formatted database. """
	# determine position (first and last indices) of wavenmber, moleculeID, isotopeID
	fields = get_geisa_fields (geisa.name)
	iw, lw, lM = fields['iw'], fields['lw'], fields['lM']
	# proceed to first requested line
	record = bisect_first_line (geisa, xLow, xHigh, iw, lw)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		wvn = float(record[:lw])
		if wvn<=xHigh:  lines.append(record)
		else:           break
		# read next record
		record = geisa.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:lM])
	if record:        print('# first line not accepted \n', record[:lM])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_Mol (geisa, xLow, xHigh, getMol):
	""" Read lines of a given molecule up to a upper wavenumber limit from Geisa formatted database. """
	# determine position (first and last indices) of wavenmber, moleculeID, isotopeID
	fields = get_geisa_fields (geisa.name)
	iw, lw, iM, lM = fields['iw'], fields['lw'], fields['iM'], fields['lM']
	# proceed to first requested line
	print ('bisect_first_line', xLow, xHigh, iw, lw)
	record = bisect_first_line (geisa, xLow, xHigh, iw, lw)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		wvn = float(record[:lw])
		if wvn>xHigh: break
		mol = int(record[iM:lM])
		if mol==getMol: lines.append(record)
		# read next record
		record = geisa.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:lM])
	if record:        print('# first line not accepted \n', record[:lM])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_MolIso (geisa, xLow, xHigh, getMol, getIso):
	""" Read lines of a given molecule/isotope up to a upper wavenumber limit from Geisa formatted database. """
	# determine position (first and last indices) of wavenmber, moleculeID, isotopeID
	fields = get_geisa_fields (geisa.name)
	iw, lw, iM, lM, iI, lI = fields['iw'], fields['lw'], fields['iM'], fields['lM'], fields['iI'], fields['lI']
	# proceed to first requested line
	record = bisect_first_line (geisa, xLow, xHigh, iw, lw)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		wvn = float(record[:lw])
		if wvn>xHigh: break
		mol = int(record[iM:lM])
		iso = int(record[iI:lI])
		if mol==getMol and iso==getIso: lines.append(record)
		# read next record
		record = geisa.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:lM])
	if record:        print('# first line not accepted \n', record[:lM])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_MolStr (geisa, xLow, xHigh, getMol, strMin):
	""" Read strong lines of a given molecule up to a upper wavenumber limit from Geisa formatted database. """
	# determine position (first and last indices) of wavenmber, moleculeID, isotopeID
	fields = get_geisa_fields (geisa.name)
	iw, lw, iM, lM, iS, lS = fields['iw'], fields['lw'], fields['iM'], fields['lM'], fields['iS'], fields['lS']
	# proceed to first requested line
	record = bisect_first_line (geisa, xLow, xHigh, iw, lw)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		wvn = float(record[:lw])
		if wvn>xHigh: break
		mol = int(record[iM:lM])
		Str = float(record[iS:lS].replace('D','e'))
		if mol==getMol and Str>=strMin: lines.append(record)
		# read next record
		record = geisa.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:lM])
	if record:        print('# first line not accepted \n', record[:lM])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_MolIsoStr (geisa, xLow, xHigh, getMol, getIso, strMin):
	""" Read strong lines of a given molecule/isotope up to a upper wavenumber limit from Geisa formatted database. """
	# determine position (first and last indices) of wavenmber, moleculeID, isotopeID
	fields = get_geisa_fields (geisa.name)
	iw, lw, iM, lM, iI, lI, iS, lS = fields['iw'], fields['lw'], fields['iM'], fields['lM'], fields['iI'], fields['lI'], fields['iS'], fields['lS']
	# proceed to first requested line
	record = bisect_first_line (geisa, xLow, xHigh, iw,lw)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		wvn = float(record[:lw])
		if wvn>xHigh: break
		mol = int(record[iM:lM])
		iso = int(record[iI:lI])
		Str = float(record[iS:lS].replace('D','e'))
		if mol==getMol and iso==getIso and Str>=strMin: lines.append(record)
		# read next record
		record = geisa.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:lM])
	if record:        print('# first line not accepted \n', record[:lM])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

if __name__ == "__main__":

	from py4cats.aux.aeiou import open_outFile
	from py4cats.aux.command_parser import parse_command, standardOptions

	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       dict(ID='i', name='isoCode',   type=int,   default=0),
               dict(ID='m', name='molNr',   type=int,   default=0),
               dict(ID='S', name='strMin',  type=float, default=0.0, constraint='strMin>0'),
	       dict(ID='x', name='xLimits', type=Interval,  default=Interval(0.0,99999.9), constraint='xLimits.lower>=0.0')
               ]

	files, options, commentChar, outFile = parse_command (opts, 1)

	for opt in opts:
		if 'name' in opt and 'type' in opt: exec(opt['name'] + ' = ' + repr(options.get(opt['name'])))

	if 'h' in options:
		raise SystemExit (__doc__ + "\n End of geisa help")
	elif options['molNr'] or options['xLimits'].size()<50000.0:
		# Read lines from GEISA line parameter file
		lines = extract_geisa (files[0], **options)

		if lines:
			# open an output file if explicitely specified (otherwise use standard output)
			out = open_outFile (outFile, commentChar='#')
			# print selected lines  (decode to transform bytes to str, otherwise open file with binary mode: 'wb')
			for line in lines:  out.write (line.decode())
			# close the output file (if its not stdout)
			if outFile: out.close()
	else:
		# at least molecule or wavenumber range needed
		raise SystemExit ('ERROR --- geisa:  neither molecule nor wavenumber range specified!')
