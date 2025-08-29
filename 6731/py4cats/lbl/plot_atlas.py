#!/usr/bin/env python3

"""
  plot_atlas
  Plot line position vs strength in 'atlas' style.

  usage:
  plot_atlas [options] files

  -h          help
  -c char     comment character(s) used in files (default '#', several characters allowed)
  -I          vertical impulses (bars) to indicate line strength (default: use '+', only for xmgr)
  -g int      show one molecule per graph (default: plot all lines of all molecules in one graph)
              (max.) number of graphs per page with one molecule per graph
  -o          output (print) file (default: interactive plot)
  -t string   title
  -v          verbose

  NOTE:
  The original implementation used the xmgr plotting tool (a.k.a. ACE/gr, further developed into the GRACE tool).
  If you prefer the new xmgrace, use the -v option (verbose) to print the command to the screen and replace xmgr -> xmgrace.
  (Unfortunately the option syntax is slightly different, but the "all in one" plot should work as before.)

  For a discussion of xmgr vs grace see
  https://github.com/mlund/xmgr-resurrection

  Currently only wavenumber vs strength is implemented, see also the atlas function in the lines module.
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
from subprocess import call
from glob import glob

verbose=0

####################################################################################################################################

def aceGr_atlas (lineFiles, printFile='', nGraphs=0, impulse=False, title=''):
	""" Plot line parameters (position vs strength) using ACE/gr = xmgr.

	    lineFiles    line parameter data file(s)
	    printFile    postscript file for hardcopy output (default none)
	    nGraphs      one graph per molecule and <= nGraphs per "page" (default 0, i.e. one plot for all)
	    impulse      use vertical bars instead of + markers
	    title        header line
	"""

	# show lines with simple plus or with vertical impulse
	if impulse:  symbol=12
	else:        symbol= 9

	if   isinstance(lineFiles,str):
		if '*' in lineFiles:  lineFiles = glob(lineFiles)
		else:                 lineFiles = lineFiles.split()
	elif isinstance(lineFiles,(list,tuple)):  lineFiles = list(lineFiles)
	else:  raise SystemExit ("ERROR aceGr_atlas:  invalid input data, expected (list of) line parameter data file(s)")

	# postscript file to be produced
	if printFile:
		aceGr =  "xmgr -log y -hardcopy -printfile " + printFile
		if printFile.endswith('eps'): aceGr =  aceGr + ' -eps'
		print('plotting in batch mode, postscript file: ', printFile)
	else:
		aceGr = "xmgr -log y"
	# title, logarithmic scale, x and y axis titles
	aceGr = aceGr + r""" -pexec 'xaxis label "position [cm\S-1\N]"' """
	if nGraphs==0:
		if title:    aceGr = aceGr + """ -pexec 'title "%s"'""" % title
		# one plot/graph for all molecules
		aceGr += " -legend load"
		aceGr += r""" -legend load -pexec 'yaxis label "strength \7S\4 [cm\S-1\N / (molec.cm\S-2\N)]"' """
		for nFile,lFile in enumerate(lineFiles):
			if not os.path.isfile(lFile):
				raise SystemExit ('ERROR aceGr_atlas:  %i. file "%s" not found' % (nFile, lFile))
			aceGr += ' -pexec "s%i linestyle 0" -pexec "s%i symbol %i" %s' % (nFile, nFile, symbol, lFile)
		if verbose: print(aceGr)
		childProcess(aceGr)
	else:
		numFiles = len(lineFiles)
		if nGraphs>numFiles or nGraphs<0: nGraphs = numFiles
		lineFiles = lineFiles[::-1]
		for i in range(int(len(lineFiles)/nGraphs)+1):
			Files = lineFiles[nGraphs*i:nGraphs*(i+1)]
			print(Files, [os.path.splitext(file)[0] for file in Files])
			nFiles = len(Files)
			graphs = ' -rows ' + str(nFiles)
			if nFiles>10: graphs += ' -maxgraph %i' % nFiles
			elif nFiles<1: break
			set   = ' -pexec "s0 linestyle 0" -pexec "s0 symbol %i"  -pexec "s0 color 2" ' % symbol
			for nFile,lFile in enumerate(Files):
				if not os.path.isfile(lFile):
					raise SystemExit ('ERROR aceGr_atlas:  %i. file "%s" not found' % (nFile, lFile))
				set += """ -pexec 'subtitle "%s"'""" % os.path.splitext(lFile)[0]
				if nFile>0:
					set += """ -pexec 'xaxis ticklabel off' """
				if nFile==nFiles-1:
					set += r""" -pexec 'yaxis label "\7S\4 [cm\S-1\N / (molec.cm\S-2\N)]"' """
					if title: set += """ -pexec 'title "%s"'""" % title
				graphs += " -graph %i -log y  %s %s" % (nFile, set, lFile)
			grCommand = aceGr+graphs
			if printFile and nGraphs>1:
				he = os.path.splitext(printFile)
				grCommand = grCommand.replace(printFile, '%s.%i%s' % (he[0], i, he[1]))
			if verbose: print('\n', grCommand)
			childProcess(grCommand)


####################################################################################################################################

def childProcess (job):
	""" System call along with exception handling. """
	#os.system(aceGr)
	#print (job)
	try:
		returnCode = call (job, shell=True)
		if returnCode!=0:  print(job.split()[0], "--- child was terminated by signal", returnCode, file=sys.stderr)
	except OSError as e:
		print("Execution failed:", e, file=sys.stderr)


####################################################################################################################################

if __name__ == "__main__":
	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

	from py4cats.aux.aeiou import commonExtension
	from py4cats.aux.command_parser import parse_command, standardOptions

	opts = standardOptions + [
	       {'ID': 'v'},
	       {'ID': 'I'},
	       {'ID': 'g', 'name': 'nGraphs', 'type': int, 'default': 0},
	       {'ID': 't', 'name': 'title', 'type': str}
	       ]

	dataFiles, options, commentChar, outFile = parse_command (opts,(1,99))

	if 'h' in options:  raise SystemExit (__doc__ + "\n End of plot_atlas help")

	verbose = 'v' in options
	impulse = 'I' in options

	if outFile and os.path.splitext(outFile)[1]==commonExtension(dataFiles):
		raise SystemExit ("output/plot file has same extension as linefiles!\n(probably you forgot to specify plot file)")

	aceGr_atlas (dataFiles, outFile, options.get('nGraphs',0), impulse, options.get('title'))
