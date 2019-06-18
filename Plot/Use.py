
def Use( which, tf=True ) : 
	'''
	which:
		(1) which='Agg'
		(2) which='tex', tf=True
		(3) which is LatexPackageName
	'''
	import matplotlib as mpl
	which = str(which).lower()
	if (which == 'agg') : mpl.use('Agg')
	elif (which == 'tex') : 
		# Some systems, 
		# 	set usetex=True , use tex
		# 	set usetex=False, don't use tex
		# But also some systems,
		# 	set usetex=True , raise error
		# 	set usetex=False, use tex!
		# Therefore, set True/False dependenting on your system
		from jizhipy.Basic import Path
		import os
		texcache=Path.AbsPath('~/.cache/matplotlib/tex.cache')
		if (Path.ExistsPath(texcache)) : 
			os.system('rm -r '+texcache)
		mpl.rcParams['text.usetex'] = bool(tf)
	#	import matplotlib.pyplot as plt
	#	plt.rc('text', usetex=True)
	else :  # which is packagename
		mpl.rcParams['text.latex.preamble'] = [r'\usepackage{'+which+r'}']

