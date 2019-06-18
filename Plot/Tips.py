
def Tips() : 
	'''
	***** Press "q" to quit, "j" to down, "k" to up
	
	legend(loc=)
		2	 9	0/1
		6	10	5/7
		3	 8	 4

	if (self.verbose and jp.SysFrame(0,2)[-3][-2]=='') : 

	hp.graticule(dlat, dlon, coord=None, local=None, **kwds)
		dlat, dlon : degree
		coord : {'E', 'G', 'C'}
		local : bool
			If True, draw a local graticule (no rotation is performed, useful for a gnomonic view, for example)

	hp.mollview(rot=(lon, lat, psi in degree))


	***** Modify healpy
	(1) max number of hp.graticule():
	# xxx/site-packages/healpy/projaxes.py  =>  def _get_interv_graticule  =>  
	#            max_n_par = 18
	#            max_n_mer = 36
	 
	(2) fontsize in hp.mollview():
	# /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/healpy/visufunc.py
	#      	def mollview(titlesize=14, unitsize=14)
	#    ax.set_title(title, fontsize=titlesize)
	#    ax.text(0.86,0.05,ax.proj.coordsysstr, fontsize=notextsize
	#    cb.ax.text(0.5,-1.0,unit, fontsize=unitsize,
	#    if (cbarsize is not None) : cb.ax.tick_params(axis='both', which='major', labelsize=cbarsize)

	----------------------------------------

	***** matplotlib valid backends:
	[u'pgf', u'ps', u'Qt4Agg', u'GTK', u'GTKAgg', u'nbAgg', u'agg', u'cairo', u'MacOSX', u'GTKCairo', u'Qt5Agg', u'template', u'WXAgg', u'TkAgg', u'GTK3Cairo', u'GTK3Agg', u'svg', u'WebAgg', u'pdf', u'gdk', u'WX']
	http://blog.sina.com.cn/s/blog_7101508c01014iy6.html
	default: TkAgg

	----------------------------------------

	***** Judge if top function: 
		if (self.verbose and jp.SysFrame(0,2)[-3][-2]=='') : 
			print('is top function')

	----------------------------------------

	***** Any shape subplot
	Command: plt.subplot2grid(shape, startloc, rowspan=1, colspan)
	fig = plt.figure(figsize=(8,6))
	ax1 = plt.subplot2grid((3,3), (0,0))
	ax2 = plt.subplot2grid((3,3), (0,1), colspan=2)
	ax3 = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2)
	ax4 = plt.subplot2grid((3,3), (1,2), rowspan=2)
	ax1.plot([1.2,2,2.9], [4.2,5,5.9], 'b-')
	ax2.plot([1.2,2,2.9], [4.2,5,5.9], 'r-')
	ax3.plot([1.2,2,2.9], [4.2,5,5.9], 'bo-')
	ax4.plot([1.2,2,2.9], [4.2,5,5.9], 'ro-')
	plt.tight_layout()
	plt.show()

	----------------------------------------

	***** \sum \int in formula:
		r$\displaystyle\sum_{n=1}^\infty$

	----------------------------------------

	***** xlabel, ylabel, title location shift:
			plt.xlabel( '', labelpad= )
			plt.ylabel( '', labelpad= )
			plt.title( '', y=1.02)  # default y=1

	----------------------------------------


	***** legend transparent
		plt.legend(loc=4, facecolor='w', framealpha=1, fontsize=12, ncol=)

	***** Press "q" to quit, "j" to down, "k" to up
	'''
	pass
