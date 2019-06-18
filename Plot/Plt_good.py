# legend(loc=)
#	2	 9	0/1
#	6	10	5/7
#	3	 8	 4

def Tips() : 
	'''
	***** Press "q" to quit, "j" to down, "k" to up

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





##################################################
##################################################
##################################################





class plt_color( object ) : 

	def __init__( self ) : 
		'''
		red:    'r' = (1,0,0) < #d62728
		orange: '#ff7f0e' = (1,0.5,0)
		yellow: '#bcbd22' = 'y'
		green:  (0,0.7843,0) < '#2ca02c' < 'g'
		cyan:   '#17becf' = 'c'
		blue:   '#1f77b4' < 'b'
		purple: '#9467bd' < (0.42,0,1)
		pink:   (1,0.702,0.816) < '#e377c2' < 'm'
		brown:  '#8c564b'
		gray:   '#f0f0f0' < '#e0e0e0' < '#d0d0d0' < '#c0c0c0' < '#b0b0b0' < '#a0a0a0' < '#7f7f7f'
		black:  'k'
		white:  'w'
		'''
		self.mycolor = {'red':('r','#d62728'), 'orange':('#ff7f0e',), 'yellow':('y',), 'green':((0,0.7843,0),'#2ca02c','g'), 'cyan':('c',), 'blue':('#1f77b4','b'), 'purple':('#9467bd',(0.42,0,1)), 'pink':((1,0.702,0.816),'#e377c2','m'), 'brown':('#8c564b',), 'gray':('#f0f0f0','#e0e0e0','#d0d0d0','#c0c0c0','#b0b0b0','#a0a0a0','#7f7f7f'), 'black':('k',), 'white':('w',)}



	def Cmap( self, cmap='gist_rainbow_r', set_under=None, set_bad=None, set_over=None, set_gamma=None, is_gray=None ) : 
		'''
		Default cmap is: matplotlib.cm.gist_rainbow_r
			Other similar: jet
		return cmap <matplotlib.colors.LinearSegmentedColormap> 

		cmap is used in hp.mollview(), plt.pcolormesh(), ...
	
		cmap:
			(1) <matplotlib.colors.LinearSegmentedColormap> 
			(2) name of cmap, like 'jet'
			(3) None: cmap name list
			
		set_under:
			str | (r,b,g)
			set the under color
	
		set_bad:
			tuple, (color, value)
			set the color of bad value
	
		matplotlib.pyplot.set_cmap(cmap)
	
		Better 'jet':  dark blue becomes slightly brighter
		*modify _jet_data in xxx/site-packages/matplotlib/_cmap.py
		_jet_data = {'red': ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.8, 0.8)),
	               'green': ((0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
	                'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
		'''
		if (cmap is None) : 
			cmapname = ['Accent','Accent_r','Blues','Blues_r', 
				'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
				'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2',
				'Dark2_r', 'GnBu', 'GnBu_r', 'Greens',
				'Greens_r', 'Greys', 'Greys_r', 'OrRd',
				'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn',
				'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
				'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG',
				'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
				'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd',
				'PuRd_r', 'Purples', 'Purples_r', 'RdBu',
				'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
				'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
				'RdYlGn_r', 'Reds', 'Reds_r', 'Set1',
				'Set1_r', 'Set2', 'Set2_r', 'Set3',
				'Set3_r', 'Spectral', 'Spectral_r', 'Wistia',
				'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
				'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd',
				'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
				'autumn_r', 'binary', 'binary_r', 'bone',
				'bone_r', 'brg', 'brg_r', 'bwr',
				'bwr_r', 'cool', 'cool_r', 'coolwarm',
				'coolwarm_r', 'copper','copper_r','cubehelix',
				'cubehelix_r', 'flag', 'flag_r', 'gist_earth',
				'gist_earth_r', 'gist_gray', 'gist_gray_r', 
				'gist_heat', 'gist_heat_r', 'gist_ncar', 
				'gist_ncar_r', 'gist_rainbow',
				'gist_rainbow_r', 'gist_stern','gist_stern_r', 
				'gist_yarg', 'gist_yarg_r', 'gnuplot', 
				'gnuplot2', 'gnuplot2_r',
				'gnuplot_r', 'gray', 'gray_r', 'hot',
				'hot_r', 'hsv', 'hsv_r', 'inferno',
				'inferno_r', 'jet', 'jet_r', 'magma',
				'magma_r', 'nipy_spectral','nipy_spectral_r', 
				'ocean', 'ocean_r', 'pink','pink_r','plasma',
				'plasma_r', 'prism', 'prism_r', 'rainbow',
				'rainbow_r','seismic','seismic_r','spectral',
				'spectral_r', 'spring', 'spring_r', 'summer',
				'summer_r', 'terrain', 'terrain_r','viridis',
				'viridis_r', 'winter', 'winter']
			return cmapname
		#----------------------------------------
		from matplotlib import cm
		from jizhipy.Basic import IsType
		default_cmap = 'gist_rainbow_r'
		if (IsType.isstr(cmap)) : 
			try : cmap = cm.get_cmap(cmap)
			except : cmap = cm.get_cmap(default_cmap)
		if (set_under is not None) : 
			try : cmap.set_under(set_under)
			except : pass
		else : cmap.set_under('w')
	#	else : cmap.set_under('none')
		if (set_bad is not None) : 
			try : cmap.set_bad(set_bad)
			except : pass
		if (set_over is not None) : 
			try : cmap.set_over(set_over)
			except : pass
		if (set_gamma is not None) : 
			try : cmap.set_gamma(set_gamma)
			except : pass
		if (is_gray is not None) : 
			try : cmap.is_gray(is_gray)
			except : pass
		return cmap



	def Linecolor( self, num, cmap='jet' ) : 
		'''
		Return is from Blue to Red

		num:
			(1) color name:
					in ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink', 'brown', 'black', 'white']
					return () of this color
			(2) int, number of colors:
					return () of colors
			(3) == 'png' : 
					open 'jizhipy_Plt_LineColor.png'
	
		cmap:
			Use which cmap to generate the linecolor
			(1) None=>'gist_rainbow_r' | Name of cmap (see plt_color.cmap(None))
			(2) <matplotlib.colors.LinearSegmentedColormap> (cmap instance)
			(3) cmap='my': use my cmap
		'''
		from jizhipy.Basic import Raise, IsType, ShellCmd, Path
		import numpy as np
		#----------------------------------------
		if (IsType.isstr(num)) : 
			num = num.lower()
			if ('png' in num) : 
				uname = ShellCmd('uname')[0].lower()
				figpath = Path.jizhipyPath('jizhipy_tool/jizhipy_Plt_Color_Linecolor.png')
				operate = 'eog ' if(uname=='linux')else 'open '
				ShellCmd(operate+figpath)
				return
			#----------------------------------------
			if (num in self.mycolor.keys()) : return self.mycolor[num] 
			else : Raise(Exception, num+' NOT in '+str(self.mycolor.keys()))
		#----------------------------------------
		if (IsType.isstr(cmap) and cmap == 'my') : 
			_jet_data_my = np.array([
				(255,   0,   0),
				(255, 128,   0),
				(200, 200,   0),
				(  0, 200,   0),
				(  0, 250, 250),
				(  0,   0, 255),
				(160,   0, 255)]) /255.
			r, g, b = _jet_data_my.T
			from scipy.interpolate import interp1d
			r, g, b = np.array([r, g, b]) * ratio
			x = np.linspace(0, 1, r.size)
			fr = interp1d(x, r)
			fg = interp1d(x, g)
			fb = interp1d(x, b)
			x = np.linspace(0, 1, num)
			r, g, b = fr(x), fg(x), fb(x)
			color = []
			for i in range(num) : 
				color += [(r[i], g[i], b[i])]
			return tuple(color)
		#----------------------------------------
		else : 
			cmap = self.Cmap(cmap)
			color=[cmap(x)[:3] for x in np.linspace(0.,1,num)]
			for i in range(len(color)) : 
				color[i] = (color[i][0], color[i][1], color[i][2])
			return tuple(color)



	def _rc( self, which, color, alpha ) : 
		'''
		alpha:
			set the transparent, 0~1

		color:
			==None: return the current color
			=='r'/../(1,0,0,alpha)/... (real color): return the original color
		'''
		which = str(which).lower()
		try : 
			alpha = float(alpha)
			if (alpha < 0 or alpha > 1) : alpha = 1.
		except : alpha = 1.
		if (type(color) in [list, tuple]) : 
			try : color = tuple(color) + (alpha,)
			except : pass
		import matplotlib
		color0 = {'ec': matplotlib.rcParams['axes.edgecolor'], 
		          'fc': matplotlib.rcParams['axes.facecolor']}
		#----------------------------------------
		if (which in ['edge', 'edgecolor']) : 
			if (color is not None) : 
				try : matplotlib.rc('axes', edgecolor=color)
				except : pass
			return color0['ec']
		#----------------------------------------
		elif (which in ['face', 'facecolor']) : 
			if (color is not None) : 
				try : matplotlib.rc('axes', facecolor=color)
				except : pass
			return color0['fc']

			
	def Edgecolor( self, edgecolor='k', alpha=1 ) : 
		''' 
		Modify edgecolor of the frame
		edgecolor=None: return current edgecolor

		If want to set the edgecolor of ax, call edgecolor() before create the ax
		Once ax is created, we can't change its edgecolor
		'''
		return self._rc('edgecolor', edgecolor, alpha)


	def Facecolor( self, facecolor='w', alpha=1 ) : 
		'''
		Modify facecolor of axes
		facecolor=None: return current facecolor

		If want to set the facecolor of ax, call facecolor() before create the ax
		Once ax is created, we can't change its facecolor
		'''
		return self._rc('facecolor', facecolor, alpha)



	def HealpixColorbar( self, orientation='vertical', ticks=None, ticklabels=None, ticksize=None, unit=None, unitsize=14, unitloc=None, shrink=None, pad=None, aspect=None, ax=None, fig=None ) : 
		'''
		orientation:
			'vertical' | 'horizontal'
	
		ticks:
			None | list/ndarray | 
			=='minmax': just show max and min
			=='+minmax': add max and min
			=='middle': just show middle value
	
		ticklabels: 
			None | list | '%.2f' like 

		ticksize:
			tick's fontsize
	
		unit, unitsize:
		unitloc:
			None | 'center'
		
		ax, fig:
			which axes and figure
	

		pad		 0.05: if vertical, 0.15 if horizontal; 
			fraction of original axes between colorbar and new image axes
			change the separation between colorbar and image

		shrink	  1.0: 
			fraction by which to multiply the size of the colorbar
			change the size of colorbar

		aspect	   20: 
			ratio of long to short dimensions
			change the shape of colorbar: long/short

		! fraction 0.15: 
			fraction of original axes to use for colorbar
		'''
		import matplotlib.pyplot as plt
		if (fig is None) : fig = plt.gcf()
		if (ax  is None) : ax  = plt.gca()
		im = ax.get_images()[0]
		fraction = 0.1
		if (orientation == 'vertical') : 
			if (pad is None) : pad = 0.03
			if (shrink is None) : shrink = 0.59
			if (aspect is None) : aspect = 18
			anchor = (0,0.5) if(unit is None)else (0,0.43)
		else : 
			if (pad is None) : pad = 0.04
			if (shrink is None) : shrink = 0.65
			if (aspect is None) : aspect = 30
			anchor = (0.5,1)
		#----------------------------------------
		cbar = fig.colorbar(im, orientation=orientation, shrink=shrink, aspect=aspect, fraction=fraction, pad=pad, anchor=anchor, ax=ax)
		#----------------------------------------
		if (unit is not None) : 
			if (orientation == 'vertical') : cbar.ax.set_title(unit, fontsize=unitsize, verticalalignment='bottom')
			else : 
				if (unitloc is None) : cbar.ax.set_ylabel(unit, fontsize=unitsize, rotation='horizontal', verticalalignment='center', horizontalalignment='right')
				elif (unitloc == 'center') : cbar.ax.set_xlabel(unit, fontsize=unitsize, verticalalignment='bottom')
		#----------------------------------------
		if (ticks is not None) : 
			vmin, vmax = cbar.vmin, cbar.vmax
			if (ticks in ['+minmax', '+maxmin']) : 
				if (orientation == 'vertical') : ticks = vmin + (vmax-vmin)*cbar.ax.get_yticks()
				else: ticks =vmin+(vmax-vmin)*cbar.ax.get_xticks()
				import numpy as np
				ticks =np.concatenate([[vmin], ticks,[vmax]])
			elif (ticks in ['minmax', 'maxmin']) : ticks = [vmin, vmax]
			elif (ticks=='middle') : ticks = [(vmin+vmax)/2.]
			cbar.set_ticks(ticks)
			if (ticklabels is not None) : 
				if (type(ticklabels) == str) : 
					ticklabels, fmt = [], ticklabels
					for i in range(len(ticks)) : 
						tl = fmt % ticks[i]
						if ('.' in tl) : 
							tls = tl.split('.')
							if (int(tls[1])==0) : tl =tls[0]
						ticklabels.append( tl )
				cbar.set_ticklabels(ticklabels)
		#----------------------------------------
		if (ticksize is not None) : plt_axes().Label(ax=cbar.ax, size=ticksize)
		return cbar



	def Colorbar( self ) : 
		'''
		return current colorbar

		colorbar of healpix:
			use plt.gca()

		colorbar of plt.pcolor:
			use plt.gci() == plt.gcf()._gci()
	
		(cbar.ax) can be as the input ax in
			jp.Plt.Axes...
		'''
		import matplotlib.pyplot as plt
		ax = plt.gca()
		if ('healpy' in str(ax.__class__)) : 
			im = ax.get_images()[-1]
		else : im = plt.gci()
		cbar = im.colorbar
		return cbar
		




##################################################
##################################################
##################################################





class plt_axes( object ) : 

	def _gca( self, ax=None ) : 
		from jizhipy.Basic import IsType
		if (IsType.isstr(ax)) : 
			if (str(ax).lower() in ['cbar', 'colorbar']) : 
				ax = plt_color().Colorbar()
			else : ax = None
		if (ax is None) : 
			import matplotlib.pyplot as plt
			ax = plt.gca()
		return ax


	def _gxy( self, xy ) : 
		xy = str(xy).lower()
		if (xy == 'both') : xy = 'xy'
		return xy


	def _gbool( self, *args ) : 
		args = list(args)
		for i in range(len(args)) : args[i] = bool(args[i])
		args = tuple(args)
		if (len(args) == 1) : args = args[0]
		return args





	def Frameoff( self, edgeon=False, ax=None ) : 
		'''
		edgeon:
			False: turn off the whole frame
			True : hold edges
		'''
		ax = self._gca(ax)
		if (not edgeon) : ax.set_axis_off()
		else : 
			self.Label(left=False, right=False, bottom=False, top=False)
			self.Tick('both', 'both', left=False, right=False, bottom=False, top=False)





	def Label( self, xy='both', color=None, pad=None, size=None, left=None, right=None, bottom=None, top=None, direction=None, ax=None ) : 
		'''
		Label means the number outside the frame

		pad=4, size=12, left=bottom=True, right=top=False
		'''
		kwargs = {}
		if(color  is not None): kwargs['color'] =color
		if(pad    is not None): kwargs['pad'] =pad
		if(size   is not None): kwargs['labelsize'] =size
		if(left   is not None): kwargs['labelleft'] =left
		if(right  is not None): kwargs['labelright'] =right
		if(bottom is not None): kwargs['labelbottom']=bottom
		if(top    is not None): kwargs['labeltop'] =top
		ax, xy = self._gca(ax), self._gxy(xy)
		if ('x' in xy) : 
		#	ax.tick_params(axis='x', which='major', labelcolor=color, pad=pad, labelsize=size, labelleft=left, labelright=right, labelbottom=bottom, labeltop=top)
			ax.tick_params(axis='x', which='major', **kwargs)
		if ('y' in xy) : ax.tick_params(axis='y', which='major', **kwargs)





	def _setTick( self, xy, which=None, fmt=None, direction=None, length=None, width=None, color=None, left=True, right=False, bottom=True, top=False, ax=None ) : 
		'''
		Set Tick by hand

		plt.xticks(), plt.yticks()

		xy:
			'x' | 'y' | 'both'

		which:
			(case 1): list to be shown / xylabel ndarray
			(case 2): ==None: return xlabel or/and ylabel

		fmt:
			'str' like '%i', '%.3f', '%6.3f'

		direction:
			'in' | 'out' | 'inout'

		length, width, color:
			of ticks' size  (length=8, width=1)

		left, right, bottom, top:
			True/False | 'on'/'off'
			Turn on/off the ticks
		'''
		import matplotlib.pyplot as plt
		from jizhipy.Basic import IsType, Raise
		import numpy as np
		from jizhipy.Optimize import Sample, Interp1d
		xy = self._gxy(xy)
		ax = self._gca(ax)
		ax0 = plt.gca()
		plt.sca(ax)
		if (len(ax.lines) != 0) : 
			xdata, ydata = ax.lines[-1].get_data()
		else : 
			xdata = ax.collections[-1]._coordinates[0,:,0]
			ydata = ax.collections[-1]._coordinates[:,0,1]
		if (which is None) : 
			xydata = []
			if ('x' in xy) : xydata.append(xdata.copy())
			if ('y' in xy) : xydata.append(ydata.copy())
			if (len(xydata) == 1) : xydata = xydata[0]
			return xydata
		#----------------------------------------
		which = np.array(which)
		if (len(which.shape) == 1) : which = which[None,:]
		if ('xy' in xy and len(which)==1) : xy = 'x'
		#----------------------------------------
		kwargs = {'direction':direction, 'left':left, 'right':right, 'bottom':bottom, 'top':top}
		if (color  is not None) : kwargs['color']  = color
		if (length is not None) : kwargs['length'] = length
		if (width  is not None) : kwargs['width']  = width
		#----------------------------------------
		if ('x' in xy) : 
			loc = plt.xticks()[0]
			xlabel = which[0]
			if (loc.size != xlabel.size) : 
				if (xdata.size != xlabel.size) : 
				#	Raise(Exception, 'xdata.size='+str(xdata.size)+' != xlabel.size='+str(xlabel.size))
					xlabel =Sample(xlabel,0, size=xdata.size)
				n =Interp1d(xdata, np.arange(xdata.size), loc)
				xlabel = Interp1d(np.arange(xlabel.size), xlabel, n)
			if (fmt is not None) : 
				try : fmt % 0
				except : fmt = '%.1f'
				xlabel = list(xlabel)
				for i in range(len(xlabel)) : 
					xlabel[i] = fmt % xlabel[i]
			plt.xticks(loc, xlabel)
			ax.tick_params(axis='x',which='major',**kwargs)
		#----------------------------------------
		if ('y' in xy) : 
			loc = plt.yticks()[0]
			ylabel = which[-1]
			if (loc.size != ylabel.size) : 
				if (ydata.size != ylabel.size) : 
					Raise(Exception, 'ydata.size='+str(ydata.size)+' != ylabel.size='+str(ylabel.size))
					ylabel =Sample(ylabel,0, size=ydata.size)
				n =Interp1d(ydata, np.arange(ydata.size), loc)
				ylabel = Interp1d(np.arange(ylabel.size), ylabel, n)
			if (fmt is not None) : 
				try : fmt % 0
				except : fmt = '%.1f'
				ylabel = list(ylabel)
				for i in range(len(ylabel)) : 
					ylabel[i] = fmt % ylabel[i]
			plt.yticks(loc, ylabel)
			ax.tick_params(axis='y',which='major',**kwargs)
		plt.sca(ax0)



	def _fmtTick( self, xy, which, fmt=None, sep=None, direction=None, length=None, width=None, color=None, left=True, right=False, bottom=True, top=False, ax=None ) : 
		'''
		Set Tick automatically

		Tick means the vertical lines on x-axis and parallel lines on y-axis
		
		xy:
			'x' | 'y' | 'both'

		which:
			'major' | 'minor' | 'both'

		sep:
			separation of ticks
			pair [0.1, 0.05] for [major, minor]
			one value 0.1 for major

		fmt:
			(1) 'str' like '%i', '%.3f', '%6.3f'
			(2) ==None: use plt.?ticks(sep[?][0], sep[?][1]) according to xy

		direction:
			'in' | 'out' | 'inout'

		length, width, color:
			of ticks' size  (length=8, width=1)

		left, right, bottom, top:
			True/False | 'on'/'off'
			Turn on/off the ticks

		ax:
			(1) can be =='cbar' | 'colorbar'
		'''
		from matplotlib.ticker import MultipleLocator, FormatStrFormatter
		import numpy as np
		from jizhipy.Array import Asarray
		import matplotlib.pyplot as plt
		from jizhipy.Basic import IsType
		if (fmt is not None) : 
			try : fmt % 0
			except : fmt = '%.1f'
		ax = self._gca(ax)
		#---------------------------------------------
		if (IsType.iscbar(ax)) : 
			vmin, vmax = ax.get_clim()
			sep = Asarray(sep)[0]
			amin = int(vmin / sep)*sep
			amax = int(vmax / sep +1)*sep
			ticks = np.arange(amin, amax+sep/2., sep)
			ticks = ticks[(vmin<=ticks)*(ticks<=vmax)]
			ticklabels = []
			for i in range(ticks.size) : 
				ticklabels.append( fmt % ticks[i] )
			ax.set_ticks(ticks)
			ax.set_ticklabels(ticklabels)
			return
		#---------------------------------------------
		xy = self._gxy(xy)
		if (IsType.isstr(which)) : 
			which = str(which).lower()
			if (which == 'both') : which = 'majorminor'
		kwargs = {'direction':direction, 'left':left, 'right':right, 'bottom':bottom, 'top':top}
		if (color  is not None) : kwargs['color']  = color
		if (length is not None) : kwargs['length'] = length
		if (width  is not None) : kwargs['width']  = width
		#---------------------------------------------
		if (sep is not None and fmt is None) : 
			dopltticks = True
			if (xy == 'x') : sep = [sep]
			if (xy == 'y') : sep = [None, sep]
		else : dopltticks = False
		#---------------------------------------------
		if ('major' in which) : 
			if (not dopltticks) : 
				try : 
					sep = Asarray(sep, float)[:2]
					loc = sep[0]
					loc = MultipleLocator(loc)
					fmt = FormatStrFormatter(fmt)
				except : pass
			if ('x' in xy) : 
				if (not dopltticks) : 
					try : 
						ax.xaxis.set_major_locator(loc)
						ax.xaxis.set_major_formatter(fmt)
					except : pass
				else : 
					ax.set_xticks(sep[0][0])
					ax.set_xticklabels(sep[0][1])
				ax.tick_params(axis='x',which='major',**kwargs)
			if ('y' in xy) : 
				if (not dopltticks) : 
					try : 
						ax.yaxis.set_major_locator(loc)
						ax.yaxis.set_major_formatter(fmt)
					except : pass
				else : 
					ax.set_xticks(sep[1][0])
					ax.set_xticklabels(sep[1][1])
				ax.tick_params(axis='y',which='major',**kwargs)
		#---------------------------------------------
		if ('minor' in which) : 
			if(length is not None): kwargs['length'] *= 0.6
		#	if(width  is not None): kwargs['width'] =width
			if (not dopltticks) : 
				try : 
					sep = Asarray(sep, float)[:2]
					loc=sep[1] if('major' in which)else sep[0]
					loc = MultipleLocator(loc)
			#	fmt = FormatStrFormatter(fmt)
				except : pass
			if ('x' in xy) : 
				if (not dopltticks) : 
					try : 
						ax.xaxis.set_minor_locator(loc)
					#	ax.xaxis.set_minor_formatter(fmt)
					except : pass
				ax.tick_params(axis='x',which='minor',**kwargs)
			if ('y' in xy) : 
				if (not dopltticks) : 
					try : 
						ax.yaxis.set_minor_locator(loc)
					#	ax.yaxis.set_minor_formatter(fmt) # this will show values on minor axis, we don't want it
					except : pass
				ax.tick_params(axis='y',which='minor',**kwargs)



	def Tick( self, xy, which=None, fmt=None, sep=None, direction=None, length=None, width=None, color=None, left=True, right=False, bottom=True, top=False, ax=None ) : 
		'''
		(1) call Tick(xy, which, fmt, sep, direction, ...)
				set the format of ticks
		(2) call Tick(xy, which, fmt, direction, ...)
				reset the ticklabels

		Tick means the vertical lines on x-axis and parallel lines on y-axis
		
		xy:
			'x' | 'y' | 'both'

		which:
			(case 1) 'major' | 'minor' | 'both'
			(case 2) list to be shown / xylabel ndarray, MUST be int/float array, NOT str array
			(case 3) ==None: return ticks

		fmt:
			(1) 'str' like '%i', '%.3f', '%6.3f'
			(2) ==None: use plt.?ticks(sep[?][0], sep[?][1]) according to xy

		sep:
			separation of ticks
			pair [0.1, 0.05] for [major, minor]
			one value 0.1 for major

		direction:
			'in' | 'out' | 'inout'

		length, width, color:
			of ticks' size  (length=8, width=1)

		left, right, bottom, top:
			True/False | 'on'/'off'
			Turn on/off the ticks
		'''
		from jizhipy.Basic import IsType
		if (IsType.isstr(which)) : 
			self._fmtTick(xy, which, fmt, sep, direction, length, width, color, left, right, bottom, top, ax )
		else : 
			return self._setTick(xy, which, fmt, direction, length, width, color, left, right, bottom, top, ax )





	def Format( self, xy, fmt, ax=None ) : 
		'''
		xy: 
			'x', 'y', 'both'
	
		fmt: 
			(1) '%d' or '%.2f' or '%.1e', '%...': 
				use plt.gca().xaxis.set_major_formatter(FormatStrFormatter(fmt))
	
			(2) 'sci'/'scientific' or 'plain'(turns off scientific notation): 
				use plt.ticklabel_format(axis=axis, style=fmt)
			(3) 'log'/'lg'
				actually log_10
				must be called before plt.xlim() and plt.ylim()
				use plt.loglog(), plt.semilogx(), plt.semilogy()
		'''
		import matplotlib.pyplot as plt
		from matplotlib.ticker import FormatStrFormatter
		ax,xy, axis =self._gca(ax), self._gxy(xy), xy
		if ('%' in fmt) : 
			if ('x' in xy) : ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
			if ('y' in xy) : ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
		#---------------------------------------------
		elif ('sci' in fmt) : 
			ax.ticklabel_format(axis=axis, style=fmt, scilimits=(0,0))
		#---------------------------------------------
		elif ('log' in fmt or 'lg' in fmt) : 
			if ('x' in xy) : ax.semilogx()
			if ('y' in xy) : ax.semilogy()





	def Period( self, x=None, y=None, period=None, dmajor=None, fmt='%i', ax=None ) : 
		'''
		return [ticks, labels]

		x, y:
			list/ndarray
			is the x/y in plotting: plt.plot(x, y)
	
		period:
			one int value
			For o'clock, set period=24  hour
			For phase,   set period=360 degree
	
		dmajor:
			Value of each major tick/label to be shown
			For o'clock, can set dmajor=12/6/...
			For phase,   can set dmajpr=30/60/...
		'''
		from jizhipy.Basic import Raise
		if (x is not None and y is not None) : Raise('jizhipy.Plt.PeriodicAxes(): can NOT set x and y are NOT None at the same time')
		if (x is None and y is None) : Raise('jizhipy.Plt.PeriodicAxes(): x = y = None')
		import matplotlib.pyplot as plt
		which = 'x' if(x is not None)else 'y'
		n = 1.*arr[0] / dmajor
		if (n == int(n)) : n = int(n)
		else : n = 1 + int(n)
		x0 = n * dmajor
		x0 = np.arange(x0, arr[-1]+dmajor/10., dmajor)
		x1 = list(x0 % period)
		for i in range(len(x1)) : x1[i] = (fmt % x1[i])
		ax0 = plt.gca()
		plt.sca(ax)
		if   (which == 'x') : plt.xticks(x0, x1)
		elif (which == 'y') : plt.yticks(x0, x1)
		plt.sca(ax0)
		return [x0, x1]





	def Twin( self, share, ax=None ) : 
		'''
		share:
			=='x': Create a twin Axes sharing the xaxis
			=='y': Create a twin Axes sharing the yaxis

		ax1 = plt.plot([1,2,3], [4,5,6])
		ax2 = Twin('x', ax1)
		ax2.plot([1,2,3], [7,8,9])
		'''
		ax = self.Gca(ax)
		if (str(share).lower() == 'x') : return ax.twinx()
		else : return ax.twiny()





	def Axes3D( self, viewinit=None, axfig=None ) : 
		'''
		Create a 3D axes
		return:
			Axes3D "<class 'mpl_toolkits.mplot3d.axes3d.Axes3D'>"

		axfig:
			(1) ==figure: "<class 'matplotlib.figure.Figure'>"
			(2) ==Axes3D: "<class 'mpl_toolkits.mplot3d.axes3d.Axes3D'>"
			(3) ==tuple(width, height): size of figure
			(4) ==None
	
		viewinit:
			elev, azim = viewinit [degree]
		'''
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D
		# (1) Give figure, set this figure to be current, then generate new Axes3D, and also set this axes to be current
		if (str(type(axfig)) == "<class 'matplotlib.figure.Figure'>") : 
			self.Scf(axfig)
			ax = Axes3D(axfig)
			plt.sca(ax)
		#----------------------------------------
		# (2) Give Axes3D, set it to be current
		elif (str(type(axfig)) == "<class 'mpl_toolkits.mplot3d.axes3d.Axes3D'>") : plt.sca(axfig)
		#----------------------------------------
		# (3) ax==None, ax=plt.gca(), if not Axes3D, use current figure to generate one and set it to be current
		elif (axfig is None) : 
			ax = plt.gca()
			if (str(type(ax)) != "<class 'mpl_toolkits.mplot3d.axes3d.Axes3D'>") : 
				ax = Axes3D(plt.gcf())
				plt.sca(ax)
		#----------------------------------------
		# (4) Give size of figure (width, height), create a new figure with figsize=(width, height), then create Axes3D
		else : 
			try : fig = plt.figure(figsize=ax)
			except : fig = plt.figure()
			ax = Axes3D(fig)
			plt.sca(ax)
		#----------------------------------------
		if (viewinit is not None) : 
			elev, azim = viewinit
			ax.view_init(elev, azim)
		return ax





##################################################
##################################################
##################################################





class Plt( object ) : 

	def __init__( self ) :  # self.Color,  self.Axes
		self.Color = plt_color()
		self.Axes  = plt_axes()


	def Tips( self ) : help(Tips)


	def Gca( self, ax=None ) : 
		import matplotlib.pyplot as plt
		if (ax is None) : ax = plt.gca()
		return ax


	def Sca( self, ax=None ) : 
		import matplotlib.pyplot as plt
		if (ax is None) : ax = plt.gca()
		plt.sca(ax)
		return ax


	def Gcf( self, fig=None ) : 
		import matplotlib.pyplot as plt
		if (fig is None) : fig = plt.gcf()
		return fig


	def Scf( self, fig=None ) : 
		'''
		set current figure	
		fig:
		  None | <matplotlib.figure.Figure> | int(fig number)
		  If give a figure, change current figure to this

		Other medhod:
			from matplotlib import _pylab_helpers
			try : 
				figManager = _pylab_helpers.Gcf.get_active()
				figManager.canvas.figure
				if (str(type(fig)) == "<class 'matplotlib.figure.Figure'>") : figManager.canvas.figure = fig
				return figManager.canvas.figure
			except : 
				fig = plt.figure()
				return fig
		'''
		import matplotlib.pyplot as plt
		if (fig is None) : fig = plt.figure()
		elif (type(fig) == int) : fit = plt.figure(fig)
		else : 
			try : fig = plt.figure(fig.number)
			except : fig = plt.figure()
		return fig





	def GetImage( self, ax=None ) : 
		ax = self.Gca(ax)
		im = ax.get_images()[0]



	def Figure( self, *args, **kwargs ) : 
		'''
		For pcolormesh(a)
			jp.Plt.Figure(pcolor=True)
			a.shape = (500, 500) is the best: eyes look good, image size is small
			a.shape = (400, 400) is 2nd
			a.shape = (<400, <400) are too small, pixels edges are clear
			a.shape = (600, 600) and (700, 700) are bad, lot of white lines in the image
			a.shape = (>700, >700), image sizes are too large, eyes can NOT distinguish between >700 and =500
		==> plt.savefig( .pdf ) is much small then .eps
		==> use magick xx.pdf pdf to make it further smaller

		Step 1: plt.figure(*args, **kwargs)
		Step 2: adjust left, right, bottom, top, hspace, vspace
		call jp.Plt.Figure(frameon=True) : plot host frame

		**kwargs:
			frameon, left, right, bottom, top, hspace, vspace, full, facecolor, edgecolor  +  other keys in plt.figure()

			if pcolor=True:
				other keys in **kwargs:
					dleft, dright, dtop, dbottom:
						set axsize['left'] += dleft  ......

		*args:
			the same as plt.figure()
		''' 
		import matplotlib.pyplot as plt
		def FrameOn( new ) : 
			if (new) : 
				fig0 = plt.gcf()
				ax0 = None if(len(fig0.axes)==0)else plt.gca()
				box={'left':0, 'right':1, 'bottom':0, 'top':1}
				self.Zoom(box)
			self.Axes.Tick('both', 'both', '%.1f', [0.1, 0.01], right=True, top=True, left=True, bottom=True, color='k', direction='in')
			plt.grid(True, alpha=0.2)
			dy, dx = 8./fig.get_size_inches()/100
			dx, dy = dx*1.4, dy*0.95
			for i in range(1, 10) : 
				# in
				plt.annotate(str(i), (i/10.-dy/2,0))    # b
				plt.annotate(str(i), (0,i/10.-dx/2))    # l
				plt.annotate(str(i), (i/10.-dy/2,1-dx)) # t
				plt.annotate(str(i), (1-dy,i/10.-dx/2)) # r
			if (new and ax0 is not None) : plt.sca(ax0)
		#----------------------------------------
		#----------------------------------------
		if (len(args)==0 and len(kwargs)==1 and 'frameon' in kwargs.keys() and kwargs['frameon']==True) : 
			fig = plt.gcf()
			FrameOn(True)
			return fig
		#----------------------------------------
		if ('full' in kwargs.keys() and kwargs['full']) : 
			kwargs.pop('full')
			kwargs['left']   = 0
			kwargs['right']  = 1
			kwargs['bottom'] = 0
			kwargs['top']    = 1
		doadjust = False
		adjust = {'left':None, 'right':None, 'bottom':None, 'top':None, 'hspace':None, 'wspace':None}
		for k in adjust.keys() : 
			if (k in kwargs.keys()) : 
				adjust[k] = kwargs[k]
				kwargs.pop(k)
				doadjust = True
		#----------------------------------------
		ispcolor = False
		if ('pcolor' in kwargs.keys()) : 
			ispcolor = bool(kwargs['pcolor'])
			kwargs.pop('pcolor')
		elif ('pcolormesh' in kwargs.keys()) : 
			ispcolor = bool(kwargs['pcolormesh'])
			kwargs.pop('pcolormesh')
		if (ispcolor) : kwargs['figsize'] = (6.1, 5)
		#----------------------------------------
		kwargspcolor = {}
		if ('dleft' in kwargs.keys()) : 
			kwargspcolor['dleft'] = kwargs['dleft']
			kwargs.pop('dleft')
		if ('dright' in kwargs.keys()) : 
			kwargspcolor['dright'] = kwargs['dright']
			kwargs.pop('dright')
		if ('dtop' in kwargs.keys()) : 
			kwargspcolor['dtop'] = kwargs['dtop']
			kwargs.pop('dtop')
		if ('dbottom' in kwargs.keys()) : 
			kwargspcolor['dbottom'] = kwargs['dbottom']
			kwargs.pop('dbottom')
		#----------------------------------------
		fig = plt.figure(*args, **kwargs)
		fig.subplots_adjust(**adjust)
		if ('facecolor' not in kwargs.keys()) : 
			self.Color.Facecolor('none')
		if('frameon' in kwargs.keys() and kwargs['frameon']):
			FrameOn(False)
		if ('edgecolor' in kwargs.keys()) : 
			self.Color.Edgecolor(kwargs['edgecolor'])
		#----------------------------------------
		if (ispcolor) : self.Zoom(self.GetSize(ax='pcolor', **kwargspcolor))
		return fig





	def FigureFrame( self, left=None, right=None, bottom=None, top=None, hspace=None, wspace=None, fig=None ) : 
		'''
		return/set parameters of largest/basic frame of figure
		default = {'left': 0.125, 'right': 0.9, 'bottom': 0.1, 'top': 0.9, 'wspace': 0.2, 'hspace': 0.2}
		'''
		import numpy as np
		if (fig is None) : 
			import matplotlib.pyplot as plt
			fig = plt.gcf()
		kwargs = {}
		if (left   is not None) : kwargs['left']   = left
		if (right  is not None) : kwargs['right']  = right
		if (bottom is not None) : kwargs['bottom'] = bottom
		if (top    is not None) : kwargs['top']    = top
		if (hspace is not None) : kwargs['hspace'] = hspace
		if (wspace is not None) : kwargs['wspace'] = wspace
		if (len(kwargs) != 0) : fig.subplots_adjust(**kwargs)
		subplotpars = fig.subplotpars.__dict__.copy()
		subplotpars.pop('validate')
		subplotpars['left'] += np.random.random()*0.001
		return subplotpars





	def Zoom( self, box1=None, box2=None, link1=None, link2=None, labeloff1=False, labeloff2=False, frameoff1=False, frameoff2=False, edgecolor1='k', edgecolor2='k', facecolor1='none', facecolor2='none' ) : 
		''' 
		Create ax on current fig
		NOT THAT on gcf(), NOT on gca()

		return:
			[ax1, lrbt1]  |  [ax1, ax2, lrbt1, lrbt2]
	
		box1, box2:
			box1={'left':, 'right':, 'bottom':, 'left':,}
			lrbt relate to the edge of figure
	
		link1, link2:
			tuple, any size
			link1 for box1, link2 for box2
				link box1 and box2
			if link2=None: link2=link1
			Example: 
				link1=(1,2), link2=None
					link box1-1 and box2=1, box1-2 and box2=2
				link1=(1,2), link2=(3,4)
					link box1-1 and box2=3, box1-2 and box2=4
	
		color=='none' mean transparent
	
		if link, lrbt1 is the box to the original data, lrbt2 is the zoom in
	
		Example 1:
			a = np.arange(20).reshape(5,4)
			x = np.arange(1000)-500
			y = np.exp(-x**2)
			x0 = np.linspace(100, 100+5*np.pi, 100)
			y0 = np.sin(x0)
	
			plt_color.face((1,0,0), alpha=0.1)
			plt.plot(x0, y0, 'b-')
			
			ax1, ax2, lrbt1, lrbt2 = plt_zoom([0.1, 0.4, 0.1, 0.4], [0.3, 0.9, 0.5, 0.95], (1,2), (4, 3))
			
			plt.sca(ax1)
			plt_axes.frameoff()
			lrbt1[1] *= 1.19
			ax1 = plt_zoom(lrbt1)[0]
			plt.pcolormesh(a)
			plt.colorbar()
			
			plt.sca(ax2)
			plt.plot([110, 230], [300, 370], 'g-')
			
			plt.show()
	
	
		Example 2:
			nside = 256
			hpmap = np.zeros(12*nside**2)
			npix = (jp.Random.Random(hpmap.size/20)*hpmap.size).astype(int)
			A = jp.Random.Random(npix.size)*2000+1000
			hpmap[npix] = A
			hpmap =hp.smoothing(hpmap, np.pi/180*2, verbose=False)
	
			lon, lat, dlon, dlat, Nlon, Nlat = 68, -2.5, 25, 25, 250, None 
			a = jp.CoordTrans.Healpix2Flat(lon, lat, dlon, dlat, Nlon, Nlat, hpmap)[-1]
			b = jp.CoordTrans.Healpix2Flat(lon, lat, dlon, dlat, Nlon, Nlat, hpmap)[-1]
		
			fig = plt.figure(figsize=(4,8))
			plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
			cmap = plt_color.cmap('jet')
		
			plt.annotate('21cm + noise', (0.1, 0.95), rotation=90)
			plt.annotate('extracted 21cm', (0.1, 0.715), rotation=90)
			plt.annotate('input 21cm', (0.1, 0.48), rotation=90)
			
			# 4 planes, each 0.25
			lrbt1 = [0.1, 1, 0.79, 0.99]
			ax = plt_zoom(lrbt1)[0]
			hp.mollview(hpmap, title='1', fig=fig.number, hold=True, cmap=cmap)
			
			lrbt2 = [0.1, 1, 0.555, 0.755]
			plt_zoom(lrbt2)
			hp.mollview(hpmap, title='2', fig=fig.number, hold=True, cmap=cmap)
			
			lrbt3 = [0.1, 1, 0.32, 0.52]
			plt_zoom(lrbt3)
			hp.mollview(hpmap, title='3', fig=fig.number, hold=True, cmap=cmap)
			
			# zoom in
			lrbt1 = [0.4, 0.45, 0.655, 0.68]
			lrbt2 = [0.02, 0.49, 0.01, 0.26]
			lrbt2 = [0.02, 0.48, 0.01, 0.25]
			plt_zoom(lrbt1, lrbt2, (3,4), (2,1), labeloff1=True, edgecolor='r')
			plt.pcolormesh(a, cmap=cmap)
			
			lrbt1 = [0.4, 0.45, 0.42, 0.445]
			lrbt2 = [0.52, 0.98, 0.01, 0.25]
			bg = (0, 0.7843, 0)
			plt_zoom(lrbt1, lrbt2, (3,4), (2,1), labeloff2=True, edgecolor='m')
			plt.pcolormesh(b, cmap=cmap)
			plt.show()
	
		hp.mollview():
	        figsize = (8.5, 5.4) => w/h=1.57 | h/w=0.64
	        lbwh = (0.02,0.05,0.96,0.9)  => 
			lrbt = (0.02,0.9,0.05,0.95)
		'''
		import numpy as np
		import matplotlib.pyplot as plt
		if (facecolor1 is None) : facecolor1 = 'none'
		if (facecolor2 is None) : facecolor2 = 'none'
		if (edgecolor1 is None) : edgecolor1 = 'none'
		if (edgecolor2 is None) : edgecolor2 = 'none'
		#----------------------------------------
		def box2lrbt( box ) : return (box['left'], box['right'], box['bottom'], box['top'])
		def lrbt2box( lrbt ) : return {'left':lrbt[0], 'right':lrbt[1], 'bottom':lrbt[2], 'top':lrbt[3]}
		#----------------------------------------
		def LBWH( lrbt ) :  # conver lrbt to lbwh and xy
			l, r, b, t = lrbt
			w, h = r-l, t-b
		#	pars = self.FigureFrame()
			pars = {'left':0, 'right':1, 'bottom':0, 'top':1}
			w0 = pars['right'] - pars['left']
			h0 = pars['top']   - pars['bottom']
			l, w = pars['left']+l*w0, w*w0
			b, h = pars['bottom']+b*h0, h*h0
			lbwh = [l, b, w, h]
			r, t = l+w, b+h
			xy = np.array([(r,t), (l,t), (l,b), (r,b)])
			return lbwh, xy
		#----------------------------------------
		lrbt1, lrbt2 = None, None
		if (box1 is not None) : lrbt1 = box2lrbt(box1)
		if (box2 is not None) : lrbt2 = box2lrbt(box2)
		#----------------------------------------
		if (link1 is None) :  # NOT link
			ec0, fc0 = self.Color.Edgecolor(edgecolor1), self.Color.Facecolor(facecolor1)
			axlist, ax1, ax2, lbwh1, lbwh2 = [], None, None, None, None
			if (lrbt1 is not None) : 
				self.Color.Edgecolor(edgecolor1)
				self.Color.Facecolor(facecolor1)
				lbwh1 = LBWH( lrbt1)[0]
				ax1 = plt.axes(lbwh1)
				axlist.append(ax1)
				if (labeloff1) : 
					self.Axes.Label('both', left=False, right=False, top=False, bottom=False)
					self.Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
				if (frameoff1) : self.Axes.Frameoff(ax1)
			#----------------------------------------
			if (lrbt2 is not None) : 
				self.Color.Edgecolor(edgecolor2)
				self.Color.Facecolor(facecolor2)
				lbwh2 = LBWH(lrbt2)[0]
				ax2 = plt.axes(lbwh2)
				axlist.append(ax2)
				if (labeloff2) : 
					self.Axes.Label('both', left=False, right=False, top=False, bottom=False)
					self.Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
				if (frameoff2) : self.Axes.Frameoff(ax2)
			#----------------------------------------
			self.Color.Edgecolor(ec0),self.Color.Facecolor(fc0)
			if (lrbt1 is not None and lrbt2 is None) : 
				axlist = [ax1, lrbt2box(lrbt1)]
			elif (lrbt1 is not None and lrbt2 is not None) : 
				axlist = [ax1, ax2, lrbt2box(lrbt1), lrbt2box(lrbt2)]
			return axlist
		#----------------------------------------
		#----------------------------------------
		else :  # link
			from jizhipy.Basic import IsType
			from jizhipy.Array import Asarray
			from mpl_toolkits.axes_grid1.inset_locator import mark_inset
			if (link2 is None) : link2 = link1
			link1, link2 = Asarray(link1), Asarray(link2) 
			n = min(len(link1), len(link2))
			link1, link2 = link1[:n], link2[:n]
			#----------------------------------------
			ec0, fc0 = self.Color.Edgecolor('none'), self.Color.Facecolor('none')
			lbwh1 = LBWH(lrbt1)[0]
			ax1 = plt.axes(lbwh1)
			self.Axes.Frameoff(ax1)
			#----------------------------------------
			lbwh2, xy2 = LBWH(lrbt2)
			w0, h0 = 0.0001, 0.0001
			for i in range(n) : 
				l1, l2 = link1[i], link2[i]
				x, y = xy2[l2-1]
				if   (l1 == 1) : lbwh = [x-w0, y-h0, w0, h0]
				elif (l1 == 2) : lbwh = [x, y-h0, w0, h0]
				elif (l1 == 3) : lbwh = [x, y, w0, h0]
				elif (l1 == 4) : lbwh = [x-w0, y, w0, h0]
				ax2 = plt.axes(lbwh)
				self.Axes.Frameoff(ax2)
				mark_inset(ax1, ax2, l1, l1, fc='none', ec=edgecolor2) # when use mark_inset() to link, frame of ax1 can NOT be remove while ax2 can. Therefore, use axlist[0] to be original image, axlist[1] to be the zoom region.
			#----------------------------------------
			# Re-plot ax1, ax2
			self.Color.Edgecolor(edgecolor1)
			self.Color.Facecolor(facecolor1)
			h1, h2 = lbwh1[-1]*0.9999, lbwh2[-1]*0.9999
			d1, d2 = lbwh1[-1]*0.0001, lbwh2[-1]*0.0001
			lbwh1[-1] = h1
			lbwh2[-1] = h2
			ax1 = plt.axes(lbwh1)  #@#@
			if (labeloff1) : 
				self.Axes.Label('both', left=False, right=False, top=False, bottom=False, ax=ax1)
				self.Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
			if (frameoff1) : self.Axes.Frameoff(ax1)
			self.Color.Edgecolor(edgecolor2)
			self.Color.Facecolor(facecolor2)
			ax2 = plt.axes(lbwh2)  #@#@
			if (labeloff2) : 
				self.Axes.Label('both', left=False, right=False, top=False, bottom=False, ax=ax2)
				self.Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
			if (frameoff2) : self.Axes.Frameoff(ax2)
			self.Color.Edgecolor(ec0),self.Color.Facecolor(fc0)
			lrbt1, lrbt2 = list(lrbt1), list(lrbt2)
			lrbt1[-1] -= d1
			lrbt2[-1] -= d2
			axlist =[ax1,ax2,lrbt2box(lrbt1),lrbt2box(lrbt2)]
			return axlist





	def Style( self, style=None, cmap=1, facecolor=1, grid=1, cap=1, math=1, font=1, zorder=1 ) : 
		'''
		style:
			=1: use 1.x style
			=2: use 2.x style
			=None: use style set here(this function)
	
		zorder:
			Ticks and grids are now plotted above solid elements such as filled contours, but below lines. To return to the previous behavior of plotting ticks and grids above lines, set mpl.rcParams['axes.axisbelow'] = False
	
		http://matplotlib.org/users/dflt_style_changes.html
		'''
		import matplotlib as mpl
		if (style==1) : 
			mpl.style.use('classic')
			return
		# Below is for style=None
		if (cmap ==1) : mpl.rcParams['image.cmap'] = 'jet'
		if (facecolor==1): mpl.rcParams['figure.facecolor']='0.75'
		if (grid==1) : 
			mpl.rcParams['grid.color'] = 'k'
			mpl.rcParams['grid.linestyle'] = ':'
			mpl.rcParams['grid.linewidth'] = 0.5
		if (cap==1) : mpl.rcParams['errorbar.capsize'] = 3
		if (math==1) : 
			# cm: Computer Modern
			mpl.rcParams['mathtext.fontset'] = 'cm'
			mpl.rcParams['mathtext.rm'] = 'serif'
		if (zorder==1) : mpl.rcParams['axes.axisbelow']=False





	def Use( self, which, tf=True ) : 
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
			texcache = Path.AbsPath('~/.cache/matplotlib/tex.cache')
			if (Path.ExistsPath(texcache)) : 
				os.system('rm -r '+texcache)
			mpl.rcParams['text.usetex'] = bool(tf)
		#	import matplotlib.pyplot as plt
		#	plt.rc('text', usetex=True)
		else :  # which is packagename
			mpl.rcParams['text.latex.preamble'] = [r'\usepackage{'+which+r'}']





	def MulticolorLine( self, x, y, z=None, lcarray=None, edge=None, ax=None, **kwargs ) : 
		'''
		plot Multicolor line
		return ax
		
		z:
			for 3D line

		lcarray:
			color from blue to red, corresponding to lcarray.min() to lcarray.max()
			In principle, lcarray can be any array from small to large. In practice, generally set lcarray=x or lcarray=y or lcarray=z
	
		edge:
			int
			Split lcarray(also from blue to red) into 'edge' pieces, each piece has one color

		lcarray + edge: create color map
	
		**kwargs:
			Can set: marker, ms, mec, ls, lw, ......
			** mec=True: mec == color of marker | mec=False: maybe 'k' or marker color dependent on matplotlib setting
	
		Example:
			x = np.linspace(0, 8*np.pi, 1000)
			y = np.sin(x)
			Plt.MulticolorLine(x, y, lcarray=x, edge=10, marker='o')
		'''
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import Raise, IsType
		if (lcarray is None) : lcarray = x
		lcarray = Asarray(lcarray)
		#---------------------------------------------
		if (edge is None) : edge = 100
		if (IsType.isint(edge)) : 
			if(edge >lcarray.size*4/5): edge=lcarray.size*4/5
			edge = np.linspace(lcarray.min(), lcarray.max(), edge)
		# Check edge
		edge2 = [edge[0]]
		for i in range(len(edge)-1) : 
			tf = (edge[i]<=lcarray)*(lcarray<=edge[i+1])
			n = lcarray[tf].size
			if (n >= 2) : edge2.append(edge[i+1])
		if (len(edge2) == 1) : edge2.append(lcarray.size)
		edge = Asarray(edge2)
		#---------------------------------------------
		color = plt_color.line(len(edge)-1, 'my')
		if ('color' in kwargs) : kwargs.pop('color')
		if ('c' in kwargs) : kwargs.pop('c')
		x, y = Asarray(x), Asarray(y)
		if (lcarray.shape != x.shape) : Raise(Exception, 'jizhipy.Plt.MulticolorLine(): lcarray.shape='+str(lcarray.shape)+' != x.shape='+str(x.shape))
		#---------------------------------------------
		if ('marker' in kwargs.keys()) : 
			if ('ls' not in kwargs.keys() and 'linestyle' not in kwargs.keys()) : kwargs['ls'] = ''
		if ('mec' in kwargs.keys() and kwargs['mec'] is True):
			mec = True
			kwargs.pop('mec')
		else : mec = False
		#---------------------------------------------
		n = np.arange(x.size)
		if (z is None) : 
			if (ax is None) : ax = plt.gca()
			for i in range(len(color)) : 
				tf = n[(edge[i]<=lcarray)*(lcarray<=edge[i+1])]
				if (tf[0] != 0) : tf[0] -= 1
				if (mec) : ax.plot(x[tf], y[tf], color=color[i], mec=color[i], **kwargs)
				else : ax.plot(x[tf], y[tf], color=color[i], **kwargs)
		#---------------------------------------------
		else :  
			if (IsType.isnum(z)) : z = z + 0*x
			if (ax is None) : ax = Axes3D(plt.gcf())
			#--------------------
			if ('viewinit' in kwargs.keys()) : 
				try : elev, azim = kwargs['viewinit']
				except : elev = azim = None
				ax.view_init(elev, azim)
				kwargs.pop('viewinit')
			#--------------------
			for i in range(len(color)) : 
				tf = n[(edge[i]<=lcarray)*(lcarray<=edge[i+1])]
				if (tf[0] != 0) : tf[0] -= 1
				if (mec) : ax.plot(x[tf], y[tf], z[tf], color=color[i], mec=color[i], **kwargs)
				else : ax.plot(x[tf], y[tf], z[tf], color=color[i], **kwargs)
		return ax





	def HealpixFigure(self, figsize=None, box=None, **kwargs):
		'''
		figsize:
			(1) ==(width, height)
			(2) isnum: scale the default size
			(3) ==True: 
				return default setting but don't plt.
				return (figsize0, box0)
	
		box:
			box = {'left': , 'right': , 'bottom': , 'top': }
		'''
		width, height = 8.5, 5.4
		l, b, r, t = 0.02, 0.05, 0.98, 0.95
		figsize0 = (width, height)
		box0 = {'left':l, 'bottom':b, 'right':r, 'top':t}
		extent0 = (l, b, r-l, t-b)
		if (figsize is True) : return (figsize0, box0)
		import matplotlib.pyplot as plt
		from jizhipy.Basic import IsType
		if (figsize is None) : figsize = figsize0
		elif (IsType.isnum(figsize)) : figsize = (width*figsize, height*figsize)
		if (box is None) : box = box0
		elif (IsType.isnum(lbrt)) : 
			w, h = (r-l)*lbrt, (t-b)*lbrt
			l, r = 0.5-w/2, 0.5+w/2
			b, t = 0.5-h/2, 0.5+h/2
			box = {'left':l, 'bottom':b, 'right':r, 'top':t}
		fig = plt.figure(figsize=figsize, **kwargs)
		return fig





	def Bar( self, x, y, align='edge', *args, **kwargs ) : 
		'''
		matplotlib.pyplot.bar(x, y, ...)

		align:
			'edge': x=>xe
			'center': x=>xc
		'''
		if ('ax' in kwargs.keys()) : 
			ax = kwargs['ax']
			kwargs.pop('ax')
		else : ax = self.Gca()
		if (align == 'center') : 
			ax.bar(x, y, align='center', *args, **kwargs)
			return ax
		from jizhipy.Array import Asarray
		xe, y = Asarray(x), Asarray(y)
		width = xe[1:] - xe[:-1]
		ax.bar(xe[:-1],y,width,align='edge', *args, **kwargs)
		return ax





	def _Pcolormesh( self, *args, **kwargs ) : 
		'''
		**kwargs:
			'shape': 
				2D shape of image, must (<=500, <=500)
				Default shape=(500, 500)

			'sample':
				True | False
				How to reduce the image?
				=True: Sample(), keep the noise level
				=False: Smooth(), depress the noise
				Default sample=True
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import IsType
		from jizhipy.Optimize import Smooth, Sample
		import matplotlib.pyplot as plt
		args = list(args)
		args[0] = Asarray(args[0])
		if (len(args[0].shape) == 1) : 
			args[1], args[2] =Asarray(args[1]), Asarray(args[2])
			X, Y, C = args[:3]
		else : X, Y, C = None, None, args[0]
		#----------------------------------------
		if ('shape' in kwargs.keys()) : 
			shape = kwargs['shape']
			kwargs.pop('shape')
			if (IsType.isnum(shape)) : shape =[shape, shape]
			shape = Asarray(shape, float)
			if (shape.max() > 500) :  # lower
				if (shape[0] > shape[1]) : 
					shape[1] *= 500./shape[0]
					shape[0] = 500
				else : 
					shape[0] *= 500./shape[1]
					shape[1] = 500
			shape = Asarray(shape, int)
		else : shape = np.array([500, 500])
		#----------------------------------------
		if('sample' in kwargs.keys()) : 
			issample = bool(kwargs['sample'])
			kwargs.pop('sample')
		else : issample = False
		if (issample) : 
			if (C.shape[0] > shape[0]) : 
				C = Sample(C, 0, size=shape[0])
			if (C.shape[1] > shape[0]) : 
				C = Sample(C, 1, size=shape[1])
		else : 
			if (C.shape[0] > shape[0]) : 
				C = Smooth(C, 0, reduceshape=shape[0])
			if (C.shape[1] > shape[0]) : 
				C = Smooth(C, 1, reduceshape=shape[1])
		#----------------------------------------
		if (X is not None) : 
			X = Sample(X, 0, size=C.shape[1])
			Y = Sample(Y, 0, size=C.shape[0])
			args[0], args[1], args[2] = X, Y, C
		else : args[0] = C
		#----------------------------------------
		return plt.pcolormesh(*args, **kwargs)





	def Savefig( self, *args, **kwargs ) : 
		'''
		For pcolor:
			figsize = (6.1, 5)
			axsize  = (0.66, 0.8)
			imgsize = (4, 4) inch
			*_temp.png to *.pdf, need dpi >= 300 for good quality, then size of *.pdf is 200 KB, 1200x1200 pixels
			however, if plt.savefig(*.pdf) for 1200x1200 image, size ~10 MB
		*** NOTE THAT some times, array is small like 100x100, jp.Plt.Savefig(*.pdf, vector=True) is smaller than jp.Plt.Savefig(*.pdf, vector=False)

		plt.figure(figsize=(8,6))
		plt.savefig('fig.png')
			fig.png's size = (800, 600)
				figsize=(8,6): 8inch x 6inch
			savefig(dpi=100): each inch 100 pixel
			so, fig.png's size = 800pixel x 600pixel

		Therefore, 
			plt.figure(figsize=(8,6))
			plt.savefig('fig.png', dpi=300)
				== (only the image equal, the text/label become small !!!)
			plt.figure(figsize=(24,18))
			plt.savefig('fig.png', dpi=100)

		kwargs:
			vector: 
				==True: plt.savefig(*.pdf)
				==False: save to *_temp.png, then convert to *.pdf
			png:
				be used only when vector=False
				==False: delete *_temp.png after converted to *.pdf
				==True: save the *_temp.png

		for hp.mollview()
			(1) plt.savefig('test.pdf')
			(2) plt.savefig('test.png') + magick test.png ./test.pdf
			method (1) and (2) have the same size of pdf
		'''
		import matplotlib.pyplot as plt
		import os
		if (not (args[0][-4:]=='.pdf' and len(plt.gca().collections)>0)) : 
			return plt.savefig(*args, **kwargs)
		#----------------------------------------
		figname = args[0]
		outdir = '' if('/' in figname)else './' 
		if ('vector' in kwargs.keys()) : 
			vector = bool(kwargs['vector'])
			kwargs.pop('vector')
		else : vector = False
		if ('png' in kwargs.keys()) : 
			png = bool(kwargs['png'])
			kwargs.pop('png')
		else : png = False
		if (not vector) : 
			try : 
				dpi = kwargs['dpi']
				if (dpi is None) : raise
			except : 
			#	im = plt.gca().collections[-1]
			#	w, h = im._coordinates.shape[:2]
			#	figsize,axsize = self.GetSize(plt.gcf(), plt.gca())
			#	wfig, hfig=figsize['width'],figsize['height']
			#	wax = axsize['right'] - axsize['left']
			#	hax = axsize['top'] - axsize['bottom']
			#	wpix, hpix = wfig*wax, hfig*hax
			#	dpi = int(max(w/wpix, h/hpix))
			#	if (dpi < 200) : dpi = 200
			#	if (dpi > 300) : dpi = 300
				dpi = 300  # fix 300
			kwargs['dpi'] = dpi
			#----------------------------------------
			args = list(args)
			args[0] = args[0][:-4] + '_temp.png'
			refig = plt.savefig(*args, **kwargs)
			os.system('magick '+args[0]+' '+outdir+figname+' verbose=False')
			if (not png) : os.system('/bin/rm '+args[0])
			return refig
		#----------------------------------------
		else : 
			refig = plt.savefig(*args, **kwargs)
			os.system('magick '+figname+' '+outdir+figname+' verbose=False')
			return refig





	def GetSize( self, fig=False, ax=False, **kwargs ) : 
		'''
		(1) GetSize( fig ): 
				fig='healpix'/'mollview': mollview size
				fig='pcolor'       : pcolor size
				fig=False          : NOT return figsize
				fig=None           : default figure size
				fig=True/plt.gcf() : current figure size
				fig=<matplotlib.figure.Figure>: this figure size
				return figsize={'width': , 'height': }
		
		(2) GetSize( ax ): 
				ax='pcolor'       : pcolor size
				ax=False          : NOT return axsize
				ax=None           : default axes size
				ax=True/plt.gca() : current axes size
				ax=<matplotlib.axes._subplots.AxesSubplot>: this axes size
				return axsize={'left': , 'right': , 'bottom': , 'top': }

		(3) GetSize( fig, ax ):
				return [figsize, axsize]

		**kwargs:
			colorbar = False/True

			dleft, dright, dtop, dbottom:
				set axsize['left'] += dleft  ......
		'''
		import matplotlib.pyplot as plt
		from jizhipy.Basic import IsType
		relist = []
		if (fig is not False) : 
			if (fig is None) : width, height = 8, 6
			elif (IsType.isstr(fig)) : 
				fig = str(fig).lower()
				if (fig in ['healpix', 'mollview']) : width, height = 8.5, 5.4
				elif ('pcolor' in fig) : width, height = 6.1, 5  # colorbar=True
			else : 
				if (fig is False) : fig = plt.gcf()
				width, height = fig.get_figwidth(), fig.get_figheight()
			figsize = {'width':width, 'height':height}
			relist.append(figsize)
		#----------------------------------------
		if (ax is not False) : 
			if (ax is None) : left, right, bottom, top = 0.125, 0.9, 0.11, 0.88
			elif (IsType.isstr(ax)) : 
				ax = str(ax).lower()
				if (ax in ['healpix', 'mollview']) : 
					if ('colorbar' in kwargs.keys() and not kwargs['colorbar']) : left, bottom, right, top = 0.02, 0.05, 0.98, 0.95  # cbar=False
					else : left, bottom, right, top = 0.02, 0.185, 0.98, 0.95  # cbar=True
				elif ('pcolor' in ax) : left, right, bottom, top = 0.14, 0.9674, 0.125, 0.93
			else : 
				if (ax is True) : ax = plt.gca()
				left, bottom, right, top = ax.get_position().get_points().flatten()
			axsize = {'left':left, 'right':right, 'bottom':bottom, 'top':top}
			#--------------------
			if ('dleft' in kwargs.keys()) : 
				try : axsize['left'] += kwargs['dleft']
				except : pass
			if ('dright' in kwargs.keys()) : 
				try : axsize['right'] += kwargs['dright']
				except : pass
			if ('dtop' in kwargs.keys()) : 
				try : axsize['top'] += kwargs['dtop']
				except : pass
			if ('dbottom' in kwargs.keys()) : 
				try : axsize['bottom'] += kwargs['dbottom']
				except : pass
			#--------------------
			relist.append(axsize)
		if (len(relist) == 1) : relist = relist[0]
		return relist





	def Graticule( self, hpmapORdlat=None, dlon=None, **kwargs ) : 
		'''
		(1) Graticule( dlat, dlon ):
				dlat, dlon in degree
	
		(2) Graticule( hpmap ):
				hpmap is healpix map
			Example:
				a = np.log10(jp.GSM2016(example=True))
				nside = hp.get_nside(a)
				hp.mollview(a, title='diffuse')
				b = np.zeros(12*nside**2)
				theta, phi = np.array(hp.pix2ang(nside, np.arange(12*nside**2)))*180/np.pi
				b[(89.5<theta)*(theta<90.5)] = 1
				b=jp.CoordTrans.HealpixRotation(b, ax=40)[-1]
				jp.Plt.Graticule(b, color='r', ls='--', lw=3)
				plt.show(), exit()

		(3) Graticule( border=True, **kwargs )
				plot border
		'''
		import matplotlib.pyplot as plt
		import numpy as np
		import healpy as hp
		from jizhipy.Basic import IsType
		from jizhipy.Array import Invalid
		fig0, ax0 = plt.gcf(), plt.gca()
		if ('border' in kwargs.keys()) : 
			theta_border = np.arange(0, 181)*np.pi/180
			border = bool(kwargs['border'])
			kwargs.pop('border')
		else : border = False
		#----------------------------------------
		if(IsType.isnum(hpmapORdlat) and IsType.isnum(dlon)):
			kwargs['verbose'] = False
			hp.graticule(hpmapORdlat, dlon, **kwargs)
			if (border) : 
				ax0.projplot(theta_border, theta_border*0-np.pi, direct=True, color='k', ls='-', lw=2)
				ax0.projplot(theta_border, theta_border*0+0.9999*np.pi, direct=True, color='k', ls='-', lw=2)
			return ax0
		#----------------------------------------
		elif (hpmapORdlat is not None) : 
			figsize, axsize = self.GetSize(fig0, ax0)
			fig = plt.figure(figsize=(figsize['width'], figsize['height']))
			hpmap = hp.mollview(hpmapORdlat, fig=fig.number, return_projected_map=True)
			plt.close(fig)
			del fig
			hpmap = Invalid(hpmap)
			hpmap.data[hpmap.mask] = 0
			hpmap = hpmap.data
			#----------------------------------------
			drow =(1.*axsize['top'] -axsize['bottom']) /hpmap.shape[0]
			dcol =(1.*axsize['right'] -axsize['left']) /hpmap.shape[1]
			x, y = [], []
			for i in range(1, hpmap.shape[1]) : 
				b = hpmap[:,i]
				n = np.arange(b.size)[b>0.5]
				if (n.size > 0) : 
					x.append(axsize['left'] + dcol*i)
					y.append(axsize['bottom'] + drow*n.mean())
			self.Scf(fig0)
			self.Sca(ax0)
			self.Zoom(axsize)
			plt.plot(x, y, **kwargs)
			plt.xlim(axsize['left'], axsize['right'])
			plt.ylim(axsize['bottom'], axsize['top'])
			self.Axes.Frameoff()
			plt.sca(ax0)
			if (border) : 
				ax0.projplot(theta_border, theta_border*0-np.pi, direct=True, color='k', ls='-', lw=2)
				ax0.projplot(theta_border, theta_border*0+0.9999*np.pi, direct=True, color='k', ls='-', lw=2)
			return ax0
		#----------------------------------------
		elif (border) : 
			key = kwargs.keys()
			if ('color' not in key) : kwargs['color'] = 'k'
			if ('lw' not in key and 'linewidth' not in key) : kwargs['lw'] = 2
			ax0.projplot(theta_border, theta_border*0-np.pi, direct=True, **kwargs)
			ax0.projplot(theta_border, theta_border*0+0.9999*np.pi, direct=True, **kwargs)
			return ax0





Plt = Plt()
