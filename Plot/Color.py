
class Color( object ) : 

	def __init__( self ) : 
		'''
		self.mycolor

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





Color = Color()
