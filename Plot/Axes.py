
class Axes( object ) : 

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
		Frame: the whole frame, includes the box, the label

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
		Label means the number outside the box, not includes the tick bars

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
		Tick means the tick bars, not include the label(number)

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





Axes = Axes()
