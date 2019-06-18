
def Figure( *args, **kwargs ) : 
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
		=> jp.Plot.Figure(figsize=(8,3))
		=> jp.Plot.Figure(frameon=True)

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
	from jizhipy.Plot import Zoom, Axes, Color, GetSize
	def FrameOn( new ) : 
		if (new) : 
			fig0 = plt.gcf()
			ax0 = None if(len(fig0.axes)==0)else plt.gca()
			box={'left':0, 'right':1, 'bottom':0, 'top':1}
			ax, lrtb = Zoom(box)
		Axes.Tick('both','both', '%.1f', [0.1, 0.01], right=True, top=True, left=True, bottom=True, color='k', direction='in')
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
		# only for: jp.Plot.Figure(frameon=True)
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
		Color.Facecolor('none')
	if('frameon' in kwargs.keys() and kwargs['frameon']):
		FrameOn(False)
	if ('edgecolor' in kwargs.keys()) : 
		Color.Edgecolor(kwargs['edgecolor'])
	#----------------------------------------
	if (ispcolor) : Zoom(GetSize(ax='pcolor', **kwargspcolor))
	return fig





def FigureFrame( left=None, right=None, bottom=None, top=None, hspace=None, wspace=None, fig=None ) : 
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





def HealpixFigure( figsize=None, box=None, **kwargs ):
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
