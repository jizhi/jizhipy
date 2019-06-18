
def GetSize( fig=False, ax=False, **kwargs ) : 
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


