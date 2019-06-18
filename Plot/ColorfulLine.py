
def ColorfulLine( x, y, z=None, lcarray=None, edge=None, ax=None, **kwargs ) : 
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
		ColorfulLine(x, y, lcarray=x, edge=10, marker='o')
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
	if (lcarray.shape != x.shape) : Raise(Exception, 'jizhipy.Plot.ColorfulLine(): lcarray.shape='+str(lcarray.shape)+' != x.shape='+str(x.shape))
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

