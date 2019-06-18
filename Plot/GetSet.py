
def Gca( ax=None ) : 
	import matplotlib.pyplot as plt
	if (ax is None) : ax = plt.gca()
	return ax


def Sca( ax=None ) : 
	import matplotlib.pyplot as plt
	if (ax is None) : ax = plt.gca()
	plt.sca(ax)
	return ax


def Gcf( fig=None ) : 
	import matplotlib.pyplot as plt
	if (fig is None) : fig = plt.gcf()
	return fig


def Scf( fig=None ) : 
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



def GetImage( ax=None ) : 
	ax = Gca(ax)
	im = ax.get_images()[0]
