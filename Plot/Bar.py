
def Bar( x, y, align='edge', *args, **kwargs ) : 
	'''
	matplotlib.pyplot.bar(x, y, ...)

	align:
		'edge': x=>xe
		'center': x=>xc
	'''
	from jizhipy.Plot import Gca
	if ('ax' in kwargs.keys()) : 
		ax = kwargs['ax']
		kwargs.pop('ax')
	else : ax = Gca()
	if (align == 'center') : 
		ax.bar(x, y, align='center', *args, **kwargs)
		return ax
	from jizhipy.Array import Asarray
	xe, y = Asarray(x), Asarray(y)
	width = xe[1:] - xe[:-1]
	ax.bar(xe[:-1],y,width,align='edge', *args, **kwargs)
	return ax

