
def Corrcoef( x, y, axis ) : 
	'''
	Different from np.corrcoef()
	'''
	from jizhipy.Basic import Raise
	import numpy as np
	from jizhipy.Array import Asarray
	x, y = Asarray(x), Asarray(y)
	try : x+y
	except : Raise(Exception, 'x.shape='+str(x.shape)+', y.shape='+str(y.shape)+' can NOT broadcast')
	x = x + y*0
	y = y + x*0
	if (axis < 0) : axis += len(x.shape)
	if (axis >= len(x.shape)) : Raise(Exception, 'axis='+str(axis)+' exceeds x.shape='+str(x.shape))
	up = ((x-x.mean(axis, keepdims=True)) * (y-y.mean(axis, keepdims=True))).sum(axis) / x.shape[axis]
	down = x.std(axis) * y.std(axis)
	r = up / down
	return r
	
	



