
def RMS( cls, a, axis=None ) : 
	''' Root Mean Square
	a is array, this function return the rms of a 
	'''
	import numpy as np
	a = np.array(a)
	if (axis is None) : 
		rms = ((a**2).mean())**0.5
	else : 
		rms = ((a**2).mean(axis))**0.5
	return rms
