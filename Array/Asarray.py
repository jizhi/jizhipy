
def Asarray( value, keep=False, dtype=None ) : 
	''' 
	Asarray() instead of npfmt() and npfmt2()

	convert any value to numpy.array([])
	(1) value=None, return None
	(2) value is np.ndarray, return value
	(3) value is np.MaskedArray, return value
	(4) value is other/mix thing, return np.array(, object)
	(5) value is number/int/float, list, tuple:
			1. keep=False, return np.array([......])
			2. keep=True, return np.array() without []
	'''
	import numpy as np
	from jizhipy.Basic import IsType, Raise
	if (keep is not True and keep is not False): Raise(Exception, 'keep must be True/False, but now keep='+str(keep))
	# (1)
	if (value is None): return None
	# (3)
	if (type(value) == np.ma.core.MaskedConstant): 
		value = np.ma.asarray([0.])
		value.mask = [True]
		return value
	elif (type(value) == np.ma.core.MaskedArray): return value
	# (4)
	try: value = np.array(value, dtype)
	except: return np.array(value, np.object)
	if (value.shape == ()): 
		if (not keep): value = np.array([value])
	return value

