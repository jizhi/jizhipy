
def SciNot( array ) : 
	'''
	Scientific Notation.
	value can be scale(int/float), list/n-D array
	Return [a, n], value = a * 10**n
	'''
	from jizhipy.Basic import IsType
	import numpy as np
	from jizhipy.Array import Asarray
	if (IsType.isint(array) or IsType.isfloat(array)) : islist = False
	else : islist = True
	array = Asarray(array)
	# Flatten
	shape = array.shape
	array = array.flatten()
	# Get the sign
	sign = np.sign(array)  # sign(0)=0
	# Convert to abs
	array = abs(array)
	# because sign(0)=0, convert 0 to 1
	array[array==0] = 1
	nlarge, nsmall = (array>=1), (array<1)  # bool, not int
	# Use log10 to get the power index
	# >=1
	if (nlarge.sum() > 0) : idxlarge = np.log10(array[nlarge]).astype(int)
	else : idxlarge = []
	# <1
	if (nsmall.sum() > 0) : 
		scalesmall = int(round(np.log10(array[nsmall].min())))-2
		array[nsmall] /= 10.**scalesmall
		idxsmall = np.log10(array[nsmall]).astype(int) + scalesmall
		array[nsmall] *= 10.**scalesmall
	else : idxsmall = []
	# valid and idx
	idx = np.zeros(array.size, int)
	idx[nlarge], idx[nsmall] = idxlarge, idxsmall
	valid = sign * (array / 10.**idx)
	valid, idx = valid.reshape(shape), idx.reshape(shape)
	if (islist) : return (valid, idx)
	else : return (valid[0], idx[0])
	
	
