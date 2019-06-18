
def Invalid( array, which=None ) : 
	'''
	nan, inf are invalid value.

	which:
		(1) ==None: 
			just mask them, shape of array won't change
			return MaskedArray

		(2) ==True:
			use array.mean() to reset the bad values
			return MaskedArray

		(3) ==False: 
			return the valid values, will flatten the array
			return ndarray

		(4) ==int/float/np.ndarray
			use these value to reset the masked
			return MaskedArray

	return:
		which==False: array1d  (np.array)
		others: array  (np.ma.MaskedArray)
	'''
	import numpy as np
	array = np.ma.masked_invalid(np.array(array))
	mask = array.mask
	if (which is None) : pass
	elif (which is False) : 
		if (mask[mask].size==0) : array = array.data.flatten()
		else : array = array.data[(1-mask).astype(bool)]
	else : 
		if (mask[mask].size > 0) : 
			if (which is True) : which = array.mean()
			array.data[mask] = which
	return array


