
def Counter( array, precision=None ):
	'''
	Count the number of each element in an array

	precision:
		None | 10**n , n is int from -inf to +inf
		e.g. array=[1.273, 1.325, 1.334, 123, 123.4, 115]
		(1) precision=0.1
			array becomes [1.3, 1.3, 1.3, 123, 123.4, 115]
			return {1.3:3, 123:1, 123.4:1, 115:1}
		(2) precision=0.01
			array becomes [1.27, 1.33, 1.33, 123, 123.4, 115]
			return {1.27:1, 1.33:2, 123:1, 123.4:1, 115:1}
		(3) precision=1
			array becomes [1, 1, 1, 123, 123, 115]
			return {1:3, 123:2, 115:1}
		(4) precision=10
			array becomes [0, 0, 0, 12, 12, 12]
			return {0:3, 12:3}
		(5) precision=100
			array becomes [0, 0, 0, 100, 100, 100]
			return {0:3, 100:3}
	'''
	import numpy as np
	from collections import Counter
	array = np.array(array)
	dtype = array.dtype
	if (precision is None):
		if ('int' in dtype.name): precision = 1
		elif ('float' in dtype.name): precision = 0.1
	precision = np.log10(precision)
	if (abs(precision-int(precision)) < 1e-6):
		precision = int(round(precision))
	elif (precision <0):
		precision = int(precision)-1
	elif (precision >0):
		precision = int(precision)
	#----------------------------------------
	if (precision <0): decimals = -precision
	else: decimals = 0
	scale = 10.**precision
	if ('int' in dtype.name):
		array = ((array /scale).round() *scale).astype(int)
	elif ('float' in dtype.name):
		array = ((array /scale).round() *scale).round(decimals)
	return Counter(array)

