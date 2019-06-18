
def Arcsinhspace( a, b, center=None, nbin=20 ) : 
	'''
	This function is generally used to get the bins when calculate the probability density.

	a, b:
		are normal value, not arcsinh !!!
		Usually the min() and max() of the sample which is used to calculate the probability density.

	center: 
		The center of the sample. Default center=(a+b)/2

	The bin size will be smaller when it closes to the center.

	Return: 
		Also normal value, but the interval is in arcsinh
	'''
	import numpy as np
	if (center is None) : center = (a+b)/2.
	a, b = np.arcsinh(a-center), np.arcsinh(b-center)
	arcsinhspace = np.sinh(np.linspace(a, b, nbin)) + center
	return arcsinhspace



