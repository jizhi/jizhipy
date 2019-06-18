
def SynIndex( freq, freq0=408 ) : 
	'''
	freq:
		float | list  (MHz)

	freq0:
		reference frequency in MHz, default freq0=408

	beta = np.array([(2.482,0.044), (2.477,0.036), (2.533,0.058), (2.501,0.079), (2.528,0.102), (2.691,0.121), (2.718,0.155), (3.095,0.096), (3.097,0.088), (3.088,0.084), (3.092,0.077), (3.077,0.071)])
	freq = np.array([10,22,45,85,150,1420,2300,23000,33000,41000,61000,94000]) /408.
	'''
	import numpy as np
	p = np.array([2.54585438, 5.99296288e-02, 3.32505876e-02, 2.57045977e-03, -1.69742991e-03, -9.13846215e-05, 2.87044311e-05])
	freq = np.array(freq, float)
	if (freq0 is None) : freq0 = 408.
	freq0 = float(freq0)
	x = np.log(freq / freq0)
	synindex = p[0] + np.zeros(x.size)
	for i in range(1, len(p)) : synindex += p[i] * x**i
	return synindex
