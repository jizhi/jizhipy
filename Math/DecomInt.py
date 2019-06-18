
def DecomInt( cls, n ) : 
	'''
	n = int1 x int2
	find int1 and int2

	return:
		[int1, int2],  .shape=(2, N)
	'''
	import numpy as np
	n = int(round(n))
	sign = 1 if(n>=0)else -1
	if (n in [-1, 0, 1]) : 
		return np.array([[1], [n]])
	n = abs(n)
	a = np.arange(1, n)
	b = n / a
	m = a * b
	c = n - m
	a, b = a[c==0], b[c==0]
	ab = np.array([a, b])[:,:a.size/2+1][:,::-1]
	ab[1] *= sign
	return ab
