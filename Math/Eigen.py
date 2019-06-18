
def Eigen( cls, a, l2s=False ) : 
	'''
	l2s:
		True: eigenvalue from large to small
		False: Default, small to large
	'''
	import scipy.linalg as spla
	v, P = spla.eigh(a)
	if (l2s) : v, P = v[::-1], P[:,::-1]
	return (v, P)
