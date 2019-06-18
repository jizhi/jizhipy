
def Diag( a=None, b=None, shape=None, n=None ) : 
	'''
	(1) Diag(a) : 
			return np.diag(a)
			a can be 2D or 1D
	(2) Diag(a, b) : replace diag of a with b(1D), the return new a
	(3) Diag(shape=(n1, n2)) : return index (when flatten) of diag

	n:
		int, left (-n) or right (+n) or center (n=0) of diag
	'''
	import numpy as np
	from jizhipy.Basic import IsType
	if (n is None) : n = 0
	def Idx( a, shape, n ) : 
		if (a is not None) : 
			a = np.array(a)
			shape = a.shape
			element = True
		else : 
			if (IsType.isnum(shape)) : shape = (shape, shape)
			element = False
		shape = np.array(shape, int)
		idx = np.arange(shape[0]*shape[1]).reshape(shape)
		if (shape[0] > shape[1]) : idx[:shape[1]-shape[0]]
		elif (shape[0] < shape[1]) : idx[:shape[0]-shape[1]]
		if (n > 0) : idx = idx[:-n,n:]
		elif (n < 0) : idx = idx[-n:,:n]
		idx = np.diag(idx)
		if (element) : idx = a.flatten()[idx]
		return idx
	#----------------------------------------
	if (a is not None) : 
		a = np.array(a)
		if (len(a.shape) == 1) : 
			a = np.diag(a)
			if (n > 0) : a = np.append(np.zeros((len(a), n), a.dtype), a, 1)
			elif (n < 0) : a = np.append(np.zeros((-n, len(a)), a.dtype), a, 0)
			return a
	#----------------------------------------
	if (b is None) : return Idx(a, shape, n)
	else : 
		a = np.array(a)
		shape = a.shape
		idx = Idx(None, shape, n)
		a = a.flatten()
		N = min(len(idx), len(b))
		a[idx[:N]] = b[:N]
		a = a.reshape(shape)
		return a
