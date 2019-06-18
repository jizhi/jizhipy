
def _DoMultiprocess_MatrixDot( iterable ) : 
	import numpy as np
	if (iterable[2][-1] == 1) : 
		a, b = iterable[1][0], iterable[2][0]
	elif (iterable[2][-1] == 2) : 
		a, b = iterable[2][0], iterable[1][0].T
	elif (iterable[2][-1] == 3) : 
		a, b = iterable[1][0].T, iterable[1][1]
	m = np.dot(a, b)
	return m



def _MatrixDot( a, b, Nprocess ) : 
	'''
	'''
	import numpy as np
	from jizhipy.Process import PoolFor, NprocessCPU
	if (a.shape[0] >= a.shape[1]) : 
		if (a.shape[0] >= b.shape[1]) :  # split a
			case, send, bcast = 1, [a], [b, 1]
		else :  # b.T and split
			case, send, bcast = 2, [b.T], [a, 2]
	else :  # a.T and split a and b
		case, send, bcast = 3, [a.T, b], [3]
	#---------------------------------------------
	pool = PoolFor(0, len(send[0]), Nprocess)
	m = pool.map_async(_DoMultiprocess_MatrixDot, send, bcast)
	#---------------------------------------------
	if (case == 1) : m = np.concatenate(m, 0)
	elif (case == 2) : m = np.concatenate(m, 1)
	elif (case == 3) : 
		for i in range(1, len(m)) : m[0] += m[i]
		m = m[0]
	return m




def MatrixDot( *args, **kwargs ) : 
	'''
	MatrixDot(a, b)
		(1) a.shape[0] >= b.shape[1]:
				Matrix(a, b) - np.dot(a, b) == 0.0
		(2) a.shape[0] < b.shape[1]:
				Matrix(a, b) - np.dot(a, b) == 1e-10 > 0.0

	kwargs:
		'Nprocess'

	Usage:
	(1) MatrixDot(a, b, c, d, e, Nprocess=12)
	(2) MatrixDot([a, b, c, d, e], Nprocess=12)

	NOT THAT all matries don't support broadcast !!!
	Shapes of all matries must be able to matrix dot !!!
	'''
	import numpy as np
	from jizhipy.Basic import IsType
	from jizhipy.Process import NprocessCPU
	if ( len(args) == 1) : 
		if (IsType.isndarray(args[0]) or IsType.ismatrix(args[0])) : return np.array(args[0])
		else : args = args[0]
	#----------------------------------------
	args = list(args)
	for i in range(len(args)) : args[i] = np.array(args[i])
	if (len(args) == 1) : return args[0]
	#----------------------------------------
	try : Nprocess = kwargs['Nprocess']
	except: Nprocess = None
	Nprocess = NprocessCPU(Nprocess)[0]
	#----------------------------------------
	if (Nprocess <= 1) : 
		a = np.dot(args[0], args[1])
		for i in range(2, len(args)) : 
			a = np.dot(a, args[i])
		return a
	#----------------------------------------
	while (len(args) >= 2) : 
		size = np.zeros(len(args)-1, int)
		for i in range(size.size) : 
			size[i] = args[i].shape[0] * args[i+1].shape[1]
		n = np.where(size==size.min())[0][0]
		a = _MatrixDot(args[n], args[n+1], Nprocess)
		args = args[:n] + [a] + args[n+2:]
	return args[0]





if (__name__ == '__main__') : 
	import numpy as np
	import jizhipy as jp
	
	a = 100+np.random.random((100,123456))
	b = a.T
	#a = 100+np.random.random((90,100))
	#b = 100+np.random.random((100,123456))
	
	print(a.shape, b.shape)
	print(jp.Time(1))
	print('matrix')
	c1 = np.array(np.matrix(a) * np.matrix(b))
	
	print(jp.Time(1))
	print('jp.MatrixProd(1, None)')
	c2 = jp.MatrixProd(a, b, 1, None)
	
	print(jp.Time(1))
	print('jp.MatrixProd(2, None)')
	c3 = jp.MatrixProd(a, b, 2, None)
	print(jp.Time(1))
	
	c2 = abs(c2-c1)
	c3 = abs(c3-c1)
	print(c2.min(), c2.max(), c2.mean())
	print(c3.min(), c3.max(), c3.mean())
    
