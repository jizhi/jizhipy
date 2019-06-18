

def ArraySplit( array, axis, which ) : 
	'''	
	axis:
		split along which axis

	which:
		=='1G' | '1GB'
		==int
	'''
	import numpy as np
	from jizhipy.Array import Asarray, ArrayAxis, Repection
	from jizhipy.Basic import IsType
	array = Asarray(array)
	if (IsType.isint(which)) : mem, N = None, which
	elif (IsType.isstr(which)) : 
		which = str(which).lower()
		if ((which[-1]=='g' and which[-2] in '0123456789')) : mem = float(which[:-1])
		elif ((which[-2:]=='gb' and which[-3] in '0123456789')) : mem = float(which[:-2])  # GB
	else : 
		array = ArrayAxis(array, axis, 0, 'move')
		array = array[None,:]
		array = ArrayAxis(array, 0, axis+1, 'move')
		return array
	#--------------------------------------------------
	if (mem is not None) : 
		bit, n = array.dtype.name, 0
		while (bit[n] not in '0123456789') : n += 1
		bit = float(bit[n:])
		size = array.size
		memtot = size / 1e8 * 0.8 * bit/64
		N = int(memtot / mem) + 1
	m = np.linspace(0, array.shape[axis], N+1).astype(int)
	m = Repetition(m, renon=True)
	b = []
	array = ArrayAxis(array, axis, 0, 'move')
	for i in range(len(m)-1) : 
		a = ArrayAxis(array[m[i]:m[i+1]], 0, axis)
		b.append(a)
	return b
