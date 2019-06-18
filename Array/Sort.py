
def Sort( array, along='[0,:]', l2s=False ) : 
	'''
	array:
		Can be any shape

	along:
		Must as format like '[n1,n2,:,n4]'
		Must have ':', use [2,:] instead of [2]
		array[n1,n2,:,n4] must 1D so that we can sort along this
		along=[:,2] : second column

		'[0,:]' => 0-row
		'[:,0]' => 0-column

	l2s: 
		l2s=False: from small to large (default)
		l2s=True : from large to small
	'''
	import numpy as np
	from jizhipy.Array import Asarray, ArrayAxis
	along = along[1:-1].split(',')
	axis = along.index(':')
	along.pop(axis)
	along = np.array(along, int)
	#--------------------------------------------------
	array = Asarray(array)
	if (len(array.shape) == 1) : 
		array = np.sort(array)
		if (l2s) : array = array[::-1]
		return array
	#--------------------------------------------------
	if (array.shape[axis] == 1) : return array
	array = ArrayAxis(array, axis, -1, 'move')
	shape = array.shape
	#--------------------------------------------------
	cumprod = np.cumprod((shape[1:-1]+(1,))[::-1])[::-1]
	along = (along*cumprod).sum()
	a = array.reshape(np.prod(shape[:-1]), shape[-1])[along]
	#--------------------------------------------------
	a = a + 1j*np.arange(a.size)
	a = np.sort(a).imag.astype(int)
	if (l2s) : a = a[::-1]
	#--------------------------------------------------
	array = ArrayAxis(array, -1, 0, 'move')
	array = array[a]
	array = ArrayAxis(array, 0, axis, 'move')
	return array





def SortFilename( string ) : 
	'''
	a is any type of list or array.
	Take the first 'whole' number, and sort.
	For example:
		a = ['a12', 'b2-3', 'c35', 'd7_0', 'e4']
		SortStrNum(a) = ['b2-3', 'e4', 'd7_0', 'a12', 'c35']
		first 'whole' numbers are 2, 4, 7, 12, 35

	This function is useful for sort filename
	a = [23, 100, 4, 'xy', 'abc0', 'a12', 'b2-3', 'c35', 'd7_0', 'e4']
	'''
	import numpy as np
	n = []
	astr = np.array(a, str)
	for i in range(len(astr)) : 
		b = astr[i]
		n1, n2 = len(b), len(b)
		for j in range(len(b)) : 
			if (48 <= ord(b[j]) <= 57) : 
				n1 = j
				break
		for j in range(n1+1, len(b)) : 
			if (48 > ord(b[j]) or ord(b[j]) > 57) : 
				n2 = j
				break
		if (n1 == len(b)) : n = n + [-1]
		else : n = n + [int(b[n1:n2])]
	n = np.array(n)
	n[n<0] = n.max()+1
	nmax = n.max()
	n = np.append([n], [np.arange(len(astr))], 0)
	n = Sort(n)
	n1 = -1
	for i in range(len(n[0])) : 
		if (n[0,i] == nmax) : 
			n1 = i
			break
	n = n[1]
	astr = []
	for i in range(len(a)) : 
		astr = astr + [a[n[i]]]
	if (n1 != -1) : 
		astr = astr[:n1] + list(np.sort(astr[n1:]))
	return astr

