
def ArrayGroup( array, step=1 ) : 
	'''
	Devide array into several groups basing on step

	array:
		Will first flatten() and Sort() (small to large)

	step:
		array[i]-array[i-1]<=step, we will consider array[i] and array[i-1] are in the same group

#	return:
#		[arraygroup, arrayrange]
#		arrayrange[:,0] (include), arrayrange[:,1] (exclude)

	return:
		[arraygroup, idx]
		idx[i] = (n1, n2), array[n1:n2+1] is group i
			NOTE THAT include n2
	'''
	import numpy as np
	from jizhipy.Asarray import Asarray
	array, group = np.sort(Asarray(array).flatten()), [0]
	for t in range(1, len(array)) : 
		if (abs(array[t]-array[t-1]) <= step) : 
			group.append(group[-1])
		else : group.append(group[-1]+1)
	group = np.array(group)
	arraygroup, arrayrange, idx = [], [], []
	n = np.arange(array.size)
	for t in range(group.max()+1) : 
		arraygroup.append(array[group==t])
		m = n[group==t]
		idx.append([m[0], m[-1]])
	#	arrayrange.append([arraygroup[-1][0],arraygroup[-1][-1]+1])
#	return [arraygroup, Asarray(arrayrange)]
	return [arraygroup, Asarray(idx)]


def ArrayGroup( array, step=1, include=True ) : 
	'''
	Devide array into several groups basing on step

	array:
		Will first flatten() and Sort() (small to large)

	step, include:
		include:
			True : <=
			False: <
		step:                   (include)
			array[i]-array[i-1]   <=/<    step, we will consider array[i] and array[i-1] are in the same group

#	return:
#		[arraygroup, arrayrange]
#		arrayrange[:,0] (include), arrayrange[:,1] (exclude)

	return:
		[arraygroup, idx]
		idx[i] = (n1, n2), array[n1:n2+1] is group i
			NOTE THAT include n2
	'''
	import numpy as np
	from jizhipy.Asarray import Asarray
	array = np.sort(Asarray(array).flatten())
	n, idx = 1, [0]
	while(n < len(array)) : 
		if (abs(array[n]-array[n-1]) > step) : idx.append(n)
		n += 1
	idx = np.array(idx+[len(array)])
	idx = np.array([idx[:-1], idx[1:]]).T
	idx[:,1] -= 1
	arraygroup = []
	for i in range(len(idx)) : 
		arraygroup.append(array[idx[i][0]:idx[i][1]+1])
	return [arraygroup, idx]


def ArrayGroup( array, step=1, include=False ) : 
	'''
	Devide array into several groups basing on step

	array:
		Will flatten(), but NOT Sort()

	step, include:
		include:
			True : <=
			False: <
		step:                   (include)
			array[i]-array[i-1]   <=/<    step, we will consider array[i] and array[i-1] are in the same group

	return:
		[idx, rep, crep, nrep, valid]
		idx == index of array, is list of ndarray like: 
			[np.array([3,5,2]), np.array([0]), np.array([1,4]), np.array[6]]

		nrep == non-repetition index, is 1d-int-array:
			np.array([0,6])

		rep == repetition index of array, is list of ndarray like: 
			[np.array([3,5,2]), np.array([1,4])]
		crep == center of each rep[i], is 1d-int-array:
			np.array([3,1])

		valid = nrep+crep => index without repetition

	Usage:
		b = array.flatten()
		b[idx[0]] is a group,  b[idx[1]] is an other group
		b[rep[0]] is a repetition, b[rep[1]] is an other repetition
		b[nrep] is all non-repetition
	'''
	import numpy as np
	from jizhipy.Asarray import Asarray
	array = Asarray(array).real.flatten()
	array = np.sort(array + 1j*np.arange(array.size))
	array, order = array.real, array.imag.astype(int)
	#---------------------------------------------
	n, idx = 1, [0]
	while(n < len(array)) : 
		if (include) : 
			if (abs(array[n]-array[n-1]) > step) : 
				idx.append(n)
		elif (not include) : 
			if (abs(array[n]-array[n-1]) >= step) : 
				idx.append(n)
		n += 1
	idx = np.array(idx+[len(array)])
	idx = np.array([idx[:-1], idx[1:]]).T
	idx[:,1] -= 1
	#---------------------------------------------
	ordergroup, rep, nrep, crep = [], [], [], []
	for i in range(len(idx)) : 
		n1, n2 = idx[i]
		odi = np.sort(order[n1:n2+1])
		ordergroup.append( odi )
		if (n1 != n2) : rep.append( odi )
		else : nrep.append( odi[0] )
	for i in range(len(rep)) : 
		a = array[rep[i]]
		a = np.sort(a + 1j*np.arange(a.size))
		a = int(a.imag[int(a.size/2)])
		crep.append(rep[i][a])
	nrep, crep = np.array(nrep,int), np.array(crep,int)
	valid = np.sort(np.concatenate([nrep, crep]))
	return [ordergroup, rep, crep, nrep, valid]

