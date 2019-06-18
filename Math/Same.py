
def Same( array, err=0 ) : 
	'''
	Find the same elements in the array

	array:
		int or float array, not complex

	err:
		Set the error/difference between two elements that will be considered as the same
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	a, err = np.sort(array), abs(err)
	n = abs(a[1:] - a[:-1])
	n = np.where(n<=err)[0]
	a = a[n]
	if (a.size == 0) : return np.array([])
	v, vm, n = [a[0]], [[a[0]]], [2]
	if (len(a) == 1) : vm[-1] = npfmt(vm[-1]).mean()
	for i in range(1, len(a)) : 
		if (a[i]-v[-1] > err) : 
			v.append(a[i])
			vm[-1] = npfmt(vm[-1]).mean()
			vm.append([a[i]])
			n.append(2)
			if (i == len(a)-1) : vm[-1] = npfmt(vm[-1]).mean()
		else : 
			v[-1] = a[i]
			vm[-1].append(a[i])
			n[-1] +=1
			if (i == len(a)-1) : vm[-1] = npfmt(vm[-1]).mean()
	v = np.concatenate([[vm],[n]]).T
	if ((v[:,0]-v[:,0].astype(int)).sum() == 0) : 
		v = v.astype(int)
	return v

