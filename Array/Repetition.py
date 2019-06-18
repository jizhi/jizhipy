
def Repetition( array, err=0, renon=False ) : 
	'''
	Find the same elements in the array

	array:
		int/float array
		if is complex array, do with its real part
		array will be flattened first

	err:
		Set the error/difference between two elements that will be considered as the same

	renon:
		True | False
		if ==True: return non-repetitive value with 1D shape

	return:
		(1) renon == False:
				return v
					where v is 2D-array 
						 col-0     col-1
						[value, total_count]
		(2) renon == True: 
				return [v, a]
				where a is 1D non-repetitive value
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	a, err = np.sort(array), abs(err)
	n = abs(a[1:] - a[:-1])
	n = np.where(n<=err)[0]
	a = a[n]
	if (a.size == 0) : 
		v = np.array([], int)
		if (not renon) : return v
		else : return [v, Asarray(array).flatten()]
	#--------------------------------------------------
	v, vm, n = [a[0]], [[a[0]]], [2]
	if (len(a) == 1) : vm[-1] = Asarray(vm[-1]).mean()
	for i in range(1, len(a)) : 
		if (a[i]-v[-1] > err) : 
			v.append(a[i])
			vm[-1] = Asarray(vm[-1]).mean()
			vm.append([a[i]])
			n.append(2)
			if (i == len(a)-1) : vm[-1] = Asarray(vm[-1]).mean()
		else : 
			v[-1] = a[i]
			vm[-1].append(a[i])
			n[-1] +=1
			if (i == len(a)-1) : vm[-1] = Asarray(vm[-1]).mean()
	v = np.concatenate([[vm],[n]]).T
	if ((v[:,0]-v[:,0].astype(int)).sum() == 0) : 
		v = v.astype(int)
	#--------------------------------------------------
	if (not renon) : return v
	else : 
		a = Asarray(array).flatten()
		b = v[:,0]
		tf = np.ones(a.size, bool)
		for i in range(b.size) : 
			tfi = (a != b[i])
			n = np.where(tfi==False)[0][0]
			tfi[n] = True
			tf *= tfi
		a = a[tf]
		return [v, a]


