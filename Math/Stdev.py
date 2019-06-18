
def Stdev( a, axis=None ) : 
	'''
	Standard Deviation 

	If axis == None, return sigma of the whole a.
	If axis == int, calculate sigma along this axis.

	Also, see Smooth()
	'''
	if (axis is None) : 
		stda = (((a - a.mean())**2).sum()/a.size)**0.5
	else : 
		shape = np.array(a.shape)
		shape[axis] = 1
		stda = (((a - a.mean(axis).reshape(shape))**2).sum(axis)/(a.shape[axis]))**0.5
	return stda
