
def Edge2Center( array ) : 
	'''
	array is the edges, return the centers
	array can be non-uniform.
	'''
	from jizhipy.Array import Asarray
	array = Asarray(array)
	if (len(array.shape) != 1) : Raise(Exception, 'array must be 1D')
	step = array[1:] - array[:-1]
	return array[:-1] + step/2.



def Center2Edge( array ) : 
	'''
	array is the centers, return the edges
	array can be non-uniform
	'''
	from jizhipy.Array import Asarray
	array = Asarray(array)
	if (len(array.shape) != 1) : Raise(Exception, 'array must be 1D')
	step = array[1:] - array[:-1]
	n = array.size
	if ((1.*step.max()-step.min())/step.max() < 0.01) : 
		edge = np.zeros([n+1,]) + array[n-1]+step.mean()/2.
		edge[:n] = array - step.mean()/2.
		return edge
	else : 
		edge = np.zeros([n-2, n+1])
		m, j = n/2, 0
		for m in range(1, n-1) : 
			edge[j,m] = array[m] - (step[m]+step[m-1])/4.
			for i in range(m-1, -1, -1) : 
				edge[j,i] = 2*array[i] - edge[j,i+1]
			for i in range(m+1, n+1) : 
				edge[j,i] = 2*array[i-1] - edge[j,i-1]
			j = j + 1
		return edge.mean(0)



