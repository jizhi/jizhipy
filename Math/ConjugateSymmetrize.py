
def ConjugateSymmetrize( array ) : 
	'''
	This function is used to symmetrize the array which will be used to do the inverse Fourier transform.

	array is a N dimension array (real matrix or complex matrix), N can just be 1,2,3
	'''
	from jizhipy.Basic import Raise
	shape = np.array(array.shape) % 2
	if (shape.max() != 0) : Raise(Exception, 'Length of each dimension of input array must be even')
	func = len(array.shape)
	if (func > 3) : Raise(Exception, 'dimension of array must be 1D or 2D or 3D')

	def conjugate_symmetrize_axis( a ) : 
		a = np.append(a[:a.size/2+1], a[1:a.size/2][::-1].conjugate())
		a[a.size/2] = 0
		return a
	
	def conjugate_symmetrize_plane( a ) : 
		# x and y axis
		x = a[0,:]
		y = a[:,0]
		a[0,:] = conjugate_symmetrize_axis( x )
		a[:,0] = conjugate_symmetrize_axis( y )
		# xy plane exclude x and y axis
		xy = a[1:y.size/2+1,1:]
		a[1:,1:] = np.append(xy, xy[:-1][::-1,::-1].conjugate(), 0)
		# vertical medium axis
		axisp = a[y.size/2]
		a[y.size/2] = conjugate_symmetrize_axis( axisp )
		return a
	
	def conjugate_symmetrize_solid( a ) : 
		# x, y, z axis
		x = a[0,0,:]
		y = a[0,:,0]
		z = a[:,0,0]
		a[0,0,:] = conjugate_symmetrize_axis( x )
		a[0,:,0] = conjugate_symmetrize_axis( y )
		a[:,0,0] = conjugate_symmetrize_axis( z )
		# xy, xz, yz planes
		xy = a[0]
		xz = a[:,0]
		yz = a[:,:,0]
		a[0]     = conjugate_symmetrize_plane( xy )
		a[:,0]   = conjugate_symmetrize_plane( xz )
		a[:,:,0] = conjugate_symmetrize_plane( yz )
		# solid
		xyz = a[1:len(a)/2+1,1:,1:]
		a[1:,1:,1:] = np.append( xyz, xyz[:-1][::-1,::-1,::-1].conjugate(), 0 )
		# medium plane
		mplane = a[len(a)/2]
		a[len(a)/2] = conjugate_symmetrize_plane( mplane )
		return a

	if ( func == 1 ) : 
		array = conjugate_symmetrize_axis( array )
		array[0] = 0
	elif ( func == 2 ) : 
		array = conjugate_symmetrize_plane( array )
		array[0,0] = 0
	elif ( func == 3 ) : 
		array = conjugate_symmetrize_solid( array )
		array[0,0,0] = 0
	return array


