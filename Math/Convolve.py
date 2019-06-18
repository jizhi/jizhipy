
def _Multiprocess_Convolve2Matrix( iterable ) : 
	import numpy as np
	n = iterable[1]
	b1, Nrow, Ncol = iterable[2]
	B = np.zeros([n.size, Nrow*Ncol], b1.dtype)
	for k in range(n.size) : 
		ni = n[k] / Ncol
		nj = n[k] - ni*Ncol
		i, j = Nrow-ni, Ncol-nj
		b = b1[i:i+Nrow, j:j+Ncol]
		B[k] = b.flatten()
	return B





class Convolve( object ) : 


	def Convolve( self, source, beam, edge='mirror' ) : 
		'''
		This function is used to convolve two nd-array with same shape (source.shape==beam.shape).
	
		beam:
			You can normalize it to beam.sum()=1 or beam.max()=1 or not normalize depending on the situations.
		
		edge: 
			How to handle the edge.
			(1) edge = 'linear', convolve directly.
			(2) edge = float (0, 1, source.min() and so on), outside of the source will be treated as this float value.
			(3) edge = 'mirror', the edge as a mirror.
		'''
		from jizhipy.Basic import Raise
		import scipy.fftpack as spfft
		import numpy as np
		from jizhipy.Array import Asarray

		#dofft   = np.fft.fft
		#dofft2  = np.fft.fft2
		#doifft  = np.fft.ifft
		#doifft2 = np.fft.ifft2
		#dofftshift = np.fft.fftshift
		dofft   = spfft.fft
		dofft2  = spfft.fft2
		doifft  = spfft.ifft
		doifft2 = spfft.ifft2
		dofftshift = spfft.fftshift
		#----------------------------------------
		source, beam = Asarray(source), Asarray(beam)
		shape, dim = source.shape, len(source.shape)
		edge = str(edge).lower()
		if (source.shape != beam.shape) : 
			Raise(Exception, 'source.shape='+str(source.shape)+' != beam.shape='+str(beam.shape))
		if (dim > 3) : Raise(Exception, 'source/beam is '+str(dim)+'D, but now can just handle <=3D. You can modify this function yourself for >=4D')
	
		def fft( x ) : 
			if   (dim == 1) : y = dofft( x ) 
			elif (dim == 2) : y = dofft2( x ) 
			else : y = dofftn( x ) 
			return y
	
		def ifft( x ) : 
			if   (dim == 1) : y = doifft( x ) 
			elif (dim == 2) : y = doifft2( x ) 
			else : y = doifftn( x ) 
			return y
	
		# npix should be even when convolve (FFT)
		# Modify here for >=4D
		if (shape[0]%2 == 1) : 
			source = np.append(source, source[-1:], 0)
			beam   = np.append(  beam,   beam[-1:], 0)
		if (dim >= 2) : 
			if (shape[1]%2 == 1) : 
				source = np.append(source, source[:,-1:], 1)
				beam   = np.append(  beam,   beam[:,-1:], 1)
		if (dim >= 3) : 
			if (shape[2]%2 == 1) : 
				source = np.append(source, source[:,:,-1:], 2)
				beam   = np.append(  beam,   beam[:,:,-1:], 2)
	
		if (edge == 'linear') : 
			image = dofftshift(ifft(fft(source)*fft(beam))).real
		else :
			if (edge == 'mirror') : v, edge = 1, 0
			else : v, edge = 0, float(edge)
			
			if (dim == 1) : 
				n0 = source.shape[0]
				n1 = int(round(n0/2.))
				source1 = source[::-1] *v+edge
				source = np.append(source1, source)
				source = np.append(source, source1)
				source = source[n0-n1:n1-n0]
				source1 = 0 #@
				# For beam, always add 0
				beam1 = np.zeros([n0+2*n1,], beam.dtype)
				beam1[n0-n1:n1-n0] = beam
				beam = beam1
				image =dofftshift(ifft(fft(source)*fft(beam))).real
				image = image[n1:-n1]
	
			elif (dim == 2) : 
				n0 = Asarray(source.shape)
				n1 = (n0/2.).round().astype(int)
				source1 = source[:,::-1] *v+edge
				source = np.append(source1, source, 1)
				source = np.append(source, source1, 1)
				source1 = source[::-1]
				source = np.append(source1, source, 0)
				source = np.append(source, source1, 0)
				source = source[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1]]
				source1 = 0 #@
				# For beam, always add 0
				beam1 = np.zeros(n0+2*n1, beam.dtype)
				beam1[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1]] = beam
				beam = beam1
				image =dofftshift(ifft(fft(source)*fft(beam))).real
				image = image[n1[0]:-n1[0],n1[1]:-n1[1]]
	
			elif (dim == 3) : 
				n0 = Asarray(source.shape)
				n1 = (n0/2.).round().astype(int)
				# axis 0
				source1 = source[::-1] *v+edge
				source = np.append(source1, source, 0)
				source = np.append(source, source1, 0)
				# axis 1
				source1 = source[:,::-1]
				source = np.append(source1, source, 1)
				source = np.append(source, source1, 1)
				# axis 2
				source1 = source[:,:,::-1]
				source = np.append(source1, source, 2)
				source = np.append(source, source1, 2)
				# result
				source = source[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1],n0[2]-n1[2]:n1[2]-n0[2]]
				source1 = 0 #@
				# For beam, always add 0
				beam1 = np.zeros(n0+2*n1, beam.dtype)
				beam1[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1],n0[2]-n1[2]:n1[2]-n0[2]] = beam
				beam = beam1
				image =dofftshift(ifft(fft(source)*fft(beam))).real
				image=image[n1[0]:-n1[0],n1[1]:-n1[1],n1[2]:-n1[2]]
	
		if   (dim == 1) : image = image[:shape[0]]
		elif (dim == 2) : image = image[:shape[0],:shape[1]]
		elif (dim == 3) : image = image[:shape[0],:shape[1],:shape[2]]
		return image
	
	
	
	
	
	def ConvolveHealpix( self, a, b, lmax=None ) : 
		'''
		a:
			can be any healpix map
	
		b:
			b must be symmetry with z-axis
			it means that b is just the function of theta
			b(theta)
		'''
		import numpy as np
		import healpy as hp
		from jizhipy.Basic import Raise
		from jizhipy.Transform import Alm
		a, b = np.array(a), np.array(b)
		if (a.shape != b.shape) : Raise(Exception, 'a.shape='+str(a.shape)+' != b.shape='+str(b.shape))
		nside = hp.get_nside(a)
		alm = hp.map2alm(a, lmax)
		blm = hp.map2alm(b, lmax)
		if (lmax is None) : lmax = Alm.Get(size=alm.size)
		l, m = Alm.Get(lmax)
		for i in range(lmax+1) : blm[l==i] = blm[i]
		clm = (4*np.pi / (2*l+1))**0.5 * alm * blm
		c = hp.alm2map(clm, nside, verbose=False)
		c = c / c.sum() * a.sum() * b.sum()
		return c
	
	
	
	

	def Convolve2Matrix( self, b ) : 
		'''
		a: 2D map
		b: 2D beam
	
		convolve  : 
			return Convolve(a, b)
	
		matrix dot: 
			a1 = MatrixDot(B, a.flatten()[:,None])
			return a1.reshape(a.shape)
	
		This function return B matrix
		'''
		Nprocess = 1 # Nprocess=1 is fastest, NOT use PoolFor
		import numpy as np
		from jizhipy.Process import PoolFor, NprocessCPU
		b = 1.*np.array(b)
		Nrow, Ncol = b.shape
		b1 = np.zeros([2*Nrow, 2*Ncol], b.dtype)
		b1[Nrow/2:Nrow/2+Nrow, Ncol/2:Ncol/2+Ncol] = b
		#----------------------------------------
		Nprocess = NprocessCPU(Nprocess)[0]
		n = np.arange(Nrow*Ncol)
		send  = n
		bcast = (b1, Nrow, Ncol)
		#----------------------------------------
		if (Nprocess <= 1) : 
			iterable = (None, send, bcast)
			B = _Multiprocess_Convolve2Matrix(iterable)
		else : 
			pool = PoolFor(0, n.size, Nprocess)
			B = pool.map_async(_Multiprocess_Convolve2Matrix, send, bcast)
			B = np.concatenate(B, 0)
		return B






Convolve = Convolve()
