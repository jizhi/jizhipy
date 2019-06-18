
class PCA( object ) :  


	def __init__( self, ymap=None, stagger=0, verbose=False ):
		'''
		'''
		self.verbose = bool(verbose)
		if (self.verbose) : print('jizhipy.PCA:')
		self.Ymap(ymap)
		try : self.stagger = int(stagger)
		except : self.stagger = 0
		if (self.verbose) : print('    self.stagger =', self.stagger)





	def Stagger( self, stagger=None ) : 
		'''
		stagger:
			For axis=1 (pixel axis), NOT axis=0 (freq axis)
			None(means =0) | >=0 int
			*** Assuming the foreground changes smoothly along pixels, but the noise is random independently, stagger 1 pixel can improve the result a lot (maybe).
		'''
		if (stagger is None) : return self.stagger
		self.stagger = int(stagger)
		if (self.verbose) : print('    Change: self.stagger =', self.stagger)
		return self.stagger





	def Ymap( self, ymap=None ) : 
		'''
		ymap: 
			Input y map, must be 2D array/matrix.
			y.shape = (nfreq, npix), row nfreq is the number of frequencies, column npix is the number of pixels at its frequency.
			(1) y is ndarray
			(2) y is list, len(y)==nfreq
		'''
		import numpy as np
		from jizhipy.Basic import Raise, SysFrame, IsType
		from jizhipy.Array import Asarray
		from jizhipy.Optimize import Smooth
		if (ymap is None) : 
			if (SysFrame(0,2)[-3][-2]!='') : 
				self.ymap = None
				if (self.verbose) : print('    self.y = None')
				return
			else : return self.ymap
		pstr = '    ' if ('ymap' not in self.__dict__.keys())else '    Change: ' 
		if (IsType.islist(ymap) or IsType.istuple(ymap)) : 
			if (not IsType.isnum(ymap[0])) : 
				ymap, N = list(ymap), 1e15
				for i in range(len(ymap)) : 
					if (len(ymap) < N) : N = len(ymap)
				for i in range(len(ymap)) : 
					ymap[i]=jp.Smooth(ymap[i], 0,reduceshape=N)
		self.ymap = Asarray(ymap, np.float64)  # force bit64
		if (len(self.ymap.shape) not in [1, 2]) : Raise(Exception, 'jizhipy.PCA: ymap must be 1D/2D, but now ymap.shape='+str(self.ymap.shape))
		if (self.verbose) : print(pstr+'self.ymap.shape =', self.ymap.shape)
		return self.ymap





	def EigenVector( self, P=None ) : 
		'''
		P:
			Each "column" of P is one eigenvector corresponding to one eigenvalue
			P can be non-square
		'''
		if (P is None) : return self.P
		pstr = '    ' if ('P' not in self.__dict__.keys())else '    Change: ' 
		self.P = P
		if (self.verbose) : print(pstr+'self.P.shape =', self.P.shape)





	def Eigen( self ) : 
		'''
		return [eigenvalue, eigenvector, correlation_matrix]
		'''
		if (len(self.ymap.shape) != 2) : 
			print('    Warning: self.ymap is 1D, NOT 2D')
			return (None, None)
		import scipy.linalg as spla
	#	# rms
	#	rms = np.diag(R)**0.5
	#	# correlation matrix
	#	# also R = C / ( rms[:,None] * rms[None,:] )
	#	R /= rms[:,None]
	#	R /= rms[None,:]
		#---------------------------------------------
		# eigenvalue decomposition for the R 
		if (self.verbose) : print('=> eigh(R)')
		v, P = spla.eigh( self.R )
		self.v = v[::-1]    # we need from large to small
		self.P = P[:,::-1]  # default from small to large
		if (self.verbose) : 
			print('        self.v.shape =', v.shape)
			print('        self.P.shape =', P.shape)
		return (self.v, self.P)





	def Remap( self, pc ) : 
		'''
		pc:
			starting from 0, NOT 1: the first PC is pc=0
			(1) pc=2, int: select the 3rd PC
			(2) pc=':2': select the front 2 PCs
			(3) pc='2:': select the PCs from 3rd(include)
			(4) pc=None/'all': select all PCs

		return:
			ndarray, whose shape=(Nf, Npix)
	
		Using cmap = plt_cmap('gist_rainbow_r','w') is batter
		'''
		import numpy as np
		from jizhipy.Math import MatrixDot
		from jizhipy.Basic import IsType, Path
		if (pc is None) : self.pc = None
		else : self.pc = str(pc)
		pc = str(pc).split(':')
		if (len(pc) == 1) : pc += ['one']
		#----------------------------------------
		if (pc[0] == '') : # pc=':n'
			n = int(round(float(pc[1])))
			if (n == len(self.v)) : 
				self.pc = None
				self.remap = self.ymap
			else : 
				self.remap = MatrixDot( np.dot(self.P[:,:n], self.P[:,:n].T), self.ymap, 2, None)
		#----------------------------------------
		elif (pc[1] == '') :  # pc='n:'
			n = int(round(float(pc[0])))
			if (n == 0) : 
				self.pc = None
				self.remap = self.ymap
			else : 
				self.remap = MatrixDot( np.dot(self.P[:,n:], self.P[:,n:].T), self.ymap, 2, None)
		#----------------------------------------
		elif (pc[0].lower() in ['none', 'all']) : 
			self.remap = self.ymap
		#----------------------------------------
		else : # pc='n'
			n = int(round(float(pc[0])))
			self.remap = MatrixDot( np.dot(self.P[:,n:n+1], self.P[:,n:n+1].T), self.ymap, 2, None)
		#----------------------------------------
		if (not (self.stagger is None or self.stagger==0)) : 
			self.remap = np.concatenate( [self.repcmap] + stagger*[self.repcmap[:,-1:]], 1 )
		if (self.verbose) : 
			pstr = 'None' if (self.pc is None)else '['+self.pc+']'
			print('    self.pc = '+pstr)
			print('    self.remap.shape =', self.remap.shape)
		return self.remap





	def Save( self, hdf5name ) : 
		'''
		hdf5name:
			(1) ==None/'': hdf5name = 'jizhipy.PCA.hdf5'
			(2) ==filename of .hdf5
		'''
		from jizhipy.Basic import Path
		import h5py
		outname = hdf5name
		if(outname in [None, '']): outname ='jizhipy.PCA.hdf5'
		else : 
			if (outname[-5:] != '.hdf5') : outname += '.hdf5'
			outname = Path.AbsPath(outname)
			Path.ExistsPath(outname, True)
			fo = h5py.File(outname, 'w')
		if (self.verbose) : print('    Saving to: ', Path.AbsPath(outname, True))
		self.hdf5name = outname
		keys = self.__dict__.keys()
		fo['class'] = 'jizhipy.PCA'
		fo['stagger'] = stagger
		fo['stagger'].attrs['comment'] = 'y1, y2 = y[:,:-stagger], y[:,stagger:]'
		if ('v' in keys) : 
			fo['eigenvalue'] = v
			fo['eigenvalue'].attrs['comment']='eign values of R matrix'
		if ('P' in keys) : 
			fo['eigenvector'] = P
			fo['eigenvector'].attrs['comment'] = 'each columns are eigen vectors of R matrix'
	#	if ('a' in keys) : 
	#		fo['a'] = a
	#		fo['a'].attrs['comment'] = 'z=P*a, y=(P*a)*rms'
	#	if ('rms' in keys) : 
	#		fo['rms'] = rms
	#		fo['rms'].attrs['comment'] = 'rms of each row of y-map'
		fo.close()
		return self.hdf5name





	def Read( self, hdf5name ) : 
		'''
		Return  instance of jp.PCA
		'''
		from jizhipy.Basic import Path
		import h5py
		hdf5name = Path.AbsPath(hdf5name)
		if (self.verbose) : print('    Reading from: ', Path.AbsPath(hdf5name, True))
		fo = h5py.File(hdf5name, 'r')
		keys = fo.keys()
		pca = PCA()
		try : 
			if (fo['class'] != 'jizhipy.PCA') : raise
		except : 
			if (self.verbose) : print('        NOT .hdf5 for class: jizhipy.PCA')
			return pca
		if ('stagger' in keys) : self.stagger = fo['stagger'].value
		if ('eigenvalue' in keys) : self.v = fo['eigenvalue'].value
		if ('eigenvector' in keys) : self.P = fo['eigenvector'].value
	#	if ('a' in keys) : self.a = fo['a'].value
	#	if ('rms' in keys) : self.rms = fo['rms'].value
		fo.close()
		return pca





	def  Corr( self ) : 
		'''
		Return the correlation
		self.ymap is 1D:
			1/(N-m) * \sum_{n=0}^{N-1} x[n]*x[n+m] = Convolve(x, y), y=-x[::-1]? y=-x?

		self.ymap is 2D:
			1/x.shape[1] * matrix(x) * matrix(x).T
		'''
		from jizhipy.Math import Convolve, MatrixDot
		if (len(self.ymap.shape) == 1) : 
		#	self.R = jp.Convolve(self.ymap, -self.ymap[::-1], 'linear') / self.ymap.size
		#	self.R = Convolve(self.ymap, -self.ymap, 'linear') / self.ymap.size
			self.R = Convolve(self.ymap, self.ymap, 'linear') / self.ymap.size
		#---------------------------------------------
		else : 
			# stagger y array
			if(self.stagger==0): y1,y2 =self.ymap, self.ymap
			else: y1, y2 = self.ymap[:,:-self.stagger], self.ymap[:,self.stagger:]
			# get pixel number
			nfreq, npix = y1.shape
			# covariance matrix C = y1 * y2.T / npix
			if (self.verbose) : print('    => R',)
			self.R = MatrixDot(y1, y2.T, 1, None) / npix
		return self.R


