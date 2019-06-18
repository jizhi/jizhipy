
class Random( object ) : 


	def Seed( self, seed ) : 
		'''
		set the random seed

		Parameters
		----------
		seed:
			[int] | 'time': use current time as seed
		'''
		import numpy as np
		if (str(seed).lower() =='time'):
			import time
			seed=int(('%.11f'% time.time()).split('.')[1][2:])
		np.random.seed(seed)
		return seed
	
	
	
	
	
	def RandomShuffle( self, a, axis=None, kFold=None ) : 
		'''
		Parameters
		----------
		a:
			(1) int: a = range(a)
			(2) ndarray with any shape, any dtype

		axis:
			[int], shuffle along which axis?
			axis=None: shuffle a.flatten() (all elements)

		kFold:
			None: don't split
			=[int]: split into xx pieces
				except (a is n-D array and axis=None)

		
		Returns
		----------
		randomly shuffle all elements
		'''
		if (axis is not None): axis = int(axis)
		if (kFold is not None): kFold = int(kFold)
		import numpy as np
		from jizhipy.Basic import IsType, Raise
		if (IsType.isint(a)) : a = np.arange(a)
		else : a = np.array(a)
		shape = a.shape
		if (len(shape) ==1): axis = None
		if (axis is not None): 
			if (axis <0): axis += len(shape)
			if (axis >= len(shape)): Raise(Exception, 'a.shape='+str(shape)+', but axis='+str(axis)+' exceeds the dimension')
			b = np.arange(shape[axis])
		else: b = np.arange(a.size)
		if (len(shape)>1 and axis is None and kFold is not None): Raise(Exception, 'when a.shape='+str(shape)+' is n-D array and axis=None, can NOT use kFold')
		#----------------------------------------
		b = np.random.random(b.size) + 1j*b
		b = np.sort(b).imag.astype(int)
		#----------------------------------------
		if (kFold is None): b = [b]
		else: 
			c, n=[], np.linspace(0,len(b),kFold+1).astype(int)
			for i in range(len(n)-1): c.append(b[n[i]:n[i+1]])
			n = np.random.random(len(c)) +1j*np.arange(len(c))
			n = np.sort(n).imag.astype(int)
			b = []
			for i in n: b.append(c[i])
		#----------------------------------------
		if (len(shape)>1 and axis is None): 
			return a.flatten()[b[0]].reshape(shape)
		if (len(shape)>1 and axis is not None and axis !=0): 
			tranaxis = range(len(shape))
			tranaxis[0], tranaxis[axis] = axis, 0
			a = np.transpose(a, tranaxis)
		for i in range(len(b)):
			b[i] = a[b[i]]
			if (axis is not None and axis !=0): 
				b[i] = np.transpose(b[i], tranaxis)
		if (kFold is None): b = b[0]
		return b

	def Shuffle( self, a, axis=None, kFold=None ) : 
		return self.RandomShuffle(a, axis, kFold)





	def Randn( self, *args ) : 
		'''
		(1) Randn(d0, d1, ..., dn, complex)
		(2) Randn( (d0,d1,...,dn) )
		'''
		from jizhipy.Basic import IsType
		import numpy as np
		if (len(args) == 0) : args = [1]
		if (IsType.isdtype(args[-1])) : 
			dtype = np.dtype(args[-1]).name
			args = args[:-1]
		else : dtype = 'float64'
		if (len(args) == 0) : args = 1
		elif (not IsType.isnum(args[0])) : args = args[0]
		a = np.random.standard_normal( args )
		if ('complex' not in dtype) : a = a.astype(dtype)
		else : a = (a + 1j*np.random.standard_normal( args )).astype(dtype)
		return a





	def Random( self, *args ) : 
		'''
		(1) Random(d0, d1, ..., dn)
		(2) Random( (d0,d1,...,dn) )

		return random value [0, 1)
		'''
		from jizhipy.Basic import IsType
		import numpy as np
		if (len(args) == 0) : args = [1]
		if (IsType.isdtype(args[-1])) : 
			dtype = np.dtype(args[-1]).name
			args = args[:-1]
		else : dtype = 'float64'
		if (len(args) == 0) : args = 1
		elif (not IsType.isnum(args[0])) : args = args[0]
		a = np.random.random( args )
		if ('complex' not in dtype) : a = a.astype(dtype)
		else : a = (a+1j*np.random.random(args)).astype(dtype)
		return a





	def RandomThetaPhi( self, *args ) : 
		'''
		(1) RandomThetaPhi(d0, d1, ..., dn)
		(2) RandomThetaPhi( (d0,d1,...,dn) )

		return 
			[theta, phi] in rad
			theta=0--np.pi
			phi=0--2*np.pi
			random point on the sphere
		'''
		from jizhipy.Basic import IsType
		import numpy as np
		if (len(args) == 0) : args = 1
		elif (not IsType.isnum(args[0])): args =args[0]
		nside = 4096*4
		pix = np.random.random(args)*12.*nside**2
		return hp.pix2ang(nside, pix)

	def ThetaPhi( self, *args ) : 
		return self.RandomThetaPhi(*args)





	def RandomLine(self, size, times=100, amax=None, amin=None):
		'''
		return 1D line with size

		times:
			int
			How many times to cumulate the sin/cos functions

		(1) amax !=None and amin !=None
			force angle np.linspace(amin, amax, size)
		(2) amin ==None: 
			amin, amax = np.sort(np.random.random(2)*amax) for each time
		'''
		import numpy as np
		if (times is None) : times = 100
		if(amax is not None and amin is not None): b=[amin,amax]
		else : 
			if (amax is None) : amax = 2*np.pi*10
			amin = None
		a = np.zeros(size)
		for i in range(times) : 
			if(amin is None): b =np.sort(np.random.random(2)*amax)
			bi = np.linspace(b[0], b[1], size)
			a += np.sin(bi)
		a -= a.min()
		a *= 2./a.max()
		a -= 1
		return a

	def Line(self, size, times=100, amax=None, amin=None):
		return self.RandomLine(size, times, amax, amin)





Random = Random()

