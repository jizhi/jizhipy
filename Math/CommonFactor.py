
class CommonFactor( object ) : 


	def gcd( self, *args ) : 
		''' gcd: greatest  common divisor '''
		import numpy as np
		from jizhipy.Basic import Raise, IsType
		from jizhipy.Array import Asarray
		a = []
		for i in range(len(args)) : 
			a.append(Asarray(args[i]).flatten())
		a = np.concatenate(a)
		if (not IsType.isint(a[0])) : Raise(Exception, 'jizhipy.CommonFactor.gcd(): input number must be "int", but now is "'+a.dtype.name+'"')
		a = abs(a.astype(float))
		a = a[a>0]
		if (a.size == 0) : return 1
		elif (a.size == 1) : return int(a[0])
		vmin = int(a.min())
		for v in range(vmin, 0, -1) : 
			b = a /  v
			b = abs(b - b.astype(int)).sum()
			if (b < 1e-6) : break
		return v





	def lcm( self, *args ) : 
		''' lcm: lowest common multiple '''
		import numpy as np
		from jizhipy.Basic import Raise, IsType
		from jizhipy.Array import Asarray
		a = []
		for i in range(len(args)) : 
			a.append(Asarray(args[i]).flatten())
		a = np.concatenate(a)
		if (not IsType.isint(a[0])) : Raise(Exception, 'jizhipy.CommonFactor.lcm(): input number must be "int", but now is "'+a.dtype.name+'"')
		a = a[a>0]
		if (a.size == 0) : return 1
		elif (a.size == 1) : return 1
		b = a.prod() / a
		v = self.gcd(b)
		v = a.prod() / v
		return v
		#Raise(Exception, 'jizhipy.CommonFactor.lcm(): NOT finish this function')





CommonFactor = CommonFactor()
