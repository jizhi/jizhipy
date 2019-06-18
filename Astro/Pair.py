
class Pair( object ) : 
	'''
	Return all pairs
	ina = [[0,1,2], [0,1]]
	return : [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]
	'''

	def __init__( self, ina, doslice=None ) : 
		'''
		ina:
			each row of ina[i] is one 1D array/list

		doslice:
			slice the whole return array along which column ?
		'''
		import numpy as np
		n, dtype = [], 0
		for i in range(len(ina)) : 
			n.append(len(ina[i]))
			dtype += np.array(ina[i][0])
		self.dtype = dtype.dtype.name
		self.ina, self.inn = ina, np.array(n, int)
		self.Nrow = self.inn.prod()
		#----------------------------------------
		if (doslice is not None) : 
			if (doslice < 0) : doslice = len(ina) + doslice
			if (doslice > len(ina)-2) : doslice = len(ina)-2
			if (doslice < 0) : doslice = 0
		self.doslice = doslice
		#----------------------------------------
		if (doslice is not None) : 
			self.totslice = int(self.inn[:doslice+1].prod())
			self.nowslice = 0
			self._outa = self._WholePair(self.ina[self.doslice+1:])
		else : self.totslice = self.nowslice = None
		self.repeat = self.divisor = False





	def _WholePair( self, a ) : 
		''' generate the whole pairs '''
		import numpy as np
		b, n = [], []
		for i in range(len(a)) : n.append(len(a[i]))
		N = int(np.prod(n))
		for i in range(len(a)-1) : 
			c = np.repeat(a[i], int(np.prod(n[i+1:])))
			b.append( np.concatenate(N/c.size *[c]) )
		b.append( np.tile(a[-1], N/len(a[-1])) )
		b = np.array(b).T
		return b





	def Get( self, nowslice=None ) : 
		'''
		If doslice, nowslice means that return which slice
		'''
		import numpy as np
		if (self.doslice is None) : 
			self.outa = self._WholePair(self.ina)
			return self.outa
		#----------------------------------------
		if (nowslice is not None) : self.nowslice = nowslice
		N = self.nowslice * self.inn[self.doslice+1:].prod()
		a = []
		for i in range(self.doslice+1) : 
			n = (N / self.inn[i+1:].prod()) % len(self.ina[i])
			a.append( self.ina[i][n]+np.zeros(len(self._outa), self.dtype) )
		self.outa = np.concatenate([np.array(a).T, self._outa], 1)
		return self.outa





	def Simplify( self, divisor=False, repeat=False ) : 
		'''
		repeat:
			True: remove all repeat rows, including multiples

		divisor:
			devide the greatest common divisors of each rows
		'''
		import numpy as np
		from CommonFactor import CommonFactor
		self.repeat, self.divisor =bool(repeat),bool(divisor)
		if (self.divisor) : 
			a = []
			for i in range(len(self.outa)) : 
				a.append(CommonFactor.gcd(self.outa[i]))
			self.outa /= np.array(a)[:,None]
		#----------------------------------------
		if (self.repeat) : 
			go = self.outa[:,0] + 1j*np.arange(len(self.outa))
			go = np.sort(go).imag.astype(int)
			back = go + 1j*np.arange(go.size)
			back = np.sort(back).imag.astype(int)
			self.outa = self.outa[go]
			#----------------------------------------
			truezero = self.outa.min()-123
			if (truezero == 0) : truezero -= 4
			falsezero = self.outa.min()-235
			if (falsezero == 0) : falsezero -= 3
			n = 1.*falsezero / truezero
			if (abs(n-int(n)) > 1e-6) : truezero -= 1
			self.outa[self.outa==0] = truezero
			self.outa = 1.*self.outa
			#----------------------------------------
			for i in range(len(self.outa)-1) : 
				if (self.outa[i,0] == falsezero) : continue
				a = self.outa[i+1:] / self.outa[i:i+1]
				tf = np.zeros(len(a), bool)
				# all int
				b = abs(a - a.astype(int)).sum(1)
				tf[b<1e-6] = True
				a = a[tf]
				b = self.outa[i+1:][tf]
				a[(a==1)*(b==truezero)] = 0
				for j in range(len(a)) : 
					a[j][a[j]==0] = a[j].max()
				# all same
				b = abs(a - a[:,:1]).sum(1)
				tf[tf==True] *= (b<1e-6)
				self.outa[i+1:][tf] = falsezero
			#----------------------------------------
			self.outa = self.outa[back].astype(self.dtype)
			self.outa = self.outa[np.arange(len(self.outa))[self.outa[:,0]!=falsezero]]
			self.outa[self.outa==truezero] = 0
		return self.outa





if __name__ == '__main__' : 
	import numpy as np
	a = [[1,2,4], [0,1,2,3,4], [1,4], [0,2,3,4]]
	
	pair0 = Pair(a)
	b0 = pair0.Get()

	pair = Pair(a, 1)
	b1 = []
	for i in range(pair.totslice) : 
		b1.append( pair.Get(i) )
	b1 = np.concatenate(b1, 0)
	
	b2, n = [], 0
	for i in range(3) : 
		for j in range(5) : 
			for k in range(2) : 
				for m in range(4) : 
					n += 1
					b2.append([[a[0][i],a[1][j],a[2][k],a[3][m]]])
	b2 = np.concatenate(b2, 0)
	
	print(abs(b0-b1).sum())
	print(abs(b0-b2).sum())
	print(abs(b1-b2).sum())

	b0 = pair0.Simplify(True, True)
	print(b0)

