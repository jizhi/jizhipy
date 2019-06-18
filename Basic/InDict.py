
class InDict( object ) : 
	'''
	Backup and Reset the dict/self.__dict__
	'''
	def __init__( self, indict ):
		self.indict = indict
		self.keys = list(self.indict.keys())[:]


	def __getitem__(self, key): 
		'''
		indict = InDict()
		indict[key] ...
		'''
		return self.indict[key]


	def __setitem__(self, key, value): 
		'''
		indict = InDict()
		indict[key] = value
		'''
		self.indict[key] = value


	def Backup( self, *args ) : 
		'''
		*args:
			keys of self.indict
			str | list/tuple of str

		Dict.Backup('a', 'b', 'c', ...)
		Dict.Backup(): backup all
		'''
		if (len(args) == 0) : 
			self.backkey = self.indict.keys()[:]
			self.backdict = self.indict.copy()
			self._backkey = self.backkey[:]
		#---------------------------------------------
		from jizhipy.Basic import IsType
		keys = []
		for i in range(len(args)) : 
			if (IsType.isstr(args[i])) : keys.append(args[i])
			else : keys += list(args[i])
		self.backkey = tuple(keys)
		self.backdict = {}
		for i in range(len(keys)) : 
			v = self.indict[keys[i]]
			if (IsType.islist(v)): self.backdict[keys[i]]=v[:]
			elif (IsType.isndarray(v) or IsType.isdict(v) or IsType.ismatrix(v) or IsType.ismaskedarray(v)) : self.backdict[keys[i]] = v.copy()
			else : self.backdict[keys[i]] = v
		self._backkey = self.backkey[:]



	def Update( self, *args ) : 
		'''
		Order of *args in self.Update() must be exactly the same as in self.Backup()

		dict = {'a':1, 'b':2, 'c':3}
		adict = Dict(dict)
		adict.Backup('a', 'c', 'b')

		adict.Update(5, 6, 7)
		print adict.indict
			=> {'a':5, 'b':7, 'c':6}

		If dont want to update some keys, give 'NAN'

		adict.Update('NAN', 6, 7)
		print adict.indict
			=> {'a':1, 'b':7, 'c':6}
		'''
		from jizhipy.Basic import IsType
		for i in range(len(self.backkey)) : 
			v = args[i]
			if (IsType.isstr(v) and str(v)=='NAN') : continue
			self.indict[self.backkey[i]] = v
		self.__dict__.pop('backkey')
		


	def Reset( self, delnew=True ) : 
		if (delnew) : 
			keys = self.indict.keys()
			for i in range(len(keys)) : 
				if (keys[i] not in self.keys) : 
					self.indict.pop(keys[i])
		self.indict.update(self.backdict)


