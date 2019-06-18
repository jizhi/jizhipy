
class IsType( object ) : 
	dtype = 'class:IsType'

	def isndarray( self, a ) : 
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'ndarray') : return True
		else : return False

	def isdataframe( self, a ) : 
		''' pandas.DataFrame() '''
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'DataFrame') : return True
		else : return False

	def isseries( self, a ) : 
		''' pandas.Series() '''
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'Series') : return True
		else : return False

	def isdatetimeindex( self, a ) : 
		''' pandas.index '''
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'DatetimeIndex') : return True
		else : return False

	def isint( self, a ) : 
		if (self.isndarray(a)) : 
			if (a.shape != ()) : return False
			if (a.dtype.name[:3] == 'int') : return True
		typestr =str(type(a)).split("'")[-2].split('.')[-1][:3]
		if (typestr == 'int') : return True
		else : return False
	
	def isfloat( self, a ) : 
		if (self.isndarray(a)) : 
			if (a.shape != ()) : return False
			if (a.dtype.name[:5] == 'float') : return True
		typestr =str(type(a)).split("'")[-2].split('.')[-1][:5]
		if (typestr == 'float') : return True
		else : return False

	def isnum( self, a ) : 
		tf = self.isint(a) + self.isfloat(a)
		if (tf == 0) : return False
		else : return True
	
	def isstr( self, a ) : 
		typestr =str(type(a)).split("'")[-2].split('.')[-1][:3]
		if (typestr in ['str', 'uni']) : return True
		else : return False

	def islist( self, a ) : 
		typestr = str( type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'list') : return True
		else : return False

	def istuple( self, a ) : 
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'tuple') : return True
		else : return False

	def islistuple(self, a):
		b = self.islist(a)
		c = self.istuple(a)
		return (b or c)
	
	def isdict( self, a ) : 
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'dict') : return True
		else : return False
	
	def ismatrix( self, a ) : 
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'matrix') : return True
		else : return False

	def ismaskedarray( self, a ) : 
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr == 'MaskedArray') : return True
		else : return False
	
	def isdtype( self, a ) : 
		''' numpy.dtype '''
		strv = str(a)
		if ('int16' in strv or 'int32' in strv or 'int64' in strv or 'float32' in strv or 'float64' in strv or 'complex64' in strv or 'complex128' in strv) : return True
		else : return False

	def isclass( self, a ) : 
		if (self.isdtype(a)) : return False
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr in ['type', 'classobj']) : return True
		else : return False
		
	def isinstance( self, a ) : 
		typestr1 = str(type(a)).split("'")[-2]
		typestr2 = str(type(a)).split(' ')[0].split('<')[-1]
		if (typestr1=='instance' or typestr2=='class') : 
			return True
		else : return False

	def isfunc( self, a ) : 
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr in ['instancemethod', 'function']): return True
		else : return False

	def isspecialinstance( self, a ) : 
		notlist = ['matrix', 'MaskedArray']
		typestr = str(type(a)).split("'")[-2].split('.')[-1]
		if (typestr in notlist) : return True
		else : False

	def isbool( self, a ) : 
		if (a is None or a is True or a is False): return True
		else : return False

	def isdir( self, a ) : 
		import os
		return os.path.isdir(a)

	def isfile( self, a ) : 
		import os
		return os.path.isfile(a)

	def isfalse( self, a ) : 
		if   (a is False or a is None  ) : return True
		elif (self.isstr(a)   and a=='') : return True
		elif (self.islist(a)  and a==[]) : return True
		elif (self.istuple(a) and a==()) : return True
		elif (self.isdict(a)  and a=={}) : return True
		else : return False

	def islaohu( self ) : 
		from jizhipy.Basic import Path
		tf = (Path.AbsPath('~')[:17] == '/public/home/wufq')
		if (tf) : 
			import sys
			from jizhipy.Plot import Plt
			Plt.Style()
			sys.path = ['/public/home/wufq/.mysoftware/lib/python2.7/site-packages'] + sys.path
		return tf

	def iscomm( self, a ) : 
		tf = False
		try : 
			a.Get_rank()
			a.Get_size()
		#	if ('mpi4py.MPI.Intracomm' in str(a)) : tf = True
			tf = True
		except : pass
		return tf

	def iscbar( self, a ) : 
		a = str(type(a)).lower()
		if ('.colorbar.' in a) : return True
		else : return False

	def ispil( self, a ):
	#	b = False
	#	try: 
	#		s = a.__module__[:4]
	#		if (s == 'PIL.'): b = True
	#	except: pass
		b = str(type(a))
		if (" 'PIL." in b): b = True
		else: b = False
		return b





IsType = IsType()

