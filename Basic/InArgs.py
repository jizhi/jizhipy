
def InArgs( local, *args ) : 
	'''
	local:
		must be given:  locals()

	*args:
		dict(s)
		will .update() local to *args
	'''
	# Example:
	# class A( object ) : 
	# 
	# 	def __init__( self, a ) : 
	# 		self.inargs = {}
	# 		InArgs(locals(), self.inargs, self.__dict__)
	# 		print self.inargs
	# 
	# 	def B( self, b ) : 
	# 		InArgs(locals(), self.inargs, self.__dict__)
	# 		print self.inargs
	# 
	# 	def C( self, c ) : 
	# 		InArgs(locals(), self.inargs, self.__dict__)
	# 		print self.inargs
	# 
	# a = A(1)
	# a.B(2)
	# a.C(3)
	try : local.pop('self')	
	except : pass
	try : local.pop('cls')	
	except : pass
	try : 
		for i in range(len(args)) : 
			args[i].update(local)
	except : pass

