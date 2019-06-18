

class Self( object ) : 


	def __init__( self ) : 
		self.inpars, self.outpars = [], []


	def Add( self, local, where ) : 
		where = str(where).lower()
		if (where == 'in') : self.inpars.update(local)
		else : self.outpars.update(local)
		
