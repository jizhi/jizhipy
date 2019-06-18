import sys
stdoutback = sys.stdout
stderrback = sys.stderr
warningfilter, numwarning = [], []


class Print( object ) : 
	'''
	jp.Print(......, precision=6, suppress=True)
	'''


	def _Print( self, *args, **kwargs ) : 
	    '''
	    Usage:
	    print 'a =', a, '  =>  b =', b, '  (OK)'
	    Print('a =', a, '  =>  b =', b, '  (OK)')
	        | Print('a =', a, '  =>  b =', b, '  (OK)', precision=6, suppress=True)
	    '''
	    import numpy as np
	    try : precision = kwargs['precision']
	    except : precision = 3
	    try : suppress = kwargs['suppress']
	    except : suppress = True
	    np.set_printoptions(precision=precision, suppress=suppress)
	    for i in range(len(args)) : 
	        ai = args[i]
	        if (type(ai) == float) : ai = ('%.'+str(precision)+'f') % ai
	        print(ai,)
	    print()


	def __init__( self, *args, **kwargs) : 
		self._Print(*args, **kwargs)





	@classmethod
	def PrintHdr( cls, *args ) : 
		'''
		Print the header of .fits
	
		*args:
			(1) PrintHdr( fo[i].header ) : give header
			(2) PrintHdr( fo, i )        : give fo and i
		'''
		if (len(args) != 1) : 
			import os
			fo, i = args[:2]
			hdr = fo[i].header
			fname = fo.filename()
			f = os.path.expanduser('~/')
			if (fname[:len(f)]==f) : fname = '~/'+fname[len(f):]
			pstr = 'filename: '+fname+'\nhdu = '+str(i)+'\n\n'
		else : hdr, pstr = args[0], ''
		k, v, c = hdr.keys(), hdr.values(), hdr.comments
		n1, n2, n3 = 0, 0, 0
		for i in range(len(k)) : 
			if (len(k[i]) > n1) : n1 = len(k[i])
			if (len(str(v[i])) > n2) : n2 = len(str(v[i]))
			if (len(c[i]) > n3) : n3 = len(c[i])
		for i in range(len(k)) : 
			sk, sv, sc = k[i], str(v[i]), c[i]
			sk = ('%'+str(n1)+'s') % sk
			if (len(sv) < n2) : sv += (n2-len(sv))*' '
			pstr += sk + ' = ' + sv + ' / ' + sc + '\n'
		print(pstr[:-1])





	@classmethod
	def PrintSet( cls, stdout=True, stderr=True ) : 
		'''
		set True : print the messages
		set False: NOT print the messages
		'''
		class Std( object ) : 
			def write( self, stream ) : pass
		if (stdout) : sys.stdout = stdoutback
		else : sys.stdout = Std()
		if (stderr) : sys.stderr = stderrback
		else : sys.stderr = Std()





	@classmethod
	def WarningSet( cls, warning=None ) : 
		'''
		Print Warning or not?
		warning:
			(1) ==None : reset to default
			(2) ==False: ignore / not print warning
			(3) ==True : print warning
		'''
		import warnings
		if (numwarning == []) : 
			warningfilter.append( warnings.filters[:] )
		numwarning.append(1)
		#--------------------------------------------------
		exceptionwarning = False
		always, ignore = [], []
		for i in range(len(warnings.filters)) : 
			w = warnings.filters[i]
			if ('warning' in w[2].__name__.lower()) : 
				if   (w[0] == 'always') : always.append(i)
				elif (w[0] == 'ignore') : 
					ignore.append(i)
					if (w[2].__name__ == 'Warning') : 
						exceptionwarning = True
		#--------------------------------------------------
		if (warning is None) : 
			warnings.filters = warningfilter[0][:]
			return exceptionwarning
		#--------------------------------------------------
		elif (not warning) : 
			for i in range(len(always)) : 
				warnings.filters[always[i]] = ('ignore',) + warnings.filters[always[i]][1:]
			warnings.filterwarnings('ignore')
		#--------------------------------------------------
		else : 
			for i in range(len(ignore)) : 
				warnings.filters[ignore[i]] = ('always',) + warnings.filters[ignore[i]][1:]
		return warning




