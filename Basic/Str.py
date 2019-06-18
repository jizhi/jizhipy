
class Str( object ) : 


	def StrRemove( self, instr, replace='-', *args ) : 
		'''
		instr:
			input instr, must be one str, type(instr)==str
	
		replace:
			replace *args with this str
			For example, replace='' | replace='-'
	
		*args:
			replace these str with "replace"
	
		return:
			str
	
		Example:
			a = '$\Delta a$, b,c d, $h$, e f,g'
			b = StrRemove(a, '', '$', '\\')
			b = StrRemove(b, '-', ' ', ', ', ',', '\, ')
			  = 'Delta-a-b-c-d-h-e-f-g'
		'''
		import numpy as np
		args, n = np.array(args), []
		for i in range(len(args)) : 
			n.append(len(args[i])+i*1j)
		n = np.sort(n).imag.astype(int)
		args = args[n[::-1]]  # from large to small
		for i in range(len(args)) : 
			instr = instr.split(args[i])
			for j in range(1, len(instr)) : 
				instr[0] += replace + instr[j]
			instr = instr[0]
		return instr
	
	
	
	
	
	def StrDir( self, dirstr, expanduser=False, abspath=False):
		'''
		Convert '../paon4/paon4_data' to '../paon4/paon4_data/'
			'~/paon4_data/' to '~/paon4_data/' (NO change)
		
		dirstr:
			str | list of str ['', '', ...]
	
		expanduser:
			True | False: os.path.expanduser('~/...')

		abspath:
			True | False: os.path.abspath('...')
	
		return:
			Same shape as 
		'''
		import os
		from jizhipy.Basic import IsType
		if (IsType.isstr(dirstr)) : islist, dirstr = False, [dirstr]
		else : islist, dirstr = True, list(dirstr)
		for i in range(len(dirstr)) : 
			if ('~' in dirstr[i]) : 
				if (expanduser or abspath) : dirstr[i] = os.path.expanduser(dirstr[i])
			elif (abspath) : dirstr[i] = os.path.abspath(dirstr[i])
			if (dirstr[i] == '') : pass
			elif (dirstr[i][-1] != '/') : dirstr[i] += '/'
		if (not islist) : dirstr = dirstr[0]
		return dirstr
	
	
	


	def StrFind( self, instr, find ) : 
		'''
		find, instr:
			Both str. Where is (str)find in the (str)instr? 
			Return the index range
	
		find:
			(1) find = str: any str that will be found
			(2) find = 'isnum': where is the number(int or float)?
	
		return: 
			list of tuple, even just one element
			(1) return nfind
			(2) return [nfind, find]

		For example:
		(1) StrFind('CygA665S1dec15', 'S') => [(7,8)]
				it means instr[7:8] == 'S'

		(2) StrFind('CygA6.65S1dec15', 'isnum') => 
			[[(4,8), (9,10), (13,15)], ['6.65', '1', '15']]
		'''
		from jizhipy.Basic import IsType
		from jizhipy.Array import ArrayGroup
		instr, find, nfind = str(instr), str(find), []
		n2, n3 = len(instr), len(find)
		if (find.lower() != 'isnum') : 
			n1 = 0
			while (find in instr[n1:n2]) : 
				n0 = instr.find(find, n1, n2)
				n1 = n0 + n3
				nfind.append((n0, n1))
			return nfind
		#--------------------------------------------------
		else : 
			num, dota, dot, find = [], [], [], []
			for i in range(n2) : 
				if (instr[i] in '0123456789') : num.append(i)
				elif (instr[i] == '.') : dota.append(i)
			for i in range(len(dota)) : 
				if (dota[i]-1 in num and dota[i]+1 in num) : 
					dot.append(dota[i])
			nfind = list(ArrayGroup(num+dot)[1])
			for i in range(len(nfind)) : 
				nfind[i] = tuple(nfind[i])
				find.append(instr[nfind[i][0]:nfind[i][1]])
			return [nfind, find]





	def StrAdd( self, *args ) : 
		'''
		(1) StrAdd( 'a', 'b', 'c', 'd' )
				= 'abcd'
		(2) StrAdd( 'a', ['b'], 'c', 'd' )
				= ['abcd']
		(3) StrAdd( 'a', ['b'], ['c','d'] )
				== StrAdd( ['a','a'], ['b',''], ['c','d'] )
				= ['abc', 'ad']
		'''
		from jizhipy.Basic import IsType
		args, n = list(args), []
		for i in range(len(args)) : 
			if (IsType.isstr(args[i])) : 
				args[i] = [args[i]]
				n.append(-1)
			elif (type(args[i]) == tuple) : 
				args[i] = list(args[i])
				n.append(len(args[i]))
			else : n.append(len(args[i]))
		nmax = max(n) if(max(n)>0)else 1
		if (max(n) > 0) : 
			for i in range(len(args)) : 
				if (n[i] < 0) : args[i] = nmax*args[i]
				elif (n[i]<nmax) : args[i] += (nmax-n[i])*['']
		for i in range(nmax) : 
			for j in range(1, len(args)) : 
				args[0][i] += args[j][i]
		if (max(n) < 0) : args[0] = args[0][0]
		return args[0]





Str = Str()
