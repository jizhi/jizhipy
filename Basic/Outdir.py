
def Outdir( *args ) : 
	'''
	(1) Outdir( string ) | Outdir( string, False ): 
		return DirStr(string)
	(2) Outdir( string, True )
		return DirStr(string) and mkdir it
	(3) Use stack as below:

	*args:
		Pair of stack and which:
			Outdir( (stack1,which1), (stack2,which2), ... )
		Each pair gets on outdir. If there are several pairs, merge these outdirs

		If just one stack and one outdir:
			Outdir( stack, which ) == Outdir( (stack, which) )

	which:
		'file' or 'func/function/class'

	stack:
		int
		stack=0    is local/current
		stack=None is the uppest stack

	For example: 
	Outdir((3,'file'), (2,'func'), (1,'file'), (0,'class'))

	return:
		outdir with '/' at the end
	'''
	import os
	from jizhipy.Basic import IsType, Mkdir, Raise, SysFrame
	if (len(args)==0) : return ''
	elif (len(args) in [1,2] and IsType.isstr(args[0])) : 
		s = args[0]
		try : tf = bool(args[1])
		except : tf = False
		if (s not in ['""', "''"] and s[-1] != '/'): s +='/'
		if (tf and not os.path.exists(s)) : Mkdir(s)
		return s
	if ((IsType.isint(args[0]) or IsType.isfloat(args[0]) or args[0] is None) and IsType.isstr(args[1])) : args = (args,)
	outdir = ''
	Nmax = len(SysFrame()[1])-1
	for i in range(len(args)) : 
		stack, which = args[i]
		# stack
		try : stack = int(round(stack)) + 1
		except : 
			if (stack is None) : stack = Nmax
			else : Raise(Exception, 'stack=args[%i] is NOT int' % i)
		if (stack > Nmax) : stacki = Nmax
		# which
		frame = SysFrame(stack, stack)
		if (which.lower() in ['func', 'function', 'class']) : 
			name = frame[2][0]
			if (name == '') : name = frame[1][0]
		else : name = frame[1][0]
		# name
		n = name.rfind('.')
		if (name[n:] in ['.py', '.pyc']) : name = name[:n]
		if (name[:2] == './') : name = name[2:]
		pwd1 = os.path.abspath('.')
		pwd2 = os.path.abspath('..')
		pwd3 = os.path.abspath('~')
		if (name[:len(pwd1)] == pwd1) : 
			outdir += name[len(pwd1)+1:]
		elif (name[:len(pwd2)] == pwd2) : 
			outdir += '../'+name[len(pwd1)+1:]
		elif (name[:len(pwd3)] == pwd3) : 
			outdir += '~/'+name[len(pwd1)+1:]
		else : outdir += name
		outdir += '_output/'
	outdirabs = os.path.abspath(os.path.expanduser(outdir))
	if (not os.path.exists(outdirabs)) : os.makedirs(outdirabs)
	return outdir


