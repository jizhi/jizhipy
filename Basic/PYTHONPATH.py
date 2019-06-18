
def PYTHONPATH( use=False, notuse=False ) : 
	'''
	$PYTHONHOME
	$PYTHONPATH

	(1) use=False
			NOT append to sys.path
		notuse=False
			NOT remove from sys.path	

	(2) use=None
			append $PYTHONPATH to sys.path	
	(3) use=str | [str, str, ...]
			append it/them to sys.path	

	(4) notuse=None
			like =False, NOT remove from sys.path	
	(5) notuse=str | [str, str, ...]
			remove it/them from sys.path

	return:
		Updated sys.path
	'''
	import sys, os
	from jizhipy.Basic import IsType, ShellCmd
	#----------------------------------------
	if (use is True or use is False) : use = []
	elif (use is None) : 
		use = ShellCmd('echo $PYTHONPATH')[0].split(':')
	elif (IsType.isstr(use)) : use = [use]
	elif (IsType.istuple(use)) : use = list(use)
	#----------------------------------------
	if (notuse is True or notuse is False or notuse is None) : notuse = []
	elif (IsType.isstr(notuse)) : notuse = [notuse]
	elif (IsType.istuple(notuse)) : notuse = list(notuse)
	#----------------------------------------
	n = 0
	while (n < len(use)) : 
		if (use[n] == '') : use.pop(n)
		else : 
			if (use[n][-1] == '/') : 
				use[n] = use[n][:-1]
				n += 1
	if (len(use) > 0) : use = [''] + use
	#----------------------------------------
	here = os.path.abspath('')
	notusehere = True if('' in notuse)else False
	for i in range(len(notuse)) : notuse[i] = os.path.abspath(os.path.expanduser(notuse[i]))
	if (notusehere) : notuse.append('')
	#----------------------------------------
	sys.path = use + sys.path
	for i in range(len(notuse)) : 
		while (notuse[i] in sys.path) : 
			sys.path.remove(notuse[i])
	return sys.path[:]

