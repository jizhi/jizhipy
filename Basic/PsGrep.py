
def PsGrep( *args ):
	'''
	PsGrep('python wait.py') => ps | grep "python wait.py"

	Parameters
	----------
	*args:
		Can match many keys

	Returns
	----------
	return pid, pid is a list of str
	'''
	from jizhipy.Basic import ShellCmd
	value, _value, pid = [], [], []
	for key in args:
		_value += ShellCmd('ps | grep "'+key+'"')
	for v in _value:
		get = True
		for key in args:
			if (key not in v): get = False
		if (get and 'grep' not in v and v not in value):
			value.append(v)
	for v in value:
		w = v.split(' ')
		for p in w:
			try: 
				pid.append( str(int(p)) )
				break
			except: pass
	return pid

