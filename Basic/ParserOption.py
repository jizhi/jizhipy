
def ParserOption() : 
	'''
	Rewrite sys.argv[1:] to options=dict{}

	Instead of class:optparse.OptionParser
	'''
	import sys
	sysargv = sys.argv[1:]
	if (len(sysargv) == 0) : return {}
	#--------------------------------------------------
	# Modify '--a' to '-a', and '-freq' to '--freq'
	nopt = []  # which are the options, the rest are the values
	for i in range(len(sysargv)) : 
		if (sysargv[i][0] != '-') : continue
		nopt.append(i)
		k = len(sysargv[i])
		for j in range(len(sysargv[i])) :
			if (sysargv[i][j] != '-') :
				k = j
				break
		a = sysargv[i][k:]
		if (len(a) <= 1) : sysargv[i] = '-' + a
		else : sysargv[i] = '--' + a
	nopt.append(len(sysargv))
	#--------------------------------------------------
	options = {}
	for i in range(len(nopt)-1) : 
		arg = sysargv[nopt[i]+1:nopt[i+1]]
		if (len(arg) == 0) : arg = True
		elif (len(arg) == 1) :
			arg = arg[0]
			if (arg.lower() == 'true') : arg = True
			elif (arg.lower() == 'false') : arg = False
			elif (arg.lower() == 'none') : arg = None
		else : arg = tuple(arg)
		options[sysargv[nopt[i]]] = arg
	#--------------------------------------------------
	# Check ',' and '='
	for k in options.keys() : 
		if (type(options[k]) != str) : continue
		if ('=' in options[k]) : 
			options[k] = options[k].split('=')
			options[k][0] += '='
			options[k][1] = options[k][1].split(',')
			for i in range(len(options[k])) : 
				if (options[k][i].lower() == 'none') : options[k][i] = None
		elif (',' in options[k]) : 
			options[k] = options[k].split(',')
			for i in range(len(options[k])) : 
				if (options[k][i].lower() == 'none') : options[k][i] = None
	return options

