'''
if (SysFrame(0,2)[-3][-2]=='') : 
'''

def SysFrame( downstack=0, upstack=None ): 
	'''
	downstack: begining of the stack. Default=0 (local,current)
	upstack: the uppest stack you want. Default to the uppest
	Stack from downstack[include] to upstack[exclude]
	'''
	import sys
	# How deep of the stack
	for d in range(1000000) : 
		try : sys._getframe(d)
		except : break
	stackmax, stackmin = d-1, 0
	if (downstack is None) : downstack = stackmin
	elif (downstack < stackmin) : downstack = stackmin
	if (upstack is None) : upstack = stackmax
	elif (upstack > stackmax) : upstack = stackmax
	if (downstack == upstack) : upstack += 1
	#--------------------------------------------------
	outstr, files, funcs, lines, sentences = '', [], [], [], []
	for i in range(upstack, downstack, -1) : 
		f = sys._getframe(i)
		filename = f.f_code.co_filename
		lineno   = f.f_lineno
		name     = f.f_code.co_name
		outstr += '  File "'+filename+'", line '+str(lineno)+', in '+name+' =>\n'
		files.append(filename)
		if (name == '<module>') : funcs.append('')
		else : funcs.append(name)
		lines.append(lineno)
		st = open(files[-1]).readlines()[lines[-1]-1]
		for i in range(0, len(st)) : 
			if (st[i] != '\t') : break
		st = st[i:]
		if (st[-1] == '\n') : st = st[:-1]
		sentences.append(st)
	outstr = outstr[:-4]
	return [outstr, files, funcs, lines, sentences]





def Raise( which=None, message='', newline=False, top=False, onlyReturn=False ): 
	'''
	Usage:
		(1) Warning, but run continue
		(2) Exception, stop at once

	message:
		str, content of the Warning/Exception message

	newline:
		True: print new-blank-line after Warning, NOT Exception

	top:
		False: print all stacks/levels
		True : only print the top stack/level
	'''
	info = SysFrame(1)[0]
	if (top) : 
		n = info.find(', line ')
		s = info[:n+7]
		info = info.split('=>\n')
		for i in range(len(info)) : 
			if (s not in info[i]) : break
			elif (i == len(info)-1) : i = len(info)
		for j in range(1, i) : 
			info[0] += '=>\n' + info[j]
		info = info[0]
	#----------------------------------------
	if (which is None or which is Exception) : 
		which = 'Exception'
	if (which is Warning): which = 'Warning'
	which = which[0].upper()+which[1:].lower()
	msg = '--------------------- '+which+' ---------------------\n'+info+'\n'+which+': '+str(message)+'\n---------------------------------------------------'
	if (which=='Warning' and newline) : msg += '\n'
	if (onlyReturn): return msg
	else: 
		print(msg)
		if (which=='Exception'): exit()

