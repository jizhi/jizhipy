
def SysMemory() : 
	'''
	Return the total physical memory (MB) of the computer
	1e8 float64 accounts for 800MB:
		array.dtype = np.float64
		array.size = 1e8
		array accounts for 800MB memory
	'''
	from jizhipy.Basic import ShellCmd
	uname = ShellCmd('uname')[0]
	if (uname == 'Linux') : 
		memory = ShellCmd('free -m')[1]
	elif (uname == 'Darwin') : 
		memory =ShellCmd('top -l 1 | head -n 10 | grep PhysMem')[0]
	else : Raise(Exception, uname+' not in (Linux, Darwin)')
	#--------------------------------------------------
	for i in range(len(memory)) : 
		if (memory[i] in '123456789') : 
			n1 = i
			break
	for i in range(n1+1, len(memory)) : 
		if (memory[i] not in '0123456789') : 
			n2 = i
			break
	totmem = int(memory[n1:n2]) # MB
	#--------------------------------------------------
	if (uname == 'Darwin') : 
		for i in range(len(memory)-1, 0, -1) : 
			if (memory[i] in '0123456789') : 
				n4 = i+1
				break
		for i in range(n4-1, 0, -1) : 
			if (memory[i] not in '0123456789') : 
				n3 = i+1
				break
		totmem += int(memory[n3:n4]) # MB
	#--------------------------------------------------
	return totmem


