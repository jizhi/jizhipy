
def OrderKwargs( stack=2 ) : 
	from jizhipy.Basic import SysFrame
	code = SysFrame(stack-1, stack+1)
	funcname = code[2][-1]
	code = code[-1][0]
	n = code.find(funcname)
	code = code[n+len(funcname):]
	n1, n2 = code.find('('), code.rfind(')')
	code = code[n1+1:n2].split(',')
	order = []
	for i in range(len(code)) : 
		if ('=' not in code[i]) : continue
		c = code[i].split('=')[0]
		n1, n2 = 0, len(c)
		for j in range(len(c)) : 
			if (c[j] != ' ') : 
				n1 = j
				break
		for k in range(n1+1, len(c)) : 
			if (c[k] == ' ') : 
				n2 = k
				break
		order.append(c[n1:n2])
	return order

