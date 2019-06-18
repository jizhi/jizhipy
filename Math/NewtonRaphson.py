
def NewtonRaphson( func, initguess, *known ) : 
	'''
	For example:
	Function:
		y = a*(1-e**2)*np.sin(t) / (1+e*np.cos(t))
	Now we know:
		y, a, e
	Solve:
		t

	Values: y, a, e, t = 4.2494672726945826, 5, 0.2, 1.234

	Usage:
		### First, define func(*args)
		### Must args[0]=t, means t (which needs to be solved) must be put on the first place!
		def func( *args ) :   
			t, y, a, e = args
			err = y - a*(1-e**2)*np.sin(t) / (1+e*np.cos(t))
			return err

		### y, a, e which put after func must as the same order as args
		t = jizhipy.NewtonRaphson( func, initguess, y, a, e )
		t = jizhipy.NewtonRaphson( func, 0, 4.2494672726945826, 5, 0.2 )

	initguess:
		initial guess
		same shape as known[?]
	'''
	from scipy.optimize import newton
	from jizhipy.Basic import IsType, Raise
	import numpy as np
	from jizhipy.Array import Asarray
	known, shape, n, slen = list(known), [], 0, []
	for i in range(len(known)) : 
		if (IsType.isnum(known[i])) : n += 1
		known[i] = Asarray(known[i])
		shape.append(known[i].shape)
		known[i] = known[i].flatten()
		slen.append(known[i].size)
	slen = np.array(slen)
	if (n == len(known)) : islist = False
	else : islist = True
	#---------------------------------------------
	n = np.where(slen==slen.max())[0][0]
	shape0 = shape[n]
	for i in range(len(shape)) : 
		if (slen[i] == 1) : continue
		if (shape[i] == shape0) : continue
		Raise(Exception, 'jizhipy.NewtonRaphson(): all elements in "known" have NOT the same shape: '+str(shape))
	for i in range(len(shape)) : 
		if (slen[i] != 1) : continue
		known[i] = known[i]+np.zeros(slen[n])
	try : initguess = initguess + np.zeros(slen[n])
	except : initguess = 2.34 + np.zeros(slen[n])
	#---------------------------------------------
	known = np.array(known)
	solved = np.zeros(slen[n])
	for i in range(known.shape[1]) : 
		def nrfunc( needsolved ) : 
			knowni = [needsolved]+list(known[:,i])
			return func(*knowni)
		solved[i] = newton(nrfunc, initguess[i])
	if (not islist) : solved = solved[i]
	else : solved = solved.reshape(shape0)
	return solved
		


