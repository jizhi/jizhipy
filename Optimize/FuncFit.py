
def _Leastsq( func, x, y, p0, sigma=None, maxfev=10000, warning=False ) : 
	'''
	scipy.optime.leastsq( residuals, p0, args, maxfev )

	p0: Initial guess of the parameters in func.

	maxfev: The maximum number of calls to the function.

	residuals:
		lestsq do "min(sum(residuals(y)**2)", so residuals is a function like:
		def residuals( para, func, xdata, ydata, weight ) : 
			err = weight * (ydata - func(xdata, para))
			return err

	func:
		The fitting function of xdata-ydata:
			ydata ~ func( xdata, para )

	sigma:
		1/sigma = weight
		The sigma or weight of each y data

	return: 
		p, type(p)==list although size==1
	'''
	from scipy.optimize import leastsq
	from Asarray import Asarray, np
	import warnings
	if (not warning) : warnings.filterwarnings("ignore")
	if (sigma is None) : sigma = 1
	x,y, p0, sigma =Asarray(x),Asarray(y), Asarray(p0), Asarray(sigma)
	def residuals(para, func, xdata, ydata, weight) :  
		return weight * (ydata - func(xdata, para))
	args = (func, x, y, 1./sigma)
	p = leastsq(residuals, p0, args=args, maxfev=maxfev)[0]
	if (not warning) : warnings.filters.pop(0)
	return np.array(p)[None,:]  # (1, N)





def _FuncFit( func, x, y, p0, sigma=None, maxfev=10000, warning=False ) : 
	'''
	Need scipy.__version__ >= 0.15
	Use scipy.optime.curve_fit()

	func:
		y = func(x, p)

	return:
		[p, perr], type(p)==type(perr)==list
	'''
	from scipy.optimize import curve_fit
	import warnings
	import numpy as np
	from jizhipy.Array import Asarray
	if (not warning) : warnings.filterwarnings("ignore")
	x, y, p0 = Asarray(x), Asarray(y), Asarray(p0)
	def fitfunc( *arg ) : return func(arg[0], arg[1:])
	if (sigma is None) : abssigma = False
	else : abssigma = True
	#--------------------------------------------------
	def CurveFit( abssigma ) : 
		if (abssigma) : res = curve_fit(fitfunc, x, y, p0, sigma, abssigma, maxfev=maxfev)
		else : res = curve_fit(fitfunc, x, y, p0, sigma, maxfev=maxfev)
		p, perr = res[:2]
		perr = abs(np.diag(perr))**0.5
		p[np.isnan(p)] = p0[np.isnan(p)]
		p[np.isinf(p)] = p0[np.isinf(p)]
		perr[np.isnan(perr)] = 0
		perr[np.isinf(perr)] = 0
		return [p, perr]

	def Least() : 
		p = Leastsq(func, x, y, p0, sigma, maxfev, warning)
		perr = np.zeros(len(p))
		return [p, perr]
	#--------------------------------------------------
	if (abssigma) : 
		try : p, perr = CurveFit(True)
		except : 
			try : p, perr = CurveFit(False)
			except : p, perr = Least()
	else : 
		try : p, perr = CurveFit(False)
		except : p, perr = Least()
	if (not warning) : warnings.filters.pop(0)
	return np.array([p, perr])




def FuncFit( func, x, y, p0, sigma=None, maxfev=10000, warning=False, fiterr=False ) : 
	'''
	func:
		(1) def func(x, p) : return ...
		(2) func is str in ['gaussian', 'gaussian0', 'gaussian1', 'gaussian10']
		(3) func='polynomial': 
				p for [x^0, x^1, x^2, x^3, ...]
			if p=[None, 1.2, None, 2.3, 3.4, None, 4.5], mean:
				func = 1.2*x^1 + 2.3*x^3 + 3.4*x^4 + 4.5*x^6

	x, p0:
		Must be real, NOT complex

	y:
		Can be real or complex
		If complex, actually fit real+imag

	return:
		(1) fiterr=True : [p, perr]
		(2) fiterr=False: [p]  # NOTE THAT have [] outside
	'''
	from jizhipy.Math import Gaussian
	import numpy as np
	if (str(func).lower() == 'gaussian') :  # len(p)=2
		def func(x, p) : 
			return Gaussian.GaussianValue(x, p[0], p[1])
	elif (str(func).lower() == 'gaussian0') :  # len(p)=1
		def func(x, p) : 
			return Gaussian.GaussianValue(x, 0, p[0])
	elif (str(func).lower() == 'gaussian1') :  # len(p)=2
		def func(x, p) : 
			return Gaussian.GaussianValue1(x, p[0], p[1])
	elif (str(func).lower() == 'gaussian10') :  # len(p)=1
		def func(x, p) : 
			return Gaussian.GaussianValue1(x, 0, p[0])
	elif (str(func).lower() in ['polynomial']) :  # len(p)>=1
		p0, p00 = list(p0), list(p0)[:]
		for i in range(len(p0)) : 
			if (p0[i] is None) : p0[i] = 0
		def func(x, p) : 
			y = 0.
			for i in range(len(p00)) : 
				if (p00[i] is None) : continue
				if (i == 0) : y += p[i]
				else : y += p[i] * x**i
			return y
	#---------------------------------------------
	y = np.array(y)
	if ('complex' in y.dtype.name) : 
		y = np.append(y.real, y.imag)
		def fitfunc( x, p ) : 
			yf = func(x, p)
			yf = np.append(yf.real, yf.imag)
			return yf
	else : fitfunc = func
	#---------------------------------------------
	if (fiterr) : return _FuncFit(fitfunc, x, y, p0, sigma, maxfev, warning)
	else : return _Leastsq(fitfunc, x, y, p0, sigma, maxfev, warning)





def IMinuit( func, p0, x, y, sigma_y='' ) : 
	'''
	func is a function of y = func(p, x)
	For example:
		def func(p, x) : 
			y = p[0] + p[1]*x
			return y

	p0:
		Initial value of fitting parameters.

	x, y:
		Data to be fit.

	sigma_y:
		For Chi^2 fitting, you must provide sigma of each data point.
		if (sigma_y == '') : sigma_y = Sigma(y)
		if (sigma_y == 'int number') : calculate sigma in each bins with number of data point equals to int nunber
		if (sigma_y == value) : set sigma_y = value

	Return:
		[value, error]
	'''
	if (sigma_y == '') : 
		sigma_y = Sigma(y)
	elif (type(sigma_y) == str) : 
		Ny = y.size
		n = int(sigma_y)
		if (n < 3) : n = 3
		if (Ny/n < 2) : sigma_y = Sigma(y)
		else : 
			sigma_y = y*0.
			n = np.arange(0, Ny, n)
			n = np.append(n, [Ny])
			for i in range(n.size-1) : 
				sigma_y[n[i]:n[i+1]] = Sigma(y[n[i]:n[i+1]])
			# sigma=0
			ms = sigma_y.mean()
			if (ms == 0) : sigma_y = Sigma(y)
			else : 
				sigma_y[sigma_y==0] = ms
			if (abs(sigma_y).max() == 0) : sigma_y = sigma_y + 1e-6
	n = len(p0)
	if   (n == 1) : 
		def fitfunc( p1 ) : 
			p = [p1]
			err = sum( (y - func(x,p))**2 /sigma_y**2 )
			return err
		m = Minuit(fitfunc, p1=p0[0], print_level=0, pedantic=False)
	elif (n == 2) : 
		def fitfunc( p1, p2 ) : 
			p = [p1, p2]
			err = sum( (y - func(x,p))**2 /sigma_y**2 )
			return err
		m = Minuit(fitfunc, p1=p0[0], p2=p0[1], print_level=0, pedantic=False)
	elif (n == 3) : 
		def fitfunc( p1, p2, p3 ) : 
			p = [p1, p2, p3]
			err = sum( (y - func(x,p))**2 /sigma_y**2 )
			return err
		m = Minuit(fitfunc, p1=p0[0], p2=p0[1], p3=p0[2], print_level=0, pedantic=False)
	elif (n == 4) : 
		def fitfunc( p1, p2, p3, p4 ) : 
			p = [p1, p2, p3, p4]
			err = sum( (y - func(x,p))**2 /sigma_y**2 )
			return err
		m = Minuit(fitfunc, p1=p0[0], p2=p0[1], p3=p0[2], p4=p0[3], print_level=0, pedantic=False)
	elif (n == 5) : 
		def fitfunc( p1, p2, p3, p4, p5 ) : 
			p = [p1, p2, p3, p4, p5]
			err = sum( (y - func(x,p))**2 /sigma_y**2 )
			return err
		m = Minuit(fitfunc, p1=p0[0], p2=p0[1], p3=p0[2], p4=p0[3], p5=p0[4], print_level=0, pedantic=False)
	elif (n == 6) : 
		def fitfunc( p1, p2, p3, p4, p5, p6 ) : 
			p = [p1, p2, p3, p4, p5, p6]
			err = sum( (y - func(x,p))**2 /sigma_y**2 )
			return err
		m = Minuit(fitfunc, p1=p0[0], p2=p0[1], p3=p0[2], p4=p0[3], p5=p0[4], p6=p0[5], print_level=0, pedantic=False)
	m.migrad()
	value = m.values
	err = m.errors
	para = m.parameters
	ve = np.zeros([2, len(para)])
	for i in range(len(para)) : 
		ve[0,i] = value[para[i]]
		ve[1,i] = err[para[i]]
	return ve





#def Leastsq( x, y, func='', p0index=3, normalized=True, spORmy='sp' ) : 
#	'''
#	Use LeastsqMatrix() or LeastsqSp() basing on spORmy
#
#	func: 
#		(1) "polynomial":
#			index is the highest order of x
#			index=0: y = a + const*x
#			index=1: y = a + b*x
#			index=2: y = a + b*x + c*x^2
#			return [a, b, c]
#		(2) "gaussian":
#			y = 1/(sigma*sqrt(2pi) * exp(-(x-mean)^2/(2sigma^2))
#			return [mean, sigma]
#		(3) "power-law":
#			y = a*x^b
#			return [a, b]
#
#	p0index:
#		func=='polynomial': p0index=index
#		func=='gaussian'  : p0index=[mean, stdev]
#		func=='power-law' : p0index=[a,b]
#
#	spORmy:
#		Use LeastsqSP() or LeastsqMatrix().
#		(1) ='sp'
#		(2) ='both'
#		Note that, LeastsqSP() performs much better than LeastsqMatrix(), so, generally we just use ='sp'(default)
#	'''
#	# because 'polynomial' and 'power-law' are very simple, we can use p0=[1,1] automatically.
#	x, y = Asarray(x), Asarray(y)
#	if (func.lower() == 'polynomial') : 
#		if (type(p0index) not in [list, np.ndarray]) : 
#			p0index = int(round(p0index))
#			p0 = np.ones(p0index+1)
#			p2 = LeastsqSP(x, y, func, p0)
#			if (spORmy == 'both') : 
#				p1 = LeastsqMatrix(x, y, func, p0index)
#				p2 = np.append([p2], [p1], 0)
#		else : 
#			p2 = LeastsqSP(x, y, func, p0index)
#			if (spORmy == 'both') : 
#				index = len(p0index) - 1
#				p1 = LeastsqMatrix(x, y, func, index)
#				p2 = np.append([p2], [p1], 0)
#		return p2
#	elif (func.lower() == 'power-law') : 
#		if (x.max()>0 and x.min()>0) : 
#			if (type(p0index) not in [list, np.ndarray]) : 
#				if (y.max()>0 and y.min()>0) : p0index=[1,1]
#				elif (y.max()<0 and y.min()<0) : p0index=[-1,1]
#				else : Raise(Exception, 'x>0 but y.max()>0 or y.min()<0, it is not a power-law')
#		elif (x.max()>0 and x.min()<0) : 
#			xy = np.append(x[:,None], y[:,None], 1)
#			xy = xy[xy[:,0]>0]
#			x, y = xy.T
#			if (y.max()>0 and y.min()>0) : p0index=[1,1]
#			elif (y.max()<0 and y.min()<0) : p0index=[-1,1]
#			else : Raise(Exception, 'x>0 but y.max()>0 or y.min()<0, it is not a power-law')
#		else : Raise(Exception, 'x<0, it may not be a power-law, please check and fit it with the function written by yourself')
#		p = LeastsqSP(x, y, func, p0index)
#		return p
#	elif (func.lower() == 'gaussian') : 
#		xy = np.append(x[:,None], y[:,None], 1)
#		xy = xy[xy[:,1]>0]
#		x, y = xy.T
#		mean = (xy[xy[:,1]==xy[:,1].max()])[:,0].mean()
#		stdev = RMS(x-mean)
#		if (stdev == 0) : stdev = 1
#		if (type(p0index) not in [list, np.ndarray]) : 
#			p2 = LeastsqSP(x, y, func, [mean, stdev], normalized=normaliszed)
#		else : 
#			p2 = LeastsqSP(x, y, func, p0index, normalized=normaliszed)
#		if (spORmy == 'both') : 
#			p1 = LeastsqMatrix(x, y, func, 2, normalized=normaliszed)
#			p2 = np.append([p2], [p1], 0)
#		return p2

