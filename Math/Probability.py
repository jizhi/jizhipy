
class Probability( object ) : 


	def Bins( self, array, nbins, weight=None, wmax2a=None, nsigma=None ) : 
		'''
		nbins:
			(1) ==list/ndarray with .size==3
				** nbins, bmin, bmax = bins
				nbins: number of bins
				bmin, bmax: min and max of bins, NOT use the whole bin
			(2) ==int_number:
				** Then use weight and wmax2a
				Give the total number of the bins, in this case, x.size=bins+1, xc.size=bins

		nsigma:
			float | None
			When generate the bins, won't use the whole range of array, set nsigma, will use |array| <= nsigma*array.std()

		weight:
			** Use this only when bins==int_number
			'G?', 'K?' | None | ndarray with size=bins
			(1) ==None: each bin has the same weight => uniform bins
			(2) ==ndarray: give weights to each bins
			(3) =='G?': 
					'?' should be an value, for example, 'G1', 'G2.3', 'G5.4', 'G12', use Gaussian weight, and obtain it from np.linspace(-?, +?, bins)
			    =='K?': 
					'?' should be an value, for example, 'K1', 'K2.3', 'K5.4', 'K12', use modified Bessel functions of the second kind, and obtain it from np.linspace(-?, +?, bins)

		wmax2a:
			Use it when weight is not None
			float | None
			(1) ==float: weight.max() corresponds to which bin, the bin which value wmax2a is in
		'''
		import numpy as np
		from jizhipy.Basic import IsType
		from jizhipy.Array import Invalid, Asarray
		from jizhipy.Math import Gaussian
		#---------------------------------------------
		array = Asarray(array)
		if (nsigma is not None) : 
			mean, sigma = array.mean(), array.std()
			array = array[(mean-nsigma*sigma<=array)*(array<=mean+nsigma*sigma)]
		amin, amax = array.min(), array.max()
		#---------------------------------------------
		if (Asarray(nbins).size==3) : nbins, bmin, bmax = nbins
		else : bmin, bmax = amin, amax
		#---------------------------------------------
		# First uniform bins
		bins = np.linspace(bmin, bmax, nbins+1)
		bstep = bins[1] - bins[0]
		#---------------------------------------------
		# weight
		if (weight is not None) : 
			if (IsType.isstr(weight)) : 
				w, v = str(weight[0]).lower(), abs(float(weight[1:]))
				if (v == 0) : v = 1
				x = np.linspace(-v, v, nbins)
				if (w == 'k') : 
					import scipy.special as spsp
					weight = spsp.k0(abs(x))
					weight = Invalid(weight)
					weight.data[weight.mask] = 2*weight.max()
				else :  # Gaussian
					weight =Gaussian.GaussianValue1(x, 0, 0.4)
			#--------------------
			# wmax2a
			if (wmax2a is not None) : 
				nmax = int(round(np.where(weight==weight.max())[0].mean()))
				nb = abs(bins - wmax2a)
				nb = np.where(nb==nb.min())[0][0]
				for i in range(bins.size-1) : 
					if (bins[i] <= wmax2a < bins[i+1] ) : 
						nb = i
						break
				d = abs(nmax - nb)
				if (nmax < nb) : weight = np.append(weight[-d:], weight[:-d])
				elif (nmax > nb) : weight = np.append(weight[d:], weight[:d])
			#--------------------
			weight = weight[:nbins]
			if (weight.size < nbins) : weight = np.concatenate([weight]+(nbins-weight.size)*[weight[-1:]])
			weight = weight.max() - weight + weight.min()
			weight /= weight.sum()
			weight = weight.cumsum()
			#--------------------
			c = bins[0] + (bmax-bmin) * weight
			bins[1:-1] = c[:-1]
			#--------------------
			bins = list(bins)
			n = 1
			while(n < len(bins)) : 
				if (bins[n] - bins[n-1] < bstep/20.) : 
					bins = bins[:n] + bins[n+1:]
				else : n += 1
			bins = Asarray(bins)
		#---------------------------------------------
		return bins
	
	
	
	
	
	def ProbabilityDensity( self, randomvariable, bins, weight=None, wmax2a=None, nsigma=6, density=True ) :
		'''
		Return the probability density or number counting of array.
		Return:
			[xe, xc, y]
			xe is the edge of the bins.
			xc is the center of the bins.
			y  is the probability density of each bin, 

		
		randomvariable==array:
			Input array must be flatten()
	
		bins:
			(1) ==list/ndarray with .size>3: 
				** Then ignore  brange, weight, wmax2a
				use this as the edge of the bins
				total number of the bins is bins.size-1 (x.size=bins.size, xc.size=bins.size-1)
			(2) ==list/ndarray with .size==3
				** nbins, bmin, bmax = bins
				nbins: number of bins
				bmin, bmax: min and max of bins, NOT use the whole bin
			(3) ==int_number:
				** Then use weight and wmax2a
				Give the total number of the bins, in this case, x.size=bins+1, xc.size=bins

		weight:
			** Use this only when bins==int_number
			'G', 'K0' | None | ndarray with size=bins
			(1) ==None: each bin has the same weight => uniform bins
			(2) ==ndarray: give weights to each bins
			(3) =='G': use Gaussian weight
			    =='K0': use modified Bessel functions of the second kind

		wmax2a:
			** Use this only when bins==int_number and weight is not None
			float | None
			(1) ==None: means weight[0]=>bins[0], weight[1]=>bins[1], weight[i]=>bins[i]
			(2) ==float: 
				uniform bin b = np.linspace(array.min(), array.max(), bins+1)
				value wmax2a is in nb-th bin: b[nb] <= wmax2a <= b[nb+1]
				weight.max() => weight[nmax]
				!!! Give weight[nmax] to the bin b[nb] (then reorder the weight array)
		
		nsigma:
			float | None (use all data)
			When generate the bins, won't use the whole range of array, set nsigma, will throw away the points beyond the mean
	
		density:
			If True, return the probability density = counting / total number / bin width
			If False, return the counting number of each bin

		Return:
			[xe, xc, y]
			xe is the edge of the bins.
			xc is the center of the bins.
			y  is the probability density of each bin, 
		'''
		import numpy as np
		from jizhipy.Process import Edge2Center
		from jizhipy.Array import Asarray
		#---------------------------------------------
		# nsigma
		# Throw away the points beyond the mean
		try : nsigma = float(nsigma)
		except : nsigma = None
		array = Asarray(randomvariable).flatten()
		sigma, mean = array.std(), array.mean()
		if (nsigma is not None) : array = array[(mean-nsigma*sigma<=array)*(array<=mean+nsigma*sigma)]
		amin, amax = array.min(), array.max()
		#---------------------------------------------
		if (Asarray(bins).size <= 3) : 
			bins =self.Bins(array, bins, weight, wmax2a, None)
		bins = Asarray(bins)
		#---------------------------------------------
		bins = bins[bins>=amin]
		bins = bins[bins<=amax]
		tf0, tf1 = False, False
		if (abs(amin-bins[0]) > 1e-6) : 
			bins = np.append([amin], bins)
			tf0 = True
		if (abs(amax-bins[-1])> 1e-6) : 
			bins = np.append(bins, [amax])
			tf1 = True
		#---------------------------------------------
		y, bins=np.histogram(array, bins=bins,density=density)
		if (tf0) : y, bins = y[1:], bins[1:]
		if (tf1) : y, bins = y[:-1], bins[:-1]
		x = Edge2Center(bins)
		return [bins, x, y]
	
	
	
	
	
	def RandomVariable( self, shape, x, pdf, norm=True) : 
		'''
		Invert operation of ProbabilityDensity()
		Provide probability density, return random variable

		shape:
			The shape of generated random variable
	
		pdf==fx, norm:
			fx: 
				isfunc | isndarray
				(1) isfunc: fx = def f(x), f(x) is the probability density function
				(2) isndarray: fx.size = x.size
			norm:
				True | False
				fx must be
					1. fx >= 0
					2. \int_{-\inf}^{+\inf} fx dx = 1
				Only if norm=False, not normal it, otherwise, always normal it.
	
		x:
			isndarray, must be 1D
			Use fx and x to obtain the inverse function of the cumulative distribution function, x = F^{-1}(y)
	
		return: 
			1D ndarray with shape, random variable
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import IsType, Raise
		from jizhipy.Optimize import Interp1d
		#---------------------------------------------
		x = Asarray(x).flatten()
		if (not IsType.isfunc(fx)) : 
			fx = Asarray(fx).flatten()
			if (x.size != fx.size) : Raise(Exception, 'fx.size='+str(fx.size)+' != x.size='+str(x.size))
		else : fx = fx(x)
		fx *= 57533.4
		#---------------------------------------------
		# sort x from small to large
		x = np.sort(x + 1j*fx)
		fx, x = x.imag, x.real
		#---------------------------------------------
		dx = x[1:] - x[:-1]
		dx = np.append(dx, dx[-1:])
		#---------------------------------------------
		# Normal fx
		if (norm is not False) : 
			fxmin = fx.min()
			if (fxmin < 0) : fx -= fxmin
			fx /= (fx.sum() * dx)
		#---------------------------------------------
		# Cumulative distribution function
		fx = fx.cumsum() * dx
		#---------------------------------------------
		# Inverse function	
		F_1 = Interp1d(fx, x, None)
		#---------------------------------------------
		# Uniform random with shape
		x = np.random.random(shape)
		#---------------------------------------------
		# Random variable with f(x)
		b = F_1(x)
		return b
	




Probability = Probability()
