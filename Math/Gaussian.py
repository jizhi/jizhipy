
class Gaussian( object ) : 


	def GaussianValue( self, x, mean, std ) : 
		import numpy as np
		y = 1/(2*np.pi)**0.5 /std *np.exp(-(x-mean)**2 /(2*std**2))
		return y


	def GaussianValue1( self, x, mean, std ) : 
		import numpy as np
		y = np.exp(-(x-mean)**2 /(2*std**2))
		return y

	 
	


	def LogNormalValue( self, x, mean, std ) : 
		from jizhipy.Basic import Raise
		import numpy as np
		if (x.min() < 0) : Raise(Exception, 'x.min() < 0')
		y = 1/(2*np.pi)**0.5 /std /x *np.exp(-(np.log(x)-mean)**2 /(2*std**2))
		return y
	
	
	
	
	
	def _GaussianSolidAngle( self, fwhmx, fwhmy=None ) : 
		'''
		1.133*fwhm*2 \approx (4/pi)**0.5 *fwhm**2

		fwhmx, fwhmy: 
			in rad
			Must be isnum
	
		return: 
			solid_angle = _GaussianSolidAngle(fwhmx, fwhmy) in sr
		'''
		from scipy.integrate import quad, dblquad
		import numpy as np
		#--------------------------------------------------
		if (fwhmx != fwhmy) : 
			def SolidAngle(y, x) : 
				sa = np.exp(-4*np.log(2) *y**2 *(np.sin(x)**2/fwhmx**2 + np.cos(x)**2/fwhmy**2)) *np.sin(y)
				return sa
			solid_angle = dblquad(SolidAngle, 0., 2*np.pi, lambda x:0., lambda x:np.pi)[0]
		#--------------------------------------------------
		else : 
			def SolidAngle(x) : 
				sa = np.exp(-4*np.log(2) *x**2 /fwhmx**2) *np.sin(x)
				return sa
			solid_angle=2*np.pi* quad(SolidAngle, 0, np.pi)[0]
		#--------------------------------------------------
		return solid_angle
	
	
	def GaussianSolidAngle( self, fwhmx, fwhmy=None ) : 
		'''
		fwhmx, fwhmy: 
			in rad
			Can be any shape, but must can broadcast
	
		return: 
			solid_angle =GaussianSolidAngle(fwhmx, fwhmy) in sr
		'''
		from jizhipy.Basic import IsType
		from jizhipy.Array import Asarray
		import numpy as np
		fwhmx, fwhmy = Asarray(fwhmx,True), Asarray(fwhmy,True)
		if (fwhmy is None) : fwhmy = fwhmx
		fwhmx = fwhmx + 0*fwhmy  # broadcast
		fwhmy = 0*fwhmx + fwhmy
		isnum = IsType.isnum(fwhmx) + IsType.isnum(fwhmy)
		if (isnum == 2) : return self._GaussianSolidAngle(fwhmx, fwhmy)
		shape = fwhmx.shape
		fwhmx, fwhmy = fwhmx.flatten(), fwhmy.flatten()
		solid_angle = np.zeros(fwhmx.size)
		for i in range(fwhmx.size) : 
			solid_angle[i] = self._GaussianSolidAngle(fwhmx[i], fwhmy[i])
		return solid_angle.reshape(shape)


	def SolidAngle( self, fwhmx, fwhmy=None ) : 
		return self.GaussianSolidAngle(fwhmx, fwhmy=None)





	def GaussianFilter( self, per=None, times=None, std=None, shape=None ) : 
		'''
		std(pixel) = fwhm(pixel) / (8*ln2)**0.5

		return: [gaussianfilter, std]
			Unit for std: pixel
			When use this std, x must pixel: np.arange(...)
		return normalized beam.sum()=1

		gf1, s1 = GaussianFilter(per, times)
		gf2, s2 = GaussianFilter(None, None, s1, gf1.size)
		gf3=GaussianValue(np.arange(gf1.size), gf1.size/2, s1)
			gf1==gf2==gf3,  s1==s2

		NOTE THAT must be GaussianFilter.sum()==1
			can NOT be other value

		Case 1: per, times, shape
			(1) Use per, times to generate gaussianfilter
					gaussianfilter.size = (per-1) * times + 1
			(2) Use shape to reshape gaussianfilter

		Case 2: std, shape
			Use std, shape to generate gaussianfilter

		First use case 1, otherwise use case 2

		shape: 
			(1) Use per, times, shape
				1. shape in [None, '1D']: 1D with original size
				2. shape isint: 1D with .size=shape
				3. shape == '2D': 1D with original size^2
				4. shape == (int, int, ...): N-D filter, std=(axis0, axis1, ...)
			(2) Use std, shape
				1. shape isint: 1D with .size=shape
				2. shape == (int, int, ...): N-D filter, std=(axis0, axis1, ...)
		'''
		from jizhipy.Optimize import GaussianFilter, Smooth
		from jizhipy.Basic import IsType
		import numpy as np
		from jizhipy.Array import Asarray
		if (per is not None and times is not None) : case = 1
		else : case = 2
		if (shape is None) : shape = ['']
		elif (IsType.isstr(shape)) : shape = ['' for i in range(int(shape[:-1]))]
		else : shape = Asarray(shape, int).flatten()
		#------------------------------------
		filt, sigma = [], []
		for i in range(len(shape)) : 
			if (case == 1) : 
				f, s = GaussianFilter(per, times)
				if (shape[i] != '') : f = Smooth(f, 0, reduceshape=shape[i])
			if (case == 2) : 
				f, s =GaussianFilter(None, None, std, shape[i])
			filt.append(f)
			sigma.append(s)
		#------------------------------------
		nd = len(shape)
		if (nd == 1) : filt, std = filt[0], sigma[0]
		else : 
			std = np.array(sigma)
			for i in range(nd) : 
				shape = [1 for j in range(nd)]
				shape[i] = filt[i].size
				filt[i] = filt[i].reshape(shape)
			for i in range(1, nd) : filt[0] = filt[0]*filt[i]
			filt = filt[0]
		return [filt, std]





	def GaussianBeam( self, fwhm, theta, normalized='sum' ) : 
		'''
		fwhm:
			rad, any shape

		theta:
			rad, any shape
		** Special case:
		** 	theta.shape = (N,1) | (1,N)
		** 	theta is 2D, one of dimension =1
		** 	This case means that generate 2D Gaussian beam

		normalized:
			False | 'int' | 'sum'
					'int' means 'integrate'
			False: max=1
			'sum': beam.sum()==1
			'int': (beam * dtheta).sum()==1
		'''
		from jizhipy.Astro import Beam
		return Beam.GaussianBeam(fwhm, theta, normalized)





	def _ReduceSTD(self, dimension, std_in, sigma_in, sigma_g):
		'''
		dimension:
			'1D' | '2D'
	
		std_in:
			standard deviation of the input data, a.std()
		sigma_in:
			unit: pixel
			resolution of the input data
			sigma_in = fwhm_in / (8 ln2)**0.5
			For signal convolved by beam, sigma_in~sigma_beam
			For pure Gaussian noise, sigma_in=0=None
	
		sigma_g:
			unit: pixel
			sigma of the Gaussian filter
		'''
		import numpy as np
		sigma_in = None if(sigma_in is None or sigma_in is False or sigma_in<1e-6)else float(sigma_in)
		if (str(dimension).lower() == '1d') : 
			if (sigma_in is None or sigma_in == 0) : 
				return std_in / (np.pi * sigma_g)**0.5
			else : 
				sig = (sigma_g**2 + sigma_in**2)**0.5
				return std_in / (sig / sigma_in)**0.5
		#----------------------------------------
		elif (str(dimension).lower() == '2d') : 
			if (sigma_in is None or sigma_in == 0) : 
				return std_in / (2 * np.pi**0.5 * sigma_g)
			else : 
				sig = (sigma_g**2 + sigma_in**2)**0.5
				return std_in / (sig / sigma_in)



	def _SigmaR( self, dimension, std_a, sigma_a, std_b, sigma_g=None, r=None ) :
		'''
		std_a:
			standard deviation of data a: a.std()
		sigma_a:
			unit: pixel
			resolution of the input data
			sigma_in = fwhm_in / (8 ln2)**0.5
		*** a can be signal convolved by beam, sigma_a~sigma_beam, OR pure noise, sigma_a=0=None
	
		std_b: 
			standard deviation of Gaussian-Noise b: b.std()
		*** b must be pure noise, therefore NOT sigma_b
	
		*** r is signal to noise ratio (SNR)

		(case 1) sigma_g != None and r == None:
			use sigma_g to smooth a then obtain a1
			use sigma_g to smooth b then obtain b1
			return: SNR=a1.std()/b1.std()
	
		(case 2) r != None and sigma_g == None:
			want r = a1.std()/b1.std()
			need how large sigma_g (return)

		(case 3) r == None and sigma_g == None:
			return [r_limit, sigma_g_limit]
		'''
		import numpy as np
		from jizhipy.Basic import Raise
		# case 1
		if (sigma_g is not None) : 
			stda = self._ReduceSTD(dimension, std_a, sigma_a, sigma_g)
			stdb = self._ReduceSTD(dimension, std_b, None, sigma_g)
			r = stda / stdb
			return r
		#----------------------------------------
		else : 
			def wei3( x ) : 
				x = ('%.9f' % x).split('.')
				if (len(x[0]) >= 3) : x = x[0]
				elif (len(x[0]) == 2) : x = x[0]+'.'+x[1][0]
				else : 
					if (x[0] != '0') : x = x[0]+'.'+x[1][:2]
					else : 
						for i in range(len(x[1])) : 
							if (x[1][i] != '0') : break
						x = '0.' + i*'0' + x[1][i:i+3]
				return x
			#----------------------------------------
			if (str(dimension).lower() == '1d') : 
				n0 = (np.pi * sigma_a)**0.5 * std_a / std_b
				n0 *= 0.9  # r_limit
				relim = True if(r is None)else False
				if (r is None) : r = n0
				elif (r > n0) : 
					n0str = wei3(n0)
					Raise(Warning, 'input n='+('%.1f' % n)+' reaches n_limit='+n0str, top=True)
					r = n0
				up = r**4 * std_b**4 * sigma_a**2
				down0 = np.pi**2 * std_a**4 * sigma_a**2
				down1 = r**4 * std_b**4
				sigma_g = (up / (down0 - down1))**0.5
				if (relim) : return [n0, sigma_g]
				else : return sigma_g
			#----------------------------------------
			elif (str(dimension).lower() == '2d') : 
				n0 = 2 * np.pi**0.5 * sigma_a * std_a / std_b
				n0 *= 0.9
				relim = True if(r is None)else False
				if (r is None) : r = n0
				elif (r > n0) : 
					n0str = wei3(n0)
					Raise(Warning, 'input n='+('%.1f' % n)+' reaches n_limit='+n0str, top=True)
					r = n0
				up = r**2 * std_b**2 * sigma_a**2
				down0 = 4*np.pi * std_a**2 * sigma_a**2
				down1 = r**2 * std_b**2
				sigma_g = (up / (down0 - down1))**0.5
				if (relim) : return [n0, sigma_g]
				else : return sigma_g



	def ReduceStd( self, dimension, std_a, sigma_a, std_b=None, sigma_g=None, r=None ) : 
		'''
		(case 1)
			ReduceStd(dimension, std_a, sigma_a, sigma_g)
				return std_a_reduced
		(case 2) 
			1. ReduceStd(dimension, std_a, sigma_a, std_b)
				return [r_lim, sigma_g_lim]
			2. ReduceStd(dimension, std_a, sigma_a, std_b, sigma_g)
				return r
			3. ReduceStd(dimension, std_a, sigma_a, std_b, r)
				return sigma_g


		\sum( Sout * Sin ) / N = (Sout * Sin).mean() = std(Sin)**2 / (1+g**2)**(3/4)   # or **(2/3)
		'''
		if (std_b is None or std_b is False) : return self._ReduceSTD( dimension, std_a, sigma_a, sigma_g )
		else : return self._SigmaR( dimension, std_a, sigma_a, std_b, sigma_g, r )





Gaussian = Gaussian()
