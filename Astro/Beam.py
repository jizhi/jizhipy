
class SyntheticBeam( object ) : 


	def WaveVector( self, thetaphi=None, widthshape=None, freq=None ) : 
		'''
		\vec{k} = 2pi/lambda * (xk, yk, zk)
		(xk, yk, zk) = thetaphi2xyz(thetaphi)
		(1) WaveVector( thetaphi, freq ) : set self.k, self.thetaphi, self.freq
		(2) WaveVector( thetaphi ) : set self.thetaphi

		self.thetaphi.shape = (2, N_sky-direction)
		self.k.shape = (3, N_sky-direction)

		Generate thetaphi matrix:
		thetaphi, widthshape: use one of them
		(1) give thetaphi argument: 
		thetaphi: 
			thetaphi.shape = (2, N_sky-direction)
			2: (theta, phi) in [rad]
				direction of incident on celestial shpere
				Ground experiment: is half celestial sphere
				Space  experiment: is all  celestial sphere

		(2) give widthshape argument:
			widthNpix = (thetawidth, shape)
				thetawidth: 1 value/float in [rad]
				shape: N | (N,) | (N1, N2)
			Call Beam.ThetaPhiMatrix(thetawidth, Npix)

		freq:
			[MHz], observation frequency
			must be 1 value
			If given, calculate \vec{k}

			k.shape = (3, N_sky-direction)
			3: (xk, yk, zk), NOTE THAT have *2pi/lambda
			N_sky-direction: direction of the incident light

		NOTE THAT (xk, yk, zk) and thetaphi must be in the SAME coordinate system of self.feedpos !!!
		'''
		import numpy as np
		from jizhipy.Transform import CoordTrans
		from jizhipy.Astro import Beam
		from jizhipy.Basic import IsType
		from jizhipy.Array import Asarray
		
		# self.thetaphi
		if (thetaphi is not None) : 
			thetaphi = np.array(thetaphi)
			if (thetaphi.size == 2) : thetaphi = thetaphi.reshape(2, 1)
		else : 
			thetawidth, shape = widthshape
			shape = Asarray(shape, int).flatten()
			self.width = float(thetawidth)
			self.shape = tuple(shape)
			if (shape.size == 1) : 
				theta = np.linspace(-thetawidth/2., thetawidth/2., shape[0])
				thetaphi = np.array([theta, 0*theta])
			else : 
				thetaphi = Beam.ThetaPhiMatrix(thetawidth, shape.max())
				n = shape.max() - shape.min()
				n1, n2 = n/2, n-n/2
				if (shape[0] < shape[1]) : 
					thetaphi = thetaphi[:,n1:-n2]
				elif (shape[0] > shape[1]) : 
					thetaphi = thetaphi[:,:,n1:-n2]
		self.thetaphi = thetaphi
		#----------------------------------------
		if (freq is not None) : 
			freq = float(freq)
			k = 2*np.pi / (300./freq) * CoordTrans.thetaphi2xyz(self.thetaphi)
			self.k, self.freq = k, freq





	def FeedPosition( self, xyz=None, thetaphir=None ) : 
		'''
		return: 
			self.feedpos .shape = (3, Nfeed)

		xyz, thetaphir: use one of them

		xyz: 
			position of each feed (note that NOT baseline) in xyz coordinate
			unit [meter]
			xyz.shape = (3, Nfeed)
			3: (x, y, z) of the position

		thetaphir:
			position of each feed (note that NOT baseline) in spherical coordinate
			thetaphir.shape = (3, Nfeed)
			3: (theta, phi, r)
				theta, phi are in [rad], r is in [meter]
				(theta, phi) gives the direction, r gives the distance
				Call jp.Coord.Trans.thetaphi2xyz()
			Nfeed: total number of feed

		NOTE THAT is NOT the baseline rij = ri - rj, only ri
		'''
		import numpy as np
		from jizhipy.Transform import CoordTrans
		if (xyz is not None) : 
			pos = np.array(xyz)
			if (pos.size == 3) : pos = pos.reshape(3, 1)
		elif (thetaphir is not None) : 
			thetaphir = np.array(thetaphir)
			if (thetaphir.size == 3) : thetaphir = thetaphir.reshape(3, 1)
			pos = CoordTrans.thetaphi2xyz(thetaphir[:2], thetaphir[2])
		self.feedpos = pos





	def Baseline( self, xyz=None, thetaphir=None ) : 
		'''
		return:
			self.baseline .shape = (3, Nbl)

		xyz, thetaphir: use one of them

		xyz: 
			baseline in xyz coordinate, note that NOT position of each feed
			unit [meter]
			xyz.shape = (3, Nbl)
			3: (x, y, z) of baseline

		thetaphir:
			baseline in spherical coordinate, note that NOT position of each feed
			thetaphir.shape = (3, Nbl)
			3: (theta, phi, r)
				theta, phi are in [rad], r is in [meter]
				(theta, phi) gives the direction, r gives the length
				Call jp.Coord.Trans.thetaphi2xyz()
			Nbl: total number of baseline

		NOTE THAT is baseline rij = ri - rj, NOT ri
		'''
		import numpy as np
		from jizhipy.Transform import CoordTrans
		if (xyz is not None) : 
			bl = np.array(xyz)
			if (bl.size == 3) : bl = bl.reshape(3, 1)
		elif (thetaphir is not None) : 
			thetaphir = np.array(thetaphir)
			if (thetaphir.size == 3) : thetaphir = thetaphir.reshape(3, 1)
			bl = CoordTrans.thetaphi2xyz(thetaphir[:2], thetaphir[2])
		self.baseline = bl





	def FeedBeam( self, *args, **kwargs ) : 
		'''
		each beam of feed OR combined beam of pair, depending on self.feedpos and self.baseline

		For baseline case, self.beam must be real, NOT complex

		*args: give fwhm: 
			in [rad]
			(1) 1 value: all beam are the same
					args = (1, )
			(2) fwhm.size = Nfeed: each feed has its fwhm
					args = (1, 2, 3, 4, ...)

		**kwargs:
			sqrt=True: return beam**0.5
			Default  : return power beam

		(1) FeedBeam(): Uniform beam
		(2)	FeedBeam(fwhm): GaussianBeam, NOT SincBeam
		(3) FeedBeam(fwhm1, fwhm2): EllipticGaussianBeam

		self.beam
			(case 1) =1
			(case 2) .shape = (1, N_incident_direction)
			(case 3) .shape = (Nfeed/Nbl, N_incident_direction)
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import IsType, Raise
		from jizhipy.Astro import Beam
		if (len(args) == 0 or args[0] is None) : 
			self.beam = 1
			return
		if ('thetaphi' not in self.__dict__.keys()) : Raise(Exception, 'NOT exists  self.thetaphi,  set it by SyntheticBeam.WaveVector()')
		if (len(args) == 1) : fwhm1 = fwhm2 = Asarray(args[0])
		else : fwhm1, fwhm2 = args[:2]
		self.beam = []
		for i in range(len(fwhm1)) : 
			self.beam.append( Beam.EllipticGaussianBeam(fwhm1[i], fwhm2[i], self.thetaphi[0], self.thetaphi[1]) )
		self.beam = np.array(self.beam)
		if ('sqrt' in kwargs.keys() and kwargs['sqrt']) : 
			self.beam = abs(self.beam)**0.5
		return self.beam





	def SyntheticBeam(self, comm=None, per=30, verbose=False):
		'''
		per:
			to speed up the calculate

		SyntheticBeam = |\sum_{i=1}^{Nfeed} \exp{1j \vec{k} \vec{r}_i} B_i }|^2

		self.k.shape = (3, N_incident_direction)
		self.feedpos.shape = (3, Nfeed)
		self.beam.shape = (Nfeed, N_incident_direction)

		kr.shape = (Nfeed, N_incident_direction)
		'''
		import numpy as np
		from jizhipy.Process import ProgressBar, MPI
		from jizhipy.Basic import Raise, IsType
		if (comm is not None) : 
			rank, Nprocess, host, info = MPI.Comm(comm)
		else : rank = 0
		#----------------------------------------
		onebeam = True
		if ('beam' not in self.__dict__.keys() or IsType.isnum(self.beam)) : beam = 1
		elif (len(self.beam)==1) : beam = self.beam[0]
		else : onebeam = False
		#----------------------------------------
		#----------------------------------------
		try : R, isbl = self.feedpos+0, False
		except : 
			try : R, isbl = self.baseline+0, True
			except : Raise(Exception, 'self.feedpos and self.baseline are NOT valid')
		k = self.k[:,None]
		R = R.reshape( R.shape + (1,)*len(k.shape[2:]) )
		#----------------------------------------
		if (isbl) : Bsyn_sqrt = np.zeros(k.shape[2:])
		else : Bsyn_sqrt = np.zeros(k.shape[2:], complex)
		n = np.arange(0, R.shape[1], per)
		n = np.append(n, [R.shape[1]])
		Nfor = n.size - 1
		if (verbose) : progressbar = ProgressBar('jizhipy.Beam.SyntheticBeam:', Nfor)
		for i in range(Nfor) : 
			if (verbose) : progressbar.Progress()
			n1, n2 = n[i], n[i+1]
			kr = (k*R[:,n1:n2]).sum(0)
			if (not isbl) : 
				if (onebeam) : Bsyn_sqrt_i = np.exp(1j*kr)
				else: Bsyn_sqrt_i = np.exp(1j*kr) *beam[n1:n2]
			else : 
				if (onebeam) : Bsyn_sqrt_i = 2*np.cos(kr)
				else : Bsyn_sqrt_i = 2*np.cos(kr) *beam[n1:n2]
			Bsyn_sqrt += Bsyn_sqrt_i.sum(0)
		if (onebeam) : Bsyn_sqrt *= beam
		#----------------------------------------
		if (comm is None) : 
			if (not isbl) : self.synbeam = abs(Bsyn_sqrt)**2
			else : self.synbeam = Bsyn_sqrt
			self.synbeam /= self.synbeam.max()  # normalsized
			return
		#----------------------------------------
		Bsyn_sqrt = MPI.Gather(Bsyn_sqrt[None,:], comm, 0, 0)
		if (rank == 0) : 
			Bsyn_sqrt = Bsyn_sqrt.sum(0)
			if (not isbl) : self.synbeam = abs(Bsyn_sqrt)**2
			else : self.synbeam = Bsyn_sqrt
			self.synbeam /= self.synbeam.max()
		else : self.synbeam = None
		self.synbeam = MPI.Bcast(self.synbeam, comm, 0)
		return self.synbeam










class Beam( object ) : 


	def __init__( self ) : 
		self.SyntheticBeam = SyntheticBeam



	def _SymmetryBeam( self, which, fwhm, theta, normalized ):
		'''
		Generate any-dimension beam

		theta, fwhm:
			in rad
			Can be any shape

		normalized:
			False | 'int' | 'sum'
					'int' means 'integrate'
			False: max=1
			'sum': beam.sum()==1
			'int': (beam * dtheta).sum()==1
		
		return:
			beam, ndarray
			beam.shape = fwhm.shape + theta.shape
	
		GaussianBeam = exp{-(theta/FWHM)^2/2/sigma^2}, its max=1, not normalized.
	
		Set theta=0.5*FWHM, compute exp{-(1/2)^2/2/sigma^2}=0.5(because the max=1, theta=0.5*FWHM will decreate to half power) and get sigma^2=1/(8ln2)
	
		\int{e^(-a*x^2)dx} = (pi/a)^0.5
		So, for the normalized GaussianBeam_normalized 
		= 1/FWHM * (4ln2/pi)^0.5 * exp(-4ln2 * theta^2 / FWHM^2)
	
		SincBeam = np.sinc(2.783/np.pi * theta**2 / fwhm**2)
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import IsType
		islistfwhm = False if(IsType.isnum(fwhm))else True
		islisttheta = False if(IsType.isnum(theta))else True
		fwhm, theta = Asarray(fwhm), Asarray(theta)
		shapefwhm, shapetheta = fwhm.shape, theta.shape
		fwhm, theta = fwhm.flatten(), theta.flatten()
		#----------------------------------------
		which = str(which).lower()
		if (which == 'gaussian') : b = np.exp(-4*np.log(2) * theta[None,:]**2 / fwhm[:,None]**2)
		elif (which == 'sinc') : b = np.sinc(2.783/np.pi * theta[None,:]**2 / fwhm[:,None]**2)**2
		#----------------------------------------
		if (normalized is not False) : 
			if (normalized == 'int') : 
				a = 1/fwhm[:,None] *(4*np.log(2)/np.pi)**0.5
				b *= a  # for 'int'
			elif (normalized == 'sum') : 
				b /= b.sum(1)[:,None]
		#----------------------------------------
		if (not islistfwhm and not islisttheta) : 
			b = b[0,0]
		elif (not islistfwhm and islisttheta) : 
			b = b.reshape(shapetheta)
		elif (islistfwhm and not islisttheta) : 
			b = b.reshape(shapefwhm)
		else : b = b.reshape(shapefwhm + shapetheta)
		#----------------------------------------
		return b





	def GaussianBeam( self, fwhm, theta, normalized=False, **kwargs ) : 
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

		**kwargs:
			only accept one key word: which='sinc'

		return:
			beam, ndarray
			beam.shape = fwhm.shape + theta.shape
		'''
		import numpy as np
		if ('which' in kwargs.keys()) : which = str(kwargs['which']).lower()
		else : which = 'gaussian'
		theta = np.array(theta)
		beam2d = True if(theta.size>1 and len(theta.shape)==2 and 1 in theta.shape)else False
		if (not beam2d) : return self._SymmetryBeam(which, fwhm, theta, normalized)
		#----------------------------------------
		b = self._SymmetryBeam(which, fwhm, theta.flatten(), normalized)
		isnumfwhm = True if(len(b.shape)==1)else False
		if (isnumfwhm) : b = b[None,:]
		beam = []
		for i in range(len(b)) : 
			beam.append( b[i][:,None] * b[i][None,:] )
		if (isnumfwhm) : beam = beam[0]
		else : beam = np.array(beam)
		return beam





	def SincBeam( self, fwhm, theta, normalized=False ) : 
		'''
		fwhm:
			rad, any shape

		theta:
			rad, any shape
		** Special case:
		** 	theta.shape = (N,1) | (1,N)
		** 	theta is 2D, one of dimension =1
		** 	This case means that generate 2D sinc beam

		normalized:
			True: sum=1   |   False: max=1

		return:
			beam, ndarray
			beam.shape = fwhm.shape + theta.shape
		'''
		return self.GaussianBeam(fwhm, theta, normalized, which='sinc')





	def EllipticGaussianBeam( self, fwhm1, fwhm2, theta, phi, normalized=False ) : 
		'''
		theta, phi, fwhm1, fwhm2:
			in rad
			theta.shape == phi.shape, can be any shape
			fwhm1.shape == fwhm2.shape, can be any shape
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import Raise, IsType
		if (IsType.isnum(fwhm1)) : islistfwhm1 = False
		else : islistfwhm1 = True
		if (IsType.isnum(fwhm2)) : islistfwhm2 = False
		else : islistfwhm2 = True
		islistfwhm = bool(islistfwhm1 + islistfwhm2)
		if (IsType.isnum(theta)) : islisttheta = False
		else : islisttheta = True
		if (IsType.isnum(phi)) : islistphi = False
		else : islistphi = True
		islisttheta = bool(islisttheta + islistphi)
		fwhm1, fwhm2, theta, phi = Asarray(fwhm1), Asarray(fwhm2), Asarray(theta), Asarray(phi)
		shape1, shape2, shapet, shapep = fwhm1.shape, fwhm2.shape, theta.shape, phi.shape
		printstr = 'fwhm1.shape='+str(shape1)+', fwhm2.shape='+str(shape2)+', theta.shape='+str(shapet)+', phi.shape='+str(shapep)
		if (shape1 != shape2) : Raise(Exception, 'fwhm1.shape != fwhm2.shape. '+printstr)
		if (shapet != shapep) : Raise(Exception, 'theta.shape != phi.shape. '+printstr)
		#--------------------------------------------------
		fwhm1, fwhm2, theta, phi = fwhm1.flatten(), fwhm2.flatten(), theta.flatten(), phi.flatten()
		b = np.exp(-4*np.log(2) * theta[None,:]**2 * ((np.cos(phi[None,:])/fwhm1[:,None])**2 + (np.sin(phi[None,:])/fwhm2[:,None])**2))
		if (normalized) : 
			a = 4*np.log(2) * ((np.cos(phi[None,:])/fwhm1[:,None])**2 + (np.sin(phi[None,:])/fwhm2[:,None])**2)
			b = b/(np.pi/a)**0.5
			a = 0 #@
		#--------------------------------------------------
		if (not islistfwhm and not islisttheta) : b = b[0,0]
		elif (not islistfwhm and islisttheta) : 
			b = b.reshape(shapet)
		elif (islistfwhm and not islisttheta) : 
			b = b.reshape(shape1)
		else : b = b.reshape(shape1 + shapet)
		return b





	def ThetaPhiMatrix( self, thetawidth, Npix ) : 
		'''
		thetawidth:
			[rad] !!!
			total field of view (from left to rignt) in rad.
			Must be one
	
		Npix:
			theta and phi matrix shape = (Npix, Npix)
			Must be one int
	
		# At the center (Npix/2, Npix/2), theta = 0
	
		return:
			[theta, phi] in rad.

		theta: -thetawidth/2 -> 0 -> +thetawidth/2
		phi: 0--2*np.pi
		'''
		import numpy as np
		from jizhipy.Basic import Print
		thetawidth = float(thetawidth)
		Npix = Npix0 = int(round(Npix))
		if (Npix %2 == 0 ) : Npix += 1
		N = Npix / 2
		# Length of each pixel
		pixlist = np.arange(-N, N+1.)
		dx = thetawidth / (Npix-1)
		theta = dx * (pixlist[:,None]**2 + pixlist[None,:]**2)**0.5
		phi = pixlist[None,:] + 1j*pixlist[::-1,None]
		Print.WarningSet(False)
		phi = np.arctan(phi.imag / phi.real)
		Print.WarningSet(True)
		phi[N,N] = 0
		phi[:,:N] += np.pi
		phi[N+1:,N:] += 2*np.pi
		invalid = (theta > np.pi)
		theta[invalid], phi[invalid] = np.nan, np.nan
		theta, phi = theta[:Npix0,:Npix0], phi[:Npix0,:Npix0]
		return np.array([theta, phi])





Beam = Beam()














def BeamMap( pointRA=None, pointDec=None, dwl=None, freq=None, uniform=False, Bmap0=None, dtype='float32', nside=None ) : 
	'''
	pointRA, pointDec:
		[degree]
		Where does the antenna point to?
		default points to (RA, Dec) = (0, 0)

	dwl:
		[meter]
		d - diameter: for dish: one value
		w - width, l - length: for cylinder: (w,l)

	freq: 
		[MHz]
		Used to get FWHM

	uniform:
		True | False
		If ==True: return =1 map

	Bmap0:
		Use this Bmap0 as the basic and rotate it to (pointRA, pointDec)
	'''
	import healpy as hp
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise
	from jizhipy.Transform import CoordTrans
	try : dtype = np.dtype(dtype)
	except : dtype = np.dtype(None)
	if (nside is not None) : nside = int(round(nside))
	elif (Bmap0 is not None) : nside = hp.get_nside(Bmap0)
	else : Raise(Exception, 'nside = None')
	#--------------------------------------------------
	if (uniform) : 
		Bmap = np.ones(12*nside**2, dtype)
		if (Bmap0 is not None) : Bmap *= Bmap0[Bmap0.size/2]
		return Bmap
	#--------------------------------------------------
	if (Bmap0 is not None) : 
		nside0 = hp.get_nside(Bmap0)
		if (nside0 != nside) : Bmap0 = hp.ud_grade(nside, Bmap0)
		Bmap0 = Bmap0.astype(dtype)
	#--------------------------------------------------
	else : 
		n = hp.ang2pix(nside, np.pi/2, 0)
		Bmap0 = np.zeros(12*nside**2, dtype)
		Bmap0[n] = 10000
		D = Asarray(dwl)[0]
		fwhm = 1.03 * 300/freq / D
		Bmap0 = hp.smoothing(Bmap0, fwhm, verbose=False)
		Bmap0[Bmap0<0] = 0
		Bmap0 /= Bmap0.sum()
	#--------------------------------------------------
	if (pointRA is None) : pointRA = 0
	if (pointDec is None) : pointDec = 0
	if (abs(pointRA)<1e-4 and abs(pointDec)<1e-4) : return Bmap0
	#--------------------------------------------------
	theta, phi = hp.pix2ang(nside, np.arange(12*nside**2))
	theta, phi = CoordTrans.thetaphiRotation([theta, phi], az=pointRA, ay=-pointDec)
	n = hp.ang2pix(nside, theta, phi)
	Bmap = Bmap0[n]
	return Bmap




