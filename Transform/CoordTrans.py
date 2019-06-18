
def _DoMultiprocess_Celestial( iterable ) : 
	import ephem
	import numpy as np
	RAl, Decb = iterable[1]
	coordin, coordout, epochin, epochout = iterable[2]
	if   (coordin =='galactic')  : funcin  = ephem.Galactic
	elif (coordin =='equatorial'): funcin  = ephem.Equatorial
	elif (coordin =='ecliptic')  : funcin  = ephem.Ecliptic
	if   (coordout=='galactic')  : funcout = ephem.Galactic
	elif (coordout=='equatorial'): funcout = ephem.Equatorial
	elif (coordout=='ecliptic')  : funcout = ephem.Ecliptic
	x, y = [], []
	for i in range(len(RAl)) : 
		xy = funcin(RAl[i], Decb[i], epoch=epochin)
		xy = funcout(xy, epoch=epochout)
		if   (coordout == 'galactic') : 
			x.append(xy.lon +0)
			y.append(xy.lat +0)
		elif (coordout == 'equatorial') : 
			x.append(xy.ra  +0)
			y.append(xy.dec +0)
		if   (coordout == 'ecliptic') : 
			x.append(xy.lon +0)
			y.append(xy.lat +0)
	return np.array([x, y])





class CoordTrans( object ) : 


	def _CoordInOut( self, coordin, coordout ) : 
		'''
		return registered coordin and coordout
			'equatorial', 'galactic', 'ecliptic'
		'''
		from jizhipy.Baise import Raise
		raisestr = "coordin='"+str(coordin)+"', coordout='"+str(coordout)+"' not in ['galactic', 'equatorial', 'ecliptic']"
		coordin, coordout = str(coordin).lower(), str(coordout).lower()
		if   ('gal' == coordin[:3]) : coordin = 'galactic'
		elif ('equ' == coordin[:3]) : coordin = 'equatorial'
		elif ('ecl' == coordin[:3]) : coordin = 'ecliptic'
		else : Raise(Exception, raisestr)
		if   ('gal' == coordout[:3]) : coordout = 'galactic'
		elif ('equ' == coordout[:3]) : coordout = 'equatorial'
		elif ('ecl' == coordout[:3]) : coordout = 'ecliptic'
		else : Raise(Exception, raisestr)
		return [coordin, coordout]



	def _EpochInOut( self, epochin, epochout ) : 
		'''
		epoch:
			give and return: str or ephem property
			(1) '2000' or ephem.J2000
			(2) '1950' or ephem.B1950
			(3) '1900' or ephem.B1900
			(4) other number
		'''
		def _Epoch( epoch ) : 
			from jizhipy.Baise import IsType
			if (IsType.isstr(epoch) or IsType.isnum(epoch)) : 
				epoch = str(epoch)
			#	if   ('2000' in epoch) : epoch = ephem.J2000
			#	elif ('1950' in epoch) : epoch = ephem.B1950
			#	elif ('1900' in epoch) : epoch = ephem.B1900
			#	else : epoch = str(epoch)
			return epoch
		return [_Epoch(epochin), _Epoch(epochout)]



	def _RAlDecb( self, RAl, Decb ) : 
		'''
		Make RAl and Decb can broadcast
		return:
			[RAl, Decb, islist]
		'''
		from jizhipy.Basic import IsType
		from jizhipy.Array import Asarray
		if (IsType.isnum(RAl) and IsType.isnum(Decb)) : 
			islist = False
		else : islist = True
		RAl, Decb = Asarray(RAl), Asarray(Decb)
		try : 
			RAl  = RAl + Decb*0
			Decb = RAl*0 + Decb
		except : Raise(Exception, 'RAl.shape='+str(RAl.shape)+', Decb.shape='+str(Decb.shape)+', can NOT broadcast')
		return [RAl, Decb, islist]





	def ToHealpix( self, nside, RA, Dec=None, T=None, replace=True ) : 
		''' 
		replace:
			=True : if there are several values in 1 pixel, use the lagest value to fill this pixel (for diffuse emission)
			=False : if there are several values in 1 pixel, fill this pixel with the sum of all values (for point sources)
	
		RA, Dec:
			in rad
			RA  or lon : 0 ~ 2pi
			Dec or lat : -pi/2 ~ pi/2
	
		if Dec=None and T=None : 
			RA, Dec, T = RA[0], RA[1], RA[2] = RA  (ROW)
		'''
		import healpy as hp
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import Raise
		replace = bool(replace)
		hp.nside2resol(nside)
		if (Dec is None and T is None) : RA, Dec, T = RA
		RA, Dec, T = 1*Asarray(RA).flatten(), 1*Asarray(Dec).flatten(), 1*Asarray(T).flatten()
		size = max(RA.size, Dec.size, T.size)
		if ((RA.size not in [size,1]) or (Dec.size not in [size,1]) or (T.size not in [size,1])) : Raise(Exception, 'jizhipy.CoordTrans.ToHealpix(): RA.size='+str(RA.size)+', Dec.size='+str(Dec.size)+', T.size='+str(T.size)+' not in ['+str(size)+',1]')
		if ( RA.size == 1) :  RA = np.zeros(size)+RA[0]
		if (Dec.size == 1) : Dec = np.zeros(size)+Dec[0]
		if (  T.size == 1) :   T = np.zeros(size)+T[0]
		#----------------------------------------
		if (replace) : 
			T = np.sort(T + 1j*np.arange(size))
			n = T.imag.astype(int)
			T = T.real
			RA, Dec = RA[n], Dec[n]
		#----------------------------------------
		hpmap = np.zeros(12*nside**2)
		pix = hp.ang2pix(nside, np.pi/2-Dec, RA)
		if (replace) : 
			hpmap[pix] = T
			return hpmap
		#----------------------------------------
		for i in range(pix.size) : 
			hpmap[pix[i]] += T[i]
		return hpmap





	def Celestial( self, RAl, Decb, coordin, coordout, epochin='2000', epochout='2000', Nprocess=None ) : 
		'''
		(RA,Dec) <=> (l,b)  conversion

		RAl, Decb:
			in rad, NOT degree
			Can be any shape
			But must can broadcast

		coordin, coordout:
			'Equatorial', 'Galactic', 'Ecliptic'

		return:
			[RAl, Decb]
				in rad
				with same shape as (RAl+Decb).shape
		'''
		from jizhipy.Process import PoolFor, NprocessCPU
		import numpy as np
		Nprocess = NprocessCPU(Nprocess)[0]
		coordin, coordout=self._CoordInOut(coordin, coordout)
		epochin, epochout=self._EpochInOut(epochin, epochout)
		if (coordin==coordout and epochin==epochout) : 
			return np.array([RAl%(2*np.pi), Decb])
		RAl, Decb, islist = self._RAlDecb(RAl, Decb)
		shape = RAl.shape
		sent = [RAl.flatten(), Decb.flatten()]
		bcast = [coordin, coordout, epochin, epochout]
		if (Nprocess <= 1) : 
			x, y=_DoMultiprocess_Celestial([None,sent, bcast])
		else : 
			pool = PoolFor(0, len(RAl), Nprocess)
			xy=pool.map_async(_DoMultiprocess_Celestial, sent,bcast)
			x, y = np.concatenate(xy, 1)
		x, y = x.reshape(shape), y.reshape(shape)
		if (not islist) : x, y = x[0], y[0]
		return np.array([x, y])


	### Healpix, (coordin, epochin) => (coordout, epochout) ###


	def _Nside( self, nside ) : 
		import numpy as np
		n = np.log(nside) / np.log(2)
		if (n != int(n)) : Raise(Exception, 'nside='+str(nside)+' != 2**n')
		return int(round(nside))



	def _Ordering( self, ordering ) : 
		ordering = str(ordering).upper()
		if('NEST' in ordering): ordering, nest ='NESTED',True
		else : ordering, nest = 'RING', False
		return [ordering, nest]



	def CelestialHealpix( self, healpixmap, ordering, coordin, coordout, epochin='2000', epochout='2000', Nprocess=None ) : 
		'''
		return:
			Case (1) [healpixmap_in2out, hpix_in2out]
			Case (2) hpix_in2out

		Usage: healpixmap_in2out = healpixmap[hpix_in2out]

		HealPIX map 'Equatorial' <=> 'Galactic'

		healpixmap:
			(1) np.ndarray with shape=(12*nside**2,)
			(2) int number =nside

		ordering:
			'RINGE' | 'NESTED'
		'''
		import healpy as hp
		import numpy as np
		coordin, coordout=self._CoordInOut(coordin, coordout)
		epochin, epochout=self._EpochInOut(epochin, epochout)
		try : healpixmap = healpixmap[:]
		except : pass
		try : nside = hp.get_nside(healpixmap)
		except : 
			nside = self._Nside(healpixmap)
			healpixmap = None
		if (coordin==coordout and epochin==epochout) : 
			if (healpixmap is None) : return np.arange(12*nside**2)
			else: return [healpixmap, np.arange(12*nside**2)]
		ordering, nest = self._Ordering(ordering)
		Decb, RAl = hp.pix2ang(nside, np.arange(12*nside**2), nest=nest)
		Decb = np.pi/2 - Decb
		RAl, Decb = self.Celestial(RAl, Decb, coordout, coordin, epochout, epochin, Nprocess)
		hpix_in2out=hp.ang2pix(nside, np.pi/2-Decb,RAl, nest=nest) 
		try : return [healpixmap[hpix_in2out], hpix_in2out]
		except : return hpix_in2out


	################# xyz, theta, Rotation #################


	def _Angle( self, kwargs, orderkwargs ) : 
		''' degree to rad, not call directly '''
		from jizhipy.Basic import IsType
		import numpy as np
		from jizhipy.Array import Asarray
		N = len(orderkwargs)
		islist = []
		raisestr = 'Shape miss-matching, '
		for i in range(N) : 
			ok = orderkwargs[i]
			if (kwargs[ok] is None) : kwargs[ok] = 0
			islist.append( 1-IsType.isnum(kwargs[ok]) )
			kwargs[ok] = Asarray(kwargs[ok]) *np.pi/180
			raisestr += ok+'.shape='+str(kwargs[ok].shape)+', '
		try : 
			for i in range(N) : 
				oki = orderkwargs[i]
				for j in range(N) : 
					okj = orderkwargs[j]
					if (i == j) : continue
					kwargs[oki] = kwargs[oki] + kwargs[okj]*0  # broadcast
		except : Raise(Exception, raisestr)
		if (np.array(islist).sum() == 0) : islist = False
		else : islist = True
		return [kwargs, islist]



	def _RotationMatrix( self, key, ang, islist ) : 
		''' return R, not call directly '''
		from jizhipy.Array import ArrayAxis
		import numpy as np
		one  = np.ones(ang.shape)
		zero = np.zeros(ang.shape)
		if (key == 'ax') : 
			R = np.array([[ one,     zero    ,    zero    ],
			              [zero,  np.cos(ang), np.sin(ang)],
			              [zero, -np.sin(ang), np.cos(ang)]])
		elif (key == 'ay') : 
			R = np.array([[np.cos(ang), zero, -np.sin(ang)],
			              [   zero   ,   one,     zero    ],
			              [np.sin(ang), zero,  np.cos(ang)]])
		elif (key == 'az') : 
			R = np.array([[ np.cos(ang), np.sin(ang), zero],
			              [-np.sin(ang), np.cos(ang), zero],
			              [    zero    ,    zero    ,  one]])
		else : Raise(Exception, "key in **kwargs not in ['ax', 'ay', 'az']")
		R = ArrayAxis(R, 0, -1, 'move')
		R = ArrayAxis(R, 0, -1, 'move')
		if (not islist) : R = R[0]
		return R




	def xyzRotationMatrix( self, which, **kwargs ) : 
		'''
		Right-hand rule:
			Your right thumb points along the +Z axis and the curl of your fingers represents a motion from +X to +Y to -X to -Y. When viewed from the top along -Z, the system is counter-clockwise.
			The angle along the fingers is positive.

		'which': 
			'system' | 'point'
			Rotate coordinate system or point

		**kwargs:
			(1) kwargs={'ax':, 'ay':, 'az':, 'order':}
				Only allow keys 'ax', 'ay', 'az', 'order', all the keys are options
				'order'=['ay', 'ax', ...] like
				Rotate about which axis with how many degree, 'order' is to determinate rotate which axis first, then which axis, ....

			(2) kwargs={'atheta':, 'aphi':, 'ang'}
				Only allow keys 'atheta', 'aphi', 'ang'
				('atheta', 'aphi') reprecents the rotation axis (any axis), 'ang' is the rotation angle (right-handed)
				NOTE THAT atheta is theta, NOT lat


		Case (1)
		xyzRotationMatrix(ay, ax, az)    OR
		xyzRotationMatrix(ax, ay, az, order=['ay','ax','az']) 
			First rotate about Y-axis, second rotate about X-axis, third rotate about Z-axis

		ax: Rotation angle about X-axis, thumb points to +X
		ay: Rotation angle about Y-axis, thumb points to +Y
		az: Rotation angle about Z-axis, thumb points to +Z
		order: give the order manually, order=['az','ax','ay']

		ax, ay, az:
			in degree, NOT rad
			Can be one or N-D array which can broadcast


		Case (2)
		xyzRotationMatrix(atheta, aphi, ang)
			Rotate about axis (atheta, aphi) with ang

		atheta, aphi, ang: 
			all in degree
			Can be one or N-D array which can broadcast


		return:
			Rotation matrix R in np.array(), NOT np.matrix()
				new_xyz = R * old_xyz

			R.shape = ax.shape+(3,3)
		'''
		from jizhipy.Basic import Raise, OrderKwargs
		# Check kwargs
		case1, case2 = False, False
		if ('ax' in kwargs.keys() or 'ay' in kwargs.keys() or 'az' in kwargs.keys()) : case1 = True
		if ('atheta' in kwargs.keys() or 'aphi' in kwargs.keys() or 'ang' in kwargs.keys()) : case2 = True
		if (case1 and case2) : Raise(Exception, 'jizhipy.CoordTrans.xyzRotationMatrix(), have both (ax,ay,az,order) and (atheta,aphi,ang), NOT allow')
		if (case1) : 
			return self._xyzRotationMatrix1(**kwargs)
		elif (case2) : 
			return self._xyzRotationMatrix2(**kwargs)





	def _xyzRotationMatrix1( self, **kwargs ) : 
		''' case 1, input (x, y, z) '''
		from jizhipy.Basic import OrderKwargs
		import numpy as np
		try : orderkwargs = kwargs['order']
		except : orderkwargs = OrderKwargs(2)
		N = len(orderkwargs)
		kwargs, islist = self._Angle(kwargs, orderkwargs)
		R = []
		for i in range(N) : 
			ok = orderkwargs[i]
			Ri = self._RotationMatrix(ok, kwargs[ok], islist)
			if (not islist) : Ri = Ri[None,:]
			shape = Ri.shape  #@
			Ri = Ri.reshape(np.prod(shape[:-2]), 3, 3)
			R.append(Ri)
		for i in range(1, N) : 
			for j in range(len(R[0])) : 
				R[0][j] = np.array(np.matrix(R[i][j]) * np.matrix(R[0][j]))
		R = R[0]
		R = R.reshape(shape)
		if (not islist) : R = R[0]
		return R





	def _xyzRotationMatrix2( self, atheta, aphi, ang ) : 
		''' case 2, input (theta, phi, ang) '''
		from jizhipy.Basic import OrderKwargs
		import numpy as np
		try : orderkwargs = kwargs['order']
		except : orderkwargs = OrderKwargs(2)
		N = len(orderkwargs)
		kwargs, islist = self._Angle(kwargs, orderkwargs)
		R = []
		for i in range(N) : 
			ok = orderkwargs[i]
			Ri = self._RotationMatrix(ok, kwargs[ok], islist)
			if (not islist) : Ri = Ri[None,:]
			shape = Ri.shape  #@
			Ri = Ri.reshape(np.prod(shape[:-2]), 3, 3)
			R.append(Ri)
		for i in range(1, N) : 
			for j in range(len(R[0])) : 
				R[0][j] = np.array(np.matrix(R[i][j]) * np.matrix(R[0][j]))
		R = R[0]
		R = R.reshape(shape)
		if (not islist) : R = R[0]
		return R




	def _xyzRotation( self, xyz, **kwargs ) : 
		'''
		(x,y,z) coordinates rotation with ax,ay,az=

		xyz:
			xyz can be any shape, but must:
			x, y, z = xyz

			x = sin(theta) * cos(phi)
			y = sin(theta) * sin(phi)
			z = cos(theta)

		**kwargs:
			See self.xyzRotationMatrix()
			ax=None: means NOT rotate this axis
			(1) xyzRotation(xyz, ay=1, ax=2, az=3)
			(2) 
				ang = {'ay':1, 'ax':2, 'az':3, 'order':['ay','ax','az']}
				xyzRotation(xyz, **ang)

		return:
			Same shape and type as input xyz

			xyz_new.shape = xyz.shape+(3,)   # (3,) for x,y,z
		'''
		from jizhipy.Basic import OrderKwargs, Raise
		from jizhipy.Array import ArrayAxis, Asarray
		xyz = Asarray(xyz, True, float)
		shapexyz = xyz.shape
		if (shapexyz[0] != 3) : Raise(Exception, 'xyz.shape='+str(shapexyz)+', shape[0] != 3')
		if (xyz.shape == (3,)) : islistx = False
		else : islistx = True
		x, y, z = xyz
		#--------------------------------------------------
		try : kwargs['order']
		except : kwargs['order'] = OrderKwargs()
		if ('which' in kwargs.keys()) : 
			which = kwargs['which']
		else : which = 'system'
		R = self.xyzRotationMatrix(which, **kwargs)
		shapeR = R.shape
		R = ArrayAxis(R, -2, 0, 'move')  #@#@
		if (R.shape == (3,3)) : islistr = False
		else : islistr = True
		Rx, Ry, Rz = R.T
		Rx, Ry, Rz = Rx.T, Ry.T, Rz.T
		#--------------------------------------------------
		if (Rx.shape == (3,)) : 
			sR = Rx.shape + len(x.shape)*(1,)
			Rx, Ry, Rz = Rx.reshape(sR), Ry.reshape(sR), Rz.reshape(sR)
		try : xyz = x*Rx + y*Ry + z*Rz
		except : Raise(Exception, 'x.shape='+str(x.shape)+', Rx.shape='+str(Rx.shape)+', can NOT broadcast')
		return xyz



	def xyzRotationSystem( self, xyz, **kwargs ) : 
		'''
		Original:
			xyz in XYZ0 coordinate system
		Then: 
			Rotate XYZ0 system to NEW system XYZ1
			point xyz doesn't move
		Return:
			xyz in NEW system XYZ1
		'''
		keys = kwargs.keys()
		if('order' not in keys) : 
			from jizhipy.Basic import OrderKwargs
			kwargs['order'] = OrderKwargs()
		return self._xyzRotation(xyz, **kwargs)



	def xyzRotationPoint( self, xyz, **kwargs ) : 
		'''
		Original:
			xyz in XYZ0 coordinate system
		Then: 
			Rotate point xyz to NEW point xyz1
			system XYZ0 doesn't move
		Return:
			xyz1 in original system XYZ0
		'''
		keys = kwargs.keys()
		if('order' not in keys) : 
			from jizhipy.Basic import OrderKwargs
			kwargs['order'] = OrderKwargs()
		if ('ax' in keys) : kwargs['ax'] *= -1
		if ('ay' in keys) : kwargs['ay'] *= -1
		if ('az' in keys) : kwargs['az'] *= -1
		return self._xyzRotation(xyz, **kwargs)



	def xyzRotationPointAny( self, xyz, atheta, aphi, ang ) : 
		'''
		The result here is the same as:
			http://blog.csdn.net/zsq306650083/article/details/8773996

		atheta, aphi, ang:
			in degree, one value/scale

		Original:
			xyz in XYZ0 coordinate system

		Rotate an angle about any axis
			angle: 'ang'
			axis:  'atheta', 'aphi'

		(1) Rotate +X0 axis to "any axis"(atheta, aphi), NEW coordinate system XYZ1: 
				Rotate system, point doesn't move
			xyz1 = xyzRotationSystem(xyz, az=aphi, ay=-(90-atheta))
		(2) Rotate point xyz1 with angle ang about +X1
			xyz2 = xyzRotationPoint(xyz1, ax=ang)
		(3) xyz2 is final point in XYZ1, but we need xyz2 in XYZ0
			xyzf = xyzRotationSystem(xyz2, ay=(90-atheta), az=-aphi)
		'''
		xyz1 = self.xyzRotationSystem(xyz, az=aphi, ay=-(90-atheta))
		xyz2 = self.xyzRotationPoint(xyz1, ax=ang)
		xyzf = self.xyzRotationSystem(xyz2, ay=(90-atheta), az=-aphi)
		return xyzf





	def thetaphi2xyz( self, thetaphi, Amp=1 ) : 
		'''
		thetaphi:
			in rad, NOT degree
			Can be any shape, but must:
				theta, phi = thetaphi
			theta, phi can NOT have the same shape, but must can broadcast

			x = sin(theta) * cos(phi)
			y = sin(theta) * sin(phi)
			z = cos(theta)

		Amp:
			Must can broadcast with theta

		return:
			xyz = [x, y, z]
			x.shape = y.shape = z.shape = theta.shape = phi.shape
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import Raise
		theta, phi = thetaphi
		theta,phi,Amp =Asarray2(theta,True), Asarray2(phi,True), Asarray2(Amp,True)
		try : 
			x = Amp * np.sin(theta) * np.cos(phi)
			y = Amp * np.sin(theta) * np.sin(phi)
			z = Amp * np.cos(theta) * np.ones(phi.shape, phi.dtype)
		except : Raise(Exception, 'theta.shape='+str(theta.shape)+', phi.shape='+str(phi.shape)+', Amp.shape='+str(Amp.shape)+', can NOT broadcast')
		xyz = np.array([x, y, z])
		del x, y, z
		return xyz





	def xyz2thetaphi( self, xyz ) : 
		'''
		xyz:
			Can be any shape, but must:
			x, y, z = xyz

		return:
			thetaphi = [theta, phi]
			in rad (NOT degree)
			theta.shape = phi.shape = x.shape

		theta: 0--np.pi
		phi: 0--2*np.pi
		'''
		from jizhipy.Basic import Print, IsType
		import numpy as np
		x, y, z = xyz
		if (IsType.isnum(x)) : islistx = False
		else : islistx = True
		if (IsType.isnum(y)) : islisty = False
		else : islisty = True
		if (IsType.isnum(z)) : islistz = False
		else : islistz = True
		islist = bool(islistx + islisty + islistz)
		x, y, z = Asarray(x), Asarray(y), Asarray(z)
		r = (x**2 + y**2 + z**2)**0.5
		r[r==0] = 1e-30
		x, y, z = x/r, y/r, z/r
		#--------------------------------------------------
		# theta from 0 to 180, sin(theta) >=0
		theta = np.arccos(z)  # 0 to 180 degree
		zero = (np.sin(theta) == 0)
		sign = np.sign(z[zero])
		theta[zero] += 1e-30
		del z
	#	Print.WarningSet(False)
		x = x / np.sin(theta)
		y = y / np.sin(theta)
	#	Print.WarningSet(True)
		x = x + 1j*y
		del y
		phi = np.angle(x) % (2*np.pi)
		theta[zero] -= 1e-30
		if (zero.sum() > 0) : phi[zero] = np.pi/2 - sign*np.pi/2	
		if (not islist) : theta, phi = theta[0], phi[0]
		return np.array([theta, phi]) %(2*np.pi)





	def thetaphiRotation( self, thetaphi, **kwargs ) : 
		'''
		(theta, phi) coordinates rotation with ax,ay,az=

		thetaphi:
			in rad, NOT degree
			thetaphi can be any shape, but must:
			theta, phi = thetaphi

		return:
			thetaphi
		'''
		from jizhipy.Basic import OrderKwargs
		import numpy as np
		theta, phi = np.array(thetaphi, float)
		shape = theta.shape
		xyz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
		del theta, phi
		try : kwargs['order']
		except : kwargs['order'] = OrderKwargs()
		xyz = self._xyzRotation(xyz, **kwargs)
		thetaphi = self.xyz2thetaphi(xyz)
		return thetaphi





	def HealpixRotation( self, hpmapORnside, **kwargs ) : 
		'''
		Rotate HealPIX map with ax,ay,az=

		return:
			npix | (npix, rotated_healpy_map)

		npix, rehpmap = HealpixRotation( inhpmap )
		rehpmap == inhpmap[npix]

		+x: RA=0
		+y: RA=90
		+z: North pole
		'''
		from jizhipy.Basic import OrderKwargs, IsType
		import healpy as hp
		import numpy as np
		kwargs['order'] = OrderKwargs()[::-1]
		keys = kwargs.keys()
		for i in range(len(keys)) : 
			if (len(keys[i])==2 and keys[i][0]=='a') : 
				kwargs[keys[i]] *= -1
		#----------------------------------------
		if (IsType.isnum(hpmapORnside)) : nside = hpmapORnside
		else : nside = hp.get_nside(hpmapORnside)
		theta, phi = hp.pix2ang(nside, np.arange(12*nside**2))
		theta, phi=self.thetaphiRotation([theta,phi],**kwargs)
		npix = hp.ang2pix(nside, theta, phi)
		#----------------------------------------
		if (IsType.isnum(hpmapORnside)) : return npix
		else : return [npix, hpmapORnside[npix]]


	##################################################


	def Healpix2Flat( self, lon=None, lat=None, dlon=None, dlat=None, Nlon=None, Nlat=None, hpmap=None, nside=None, ordering='RING' ) : 
		'''
		npix, rehpmap = Healpix2Flat( hpmap=inhpmap )
		rehpmap == inhpmap[npix]

	(1) Healpix2Flat(lon, lat, dlon, dlat, Nlon, Nlat, hpmap)
	(2) Healpix2Flat(lon, lat, None, dlat, None, Nlat, hpmap)
	(3) Healpix2Flat(lon, lat, dlon, dlat, Nlon, Nlat, nside)
	(4) Healpix2Flat(lon, lat, None, dlat, None, Nlat, nside)
	(5) Healpix2Flat(hpmap): Nrow*Ncol = 4*12*nside**2
	(6) Healpix2Flat(nside)
	(7) Healpix2Flat( hpmap=remap, nside= ):
			give 2D flat map, return healpixmap
	return: npix | (npix, hpmap[npix])
		****** Order of maps:
			hp.mollview(): 
				left -> right (RA) : 180 -> 0 -> -180
				bottom -> top (Dec): -90 -> 0 -> 90

			plt.pcolormesh():
				left -> right (RA) : 180 -> 0 -> -180
				bottom -> top (Dec): -90 -> 0 -> 90

			np.array( hpmap[npix] ):
				col[0] -> col[N-1] (RA) : 180 -> 0 -> -180
				row[0] -> row[N-1] (Dec): -90 -> 0 -> 90

				(RA+180, Dec-90)  ...  (RA-180, Dec-90)
				(RA+180, Dec-80)  ...  (RA-180, Dec-80)
		array =	                  ... 
				(RA+180, Dec+90)  ...  (RA-180, Dec+90)
		****************************************


		lon, lat:
			in degree
			lon=RA or l, lat=Dec or b
			Center of the region

		dlon, dlat:
			in degree
			size of the region
			(1) dlon !=None, dlat !=None
			(2) dlon ==None, dlat !=None

		Nlon, Nlat:
			if (is None) : N = 2*int((12*nside**2)**0.5)
			if (is str): 
				'2'  : N = 2   * int((12*nside**2)**0.5)
				'5.3': N = 5.3 * int((12*nside**2)**0.5)

		hpmap, nside:
			One of them must be !=None

		return: npix | (npix, hpmap[npix])

		NOTE THAT maybe not all npix are valid, use 
			mask = Invalid(theta).mask
		'''
		import numpy as np
		import healpy as hp
		from jizhipy.Astro import Beam
		from jizhipy.Array import Invalid
		from jizhipy.Basic import IsType
		if (lon is None and lat is None and nside is not None and hpmap is not None and len(hpmap.shape)==2) : 
			# case 7
			NDec, NRA = hpmap.shape
			Dec0, RA0 = -np.pi/2, np.pi
			dDec, dRA = np.pi/NDec, -2*np.pi/NRA
			theta, phi = hp.pix2ang(nside, np.arange(12*nside**2))
			theta = np.pi/2 - theta
			phi[phi>np.pi] -= 2*np.pi
			ntheta, nphi = ((theta-Dec0)/dDec).astype(int), ((phi-RA0)/dRA).astype(int)
			n = NRA*ntheta + nphi
			hpmap = hpmap.flatten()[n]
			return [n, hpmap]
		#---------------------------------------------
		#---------------------------------------------
		if (hpmap is None) : 
			hpmap, isnside = np.arange(12*nside**2), True
		else : 
			nside, isnside = hp.get_nside(hpmap), False
		nest =False if(str(ordering).lower()=='ring')else True
		#---------------------------------------------
		# Flat full sky
		if (lon is None and lat is None) : 
			N0 = 2*int(12**0.5*nside)
			if (Nlon is None) : Nlon = N0
			elif(IsType.isstr(Nlon)): Nlon=int(float(Nlon)*N0)
			if (Nlat is None) : Nlat = N0
			elif(IsType.isstr(Nlat)): Nlat=int(float(Nlat)*N0)
			theta = np.linspace(0, np.pi, Nlat)
			phi = np.linspace(-np.pi, np.pi, Nlon)
			theta = theta[:,None] + 0*phi[None,:]
			phi = phi[None,:] + 0*theta
			npix = hp.ang2pix(nside, theta, phi, nest=nest)
			npix = npix[::-1,::-1]
			if (isnside) : return npix
			else : return [npix, hpmap[npix]]
		#--------------------------------------------------
		#--------------------------------------------------
		elif (lon is not None and lat is not None and dlon is not None) : # Healpix2Flat(lon, lat, dlon, dlat, Nlon, Nlat)
			lon, lat, dlon, dlat = np.array([lon, lat, dlon, dlat])*np.pi/180  # rad
			if (Nlat is None) : Nlat = Nlon
			# left -> right (RA) : 180 -> 0 -> -180
			# bottom -> top (Dec): -90 -> 0 -> 90
			lat = np.linspace(lat-dlat/2, lat+dlat/2, Nlat)
			lat[lat>np.pi/2]  =  np.pi - lat[lat>np.pi/2]
			lat[lat<-np.pi/2] = -np.pi - lat[lat<-np.pi/2]
			lon = np.linspace(lon-dlon/2, lon+dlon/2, Nlon)
			lon %= 2*np.pi
			lon[lon>np.pi] -= 2*np.pi
			if ((lon[:-1]-lon[1:]).sum()<0) : lon = lon[::-1]
			lon = lon[None,:] + 0*lat[:,None]
			lat = lat[:,None] + 0*lon
			npix = hp.ang2pix(nside, np.pi/2-lat, lon)
			if (isnside) : return npix
			else : return [npix, hpmap[npix]]
		#--------------------------------------------------
		#--------------------------------------------------
		elif (lon is not None and lat is not None and dlon is None) :   # Healpix2Flat(lon, lat, dlat, Nlat)
			thetawidth, Npix = dlat*np.pi/180, Nlat
			theta, phi = Beam.ThetaPhiMatrix(thetawidth, Npix) # rad, np.nan may in theta,phi
			#---------------------------------------------
			mask = Invalid(theta).mask
			theta[mask], phi[mask] = 0, 0
			thetar, phir = self.thetaphiRotation([theta, phi], ax=-(90-lat), az=-(lon+90))
			npix = hp.ang2pix(nside, thetar, phir, nest=nest)
			npix = npix[::-1,::-1]
			#---------------------------------------------
			if (isnside) : 
				npix[mask] = -1
				return npix
			else : 
				rehpmap = hpmap[npix]
				rehpmap[mask] = np.nan
				npix[mask] = -1
				return [npix, rehpmap]





	def CoplaneRotation( self, xyz1, xyz2, rotaxis=None, rotang=None ) : 
		'''
		xyz1, xyz2:
			xyz1 and xyz2 have the same shape
			Must xyz1=(x1,y1,z1), xyz2=(x2,y2,z2)
			x, y, z can be any shape

		(1) rotaxis = rotang = None:
			Two points xyz1=(x1,y1,z1), xyz2=(x2,y2,z2) in XYZa coordinate system.
			Find another coordinate, xyz1 on the new +x-axis, xyz2 on the new xy-plane
				(Original point doesn't change)
			return [xyz1_new, xyz2_new, rotaxis, rotang]

		(2) rotaxis and rotang != None, and len()=3
			Inverse operation of (1)
				Rotate xyz1_new, xyz2_new back to xyz1, xyz2
		'''
		import numpy as np
		if (rotaxis is None or rotang is None) : 
			rotaxis, rotang = [], []  # degree
			# Make point1 as new x-axis
			theta, phi = self.xyz2thetaphi(xyz1) *180/np.pi
			# First rotate phi about z-axis, second rotate -(90-theta) about y-axis
			xyz1 = self._xyzRotation(xyz1, az=phi, ay=-(90-theta))
			xyz2 = self._xyzRotation(xyz2, az=phi, ay=-(90-theta))
			rotaxis += ['az', 'ay']
			rotang += [phi, -(90-theta)]
			# Now, point1 (xyz1) is new x-axis, point2 is in new coordinate
			# Angle of point2 to new x-axis
			ang = np.angle(xyz2[1] + 1j*xyz2[2]) *180/np.pi
			xyz2 = self._xyzRotation(xyz2, ax=ang)
			rotaxis += ['ax']
			rotang += [ang]
			return [xyz1, xyz2, rotaxis, rotang]
		#---------------------------------------------

		else : # Rotate back
			def RotBack( xyz, rotaxis, rotang ) : 
				for i in range(len(rotaxis)-1, -1, -1) : 
					if   (rotaxis[i] == 'ax') : xyz = self._xyzRotation(xyz, ax=-rotang[i])
					elif (rotaxis[i] == 'ay') : xyz = self._xyzRotation(xyz, ay=-rotang[i])
					elif (rotaxis[i] == 'az') : xyz = self._xyzRotation(xyz, az=-rotang[i])
				return xyz
			xyz1 = RotBack(xyz1, rotaxis, rotang)
			xyz2 = RotBack(xyz2, rotaxis, rotang)
			return [xyz1, xyz2]





	def ArrayRotation( self, bound, array, **kwargs ):
		'''
		bound:
			True | False

		For 1D/2D/3D array
		array[0,0,0] is the original point
		ax: axis-0 of array: array.shape[0]
		ay: axis-1 of array: array.shape[1]
		az: axis-2 of array: array.shape[2]
		'''
		from jizhipy.Array import Invalid
		import numpy as np
		shape0 = array.shape
		r = max(shape0)*2**0.5/2+2
		for i in range(len(shape0)):
			d = int(r-shape0[i]/2)
			if   (i==0): halfshape = (d,)+shape0[1:]
			elif (i==1): halfshape = (shape0[0], d, shape0[2])
			elif (i==2): halfshape = shape0[:-1]+(d,)
			half = np.nan + np.zeros(halfshape)
			array =np.concatenate([half, array, half], i)
		shape = array.shape
		#----------------------------------------
		xyzAfter = [array*0. for i in range(3)]
		axis = 0
		for xyz in xyzAfter:
			if   (len(shape)>=1 and axis==0): 
				for i in range(1, shape[0]):
					xyz[i] = i
			elif (len(shape)>=1 and axis==1): 
				for i in range(1, shape[1]):
					xyz[:,i] = i
			elif (len(shape)>=1 and axis==2): 
				for i in range(1, shape[2]):
					xyz[:,:,i] = i
			axis += 1
		xyzBefore = self.xyzRotationPoint(xyzAfter, **kwargs)
		#----------------------------------------
		xyzBefore = xyzBefore.astype(int)
		n = xyzBefore[-1]
		for i in range(-2, -len(shape)-1, -1):
			n += xyzBefore[i]*np.prod(shape[:i+1])
		array = array.flatten(n.astype(int)).reshape(shape)
		#----------------------------------------
		mask = Invalid(array, None).mask
		n = []
		for i in range(len(shape)):
			s = range(len(shape)).remove(i)
			if (bound): tf = mask.prod(axis=s)
			else: tf = mask.sum(axis=s)
			n.append(np.arange(tf.size)[tf>0])
		for i in range(len(n)):
			if   (i==0): array = array[n[i]]
			elif (i==1): array = array[:,n[i]]
			elif (i==2): array = array[:,:,n[i]]
		return array





	def ImageRotation( self, bound, image, angle ):
		'''
		self.ArrayRotation() is 3D rotation
		self.ImageRotation() is 2D rotation

		Parameters
		----------
		bound:
			True / False

		image:
			Must be 2D array or 3D with .shape[2]==3

		angle:
			[float] in degree
			angle>=0: anticlockwise
			angle <0: clockwise
		'''
		angle %= 360
		ang = angle-360 if(angle>180)else angle
		if (abs(ang) <1e-4): return image
		import numpy as np
		from jizhipy.Basic import IsType, Raise
		import imutils
		image = np.array(image)
		if  (len(image.shape)==2): pass
		elif(len(image.shape)==3 and image.shape[2]==3):pass
		else: Raise(Exception, 'image must be 2D, or 3D with shape[2]==3, but now image.shape='+str(image.shape))
		if (bound): image=imutils.rotate_bound(image, -angle)
		else: image = imutils.rotate(image, angle)
		return image





CoordTrans = CoordTrans()

