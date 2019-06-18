
def _DoMultiprocess_Ylm( iterable ) : 
	import scipy.special as spsp
	from jizhipy.Array import Invalid
	import numpy as np
	m, l, phi, theta = iterable[1]
	ylm = np.zeros(m.size, np.dtype(iterable[2]))
	for i in range(m.size) : 
		ylmi = spsp.sph_harm( m[i], l[i], phi[i], theta[i] )
		if (Invalid(ylmi).mask == True) : break
		ylm[i] = ylmi.take(0)
	return ylm





def _DoMultiprocess_Ylm_special( iterable ) : 
	import scipy.special as spsp
	from jizhipy.Array import Invalid
	import numpy as np
	m, l = iterable[1]
	theta, phi, dtypename = iterable[2]
	ylm = np.zeros((m.size,)+phi.shape, np.dtype(dtypename))
	for i in range(m.size) : 
		ylmi = spsp.sph_harm(m[i], l[i], np.pi/1.5, np.pi/2.5)
		if (Invalid(ylmi).mask == True) : continue
		ylmi = spsp.sph_harm( m[i], l[i], phi, theta )
		ylm[i] = ylmi
	return ylm





class Alm( object ) : 


	def LM2Index( self, which, lmax, l, m, order=None, symmetry=True ) : 
		'''
		Usage:
			See MapMaking.py

		lmax:
			int, must be one
	
		l, m: 
			Can be one or 1D array
	
		order:
			'2D': return 2D matrix with np.nan
			None: remove np.nan and return 1D(flatten) ndarray
			'l': return 1D array ordered by l from small to large
			'm': return 1D array ordered by m from small to large
			'i': return 1D array ordered by index from small to large
	
		which == 'Alm' : 
			if (self.fgsymm) : index = (m*lmax - m*(m-1)/2 + l)
			else : 
				if (m >= 0)   : index = (m*lmax - m*(m-1)/2 + l)
				else : 
					m = -m
					index = (m-1)*self.lmax-m*(m-1)/2+l+self.offnm-1
	
		which == 'AlmDS' : 
			index = l*(l+1) + m
	
		return:
			[index, morder, lorder]
			'2D': index.shape=(l.size, m.size), lorder=l[:,None], morder=m[None,:]
			Others: 1D with index.size==lorder.size==morder.size
		'''
		import numpy as np
		from jizhipy.Array import Asarray, Invalid, Sort
		from jizhipy.Basic import Raise
		l, m = Asarray(l).flatten(), Asarray(m).flatten()
		if (l[l>lmax].size > 0) : Raise(Exception, 'l=['+str(l.min())+', ..., '+str(l.max())+'] > lmax='+str(lmax))
		if (abs(m)[abs(m)>lmax].size > 0) : Raise(Exception, 'm=['+str(m.min())+', ..., '+str(m.max())+'] > lmax='+str(lmax))
		#--------------------------------------------------
		#--------------------------------------------------
		if (str(which).lower() == 'alm') : 
			if (symmetry) : index = lmax*m[None,:] - m[None,:]*(m[None,:]-1)/2. + l[:,None]
			else : 
				morder = np.append(np.arange(m.size)[m>=0], np.arange(m.size)[m<0])
				mpos, mneg = m[m>=0], abs(m[m<0])
				indexpos = lmax*mpos[None,:] - mpos[None,:]*(mpos[None,:]-1)/2. + l[:,None]
				indexneg = lmax*(mneg[None,:]-1) - mneg[None,:]*(mneg[None,:]-1)/2. + l[:,None] + self.offnm-1
				index = np.append(indexpos, indexneg, 1)
				indexpos = indexneg = mpos = mneg = 0 #@
				index = index[:,morder]
			#--------------------------------------------------
		elif (str(which).lower() == 'almds') : 
			index = l[:,None]*(l[:,None]+1) + m[None,:] +0.
		#--------------------------------------------------
		#--------------------------------------------------
		for i in range(l.size): index[i,abs(m)>l[i]] = np.nan
		#--------------------------------------------------
		if (order=='2D'): return [index, m[None,:], l[:,None]]
		#--------------------------------------------------
		lorder = (index*0+1) * l[:,None]
		morder = (index*0+1) * m[None,:]
		lorder = Invalid(lorder, False).astype(int)
		morder = Invalid(morder, False).astype(int)
		index  = Invalid(index , False).astype(int)
		#--------------------------------------------------
		if (order is not None) : 
			if   (str(order).lower() == 'i') : along = '[0,:]'
			elif (str(order).lower() == 'l') : along = '[1,:]'
			elif (str(order).lower() == 'm') : along = '[2,:]'
			index = np.array([index, lorder, morder])
			index, lorder, morder = Sort(index, along)
		if (index.size == 1) : 
			index, lorder, morder = index[0], lorder[0], morder[0]
		return [index, morder, lorder]





	def Get( self, lmax=None, l=None, m=None, idx=None, size=None, m_neg=False ) : 
		'''
		lm in healpy: m>=0, doesn't include m<0
		l: 0,1,2,..,lmax,  1,2,..,lmax, 2,..,lmax, .., lmax
		m: 0,0,0,..,0   ,  1,1,..,1   , 2,..,2   , .., lmax

		For m from -l to +l
		l: 0,   1,1,1,   2, 2,2,2,2,  ..,   lmax,lmax,..,lmax
		m: 0,  -1,0,1,  -2,-1,0,1,2,  ..,  -lmax,..,0,..,lmax

		All functions of hp.Alm have been expressed here

		(1) Get(lmax)
			return l, m = hp.Alm.getlm( lmax )

		(2) Get(lmax, idx)
			return l, m = hp.Alm.getlm( lmax, idx )
				idx can be any shape

		(3) Get(lmax, l, m)
			return idx = hp.Alm.getidx( lmax, l, m )
				l.shape==m.shape | l.size==1 | m.shape==1

		(4) Get(lmax, l)
			return all idx with this l (m=0,1,2,...,l)
				l will be forced to flatten()

		(5) Get(lmax, m)
			return all idx with this m (l=m,m+1,...,lmax)
				m will be forced to flatten()

		(6) Get(lmax, size=True)
			return size = hp.Alm.getsize(lmax)

		N+ = (lmax+1)*(lmax+2)/2    (include 0)
		N- = (lmax+1)*lmax
		Ntot = N+ + N- = (lmax+1)**2

		(7) Get(size=int)
			return lmax = hp.Alm.getlmax(size)
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import IsType
		import healpy as hp
		# (1) Get(lmax), (2) Get(lmax, idx)
		if (l is None and m is None and size is None and m_neg==False) : return np.array(hp.Alm.getlm(lmax, idx))
		# (3) Get(lmax, l, m)
		if (l is not None and m is not None and m_neg==False) : return hp.Alm.getidx( lmax, l, m )
		# (6) Get(lmax, size=True)
		if (size == True and m_neg==False) : return hp.Alm.getsize(lmax)
		# (7) Get(size)
		if (IsType.isnum(size) and m_neg==False) : 
			lmax = hp.Alm.getlmax(size)
			if (lmax<0): lmax =int(round((2*size)**0.5)-1)
			return lmax
		#---------------------------------------------
		la, ma = hp.Alm.getlm(lmax)
		n = np.arange(la.size)
		#---------------------------------------------
		# (4) Get(lmax, l)
		if (l is not None and m is None and idx is None and m_neg==False) : 
			tf1 = True if(IsType.isnum(l))else False
			l, idx = Asarray(l).flatten(), []
			for i in range(l.size) : 
				m = n[la==l[i]]
				idx.append( m )
			if (tf1) : idx = idx[0]
			return idx
		#---------------------------------------------
		# (5) Get(lmax, l=None, m)
		if (m is not None and l is None and idx is None and m_neg==False) : 
			tf1 = True if(IsType.isnum(m))else False
			m, idx = Asarray(m).flatten(), []
			for i in range(m.size) : 
				idx.append( n[ma==m[i]] )
			if (tf1) : idx = idx[0]
			return idx
		#---------------------------------------------
		# (1) Get(lmax, m_neg=True) : 
		if (l is None and m is None and size is None and m_neg==True) : 
			l, m = [], []
			for i in range(lmax+1) : 
				l.append( i + np.zeros(2*i+1, int) )
				m.append( np.arange(-i, i+1) )
			l, m = np.concatenate(l), np.concatenate(m)
			lm = np.array([l, m])
			if (idx is not None) : 
				idx = np.array(idx, int)
				lm = lm[:,idx]
			return lm





	def LMreduce( self, lmax0, lmax=None, mmax=None ) : 
		'''
		lmax0:
			int, original lmax
	
		lmax:
			int, reduced lmax < lmax0
	
		mmax:
			mmax <= lmax

		return:
			idx: int array
		'''
		import numpy as np
		size = self.Get(lmax0, size=True)
		idx = np.arange(size)
		if (lmax is None) : lmax = lmax0
		if (mmax is None) : mmax = lmax
		if (mmax  > lmax) : mmax = lmax
		l, m = self.Get(lmax0)
		tf = (l <= lmax)
		m, idx = m[tf], idx[tf]
		tf = (abs(m) <= mmax)
		idx = idx[tf]
		return idx





	def LMorder( self, lm, m_neg=None ) : 
		'''
		re-order lm to healpy order OR +-m order
	
		lm:
			l, m = lm, l.size==m.size

		m_neg:
			==None: include m>=0 and m<0
			==False: just include m>=0
			==True:  just include m<0
	
		return:
			[l, m, n], n is the order/index of input lm
				lm_out = return[:2]
				n = return[-1]
				lm_out = lm_in[:,n]
		'''
		import numpy as np
		lm = np.concatenate([lm, [np.arange(len(lm[1]))]], 0)
		mmin, mmax = lm[1].min(), lm[1].max()
		#----------------------------------------
		def mgeq0( lm ) : 
			lm, l, m, n = np.array(lm,int), [], [], []
			while (lm.size > 0) : 
				tf = lm[1] == lm[1].min()
				m.append(lm[1][tf])
				lo = np.sort(lm[0][tf] + 1j*lm[2][tf])
				l.append( lo.real.astype(int) )
				n.append( lo.imag.astype(int) )
				lm = lm[:,(1-tf).astype(bool)]
			l, m, n = np.concatenate(l), np.concatenate(m), np.concatenate(n)
			return np.array([l, m, n])
		#----------------------------------------
		def mall( lm ) : 
			lm, l, m, n = np.array(lm,int), [], [], []
			while (lm.size > 0) : 
				tf = lm[0] == lm[0].min()
				l.append( lm[0][tf] )
				mo = np.sort(lm[1][tf] + 1j*lm[2][tf])
				m.append( mo.real.astype(int) )
				n.append( mo.imag.astype(int) )
				lm = lm[:,(1-tf).astype(bool)]
			l, m, n = np.concatenate(l), np.concatenate(m), np.concatenate(n)
			return np.array([l, m, n])
		#----------------------------------------
		if (m_neg is None) : 
			if (mmin >= 0) : return mgeq0(lm)
			elif (mmax <= 0) : 
				lm = mgeq0(abs(lm))
				lm[1,lm[1]>0] *= -1
				return lm
			else : return mall(lm)
		#----------------------------------------
		elif (m_neg is False) :  # m>=0
			return mgeq0(lm[:,lm[1]>=0])
		elif (m_neg is True) :  # m<=0
			lm = abs(lm[:,lm[1]<=0])
			lm = mgeq0(lm)
			lm[1,lm[1]>0] *= -1
			return lm





	def Ylm( self, m, l, phi, theta, Nprocess=None ) : 
		'''
		Basing on scipy.special.sph_harm()
		When m is to large, can NOT compute
				d^m P_l(x) / d x^m
			and return nan+nanj or 0+0j
			and independent of theta,phi
	
		For example, 
			l=512, when m>=113, return nan+nanj, no matter what values of theta and phi are
			l=256, m>=129 will return nan+nanj
		
		l<=150, won't return nan+nanj
		when l>=151, large m will return nan+nanj

		This function is for the special case: 
			m = m + 0*l           # Broadcast l and m
			l = l + 0*m
			phi = phi + 0*theta   # Broadcast phi and theta
			theta = theta + 0*phi

		Special case:
			(m.shape==l.shape, phi.shape==theta.shape after broadcast)
			len(m.shape) == len(phi.shape)  # same dimension
			m.shape   = (n1, n2, n3, ...)
			phi.shape = (n4, n5, n6, ...)
			Same axis of m and phi, when size of this axis of m>=1, size of this axis of phi ==1
				also, when size of this axis of phi >=1, size of this axis of m ==1
			For example:
				m.shape   = (12,  1)
				phi.shape = ( 1, 73)
			axis=0 of m =12>=1, that of phi ==1
			axis=1 of phi =73>=1, that of m ==1

		In this special case, use this function Ylm_special() will much faster than use Ylm_general()
		'''
		from jizhipy.Process import PoolFor, NprocessCPU
		import numpy as np
		from jizhipy.Array import Asarray
		import scipy.special as spsp
		Nprocess = NprocessCPU(Nprocess)[0]
		dtype, dtypec =np.array(1.).dtype, np.array(1j).dtype
		#---------------------------------------------
		m, l, phi, theta = np.array(m,np.int32), np.array(l,np.int32), np.array(phi,dtype), np.array(theta,dtype)
		shape = (m.shape, l.shape, phi.shape, theta.shape)
		#---------------------------------------------
		# Broadcast all arguments !!!
		m = m + 0*l
		l = l + 0*m
		phi = phi + 0*theta
		theta = theta + 0*phi
		mshape, phis = np.array(m.shape), np.array(phi.shape)
		phis[phis==1] = mshape[phis==1]
		shapef, mshape = tuple(phis), tuple(mshape)
		#---------------------------------------------
		# Flatten
		m, l = m.flatten(), l.flatten()
		# Order m from small to large
		m = np.sort(m + 1j*np.arange(m.size))
		m, morder = m.real.astype(np.int32), m.imag.astype(np.int32)
		l = l[morder]
		#---------------------------------------------
		# Split into Nprocess piece
		if (m.size < Nprocess) : Nprocess = m.size
		nrow = 1.*m.size / Nprocess
		if (nrow-int(nrow) > 1e-6) : nrow = int(nrow)+1
		else : nrow = int(nrow)
		ns = np.arange(nrow*Nprocess).reshape(nrow,Nprocess).T
		send, sorder = [], []
		for i in range(len(ns)) : 
			n = ns[i]
			n = n[n<m.size]
			sorder.append(n)
			send.append( (m[n], l[n]) )
		#---------------------------------------------
		bcast = (theta, phi, dtypec.name)
		pool = PoolFor()
		ylm = pool.map_async(_DoMultiprocess_Ylm_special, send, bcast)
		ylm = np.concatenate(ylm)
		sorder = np.concatenate(sorder)
		#---------------------------------------------
		# Reorder
		ylmr = np.sort(sorder + 1j*ylm.real).imag
		ylmi = np.sort(sorder + 1j*ylm.imag).imag
		ylmr = np.sort(morder + 1j*ylmr).imag
		ylmi = np.sort(morder + 1j*ylmi).imag
		ylm = ylmr + 1j*ylmi
		#---------------------------------------------
		# Reshape
		isnum = True
		for i in range(len(shape)) : 
			if (shape[i] != ()) : isnum = False
		if (isnum) : ylm = ylm[0]
		else : ylm = ylm.reshape(shapef)
		return ylm





	def Healpix2Alm( self, hpmap, lmax=None, m_neg=True ) : 
		'''
		Like hp.map2alm(), but from -l to +l for m_neg=True
		Spherical harnomic expand hpmap to lm with lmax
	
		hpmap:
			one healpix map, hpmap.size == 12*nside**2
	
		m_neg:
			include -m or not?

		return: [almds, lm]
		'''
		import healpy as hp
		import numpy as np
		if (lmax is None) : lmax = 4*hp.get_nside(hpmap)
		if('complex' not in hpmap.dtype.name and m_neg==False):
			alm = hp.map2alm(hpmap, lmax)
			lm = self.Get(lmax)
			return [alm, lm]
		#--------------------------------------------------
		l = np.arange(lmax+1)  # -lmax -> 0 -> +lmax
		# Negetive: -m
		mn = np.arange(-lmax, 0)
		idxdsn, morder =self.LM2Index('AlmDs',lmax,l,  mn)[:2]
		idxmn          =self.LM2Index('Alm'  ,lmax,l, -mn)[0]
		# morder: [1, 2,1, 3,2,1, 4,3,2,1, lmax,lmax-1,...,1]
		# Positive: +m
		mp = np.arange(0, lmax+1)
		idxdsp = self.LM2Index('AlmDs', lmax, l, mp)[0]
		idxmp  = self.LM2Index('Alm'  , lmax, l, mp)[0]
		#--------------------------------------------------
		almR = hp.map2alm(hpmap.real, lmax)  # real part
		almI = hp.map2alm(hpmap.imag, lmax)  # imag part
		almds = np.zeros((lmax+1)**2, complex)
		#--------------------------------------------------
		almds[idxdsn] = (-1)**abs(morder) * (np.conj(almR[idxmn]) + 1j*np.conj(almI[idxmn]))
		almds[idxdsp] = almR[idxmp] + 1j*almI[idxmp]
		#--------------------------------------------------
		l, m = [], []
		for i in range(lmax+1) : 
			l.append( i+np.zeros(2*i+1, int) )
			m.append( np.arange(-i, i+1) )
		lm = np.array([np.concatenate(l), np.concatenate(m)])
		#--------------------------------------------------
		if (m_neg == False) : 
			almds = almds[lm[1]>=0]
			lm = lm[:,lm[1]>=0]
			lm = self.LMorder(lm)
			almds = almds[lm[-1]]
			lm = lm[:2]
		return [almds, lm]

	



	def BeamPhaseLM( self, hpmap=None, lmax=None, alm=None, flat=True ) : 
		'''
		Old name: Healpix2Alm4Pinv

		vispix = (beamphase * Tmap).sum(-1)
				   .shape = (Nbl, Nt)
		vissph = (beamphaselm * almT).sum(-1)
				   .shape = (2, Nbl, Nt)
		vispix == vissph[0] + conj(vissph[1])
								conj !!!

		return: 
			beamlm = [ (-1)^m *Beam(l,-m), Beam*(l,m) ]
			                                 (conj)

		V(m)   = (-1)^m * Beam(l,-m) * Tmap(l,m)
		V*(-m) =          Beam*(l,m) * Tmap(l,m)
		(conj)              (conj)

		V(t)  = (-1)^m * Beam(l,-m)(t) * Tmap(l,m)
		V*(t) =          Beam*(l,m)(t) * Tmap(l,m)
	
		Tmap(l, m): l>=0 and m>=0, healpy order
	
		Beam*(l, m): healpy order
		Beam(l, -m):
			l: 0,  1,1,1,  2, 2,2,2,2,  3, 3, 3,3,3,3,3
			m: 0, -1,0,1, -2,-1,0,1,2, -3,-2,-1,0,1,2,3
	
		hpmap, alm:
			Choose one of them
			alm = Healpix2Alm(hpmap, lmax, True)
		alm: must include m<0
	
		lmax:
			if set hpmap, then use lmax
	
		flat:
			True: alm4pinv.shape = (2, (lmax+1)*(lmax+2)/2), healpy order
			False: alm4pinv.shape = (2, l(lmax+1), m(lmax+1)), lower triangular matrix
	
		return: 
			beamlm = [ (-1)^m *Beam(l,-m), Beam*(l,m) ]
			                                 (conj)
		'''
		import numpy as np
		from jizhipy.Array import Invalid
		if (hpmap is not None) : alm = self.Healpix2Alm(hpmap, lmax, True)[0]
		lmax = int(round(alm.size**0.5 - 1))
		#----------------------------------------
		def Method1() : 
			l = np.arange(lmax+1)
			m = l.copy()
			#----------------------------------------
			# Vij(m), for beamlm ('AlmDS')
			idxBp = self.LM2Index('AlmDS',lmax,l, -m, '2D')[0]
			# Vij^{*}(-m)
			idxBn = self.LM2Index('AlmDS',lmax,l,  m, '2D')[0]
			valid = 1-Invalid(idxBp).mask  # 01, not TrueFalse
			''' row-l, col-m '''
			#----------------------------------------
			idxBp = Invalid(idxBp, 0).data.astype(int)
			idxBn = Invalid(idxBn, 0).data.astype(int)
			''' row-l, col-m '''
			#----------------------------------------
			m = m[None,:]
			alm4pinv1 = (-1)**m * alm[idxBp]  * valid
			alm4pinv2 = np.conj(  alm[idxBn]) * valid
			alm4pinv = np.array([alm4pinv1, alm4pinv2])
			# alm4pinv.shape =(2, l(lmax+1), m(lmax+1))
			#                  => [0]:for m>=0,  [1]:for m<=0
			#----------------------------------------
			if (flat) :  # healpy order
				alm4pinv = alm4pinv.T[valid.T>0.5].T
			return alm4pinv
		#----------------------------------------
		#----------------------------------------
		def Method2() : 
			lm = self.Get(lmax, m_neg=True)
			l1, m1, n1 = self.LMorder(lm, m_neg=False)
			l0, m0, n0 = self.LMorder(lm, m_neg=True)
			alm4pinv1 = (-1)**abs(m0) * alm[n0]
			alm4pinv2 = np.conj(alm[n1])
			alm4pinv = np.array([alm4pinv1, alm4pinv2])
			#----------------------------------------
			if (not flat) : 
				n = np.zeros([lmax+1, lmax+1], int)-1
				n[0] = np.arange(lmax+1)
				for i in range(1, lmax+1) : 
					n[i,i:] =n[i-1,-1]+1 +np.arange(lmax+1-i)
				alm4pinv = alm4pinv.T[n].T
				n[n>=0], n[n<0] = 1, 0
				alm4pinv *= n.T
			return alm4pinv
		#----------------------------------------
	#	alm4pinv = Method1()
		alm4pinv = Method2()
		#----------------------------------------
		return alm4pinv





Alm = Alm()
