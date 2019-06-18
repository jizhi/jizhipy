
def RemoveSource( inmap, lb0, lb1, fwhm, times=2, same=False, onebyone=True, verbose=False ) : 
	'''
	return:
		 (npix_source, inmap[npix_source])

	Remove bright source from the healpix map

	inmap:
		input healpix map, 1D

	lb0:
		[degree], the center of the sources to be removed
		l is longitude, b is latitude (theta=90-b)
		(1) 2D: l0, b0 = lb0[:,0], lb0[:,1]
		(2) .size==2: l0, b0 = lb0

	lb1:
		[degree], the center of the pixels which are used to fill the hole from sources removing
		lb1.shape == lb0.shape

	fwhm:
		[degree]
		isnum | islist
		fwhm size of the sources

	times:
		fwhm*times to smooth the filling pixels	

	same:
		(1) ==False: fill the hole of sources by the same shape regions centering at lb1
		(2) ==True: use one value at lb1 to fill the hole for each sources

	onebyone:
		(1) ==True: handle the sources one by one, the result must be correct, but slower
		(2) ==False: hand all sources together at once. The result is correct when all holes don't overlap. If they overlap, the result may be not correct
	'''
	import numpy as np
	import healpy as hp
	from jizhipy.Array import Invalid
	from jizhipy.Process import ProgressBar
	from jizhipy.Basic import IsType
	if (verbose is True) : pstr = 'jizhipy.RemoveSource:'
	elif (verbose == '123') : pstr, verbose = '    ', True
	if (same) : onebyone = True
	lb0, lb1 = np.array(lb0, float)*np.pi/180, np.array(lb1, float)*np.pi/180
	if (lb0.size == 2) : 
		lb0 = lb0.flatten()[None,:]
		lb1 = lb1.flatten()[None,:]
	if (IsType.isnum(fwhm)) : fwhm = fwhm+np.zeros(len(lb0))
	fwhm = np.array(fwhm) * np.pi/180
	same, onebyone = bool(same), bool(onebyone)
	inmap = np.array(inmap)
	nside = hp.get_nside(inmap)
	#--------------------------------------------------
	def OneByOne( lb0, lb1, fwhm, nside, same ) : 
		l0, b0 = lb0
		l1, b1 = lb1
		# center of source
		n0 = hp.ang2pix(nside, np.pi/2-b0, l0)
		n1 = hp.ang2pix(nside, np.pi/2-b1, l1)
		# source's region
		a0 = np.zeros(12*nside**2)
		a0[n0] = 1000
		a0 = hp.smoothing(a0, fwhm, verbose=False)
		a0 /= a0.max()
		pix = np.arange(12*nside**2)
		pix0 = pix[a0>0.4]
		if (same) : pix1 = n1 + 0*pix0
		else : 
			a1 = a0.copy()
			a1[n1] = 1000
			a1 = hp.smoothing(a1, fwhm, verbose=False)
			a1 = a1/a1.max()
			pix1 = pix[a1>0.4][:pix0.size]
			if (pix1.size < pix0.size) : pix1 = np.append(pix1, pix1[(pix1.size-pix0.size):][::-1])
		return np.array([pix0, pix1])
	#--------------------------------------------------
	if (onebyone) : 
		if (verbose) : progressbar = ProgressBar(pstr, len(lb0))
		pix = []
		for i in range(len(lb0)) : 
			if (verbose) : progressbar.Progress()
			pix.append(OneByOne(lb0[i], lb1[i], fwhm[i], nside, same))
		pix = np.concatenate(pix, 1)
	else: pix=OneByOne(lb0.T, lb1.T, fwhm.mean(), nside, same)
	#--------------------------------------------------
	inmap[pix[0]] = inmap[pix[1]]
	inmap = Invalid(inmap, True).data
	inmap=hp.smoothing(inmap, fwhm.mean()*times,verbose=False)
	return (pix[0], inmap[pix[0]])
