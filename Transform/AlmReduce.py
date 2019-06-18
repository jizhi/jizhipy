
def AlmReduce( alm, lmax=None, mmax=None ) : 
	'''
	alm:
		l from 0 to lmax0
		NOTE THAT input alm must be full: lmax==mmax

	lmax:
		lmax of the reduced alm
		Reduced alm's l from 0 to lmax

	mmax:
		mmax of the reduced alm
		Reduced alm's m from 0 to mmax

	if lmax=None: lmax=lmax0
	if mmax=None: mmax=mmax0
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	import healpy as hp
	lmax0 = hp.Alm.getlmax(alm.size)
	mmax0 = hp.Alm.getidx(lmax0, 1, 1) -1
	lmax = lmax0 if(lmax is None)else int(lmax)
	mmax = mmax0 if(mmax is None)else int(mmax)
	l, m = hp.Alm.getlm(lmax0)
	tf = (l<=lmax)*(m<=mmax)
	alm = Asarray(alm)[tf]
	return alm



