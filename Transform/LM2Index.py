
def LM2Index( which, lmax, l, m, order=None, symmetry=True ) : 
	'''
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
	from jizhipy.Basic import Raise
	from jizhipy.Array import Asarray, Invalid, Sort
	l, m = npfmt(l).flatten(), npfmt(m).flatten()
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
	for i in range(l.size) : index[i,abs(m)>l[i]] = np.nan
	#--------------------------------------------------
	if (order == '2D') : return [index, m[None,:], l[:,None]]
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
