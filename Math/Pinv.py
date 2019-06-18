
def Pinv( array, leastsq=False ) : 
	'''
	return:
		pseudo-inverse (np.ndarray format)
	'''
	import scipy.linalg as spla
	from jizhipy.Basic import Raise
	import numpy as np
	if (len(array.shape) != 2) : Raise(Exception, 'jizhipy.Pinv(): len(array.shape) = '+str(len(array.shape))+' != 2')
	if (not leastsq) : return spla.pinv2(array)
	else : 
		At = np.matrix(np.conj(array.T))
		b = At * np.matrix(array)
		b = np.matrix(spla.inv(b)) * At
		return b
	



def InvInter( A, level=None ) : 
	'''
	See: ./pdf_doc/xxx.pdf
	     https://wenku.baidu.com/view/df54bb210722192e4536f64f.html?from=search

	A: 
		dtype: float | complex
		shape: any shape ! for nrow<ncol: right inverse
		                 | for nrow>ncol: left  inverse

	return:
		inverse | Moore-Penrose pseudo inverse 
	'''
	import numpy as np
	from MatrixDot import MatrixDot
	# A.shape=(nrow, ncol)
	# B.shape=(ncol, nrow)
	# up = B * u * v * B,  u.shape=(nrow,1), v.shape=(1,ncol)
	# down = 1 + v * B * u
	A = 0.+np.array(A)
	nrow, ncol = A.shape
	B = np.diag(np.ones(min(nrow, ncol)))
	if (nrow < ncol) :  # right-inverse
		which = 'right'
		B = np.append(B, np.zeros((ncol-nrow, nrow)), 0)
	elif (nrow > ncol) :  # left-inverse
		which = 'left'
		B = np.append(B, np.zeros((ncol, nrow-ncol)), 1)
	else : which = 'both'
	A = A - B.T
#	for i in range(min(nrow, ncol)) :  # for min axis
	Nfor = min(nrow, ncol)
	if (level is not None and level < Nfor) : Nfor = level
	for i in range(level) :  # for min axis
		if (nrow < ncol) :  # for each row
			v = A[i:i+1]  # (1, ncol), A data
			u = np.zeros([nrow, 1])  # (nrow, 1), base
			u[i] = 1
		else :  # for each col
			u = A[:,i:i+1]  # (nrow, 1), A data
			v = np.zeros((1, ncol))  # (1, ncol), base
			v[:,i] = 1
		up = MatrixDot(B, u, v, B)
		down = 1 + MatrixDot(v, B, u).take(0)
		B = B - up / down
	return B
