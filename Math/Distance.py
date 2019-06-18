
def Distance( X, Y=None, metric='euclidean', p=2, pairwise=True ):
	'''
	Y:
		if Y=None, return scipy.spatial.distance.pdist(X)

	pairwise:
		=True: 
			use sklearn.metrics.pairwise_distances() or scipy.spatial.distance.cdist()
			the same as ((X[:,None]-Y[None,:])**2).sum(-1)**0.5
		=False: ((X-Y)**2).sum(-1)**0.5

	metric:
		['braycurtis',  'canberra',    'chebyshev', 
		 'cityblock',   'correlation', 'cosine', 
		 'dice',        'euclidean',   'hamming', 
		 'jaccard',     'kulsinski',   'mahalanobis', 
		 'matching',    'minkowski',   'rogerstanimoto', 
		 'russellrao',  'seuclidean',  'sokalmichener', 
		 'sokalsneath', 'sqeuclidean', 'yule']
	'''
	import numpy as np
	import scipy.spatial
	from jizhipy.Basic import Raise
	#import sklearn.metrics
	X = np.array(X)
	if (Y is None): 
		D = scipy.spatial.distance.pdist(X, metric=metric, p=p)
		return D
	Y = np.array(Y)
	if (not pairwise):
		# Broadcase
		X = X + 0*Y
		Y = Y + 0*X
		shape = X.shape
		if (X.shape != Y.shape): Raise(Exception, 'pairwise=False, must need X.shape==Y.shape, but now X.shape='+str(X.shape)+', Y.shape='+str(Y.shape))
		if (len(shape) !=2):
			X = X.reshape(int(np.prod(shape[:-1])), shape[-1])
			Y = Y.reshape(int(np.prod(shape[:-1])), shape[-1])
		D, n, step = [], 0, 10000
		while (n < len(X)):
			d = scipy.spatial.distance.cdist(X[n:n+step], Y[n:n+step], metric=metric, p=p)
			D.append(np.diag(d))
			n += step
		D = np.concatenate(D)
		if (len(shape) !=2): D = D.reshape(shape[:-1])
		return D
	else: 
		D = scipy.spatial.distance.cdist(X, Y, metric=metric, p=p)
		return D


