
def Pcolormesh( *args, **kwargs ) : 
	'''
	**kwargs:
		'shape': 
			2D shape of image, must (<=500, <=500)
			Default shape=(500, 500)

		'sample':
			True | False
			How to reduce the image?
			=True: Sample(), keep the noise level
			=False: Smooth(), depress the noise
			Default sample=True
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import IsType
	from jizhipy.Optimize import Smooth, Sample
	import matplotlib.pyplot as plt
	args = list(args)
	args[0] = Asarray(args[0])
	if (len(args[0].shape) == 1) : 
		args[1], args[2] =Asarray(args[1]), Asarray(args[2])
		X, Y, C = args[:3]
	else : X, Y, C = None, None, args[0]
	#----------------------------------------
	if ('shape' in kwargs.keys()) : 
		shape = kwargs['shape']
		kwargs.pop('shape')
		if (IsType.isnum(shape)) : shape =[shape, shape]
		shape = Asarray(shape, float)
		if (shape.max() > 500) :  # lower
			if (shape[0] > shape[1]) : 
				shape[1] *= 500./shape[0]
				shape[0] = 500
			else : 
				shape[0] *= 500./shape[1]
				shape[1] = 500
		shape = Asarray(shape, int)
	else : shape = np.array([500, 500])
	#----------------------------------------
	if('sample' in kwargs.keys()) : 
		issample = bool(kwargs['sample'])
		kwargs.pop('sample')
	else : issample = False
	if (issample) : 
		if (C.shape[0] > shape[0]) : 
			C = Sample(C, 0, size=shape[0])
		if (C.shape[1] > shape[0]) : 
			C = Sample(C, 1, size=shape[1])
	else : 
		if (C.shape[0] > shape[0]) : 
			C = Smooth(C, 0, reduceshape=shape[0])
		if (C.shape[1] > shape[0]) : 
			C = Smooth(C, 1, reduceshape=shape[1])
	#----------------------------------------
	if (X is not None) : 
		X = Sample(X, 0, size=C.shape[1])
		Y = Sample(Y, 0, size=C.shape[0])
		args[0], args[1], args[2] = X, Y, C
	else : args[0] = C
	#----------------------------------------
	return plt.pcolormesh(*args, **kwargs)

