
def _GaussianFilter( per, times, method=2 ) : 
	'''
	a(n) = S(n) - S(n-1)
	S(n) = a(n) + a(n-1) + a(n-2) + ... + a(2) + a(1)

	per:
		per >=2, can be odd or even

	times:
		times >=1

	method:
		method = 1 : non-normalized
		method = 2 : normalized: 1/(sigma*(2*pi)**0.5)

	return:
		gaussianfilter.size = (per-1) * times + 1
	'''
	import numpy as np
	from jizhipy.Basic import Raise
	per, times = np.array([per, times]).round().astype(int)
	if (per<2 or times<1) : 
		Raise(Warning, 'GaussianFilter(per, times), per='+str(per)+', times='+str(times)+', return gaussianfilter=[1]')
		return np.array([1])
	if   (method == 1) : a0 = np.ones(per)
	elif (method == 2) : a0 = np.ones(per) / per
	for n in range(2, times+1) : 
		n1 = (per-1) * n + 1
		a1 = np.zeros([per, n1])
		for i in range(per) : 
			a1[i,i:i+a0.size] = a0
		if   (method == 1) : a0 = a1.sum(0)
		elif (method == 2) : a0 = a1.sum(0) / per
#	if   (method == 1) : gaussianfilter = a0 / (1.*per)**times
	gaussianfilter = a0
	std = 1. / ((2*np.pi)**0.5 * gaussianfilter.max())
	return [gaussianfilter, std]





def GaussianFilter(per=None, times=None, std=None, Npix=None):
	'''
	call jp.Gaussian.GaussianFilter()

	return: [gaussianfilter, std]
		Unit for std: pixel
		When use this std, x must pixel: np.arange(...)

		gaussianfilter.size = (per-1) * times + 1

	gf1, s1 = GaussianFilter(per, times)
	gf2, s2 = GaussianFilter(None, None, s1, gf1.size)
	gf3 = GaussianValue(np.arange(gf1.size), gf1.size/2, s1)
		gf1==gf2==gf3,  s1==s2

	NOTE THAT must be GaussianFilter.sum()==1
		can NOT be other value

	(1) per, times
		Use per, times to generate gaussian filter
		then use Npix to reshape
	(2) sigma, Npix
		Use sigma, npix to generate gaussian filter
	'''
	from jizhipy.Math import Gaussian
	import numpy as np
	if (per is not None or times is not None) : 
		gaussianfilter, std = _GaussianFilter(per, times)
	else : 
		x = np.arange(Npix)
		x = x - x.mean()
		gaussianfilter = Gaussian.GaussianValue(x, 0, std)
	return [gaussianfilter, std]





def _Multiprocess_Smooth( iterable ) : 
	'''
	nlr:
	If reduceshape==False, large times will make the left and right edge worse and worse
	(1) ==None: append first/last element
	(2) ==True: set nlr=[len(array)/100, len(array)/100]
	(3) ==[int, int]: set nlr=[int, int]
	(4) isnum (float or int): as the outside value
	(5) =='periodic': append right end to the left head, left head to right end
	(6) ==False
	(7) =='mirror': append mirror
	'''
	import numpy as np
	from jizhipy.Optimize import Interp1d
	import matplotlib.pyplot as plt
	from jizhipy.Array import Invalid
	array = iterable[1].T  # smooth along axis-0
	weight, nla, nra, sigma, nlr, case, fft = iterable[2]
	N0, N1 = array.shape
	#---------------------------------------------
	if (case == 1) : # nlr=None, append the first/last element
		aleft = np.zeros((nla, N1), array.dtype)
		if (nla != 0) : aleft = aleft + array[:1]
		aright = np.zeros((nra, N1), array.dtype) + array[-1:]
	#---------------------------------------------
	elif (case in [2, 3]) : # nlr=True | (nl,nr), interp1d
		nlr = np.array(nlr, int)
		nlr[nlr>int(len(array)/2)] = int(len(array)/2)
		nlr[nlr<2] = 2
		aleft0 = np.array([array[:int(nlr[0]/2)].mean(0), array[int(nlr[0]/2):nlr[0]].mean(0)])
		xl0 = np.array([nlr[0]/4., nlr[0]*3/4.])
		aright0 = np.array([array[-nlr[1]:-int(nlr[1]/2)].mean(0), array[-int(nlr[1]/2):].mean(0)])
		xr0 = np.array([-nlr[1]*3/4., -nlr[1]/4.])+len(array)
		xl = np.arange(-nla, 0)
		xr = np.arange(N0, N0+nra)
		aleft, aright = [], []
		for i in range(N1) : 
			if (nla == 0) : aleft.append( np.zeros([0, N1], array.dtype) )
			else: aleft.append( Interp1d(xl0, aleft0[:,i], xl))
			aright.append( Interp1d(xr0, aright0[:,i], xr) )
		aleft, aright = np.array(aleft).T, np.array(aright).T
	#---------------------------------------------
	elif (case in [4, 6]) : # nlr=isnum | False
		aleft  = nlr + np.zeros((nla, N1), array.dtype)
		aright = nlr + np.zeros((nra, N1), array.dtype)
	#---------------------------------------------
	elif (case == 5) : # nlr='periodic'
		if (nla == 0) : aleft =np.zeros((nla, N1), array.dtype)
		else : aleft = array[-nla:]
		aright = array[:nra]
	elif (case == 7) : # nlr='mirror'
		if (nla == 0) : aleft =np.zeros((nla, N1), array.dtype)
		else : aleft = array[:nla][::-1]
		aright = array[-nra:][::-1]
	#---------------------------------------------
	#---------------------------------------------
	if (not fft) :
		array = np.concatenate([aleft, array, aright], 0)
		if (sigma is True) : arrstd = array*0
		n1, n2, n = nla, len(array)-nra, len(weight)
		arr = array*0
		weight = weight[:,None]  # 2D
		if (case != 6) : w = weight
		for i in range(n1, n2) : 
			a = array[i-nla:i-nla+n]
			if (case == 6) : 
				tf = (1-Invalid(a[:,0]).mask).astype(bool)
				if (tf.sum() < tf.size) : 
					w = weight.copy()
					w, a = w[tf], a[tf]
					w /= w.sum()
				else : w = weight
			arr[i] = (a * w).sum(0)
			if (sigma is True): arrstd[i]=((a-arr[i])*w).std(0)
		arr = arr[n1:n2]
		if (sigma is True) : 
			arrstd = arrstd[n1:n2]
			arr = np.concatenate([arr, arrstd], 0)
	#---------------------------------------------
	#---------------------------------------------
	else : 
		from jizhipy.Math import Convolve
		if (case != 6) : 
			array = np.concatenate([aleft, array, aright], 0)
			n1, n2 = nla, len(array)-nra
		if (sigma is True) : arrstd = array*0
		n = weight.size - array.shape[0]
		nl, nr = int(abs(n)/2), int(abs(n)-abs(n)/2)
		if (n > 0) : weight = weight[nl:-nr]
		elif (n < 0) : weight = np.concatenate([np.zeros(nl), weight, np.zeros(nr)])
		for i in range(array.shape[1]) : 
			array[:,i] = Convolve(array[:,i], weight, 'linear')
		if (case != 6) : 
			array = array[n1:n2]
			if (sigma is True) : arrstd = arrstd[n1:n2]
		arr = array
	#---------------------------------------------
	#---------------------------------------------
	return arr





def _Multiprocess_SmoothReduce( iterable ) : 
	import numpy as np
	array = iterable[1]
	per, sigma = iterable[2]
	n1, n2, count, arr = 0, 0, 0, []
	if (sigma) : arrstd = []
	while (n2 < array.shape[0]) : 
		n1, n2 = count*per, (count+1)*per
		if (n2 > array.shape[0]) : a2 = array.shape[0]
		arr.append( array[n1:n2].mean(0) )
		if (sigma) : arrstd.append( array[n1:n2].std(0) )
		count += 1
	arr = np.array(arr)
	if (sigma) : 
		arrstd = np.array(arrstd)
		arr = np.concatenate([arr[:,None], arrstd[:,None]], 1)
	return arr





def _Multiprocess_SmoothSmall( iterable ) : 
	import numpy as np
	from jizhipy.Optimize import Interp1d
	array = iterable[1]
	N = iterable[2]
	x = np.arange(array.shape[1])
	xnew = np.linspace(0, x[-1], N)
	a = []
	for i in range(array.shape[0]) : 
		a.append( Interp1d(x, array[i], xnew) )
	a = np.array(a).T
	return a





def _Multiprocess_SmoothLarge( iterable ) : 
	import numpy as np
	array, n = iterable[1]
	a = []
	for i in range(len(n)) : 
		a.append( array[n[i,0]:n[i,1]].mean(0) )
	array = np.array(a)
	return array


	


def _Multiprocess_medfilt( iterable ) : 
	import scipy.signal as spsn
	array = iterable[1]
	kernel_size, tf = iterable[2]
	for i in range(len(array)) : 
		if (tf) : array[i] = spsn.medfilt(array[i], kernel_size)
		else : array[i] = spsn.medfilt(array[i].real, kernel_size) + 1j*spsn.medfilt(array[i].imag, kernel_size)
	return array





def Medfilt( array, axis, kernel_size, Nprocess=1 ) : 
	'''
	array:
		Any shape
		dtype: 
			int/float/complex
			NOT MaskedArray

	axis:
		None | int number
		(1) ==None: 
			use scipy.signal.medfilt(array, kernel_size)
			do 1D, 2D, 3D, ..., N-D medfilt
		(2) ==int number: 
			mediam filter along this axis
			do 1D medfilt for all axes
	'''
	if (kernel_size is None or kernel_size<3) : return array
	import scipy.signal as spsn
	import numpy as np
	from jizhipy.Array import Asarray, ArrayAxis
	from jizhipy.Process import PoolFor, NprocessCPU
	Nprocess = NprocessCPU(Nprocess)[0]
	array,kernel_size=Asarray(array),Asarray(kernel_size).flatten()
	shape, dtype = array.shape, array.dtype
	array = 1.*array

	tf = (kernel_size % 2 == 0)
	kernel_size[tf] += 1
	kernel_size = kernel_size[:len(shape)]
	kernel_size = np.append(kernel_size, [kernel_size[0] for i in range(len(shape)-len(kernel_size))])
	kernel_size = kernel_size.astype(int)
	tf = False if(dtype.name[:7]=='complex')else True

	if (axis is None) : 
		if (len(shape) == 2) : 
			if (tf): array =spsn.medfilt2d(array, kernel_size)
			else : array = spsn.medfilt2d(array.real, kernel_size) + 1j*spsn.medfilt2d(array.imag, kernel_size)
		else : 
			if (tf) : array = spsn.medfilt(array, kernel_size)
			else : array = spsn.medfilt(array.real, kernel_size) + 1j*spsn.medfilt(array.imag, kernel_size)

	else : 
		axis = int(round(axis))
		if (axis < 0) : axis = len(shape) + axis
		if (axis >= len(shape)) : Raise(Exception, 'axis='+str(axis)+' out of '+str(len(shape))+'D array.shape='+str(shape))
		kernel_size = kernel_size[axis]
		if (shape[axis] == 1) : return array

		if (len(shape) == 1) : 
			if (tf) : array = spsn.medfilt(array, kernel_size)
			else : array = spsn.medfilt(array.real, kernel_size) + 1j*spsn.medfilt(array.imag, kernel_size)
		else : 
			array = ArrayAxis(array, axis, -1)
			shape = array.shape
			array=array.reshape(np.prod(shape[:-1]), shape[-1])
			sent = array
			bcast = [kernel_size, tf]
			if (Nprocess == 1) : 
				array=_Multiprocess_medfilt([None,sent, bcast])
			else : 
				pool = PoolFor(0, len(array), Nprocess)
				array = pool.map_async(_Multiprocess_medfilt, sent, bcast)
				array = np.concatenate(array, 0)

			array = array.reshape(shape)
			array = ArrayAxis(array, -1, axis)

	array = array.astype(dtype)
	return array





def Smooth( array, axis, per=3, times=None, std=None, filt=None, sigma=False, reduceshape=False, nlr=True, fft=False, Nprocess=1 ) : 
	'''
	Smooth/Average/Mean array along one axis.
	We can also use spsn.convolve() to do this, but spsn.convolve() will cost much more memory and time, so, the function written here is the best and fastest.

	array:
		Any shape, any dtype
		(1) int/float/complex array: faster
		(2) MaskedArray: slower

	nlr:
		Large times/std will make the left and right edge worse and worse
		Select different nlr, the edge effect wil be different. Try different nlr and use the best case !!!
		(1) ==None: append first/last "element"
		(2) ==True: set nlr=[len(array)/100, len(array)/100]
		(3) ==[int, int]: set nlr=[int, int]  # interp1d
		(4) isnum (float or int): as the outside value (nlr=0)
		(5) =='periodic': append right end to the left head, left head to right end
		(6) ==False: don't append, always filt.sum()==1, completely ==Convolve(array, filt)
		(7) =='mirror': append mirror

	axis:
		array will be smoothed/averaged along this axis.

	** First use (per, times), second use std, third use filt

	per, times:
		int, int/'stock'
		Smooth times, for each time, the window size=per: equivalent to use GaussianFilter(per, times) to convolve array

	std:
		isnum, unit is "pixel"
		The std of GaussianFilter
		std = fwhm / (8*np.log(2))**0.5

	filt:
		Use this filter directly, any size
		NOTE THAT must filt.sum()==1 (normalized), otherwise the result may be bad

	sigma:
		Used in case 1,2,3,4, NOT 5
		False, True, int, np.array
		(1) ==False: don't return the error
		(2) ==True: calculate the error of the output
		(3) ==isnum or np.array: use this sigma to calculate the error of the output

	reduceshape:
		(1) ==False: return.shape = array.shape
		(2) ==True: used togather with per, reduce the data
		(3) ==int: return.shape[axis]==this int, this int can >= or < array.shape[axis], any value you want

	fft:
		True | False
		Use FFT to convolve or NOT?

	Note that:
		Wide of 2 windows: w1 < w2
		a1  = Convolve(a,  w1)
		a2  = Convolve(a,  w2)
		a12 = Convolve(a1, w2)
		=> a12 = Smooth(a2)
		But the beam sizes of the result maps are similar (roughly the same), Beamsize(a12) >= Beamsize(a2).

	(1) Smooth( a, axis, per, times )
	(2) Smooth( a, axis, std=std )
	(3) Smooth( a, axis, filt=filt )
	(4) Smooth( a, axis, per, reduceshape=True )
	(5) Smooth( a, axis, reduceshape=int number )
	'''
	import numpy as np
	from jizhipy.Basic import IsType, Raise
	from jizhipy.Array import ArrayAxis, Asarray
	from jizhipy.Process import PoolFor, NprocessCPU
	array = Asarray(array)
	dtype, shape = array.dtype.name, array.shape
	if (len(shape) == 1) : axis = 0
	if (axis < 0) : axis = len(shape) + axis
	if (axis >= len(shape)) : Raise(Exception, 'axis='+str(axis)+' out of array.shape='+str(shape))
	#--------------------------------------------------
	if (per is not None and IsType.isstr(times)) : 
		per, times = int(per), str(times).lower()
		if ('stock' not in times) : Raise(Exception, 'per is int, but times = '+times+' != stock')
		case = 6
	#--------------------------------------------------
	elif (per is not None and times is not None) : 
		per, times, case = int(per), int(times), 1
	elif (std is not None) : std, case = float(std), 2
	elif (filt is not None) : filt, case = Asarray(filt), 3
	elif (per is not None and reduceshape is True) : 
		per, case = int(per), 4
	elif (IsType.isnum(reduceshape)) : 
		N, sigma, case = int(reduceshape), False, 5
	#--------------------------------------------------
	if (case == 1 and (per<=1 or times<=0)) : return array
#	elif (case == 2 and std < 1) : return array
	elif (case == 3 and abs(filt).max() < 1e-9) : return array
	elif (case == 4 and per <= 1) : return array
	elif (case == 5 and N == shape[axis]) : return array
	#--------------------------------------------------
	#--------------------------------------------------

	# int array to float32
	if ('int' in dtype) : 
		dtype = 'float32'
		array = array.astype(dtype)
	#--------------------------------------------------
	# Move axis to axis=0, smooth along axis=0
	array = ArrayAxis(array, axis, 0, 'move')
	shape0 = array.shape
	# N-D array to 2D | matrix
	if   (len(shape0) == 1) : array = array[:,None]
	elif (len(shape0)  > 2) : array = array.reshape(shape0[0], int(np.prod(shape0[1:])))
	shape = array.shape
	# shape0: after ArrayAxis(), smooth along shape0[0]
	# shape: 2D
	#--------------------------------------------------
	Nprocess = NprocessCPU(Nprocess)[0]
	if (Nprocess > shape[1]) : Nprocess = shape[1]
	#--------------------------------------------------
	#--------------------------------------------------

	if (case == 5) :  # Smooth(array, axis, reduceshape=int)
		# Now smooth along axis=0
		if (shape[1] > N) : 
			n = np.linspace(0, shape[0], N+1).astype(int)
			n = np.array([n[:-1], n[1:]]).T
			if (Nprocess <= 1) : 
				iterable = (None, (array, n), None)
				array = _Multiprocess_SmoothLarge(iterable)
			else : 
				pool, send = PoolFor(0, N, Nprocess), []
				ns, a = pool.nsplit, []
				for i in range(len(ns)) : 
					m = n[ns[i,0]:ns[i,1]]
					a = array[m[0,0]:m[-1,1]]
					m -= m[0,0]
					send.append( [a, m] )
				pool = PoolFor()
				array = pool.map_async(_Multiprocess_SmoothLarge, send)
				array = np.concatenate(array, 0)
		#----------------------------------------
		else : 
			if (Nprocess <= 1) : 
				iterable = (None, array.T, N)
				array = _Multiprocess_SmoothSmall(iterable)
			else : 
				pool = PoolFor(0, array.shape[1], Nprocess)
				array = pool.map_async(_Multiprocess_SmoothSmall, array.T, N)
				array = np.concatenate(array, 1)
		array = array.reshape((len(array),)+shape0[1:])
	#--------------------------------------------------
	#--------------------------------------------------

	# per is even, b[i] = a[i+1-per/2:i+1+per/2]
	# per is  odd, b[i] = a[i  -per/2:i+1+per/2]
	# per is even, left end + [:per/2-1], right end + [-per/2:]
	# per is  odd, left end + [:per/2  ], right end + [-per/2:]
	#--------------------------------------------------
	# nlr:
	# If reduceshape==False, large times will make the left and right edge worse and worse
	# (1) ==False/None: append first/last element
	# (2) ==True: set nlr=[len(array)/100, len(array)/100]
	# (3) ==[int, int]: set nlr=[int, int]
	# (4) isnum (float or int): as the outside value
	# (5) =='periodic': append right end to the left head, left head to right end
	elif (case in [1, 2, 3, 6]) : 
		# Smooth(array, axis, per, times)
		# Smooth(array, axis, std=std)
		# Smooth(array, axis, filt=filt)
		# Smooth(array, axis, per, 'stock')
		if (case == 1) : weight =GaussianFilter(per, times)[0]
		elif (case == 2) : weight = GaussianFilter(None,None, std, shape[0])[0]
		elif (case == 3) : weight = filt
		elif (case == 6) : weight =GaussianFilter(per,1)[0] #@!
		# Normalized
		weight /= weight.sum()
		# Cut the filter in order to faster
		weight = weight[weight>3e-5*weight.max()]  #@#@#@
		Nw = len(weight)
		#--------------------------------------------------
		if (case == 6) : nla, nra, nlr = Nw-1, 0, None  #@!
		else : 
			nla = int(Nw/2) if(Nw%2==1)else int(Nw/2)-1
			nra = int(Nw/2)
		if (nlr is None) : casenlr = 1
		elif (nlr is True) : 
			n = int(len(array)/100)
			if (n < 5) : n = 5
			nlr, casenlr = [n, n], 2
		elif (IsType.isnum(nlr)) : casenlr = 4
		elif (IsType.isstr(nlr) and (str(nlr).lower()=='periodic' or str(nlr).lower()[:3]=='per')) : casenlr = 5
		elif (IsType.isstr(nlr) and str(nlr).lower()=='mirror') : casenlr = 7
		elif (nlr is False) : nlr, casenlr = np.nan, 6
		else : nlr, casenlr = nlr[:2], 3
		#--------------------------------------------------
	#	nla, nra = nla+1, nra+1
		bcast = [weight, nla, nra, sigma, nlr, casenlr, fft]
		if (Nprocess == 1) : 
			iterable = ((None,None), array.T, bcast)
			array = _Multiprocess_Smooth(iterable)
		else : 
			pool = PoolFor(0, array.shape[1], Nprocess)
			array = pool.map_async(_Multiprocess_Smooth, array.T, bcast)
			array = np.concatenate(array, 1)
		if (sigma is True) : 
			arrstd = array[shape[0]:].reshape(shape0)
			array  = array[:shape[0]]
		array = array.reshape(shape0)
	#--------------------------------------------------
	#--------------------------------------------------

	elif (case == 4) :  # Smooth(array, axis, per, reduceshape=True)
		shape = array.shape
		if (Nprocess > shape[0]) : Nprocess = shape[0]
		bcast = (per, sigma)
		#--------------------------------------------------
		if (Nprocess <= 1) : 
			iterable = (None, array, bcast)
			array = _Multiprocess_SmoothReduce(iterable)
		#--------------------------------------------------
		else : 
		#	n2 = PoolFor(0, len(n1), Nprocess).nsplit
		#	send = []
		#	for i in range(len(n2)) : 
		#		j, k = n2[i]
		#		n3 = n1[j:k]
		#		send.append([n3, array[n3.min():n3.max()]])
			n1 =np.append(np.arange(0,shape[0],per), [shape[0]])
			n1 = np.array([n1[:-1], n1[1:]]).T
			n2 = np.linspace(0, len(n1), Nprocess+1).round().astype(int)
			nsplit, send = [], []
			for i in range(len(n2)-1) : 
				n = n1[n2[i]:n2[i+1]]
				if (n.size > 0) : nsplit.append( n )
			for i in range(len(nsplit)) : 
				n1, n2 = nsplit[i].min(), nsplit[i].max()
				send.append( array[n1:n2] )
			pool = PoolFor()
			array = pool.map_async(_Multiprocess_SmoothReduce, send, bcast)
			array = np.concatenate(array, 0)
		if (sigma) : 
			array, arrstd = array[:,0], array[:,1]
			arrstd=arrstd.reshape((len(array),)+shape0[1:])
		array = array.reshape((len(array),)+shape0[1:])
	#--------------------------------------------------
	#--------------------------------------------------

	array = ArrayAxis(array, 0, axis, 'move')
	array = array.astype(dtype)
	if (sigma) : 
		arrstd = ArrayAxis(arrstd, 0, axis, 'move')
		arrstd = arrstd.astype(dtype)
		return [array, arrstd]
	return array




def Sample( array, axis, per=None, size=None ) : 
	'''
	Reduce the size of array

	array:
		Any shape, any dtype
		int/float/complex ndarray
		MaskedArray

	axis: 
		Sampling along which axis?

	per, size:
		Use one of them
		(1) per, size=None:
				Sampling every per ponts
		(2) per=None, size:
				Sampled array.shape[axis] == size
	'''
	import numpy as np
	from jizhipy.Basic import IsType, Raise
	from jizhipy.Array import ArrayAxis
	if (not IsType.isndarray(array) and not IsType.isnmaskedarray(array) and not IsType.ismatrix(array)) : array = np.array(array)
	if (len(array.shape) == 1) : axis = 0
	elif (axis < 0) : axis += len(array.shape)
	if (axis >= len(array.shape)) : Raise(Exception, 'axis='+str(axis)+' out of array.shape='+str(array.shape))
	#----------------------------------------
	if (per is None and size is None) : return array
	#----------------------------------------
	elif (per is None and array.shape[axis] >= size) : 
		return Smooth(array, axis, reduceshape=size)
	#----------------------------------------
	elif (per is not None) : 
		per += 0
		n = np.arange(0, array.shape[axis], per)
	else : 
		size += 0
		n = np.linspace(0, array.shape[axis]-1, size)
	n = n.astype(int)
	#----------------------------------------
	array = ArrayAxis(array, axis, 0)
	array = array[n]
	array = ArrayAxis(array, 0, axis)
	return array
	


