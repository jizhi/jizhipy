
def CompressArray( array, dtype ) : 
	'''	
	dtype:
		'int8' or np.int8 or 'int16' or np.int16
		Must be 'int'

	arcsinh:
		=True : array = np.arcsinh(array)
		=False: don't use arcsinh()

	return: 
		r
		array_original = np.sinh(r.array * r.bscale + r.bzero) + np.random.random(r.array.size)*r.random*2-r.random

	Because float32 and float64 can save any value, 
	there is no advantage to compress to int32 and int64. 
	Therefore, when dtype in [int32, int64], return float32 or float64

	intN             <=2**(N-1)-1
	int8      3 wei  <=2**(8-1)-1=127
	int16     5 wei  <=2**(16-1)-1=32767
	int32    10 wei  <=2**(32-1)-1=2147483647
	int64    19 wei  <=2**(64-1)-1=9223372036854775807

	float16   4 wei  <=2**16-18=65518
	float32   8 wei  any value
	float64  16 wei  any value
	'''	
	arcsinh = True
	limit = {'int8':126, 'int16':32766, 'int32':2147483646, 'int64':8223372036854775806}
	import numpy as np
	from jizhipy.Basic import Struct
	if   (dtype == 'int128') : dtype = 'float64'
	elif (dtype == 'float8') : dtype = 'int8'
	dtype = np.dtype(dtype).name
	if   ('32' in dtype) : dtype = 'float32'
	elif ('64' in dtype) : dtype = 'float64'
	#----------------------------------------
	if ('32' in dtype or '64' in dtype) : 
		array = array.astype(dtype)
		return Struct(array=array, bscale=1, bzero=0, random=0)
	else : 
		arcsinh = bool(arcsinh)
		array = np.array(array, 'float64')
		array0 = abs(array)
		array0 = np.sort(array0 + 1j*np.arange(array.size))
		array0, n = array0.real, array0.imag.astype(int)
		n = n[array0>0][:3]
		array0 = array[n]
		if (arcsinh) : array = np.arcsinh(array) # -50 ~ 50
	#----------------------------------------
	vmin, vmax = array.min(), array.max()
	bscale = (vmax - vmin) / 2 / limit[dtype]
	bzero = vmin + limit[dtype] * bscale
	array = (array - bzero) / bscale
	array = array.round().astype(dtype)
	a = np.sinh(array[n]*bscale+bzero)
	random = abs(a - array0).mean()
	return Struct(array=array, bscale=bscale, bzero=bzero, random=random)



