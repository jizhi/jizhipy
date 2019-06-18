
def _Interp1d( xdata, ydata, xnew, kind='linear' ) : 
	'''
	1D interpolation. Note that all array must be 1D
	Outside the xdata, will use  Linear interpolation

	xdata:
		must be 1D, must real

	ydata:
		must be 1D, must real here

	xnew:
		any shape ndarray (not need to be 1D) | None
		* ==None: return def func(x) => a function

	kind:
		'linear' or 'cubic'
	'''
	from scipy import interpolate
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise
	#--------------------------------------------------
	# xdata must from min to max
	xdata = xdata + 1j*ydata
	xdata = np.sort(xdata)
	ydata, xdata = xdata.imag, xdata.real  
#	ydata = np.arcsinh(ydata)  # smoother
	#--------------------------------------------------
	#--------------------------------------------------
	if (xnew is None) : 
		return interpolate.interp1d(xdata, ydata, kind=kind)
	#--------------------------------------------------
	#--------------------------------------------------
	# Re-order xnew
	xnew = np.array(xnew)
	shape_xnew = xnew.shape
	xnew = xnew.flatten()
	xnew = np.sort(xnew + 1j*np.arange(xnew.size))
	xnew, order = xnew.real, xnew.imag
	order = np.sort(order + 1j*np.arange(order.size))
	order = order.imag.astype(int)
	# xnew_new[order] = xnew_original
	#--------------------------------------------------
	# xnew inside xdata
	tf = (xdata[0]<=xnew)*(xnew<=xdata[-1])
	xin, orderin = xnew[tf], order[tf]
	# xnew outside, left
	tf = (xnew<xdata[0])
	xl, orderl = xnew[tf], order[tf]
	# xnew outside, right
	tf = (xnew>xdata[-1])
	xr, orderr = xnew[tf], order[tf]
	#--------------------------------------------------
	# Interp1d inside
	if (xin.size > 0) : 
		funcin =interpolate.interp1d( xdata, ydata, kind=kind)
		yin = funcin(xin)
	else : yin = []
	#--------------------------------------------------
	# Interp1d left
	if (xl.size > 0) : 
		for i in range(1, len(xdata)) : 
			if (xdata[i] != xdata[0]) : break
		yl = (ydata[i]-ydata[0])/(xdata[i]-xdata[0]) * (xl-xdata[0]) + ydata[0]
	else : yl = []
	#--------------------------------------------------
	# Interp1d right
	if (xr.size > 0) : 
		for i in range(-2, -len(xdata)-1, -1) : 
			if (xdata[i] != xdata[-1]) : break
		yr = (ydata[i]-ydata[-1])/(xdata[i]-xdata[-1]) * (xr-xdata[-1]) + ydata[-1]
	else : yr = []
	#--------------------------------------------------
	ynew = np.concatenate([yl, yin, yr])
	ynew = ynew[order].reshape(shape_xnew)
	return ynew





def Interp1d( xdata, ydata, xnew, kind='linear' ) : 
	'''
	1D interpolation. Note that all array must be 1D
	Outside the xdata, will use  Linear interpolation

	xdata:
		must be 1D, must real

	ydata:
		must be 1D, real | complex

	xnew:
		any shape ndarray (not need to be 1D) | None
		* ==None: return def func(x) => a function

	kind:
		'linear' or 'cubic'
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	xdata,ydata =Asarray(xdata).flatten(), Asarray(ydata).flatten()
	if ('complex' in ydata.dtype.name) : 
		yr = _Interp1d(xdata, ydata.real, xnew, kind)
		yi = _Interp1d(xdata, ydata.imag, xnew, kind)
		ynew = yr + 1j*yi
	else : 
		ynew = _Interp1d(xdata, ydata, xnew, kind)
	return ynew
