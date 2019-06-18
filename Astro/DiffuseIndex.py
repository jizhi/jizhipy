
def DiffuseIndex( freq, Npix=None, random=False, scalestd=1 ) : 
	'''
	spectrum index of Galactic diffuse emission
	A = A0 * (freq / freq0)**(-beta)
	return beta, .shape=(Npix, freq.size)

	beta0 = [(2.482,0.044), (2.477,0.036), (2.533,0.058), (2.501,0.079), (2.528,0.102), (2.691,0.121), (2.718,0.155), (3.095,0.096), (3.097,0.088), (3.088,0.084), (3.092,0.077), (3.077,0.071)]

	pf = [2.54585438, 5.99296288e-02, 3.32505876e-02, 2.57045977e-03, -1.69742991e-03, -9.13846215e-05, 2.87044311e-05]

	pf_err = [0.035, 0.018, 1.21e-2, 2.38e-3, 0.93e-3, 6.73e-5, 1.92e-5]

	plt.plot(freq, index, 'b-')
	plt.xlabel(r'$\nu$ [MHz]', size=14)
	plt.ylabel(r'$\beta$', size=14)
	plt.xlim(5, 2e5)
	plt.ylim(2.4, 3.2)
	jp.Plt.Axes.Tick('y', 'both', '%.1f', [0.1, 0.05], length=8, right=True)
	jp.Plt.Axes.Tick('x', 'both', length=8, top=True)
	jp.Plt.Axes.Label(size=12)
	jp.Plt.Axes.Format('x', 'lg')
	plt.show()
	'''
	import numpy as np
	from jizhipy.Basic import IsType
	pf = [2.55, 0.06, 3.33e-2, 2.57e-3, -1.7e-3, -9.14e-5, 2.86e-5]
	pf_err = np.array([0.035, 0.002, 0.069e-2, 0.093e-3, 0.028e-3, 0.162e-5, 0.038e-5])*scalestd
	islistfreq = False if(IsType.isnum(freq))else True
	freq = np.array(freq).flatten()
	#---------------------------------------------
	if (not random) : 
		if (not islistfreq) : freq = freq[0]
		index = pf[0]
		for i in range(1, len(pf)) : 
			index = index + pf[i]*(np.log(freq/408.))**i
		if (Npix is not None) : index = index + np.zeros(Npix)
		return index
	#---------------------------------------------
	else : 
		if (Npix is not None) : islistpix = True
		else : islistpix, Npix = False, 1
		index =np.random.randn(Npix,freq.size)*pf_err[0]+pf[0]
		for i in range(1, len(pf)) : 
			index += (np.random.randn(Npix, freq.size)*pf_err[i] + pf[i])*(np.log(freq[None,:]/408.))**i
		if (not islistpix and islistfreq) : 
			index = index[0]
		elif (islistpix and not islistfreq) : 
			index = index[:,0]
		elif (not islistpix and not islistfreq) : 
			index = index[0,0]
		return index
		
		

