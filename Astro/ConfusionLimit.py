
def ConfusionLimit( freq, fwhm=None, D=None, Q=3.35, Sobs=None ):
	'''
	freq: MHz
	fwhm: rad
	D   : meter
	Q   : 3~5
	Sobs: 
		which flux density (Jy) do you want to observe
		Use to select k and gamma

	return:
		confusion limit sigma_c [Jy/beam]
	'''
	import numpy as np
	from jizhipy.Basic import Raise
	if (Sobs is None) : Sobs = 1
	if (Sobs <= 1e-4) : gamma, k = 1.57, 29500
	elif (1e-4 < Sobs < 1e-3) : gamma, k = 2.23, 60.5
	elif (Sobs > 1e-3) : gamma, k = 1.77, 1300
	#--------------------------------------------------
	if (fwhm is None and D is None) : Raise(Exception, 'One of [fwhm, D] must not be None')
	if (fwhm is None) : fwhm = 1.03*300/freq/D
	#--------------------------------------------------
	sigmac1 = (Q**(3-gamma) * k * 1.133*(gamma-1)*fwhm**2 / (3-gamma))**(1./(gamma-1))
	# http://www.cv.nrao.edu/course/astr534/Radiometers.html
	# At cm wavelengths, the rms confusion in a telescope beam with FHWM is observed to be
	sigmac2 = 1e-3 * 0.2 * (freq*1e-3)**(-0.7) * (fwhm*180/np.pi*60)**2
	return np.array([sigmac1, sigmac2])



