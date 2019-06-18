
def Jy2K( S, freq, Dx=None, Dy=None, fwhmx=None, fwhmy=None, **kwargs ):
	'''
	Convert flux density S(Jy) to brightness temperature T(K) with Gaussian Beam with FWHM resolution

	Tb = S*1e-26 / 2 / kB * wavelength**2 / solid_angle
	Tb = S*1e-26 / 2 / kB / (1.133*1.029**2) * Dx**2 

	sigma = fwhm / (8ln2)**0.5
	sigma \approx lambda/D /ksd, ksd=1.962~2.286, Reza=2

	S, freq, Dx, Dy, fwhmx, fwhmy:
		Can be any shape, but must can broadcast

	freq:
		None | MHz
		(1) None: solid_angle = 1.133 * fwhm**2
		(2) in MHz: solid_angle = GaussianSolidAngle()

	fwhmx, fwhmy:
		in rad

	(1) Jy2K(S, None, D)
	(2) Jy2K(S, freq, D)
	(3) Jy2K(S, freq, Dx, Dy)
	(4) Jy2K(S, freq, fwhm)
	(4) Jy2K(S, freq, fwhmx, fwhmy)
	'''
	import numpy as np
	from jizhipy.Math import Gaussian
	from jizhipy.Basic import Raise
	from jizhipy.Array import Asarray
	if ('fwhm' in kwargs.keys()) : fwhmx = kwargs['fwhm']
	S, freq, Dx, Dy, fwhmx, fwhmy = Asarray2(S,True), Asarray2(freq,True), Asarray2(Dx,True), Asarray2(Dy,True), Asarray2(fwhmx,True), Asarray2(fwhmy,True)
	kB = 1.3806e-23
	const = S*1e-26 / 2 / kB
	if (freq is None and Dx is not None) : 
		if (Dy is not None and Dy != Dx) : Raise(Exception, 'Wrong parameters: Jy2K( S, None, Dx, Dy ). When use Dx and Dy, freq can NOT be None')
		Tb = const * Dx**2 / (1.133*1.029**2)
		return Tb
	#--------------------------------------------------

	if (Dx is not None) : 
		fwhmx = 1.029 * 300/freq / Dx
		if (Dy is not None) : fwhmy = 1.029 * 300/freq / Dy
		else : fwhmy = fwhmx

	elif (Dy is not None) : 
		fwhmy = 1.029 * 300/freq / Dy
		fwhmx = fwhmy

	elif (fwhmx is not None) : 
		if (fwhmy is not None) : fwhmy = Num(fwhmy, float)
		else : fwhmy = fwhmx

	elif (fwhmy is not None) : fwhmx = fwhmy
	else : Raise(Exception, 'Dx=Dy=fwhmx=fwhmy=None')
	#--------------------------------------------------
	case = 2
	if (case == 1) : 
		solid_angle =Gaussian.GaussianSolidAngle(fwhmx, fwhmy)
	elif (case == 2) : solid_angle = 1.133 * fwhmx * fwhmy
	#--------------------------------------------------
	Tb = const * (300./freq)**2 / solid_angle
	return Tb





def K2Jy( T, freq, Dx=None, Dy=None, fwhmx=None, fwhmy=None ):
	'''
	Convert brightness temperature T(K) to flux density S(Jy)

	Tb = S*1e-26 * wavelength**2 / 2 / kB / solid_angle
	S = Tb * 2*bB * 1e26 * solid_angle / wavelength**2

	T, freq, Dx, Dy, fwhmx, fwhmy:
		Can be any shape, but must can broadcast

	freq:
		None | MHz
		(1) None: solid_angle = 1.133 * fwhm**2
		(2) in MHz: solid_angle = GaussianSolidAngle()

	fwhmx, fwhmy:
		in rad

	(1) Jy2K(S, None, D)
	(2) Jy2K(S, freq, D)
	(3) Jy2K(S, freq, Dx, Dy)
	(4) Jy2K(S, freq, fwhm)
	(4) Jy2K(S, freq, fwhmx, fwhmy)
	'''
	import numpy as np
	from jizhipy.Math import Gaussian
	from jizhipy.Basic import Raise
	from jizhipy.Array import Asarray
	T, freq, Dx, Dy, fwhmx, fwhmy = Asarray2(T,True), Asarray2(freq,True), Asarray2(Dx,True), Asarray2(Dy,True), Asarray2(fwhmx,True), Asarray2(fwhmy,True)
	kB = 1.3806e-23
	const = T * 2*kB * 1e26
	if (freq is None and Dx is not None) : 
		if (Dy is not None and Dy != Dx) : Raise(Exception, 'Wrong parameters: Jy2K( S, None, Dx, Dy ). When use Dx and Dy, freq can NOT be None')
		S = const / Dx**2 * (1.133*1.029**2)
		return S
	#--------------------------------------------------

	if (Dx is not None) : 
		fwhmx = 1.029 * 300/freq / Dx
		if (Dy is not None) : fwhmy = 1.029 * 300/freq / Dy
		else : fwhmy = fwhmx

	elif (Dy is not None) : 
		fwhmy = 1.029 * 300/freq / Dy
		fwhmx = fwhmy

	elif (fwhmx is not None) : 
		if (fwhmy is not None) : fwhmy = Num(fwhmy, float)
		else : fwhmy = fwhmx

	elif (fwhmy is not None) : fwhmx = fwhmy
	else : Raise(Exception, 'Dx=Dy=fwhmx=fwhmy = None')
	#--------------------------------------------------
	solid_angle = Gaussian.GaussianSolidAngle(fwhmx, fwhmy)
	#--------------------------------------------------
	S = const / (300./freq)**2 * solid_angle
	return S



#def Jy2K( S, DorResol, freq=None, ksd=None, kind=None ) : 
#	'''
#	Convert flux density S(Jy) to brightness temperature T(K) with Gaussian Beam with FWHM resolution or Rayleigh_criterion resolution (base on the DorResol).
#
#	S:
#		the flux density in Jy at this frequency (S will change with frequency). 1Jy = 1e-26 w/m^2/Hz
#
#	DorResol:
#		Diameter or resolution in rad.
#
#	freq:
#		If DorResol is set to be Diameter, you don't need to set the freq.
#		But if DorResol is set to be resolution, you must also set the freq in MHz.
#
#	ksd:
#		Beam of antenna maybe not a perfect Gaussian beam.
#		sigma = lambda/D /ksd, ksd=1.962~2.286
#		Default, ksd=None, 1.03
#		For real case, ksd=2
#
#	kind: 
#		Just for DorResolu is Diameter.
#		We will use D to calculate the solid angle. But for the solid angle, use "FWHM" or "Rayleigh_criterion" ?
#
#	return T in K.
#
#	* S: actually is Jy/beam, because here, we assume the size of the image of the source is equal to the FHWM of Gaussian beam.
#
#	* \Omega = 1.1244*FWHM^2 is very fit by NVSS paper!! Not the \Omega = pi/4 * FWHM^2
#	* The result of ksd=None, kind=None is also fit by that of Planck 2015 paper.
#
#	**** 2D flux to K
#	Flux2D at any FWHM.
#	beam = BeamModel('gaussian', 'dish', D, freq, fov, N)[0] 
#	beam = beam / beam.max()  # center is 1
#	Then: 
#		T2D = Jy2K(Flux2D, D)
#		T2D = Convolve(T2D, beam)  # beam, not beam/beam.sum()
#	Or:
#		Flux2D = Convolve(Flux2D, beam)
#		T2D = Jy2K(Flux2D, D)
#		(the result is the same)
#	'''
#	k = 3.2205e-4
#	if (freq is not None) : 
#		T = (300./freq)**2 * k * S / DorResol**2
#	elif (freq is None) : 
#		if ksd is None : ksd = (8*np.log(2))**0.5 /1.03
#		#	if (kind=='fwhm' or kind=='FWHM') : k1 = 1.03
#		#	elif (kind == 'Rayleigh_criterion') : k1 = 1.22
#		#	else : raise Exception(efn()+'name must = "FWHM" or "Rayleigh_criterion"')
#	#	if (not(1.962 <= ksd <= 2.287)) : raise Exception(efn()+'ksd='+str(ksd)+' is out of range [1.962 ~ 2.287]')
#		DorResol = Asarray(DorResol)
#		if (DorResol.size == 1) : d2 = DorResol[0]**2
#		else : d2 = DorResol[:2].prod()
#		T = k * S * d2 * ksd**2 / (8*np.log(2))
##	else : raise Exception(efn()+'Both kind="" and freq="" is not allowed')
#	return T
