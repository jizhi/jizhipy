
def NVSSSUMSS( fwhm, lon=None, lat=None, nside=None, coordsys='Galactic', galcenter=True, nvsssumsspath='nvss-sumss_1.4GHz_15mJy_fullsky.hdf5' ) : 
	'''
	return radio source map
		* merged from NVSS and SUMSS
		* healpix, RING, full sky, 1.4GHz, 15mJ limit for completeness
		* fill the missing region

	fwhm:
		[degree]
		fwhm==None: don't smooth the map

	lon, lat:
		[degree]
		lon: RA  | l
		lat: Dec | b
		Must can be broadcast
		Use lon, lat first, otherwise, use nside

	nside:
		Healpix full sky map instead of lon, lat
		nside0 = 1024

	coordsys:
		'Equatorial' | 'Galactic' | 'Ecliptic'

	galcenter:
		True | False
		==True: Retain the galactic center
		==False: discard the galactic center

	nvsssumsspath:
		str, where is the file 'nvss-sumss_1.4GHz_15mJy_fullsky.hdf5'
	'''
	import numpy as np
	import healpy as hp
	import h5py
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise, Path, IsType
	from jizhipy.Transform import CoordTrans
	if (nvsssumsspath is None) : nvsssumsspath = 'nvss-sumss_1.4GHz_15mJy_fullsky.hdf5'
	nvsssumsspath = Path.AbsPath(str(nvsssumsspath))
	if (not Path.ExistsPath(nvsssumsspath)) : 
		nvsssumsspath = Path.jizhipyPath('jizhipy_tool/nvss-sumss_1.4GHz_15mJy_fullsky.hdf5')
		if (not Path.ExistsPath(nvsssumsspath)) : Raise(Exception, 'nvsssumsspath="'+nvsssumsspath+'" NOT exists')
	#--------------------------------------------------

	nside0 = 1024
	if (lon is not None and lat is not None) : 
		if (np.sum((lat<-90)+(lat>90)) > 0) : Raise(Exception, 'lat out of [-90, 90] degree')
		nside = None
	else : 
		islist = True
		try : nside = int(nside)
		except : nside = nside0
	#--------------------------------------------------

	fo = h5py.File(nvsssumsspath, 'r')
	a = fo['nvsum'].value
	n, void = fo['void'].value
	a[n.astype(int)] = void
	resol = 0.06  # degree

	if (not galcenter) : 
		theta, phi = hp.pix2ang(nside0, np.arange(a.size))
		theta, phi = 90-theta*180/np.pi, phi*180/np.pi
		tf = (-5<theta)*(theta<5) * ((phi<55)+(phi>345))
		n = np.arange(a.size)[tf]
		del theta, phi, tf
		m = (np.random.random(n.size)*void.size).astype(int)
		a[n] = void[m]
	#--------------------------------------------------
	
	coordsys = str(coordsys).lower()
	if (nside is None) :  # use lon and lat to select region
		if (fwhm is not None and fwhm > resol) : 
			a = hp.smoothing(a, fwhm*np.pi/180, verbose=False)
			a[a<0] = 0
		islist = False if(IsType.isnum(lon+lat))else True
		lon, lat = lon+lat*0, lon*0+lat  # same shape
		lon %= 360
		lon, lat = CoordTrans.Celestial(lon*np.pi/180, lat*np.pi/180, coordsys, 'galactic')  # rad
		npix = hp.ang2pix(nside0, np.pi/2-lat, lon)
		del lon, lat
		a = a[npix]
		del npix
	else :   # full sky healpix map
		if (coordsys != 'galactic') : a = CoordTrans.CelestialHealpix(a, 'RING', 'galactic', coordsys)[0]
		if (fwhm is not None and fwhm > resol) : 
			a = hp.smoothing(a, fwhm*np.pi/180, verbose=False)
			a[a<0] = 0
		if (nside != nside0) : a = hp.ud_grade(a, nside)
	a = np.float32(a)
	if (not islist) : a = a[0]
	return a


