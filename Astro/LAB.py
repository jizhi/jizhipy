
def LAB( fwhm, freqmin, freqmax, lon=None, lat=None, nside=None, coordsys='Galactic', labpath='labh.fits', verbose=True ) : 
	'''
	fwhm:
		[degree]
		fwhm==None: don't smooth the map

	freqmin, freqmax:
		[MHz]
		Average from freqmin (include) to freqmax (include)
		LAB freq resol: 0.00488281 MHz
		Must be isnum, NOT list
		If freqmax=None, then freqmax=freqmin

	lon, lat:
		[degree]
		lon: RA  | l
		lat: Dec | b
		Must can be broadcast
		Use lon, lat first, otherwise, use nside

	nside:
		Healpix full sky map instead of lon, lat

	coordsys:
		'Equatorial' | 'Galactic'

	labpath:
		str, where is the file "labh.fits"


	LAB: [1418.2329, 1422.5786] MHz
	'''
	import pyfits
	import healpy as hp
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise, IsType, Path
	from jizhipy.Astro import DopplerEffect
	from jizhipy.Transform import CoordTrans
	if (freqmin is not None) : 
		freqmin = Asarray(freqmin).flatten()
		if (freqmin.size != 1 ) : Raise(Exception, 'freqmin must isnum, but now freqmin.size='+str(freqmin.size))
		freqmin = freqmin[0]
	if (freqmax is not None) : 
		freqmax = Asarray(freqmax)
		if (freqmax.size != 1 ) : Raise(Exception, 'freqmax must isnum, but now freqmax.size='+str(freqmax.size))
		freqmax = freqmax[0]
	if   (freqmin is not None and freqmax is None) : 
		freqmax = freqmin
	elif (freqmin is None and freqmax is not None) : 
		freqmin = freqmax
	elif (freqmin is None and freqmax is None) : 
	#	if (verboase) : Raise(Warning, 'freqmin = freqmax = None')
		if (verboase) : print('Warning: jp.LAB(), freqmin = freqmax = None')
		return np.array(0)

	if (freqmin < 1418.2329 or freqmax > 1422.5786) : 
		if (verbose) : print('Warning: jp.LAB(), freqmin=%.3f, freqmax=%.3f out of [1418.2329, 1422.5786] MHz' % (freqmin, freqmax))
		return np.array(0)
	#--------------------------------------------------

	if (labpath is None) : labpath = 'labh.fits'
	if (not Path.ExistsPath(labpath)) : 
		labpath = Path.jizhipyPath('jizhipy_tool/labh.fits')
		if (not Path.ExistsPath(labpath)) : Raise(Exception, 'labath="'+labpath+'" NOT exists')
	fo = pyfits.open(labpath)
	#--------------------------------------------------

	if (lon is not None and lat is not None) : 
		if (np.sum((lat<-90)+(lat>90)) > 0) : Raise(Exception, 'lat out of [-90, 90] degree')
		nside = None
	else : nside, islist = int(nside), True
	#--------------------------------------------------

	hdr = fo[0].header
	bscale, bzero, blank = hdr['BSCALE_'], hdr['BZERO_'], hdr['BLANK_']
	freq0 = hdr['FREQ0'] *1e-6  # MHz
	Nlon, Nlat, Nfreq = hdr['NAXIS1'], hdr['NAXIS2'], hdr['NAXIS3']
	vmin, dv = hdr['CRVAL3']/1000., hdr['CDELT3']/1000.  # km/s
	#--------------------------------------------------

	vlist = np.arange(vmin, vmin+Nfreq*dv, dv)
	freqlist = freq0 + DopplerEffect(freq0, None, vlist)
	df = freqlist[Nfreq/2] - freqlist[Nfreq/2-1]
	if (freqmin < freqlist.min() or freqmax > freqlist.max()) : 
	#	if (verbose) : Raise(Warning, 'freqmin=%.3f, freqmax=%.3f out of [%.4f, %.4f] MHz' % (freqmin, freqmax, freqlist.min(), freqlist.max()))
		if (verbose) : print('Warning: jp.LAB(), freqmin=%.3f, freqmax=%.3f out of [%.4f, %.4f] MHz' % (freqmin, freqmax, freqlist.min(), freqlist.max()))
		return np.array(0)

	nfreqmin = abs(freqlist - freqmin)
	nfreqmin = np.where(nfreqmin==nfreqmin.min())[0][0]
	nfreqmax = abs(freqlist - freqmax)
	nfreqmax = np.where(nfreqmax==nfreqmax.min())[0][0]
	nfreqmin, nfreqmax = np.sort([nfreqmin, nfreqmax])
	nfreq = np.arange(nfreqmin, nfreqmax+1)
	#--------------------------------------------------

	# Take healpix map in Galactic
	lab = fo[0].data[nfreq]+0
	tf = (lab==blank)
	lab = np.float32(lab * bscale + bzero)
	lab[tf] = np.nan
	lab = np.ma.MaskedArray(lab, tf)
	lab = np.float32(lab.mean(0).data)  # (361, 721)
	del tf
	#--------------------------------------------------

	nsidelab = 256  # 512
	latlab, lonlab=hp.pix2ang(nsidelab, np.arange(12*nsidelab**2))
	latlab, lonlab = 90-latlab*180/np.pi, lonlab*180/np.pi
	lonlab %= 360
	lonlab[lonlab>180] = lonlab[lonlab>180] - 360
	nlon = ((lonlab - 180) / -0.5).round().astype(int)
	nlon[nlon>=Nlon] = Nlon-1
	del lonlab
	nlat = ((latlab + 90) / 0.5).round().astype(int)
	nlat[nlat>=Nlat] = Nlat-1
	del latlab
	npix = nlat*Nlon + nlon
	del nlat, nlon
	lab = lab.flatten()[npix]  # healpix map
	del npix
	if (fwhm not in [0, None] and fwhm > 0.5) : 
		fwhm = (fwhm**2 - 0.5**2)**0.5
		lab = np.float32(hp.smoothing(lab, fwhm*np.pi/180, verbose=False))  # hp.smoothing() is slow !
	#--------------------------------------------------

	coordsys = str(coordsys).lower()
	if (nside is None) : 
		islist = False if(IsType.isnum(lon+lat))else True
		lon, lat = lon+lat*0, lon*0+lat  # same shape
		lon %= 360
		lon, lat = CoordTrans.Celestial(lon*np.pi/180, lat*np.pi/180, coordsys, 'Galactic')  # rad
		npix = hp.ang2pix(nsidelab, np.pi/2-lat, lon)
		del lon, lat
		lab = lab[npix]
	else : 
		if (nside != nsidelab) : lab = hp.ud_grade(lab, nside)
		if (coordsys != 'galactic') : lab = CoordTrans.CelestialHealpix(lab, 'RING', 'Galactic', coordsys)[0]
	lab = np.float32(lab)
	if (not islist) : lab = lab[0]
	return lab





LABfreq = (1418.2329, 1422.5786, 0.00488281)

def LABdo( freq ) : 
	from jizhipy.Array import Asarray
	freq = Asarray(freq)
	fmin, fmax = freq.min(), freq.max()
	if (fmax < LABfreq[0]) : return False
	elif (fmin > LABfreq[1]) : return False
	else : return True


