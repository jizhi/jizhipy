
def GSM( fwhm, freq, lon=None, lat=None, nside=None, coordsys='Galactic', gsmpath='gsm_parameter.out', save=False ) : 
	'''
	Already -2.726

	fwhm:
		[degree]
		fwhm==None: don't smooth the map

	freq, dfreq:
		[MHz]
		Must be one/isnum, NOT list
		* freq: center frequency
		* dfreq: averaged from freq-dfreq/2 (include) to freq+dfreq/2 (include). For GSM(), dfreq effect is very small, so remove this argument

	lon, lat:
		[degree]
		lon: RA  | l
		lat: Dec | b
		Must can be broadcast
		Use lon, lat first, otherwise, use nside

	nside:
		Healpix full sky map instead of lon, lat

	coordsys:
		'Equatorial' | 'Galactic' | 'Ecliptic'

	gsmpath:
		str, where is the program "gsm_parameter.out"
		./gsm_parameter.out  freq  outpath  gsmdir

	save:
		True | False
		Save gsm_xxx.npy or not?

	Compile:
		gfortran -ffixed-line-length-none gsm_parameter.f -o gsm_parameter.out

	return:
		gsm, ndarray
	'''
	import os
	import healpy as hp
	import numpy as np
	from jizhipy.Transform import CoordTrans
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise, Path, ShellCmd, IsType
	if (gsmpath is None) : gsmpath = 'gsm_parameter.out'
	freq = Asarray(freq).flatten()
	if (freq.size != 1) : Raise(Exception, 'freq must isnum, but now freq.size='+str(freq.size))
	freq = freq[0]
	gsmpath = Path.AbsPath(str(gsmpath))
	if (not Path.ExistsPath(gsmpath)) : 
		gsmpath = Path.jizhipyPath('jizhipy_tool/GSM/gsm_parameter.out')
		if (not Path.ExistsPath(gsmpath)) : Raise(Exception, 'gsmpath="'+gsmpath+'" NOT exists')
	n = gsmpath.rfind('/')
	if (n < 0) : gsmdir = ''
	else : gsmdir = gsmpath[:n+1]

	if (lon is not None and lat is not None) : 
		if (np.sum((lat<-90)+(lat>90)) > 0) : Raise(Exception, 'lat out of [-90, 90] degree')
		nside = None
	else : 
		islist = True
		try : nside = int(nside)
		except : nside = 512
	#--------------------------------------------------

	# list local "gsm_*.npy"
	gsmlist = ShellCmd('ls gsm_*.npy')
	ngsm = None
	for i in range(len(gsmlist)) : 
		try : f = np.sort([float(gsmlist[i][4:-4]), freq])
		except : continue
		r = (f[1] / f[0])**3
		if (r < 1.01) : 
			ngsm = i
			break
	if (ngsm is None) : 
		freqstr = ('%.2f' % freq)
		if (freqstr[-1] == '0') : freqstr = freqstr[:-1]
		if (freqstr[-1] == '0') : freqstr = freqstr[:-2]
		outname = 'gsm_'+freqstr
		os.system(gsmpath+' '+freqstr+' '+outname+'.dat '+gsmdir)
		gsm = np.float32(np.loadtxt(outname+'.dat'))
		os.system('rm '+outname+'.dat')
		if (save) : np.save(outname+'.npy', gsm)
	else : 
		gsm = np.float32(np.load(gsmlist[ngsm]))
		if (not save) : os.system('rm '+gsmlist[ngsm])
	if (Path.ExistsPath('qaz_cols.dat')) : os.system('rm qaz_cols.dat')
	nsidegsm = hp.get_nside(gsm)
	#--------------------------------------------------
		
	coordsys = str(coordsys).lower()
	if (nside is None) : 
		if (fwhm is not None and fwhm > 1) : 
			fwhm = (fwhm**2-1)**0.5
			gsm = hp.smoothing(gsm, fwhm*np.pi/180, verbose=False)
			gsm[gsm<0] = 0
		islist = False if(IsType.isnum(lon+lat))else True
		lon, lat = lon+lat*0, lon*0+lat  # same shape
		lon %= 360
		lon, lat = CoordTrans.Celestial(lon*np.pi/180, lat*np.pi/180, coordsys, 'galactic')  # rad
		npix = hp.ang2pix(nsidegsm, np.pi/2-lat, lon)
		del lon, lat
		gsm = gsm[npix]
		del npix
	else : 
		if (nside != nsidegsm) : gsm = hp.ud_grade(gsm, nside)
		if (coordsys != 'galactic') : gsm = CoordTrans.CelestialHealpix(gsm, 'RING', 'galactic', coordsys)[0]
		if (fwhm is not None and fwhm > 1) : 
			fwhm = (fwhm**2-1)**0.5
			gsm = hp.smoothing(gsm, fwhm*np.pi/180, verbose=False)
			gsm[gsm<0] = 0
	gsm = np.float32(gsm)
	if (not islist) : gsm = gsm[0]
	return gsm


