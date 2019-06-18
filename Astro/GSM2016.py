
def GSM2016( freq=None, outname=None, save=False, nside=None, coordsys='Galactic', unit='K', lowresol=False, dtype=None, verbose=False, rmnegative=False, rmsource=False, gsm2016path='GSM2016_data/', example=False ) : 
	'''
	An improved model of diffuse galactic radio emission from 10 MHz to 5 THz; arxiv:1605.04920; 2017MNRAS.464.3486Z

	Already remove CMB: -2.726

	example:
		True | False
		==True: read and return 'gsm2016_example.hdf5'
			jp.GSM2016(example=True)
			Only diffuse map, had remove CMB, 408MHz, K, nside=512

	compname = ['Total', 'Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']  # 'Dust1'=cold dust, 'Dust2'=warm dust

	freq:
		in MHz
		Must be one value, int | float, NOT list

	outname:
		Force to use .hdf5 format
		(1) ==str: complete output name
		(2) else: outname = 'gsm2016_'+strfreq+'.hdf5'

	save:
		==False: 
			    exists, read,     NOT save
			NOT exists, generate, NOT save
		==True: 
			    exists, generate, save
			NOT exists, generate, save
		==None: 
			    exists, read,     NOT save
			NOT exists, generate,     save

	lowresol:
		True | False
		(1) ==False:
				nside=1024
				*  freq< 10GHz, force resol=56arcmin
				** freq<=10GHz, force resol=24arcmin
		(2) ==False
				nside=64, resol=5deg=300arcmin
				Use './data/lowres_maps.txt'

	unit:
		'K'/'Trj' | 'Tcmb' | 'MJysr'

	dtype:
		'float64' | 'float32'

	nside:
		Default nside=1024

	rmnegative:
		reset negative
		True | False

	rmsource:
		remove sources from the output map
		0/False | 1 | 2 | 3 | 4
		0/False: NOT remove sources
		1: remove CygA, CasA, Crab
		2: remove sources outside center plane
		3: remove sources inside center plane

	return:
		gsm.shape = (6, 12*nside**2)

	compname = ['Total', 'Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']  # 'Dust1'=cold dust, 'Dust2'=warm dust
	'''
	import numpy as np
	import healpy as hp
	import h5py
	from jizhipy.Astro import RemoveSource
	from jizhipy.Basic import Raise, SysFrame, Mkdir, Path, Time, IsType
	if (verbose) : print('jizhipy.GSM2016:')
	if (nside is not None) : 
		n = int(round(np.log(nside) / np.log(2)))
		nside = 2**n
	else : nside = 512
	coordsys = str(coordsys)
	if (verbose) : 
		print('    freq =', freq, 'MHz')
		print('    coordsys =', coordsys)
		print('    nside =', nside)
	#----------------------------------------

	if (gsm2016path is None) : gsm2016path = 'GSM2016_data/'
	if (not Path.ExistsPath(gsm2016path)) : 
		gsm2016path = Path.jizhipyPath('jizhipy_tool/GSM2016_data/')
		if (not Path.ExistsPath(gsm2016path)) : Raise(Exception, 'gsmp2016path="'+gsm2016path+'" NOT exists')
	fdir = Path.AbsPath(gsm2016path)
	#----------------------------------------

	if (example) : 
		if (verbose) : print('    Use "gsm2016_example.hdf5"')
		fo = h5py.File(fdir+'gsm2016_example.hdf5', 'r')
		gsm = fo['gsm2016'].value
		fo.close()

		# ud_grade to nside
		if (nside is not None and gsm[0].size != 12*nside**2):
			gsm2 = []
			for i in range(len(gsm)) : 
				gsm2.append( hp.ud_grade(gsm[i], nside) )
			gsm = np.array(gsm2)

		if (coordsys.lower() != 'galactic') : 
			for i in range(len(gsm)) : gsm[i] = CoordTrans.CelestialHealpix(gsm[i], 'ring', 'Galactic', coordsys)[0]
		return gsm
	#---------------------------------------------
	#---------------------------------------------

	freq, unit, lowresol, verbose, rmnegative, rmsource = float(freq), str(unit).lower(), bool(lowresol), bool(verbose), bool(rmnegative), int(rmsource)
	vb = '123' if(verbose)else False
	if (verbose) : 
		print('    Total, Synchrotron, CMB, HI, Cold dust, Warm dust, Free-Free')
	if (freq <= 0) : 
		print('freq = '+str(freq)+' <=0')
		exit()
	dtype = np.dtype(dtype)
	#---------------------------------------------
	#---------------------------------------------

	if (IsType.isstr(outname)) : outname = str(outname)
	else : 
		strfreq = str(freq).split('.')
		while (strfreq[1]!='' and strfreq[1][-1]=='0') : 
			strfreq[1] = strfreq[1][:-1]
		if (strfreq[1]=='') : strfreq = strfreq[0]
		else : strfreq = strfreq[0] + '.' + strfreq[1]
		outname = 'gsm2016_' + strfreq + '.hdf5'
	if (save is not False and '/' in outname) : 
		for i in range(len(outname)-1, -1, -1) : 
			if (outname[i] == '/') : break
		Mkdir(outname[:i+1])
	#--------------------------------------------------
	#--------------------------------------------------

	freq /= 1000.  # GHz
	kB, C, h, T = 1.38065e-23, 2.99792e8, 6.62607e-34, 2.725
	hoverk = h / kB
	
	def K_CMB2MJysr(K_CMB, nu) :  # in Kelvin and Hz
	    B_nu =2*(h *nu) *(nu /C)**2 /(np.exp(hoverk *nu /T)-1)
	    conversion_factor = (B_nu *C /nu /T)**2 /2 *np.exp(hoverk *nu /T) /kB
	    return  K_CMB * conversion_factor * 1e20  # 1e-26 for Jy | 1e6 for MJy
	
	def K_RJ2MJysr(K_RJ, nu) :  # in Kelvin and Hz
	    conversion_factor = 2 *(nu /C)**2 *kB
	    return  K_RJ * conversion_factor * 1e20  # 1e-26 for Jy | 1e6 for MJy

	mjysr2k = 1. / K_RJ2MJysr(1., 1e9*freq)
	if (unit == 'tcmb') : 
		unit = 'Tcmb'
		conversion = 1. / K_CMB2MJysr(1., 1e9*freq)
	elif (unit == 'mjysr') : 
		unit = 'MJysr'
		conversion = 1.
		if (verbose) : print(('    Unit Conversion at %.3f MHz: 1 MJysr == %f '+unit) % (freq*1000, mjysr2k))
	else :  # (unit in ['k', 'trj']) : 
		unit = 'K'
		conversion = mjysr2k
	#--------------------------------------------------
	#--------------------------------------------------

	exist = True if(Path.ExistsPath(outname))else False
#	if (exist and save is not True) : 
	if (exist) : 
		try : 
			fo = h5py.File(outname, 'r')
			gsm = fo['gsm2016'].value
			fo.close()
			if (verbose) : print('    Exists: ', outname)
			return gsm
		except : pass
	#--------------------------------------------------

	exist = False
	if (save is not False) : save = True

	if (not exist) :  # generate
		spec_nf = np.loadtxt(fdir+'spectra.txt')
		nfreq = spec_nf.shape[1]
		left_index = -1
		for i in range(nfreq-1):
		    if (freq>=spec_nf[0,i] and freq<=spec_nf[0,i+1]):
		        left_index = i
		        break
		if (left_index < 0) : 
			print("FREQUENCY ERROR: %.3f MHz is outside supported frequency range of %.3f MHz to %.3f MHz." % (freq*1000, spec_nf[0,0]*1000, spec_nf[0,-1]*1000))
			exit()
		#--------------------------------------------------
		# compname must be this order, do NOT modify
		compname = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']  # 'Dust1'=cold dust, 'Dust2'=warm dust
		#--------------------------------------------------
		if (lowresol) : 
			nside_data, op_resolution = 64, 300.
		else : 
			nside_data = 1024  # forced because of *.bin files
			op_resolution = 56. if(freq<10)else 24.
		if (not lowresol) : 
		    gsm = np.array([ np.fromfile(fdir+'highres_'+cname+'_map.bin', dtype='float32') for cname in compname ])
		else:
		    gsm = np.loadtxt(fdir+'lowres_maps.txt')
		#--------------------------------------------------
		interp_spec_nf = np.copy(spec_nf)
		interp_spec_nf[0:2] = np.log10(interp_spec_nf[0:2])
		x1 = interp_spec_nf[0,left_index]
		x2 = interp_spec_nf[0,left_index+1]
		y1 = interp_spec_nf[1:,left_index]
		y2 = interp_spec_nf[1:,left_index+1]
		x = np.log10(freq)
		interpolated_vals =(x*(y2-y1) + x2*y1 - x1*y2) /(x2-x1)
		gsm = 10.**interpolated_vals[0] * interpolated_vals[1:,None] * gsm
		gsm = gsm * conversion
	#--------------------------------------------------
	if (len(gsm.shape) == 1) : gsm = gsm[None,:]
	else : gsm = np.concatenate([gsm.sum(0)[None,:], gsm], 0)
	#--------------------------------------------------
	#--------------------------------------------------

	# generate is nest, convert to ring
	# First ud_grade so that faster
	for i in range(len(gsm)) : 
		gsm[i] = hp.reorder(gsm[i], n2r=True)

	# ud_grade to nside
	if (gsm[0].size != 12*nside**2) : 
		gsm2 = []
		for i in range(len(gsm)) : 
			gsm2.append( hp.ud_grade(gsm[i], nside) )
		gsm = np.array(gsm2)
	#--------------------------------------------------
	#--------------------------------------------------

	# remove negative value in the hpmaps
	for i in range(len(gsm)) : 
		if (not rmnegative and i != 0) : break
		tf = gsm[i] < 0
		if (tf[tf].size == 0) : continue
		g = gsm[i].copy()
	#	g[tf] = g.mean()
		g[tf] = g[g>0].min()
		g = hp.smoothing(g, np.pi/180*10, verbose=False)
		gsm[i][tf] = g[tf]
	#--------------------------------------------------
	#--------------------------------------------------


	casa = np.array([[(111.44,-1.37),(93.89,2.37)]]) # fwhm=7
	casa1 = np.array([[(118.12,4.78),(121.83,1.09)]])
	cyga = np.array([[(81,2),(26.33,8.38)]]) # fwhm=7
	cyga1 = np.array([[(84.9,-0.55),(71.8,0.4)]]) # fwhm=3
	cyga2 = np.array([[(76.6,0.4),(71.8,0.4)]]) # fwhm=5
	cyga3 = np.array([[(88.82,4.69),(88.69,2)]]) # fwhm=3
	crab = np.array([[(-96,-2),(26.33,8.38)]]) # fwhm=4.5
	crab1 = np.array([[(-91.82,-1),(26.33,8.38)]]) # fwhm=3
	crab2 = np.array([[(-99.31,-3.43),(26.33,8.38)]]) # fwhm=3
	lb1 = np.concatenate([casa, casa1, cyga, crab, cyga1, cyga2, cyga3, crab1, crab2], 0)
	fwhm1 = np.array([7, 3, 7, 4.5, 3, 5, 3, 2, 2])
	lb10, lb11 = lb1[:,0], lb1[:,1]
	#--------------------------------------------------

	other = np.array([[(-153.41,-1.91),(-163.58,-9.29)], [(-170.31,2.08),(-163.58,-9.29)], [(-175.26,-5.88),(-163.58,-9.29)], [(-150.64,-19.38),(-157,-20)], [(-153.34,-16.39),(-157,-20)], [(133.73,1.07),(126.89,0.83)], [(160.4,2.78),(157.61,-1.24)], [(73.2,-9.06),(68.38,-7.56)], [(63.39,-5.98),(68.38,-7.56)]]) 
	fwhm2 = np.array([4, 4, 4, 1, 1, 2, 2, 3, 2])
	lb20, lb21 = other[:,0], other[:,1]
	#--------------------------------------------------

	center = np.array([[(-54.49,0.08),(-58.03,-0.04)], [(-61.38,-0.18),(-58.03,-0.04)], [(-69.23,-0.59),(-58.03,-0.04)], [(-72.15,-0.73),(-58.03,-0.04)], [(-75.88,-0.42),(-58.03,-0.04)], [(-77.63,-1.2),(-58.03,-0.04)], [(-65.21,-1.2),(-58.03,-0.04)], [(-63.1,9.8),(-60.6,8.89)], [(-6.93,16.9),(-9.35,16.06)]])
	fwhm3 = np.array([2, 2, 3, 3, 2, 2, 2, 2, 2])
	lb30, lb31 = center[:,0], center[:,1]

	if (rmsource in [1, 2, 3, 4]) : 
		if (rmsource == 4) : 
			lb0 = np.concatenate([lb30, lb20, lb10], 0)
			lb1 = np.concatenate([lb31, lb21, lb11], 0)
			fwhm = np.concatenate([fwhm3, fwhm2, fwhm1])
		elif (rmsource==1): lb0, lb1, fwhm = lb10, lb11, fwhm1
		elif (rmsource==2): lb0, lb1, fwhm = lb20, lb21, fwhm2
		elif (rmsource==3): lb0, lb1, fwhm = lb30, lb31, fwhm3
		gsm[0] = np.log10(gsm[0])
		times, same, onebyone = 1, False, True
		pix, revalue = RemoveSource(gsm[0], lb0, lb1, fwhm, times, same, onebyone, vb)
		gsm[0][pix] = revalue
		gsm[0]=hp.smoothing(gsm[0], np.pi/180,verbose=False)
		gsm[0] = 10.**gsm[0]

	gsm = gsm.astype(dtype)
	#--------------------------------------------------
	#--------------------------------------------------

	if (coordsys.lower() != 'galactic') : 
		for i in range(len(gsm)) : gsm[i] = CoordTrans.CelestialHealpix(gsm[i], 'ring', 'Galactic', coordsys)[0]

	if (save) : 
		if (verbose) : print('    Saving to: ', outname)
		Path.ExistsPath(outname, old=True)
		fo = h5py.File(outname, 'w')
		fo['gsm2016'] = gsm
		fo['gsm2016'].attrs['freq'] = freq*1000
		fo['gsm2016'].attrs['unit'] = unit
		fo['gsm2016'].attrs['MJysr2K'] = mjysr2k
		fo['gsm2016'].attrs['lowresol'] = int(lowresol)
		fo['gsm2016'].attrs['nside'] = nside
		fo['gsm2016'].attrs['pixtype'] = 'healpix'
		fo['gsm2016'].attrs['ordering'] = 'ring'
		fo['gsm2016'].attrs['coordsys'] = 'Galactic'
		fo['gsm2016'].attrs['rmnegative'] = int(rmnegative)
		fo['gsm2016'].attrs['rmsource'] = int(rmsource)
		fo['gsm2016'].attrs['creadate'] = Time(0)
		fo.close()
	return gsm


