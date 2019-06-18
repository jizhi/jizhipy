class BrightSource( object ) : 

	def __init__(self):
		self.doinit = True


	def __doinit__( self ) : 
		'''
		self.regname
		self.regRADec (degree)
		self.regfreq  (MHz)
		self.regflux  (Jy)

		Orion_A is Orion_Nebula
		'''
		self.doinit = False
		import numpy as np
		allsource = {'CasA': (('Cassiopeia_A','CasA'), ((23+23/60.+24/3600.)*15, 58+48/60.+54/3600.)),  
			'CygA': (('Cygnus_A', 'CygA'), ((19+59/60.+28.4/3600.)*15, 40+44/60.+2/3600.)), 
			'CygX': (('Cygnus_X', 'CygX'), ((20+28/60.+41/3600.)*15, 41+10.2/60)),
			'Crab': (('Crab_Nebula','Crab', 'Taurus_A','TauA'), ((5+34/60.+31.9/3600.)*15, 22+0+52.2/3600.)), 
			'VirA': (('Virgo_A', 'VirA'), ((12+30/60.+49.4/3600.)*15, 12+23/60.+28/3600.)), 
			'OriA': (('Orion_A', 'OriA'), ((5+37/60.+17.3/3600.)*15, -(5+23/60.+28/3600.))), 
			'ForA': (('Fornax_A', 'ForA'), ((3+22/60.+42/3600.)*15, -(37+12/60.+30/3600.))), 
			'CenA': (('Centaurus_A', 'CenA'), ((13+25/60.+27.6/3600.)*15, -(43+1/60.+8.8/3600.))), 
			'Rose': (('Rosette_Nebula', 'Rose'), ((6+31/60.+40/3600.)*15, 4+57.8/60))}
		self.regname, self.regRADec = {}, {}
		for key in allsource.keys() : 
			self.regname[key] = allsource[key][0]
			self.regRADec[key]=np.array(allsource[key][1])
		#----------------------------------------

		# Brightness temperature at 408MHz in fwhm=1deg
		# Rose=107K, ForA=185K, OriA=110K, CygX=348K, CenA=555K
		self.regfreq = np.array([300.,400,408,500,600,700,
750,800,900,
1000,1100,1200,1300,1400,1410,1420,1500,1600,1700,1800,1900,
2000,2500,2695,3000,3500,4000,4500,4995,5000,5500,6000,6500,
7000,7500,8000,8500,9000,9500,10000,10500,10690,11000,11500,
12000,12500,13000,13500,14000,14500,15000,15375,15500,16000,
16500,17000,17500,18000,18500,19000,19350,19500,20000,21000,
22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,
31400,32000,33000,34000,35000,36000,37000,38000,39000,40000,
85000])
	
		flux_Cassiopeia_A = np.array([7609.,6145,6055,5206,
4546,4054,3852,3671,3364,3110,2898,2716,2560,2422,2410,2397,
2301,2194,2097,2010,1931,1858,1575,1489,1375,1226,1110,1017,
941,941,876,822,774,754,695,644,600,560,526,495,467,457,442,
420,399,380,363,347,333,319,307,298,295,284,274,265,256,247,
239,232,227,225,218,206,195,185,176,168,160,153,147,141,135,
130,128,125,121,117,113,109,106,102,99,96,40])
	
		flux_Cygnus_A = np.array([6375.,4948,4862,4065,3462,
3022,2844,2687,2422,2207,2029,1880,1752,1641,1631,1621,1544,
1459,1355,1266,1187,1116,856,783,689,574,489,425,376,375,335,
302,275,251,232,214,200,186,175,164,155,152,147,139,132,126,
120,115,110,106,102,99,98,94,91,87,84,82,79,77,75,74,72,68,
64,61,58,55,53,50,48,46,44,43,42,41,40,38,37,36,35,34,33,
32,13])
	
		flux_Crab_Nebula = np.array([1311,1221,1215,1156,1105,
1064,1046,1029,1000,974,951,931,913,896,895,893,881,867,854,
842,831,821,777,762,742,715,692,672,655,654,639,626,613,602,
592,583,574,566,558,551,545,542,539,533,527,522,517,512,507,
503,499,496,495,491,487,484,480,477,474,471,468,468,465,459,
454,449,444,440,436,431,428,424,420,417,416,414,411,408,405,
402,399,397,394,392,325.])
	
		flux_Virgo_A = np.array([736.,583,573,486,419,369,349,
331,301,276,256,238,223,210,209,208,199,188,179,171,164,157,
131,123,113,100,89,81,75,75,69,64,60,57,54,51,48,46,44,42,
41,40,39,38,37,35,34,33,32,31,30,30,30,29,28,28,27,26,26,
25,25,25,24,23,22,22,21,20,19,19,18,18,17,17,17,16,16,16,15,
15,15,14,14,14,7])

		self.regflux = {'CasA': flux_Cassiopeia_A, 
		                'CygA': flux_Cygnus_A, 
						'Crab': flux_Crab_Nebula, 
						'VirA': flux_Virgo_A}





	def RADec( self, sourcename ) : 
		''' Return the RADec (degree) of the source '''
		if (self.doinit): self.__init__()
		return self.regRADec[sourcename]





	def Region( self, sourcename, size, nside, coord, ordering='ring' ) : 
		'''
		size: 
			degree, size of region

		return region of the source in healpix_index
		'''
		if (self.doinit): self.__init__()
		from jizhipy.Transform import CoordTrans
		import numpy as np
		import healpy as hp
		nest=False if(str(ordering).lower()=='ring')else True
		size = float(size)*np.pi/180
		RA, Dec = self.RADec(sourcename)*np.pi/180
		RA, Dec = CoordTrans.Celestial(RA, Dec, 'Equatorial', coord)  # rad
		pix = np.arange(12*nside**2)
		theta, phi = hp.pix2ang(nside, pix, nest=nest)  # rad
		theta = np.pi/2 - theta
		dDec, dRA = size, size/np.cos(Dec)
		Decmax, Decmin = Dec+dDec*2, Dec-dDec*2
		RAmax,  RAmin  = RA+dRA*2,   RA-dRA*2
		tf = (Decmin<=theta)*(theta<=Decmax) * (RAmin<=phi)*(phi<=RAmax)
		theta, phi, pix = theta[tf], phi[tf], pix[tf]
		phi = (phi - RA) * np.cos(theta)
		theta = theta - Dec
		tf = (theta**2 + phi**2) < size**2
		pix = pix[tf]
		return pix





	def Convert( self, angle, what ) : 
		'''
		what: 'h2d' => hour     to  degree
		      'h2s' => hour     to  hms_str
		      'd2h' => degree   to  hour
            'd2s' => degree   to  hms_str
		      's2h' => hms_str  to  hour
            's2d' => hms_str  to  degree
		'''
		if (self.doinit): self.__init__()
		from jizhipy.Basic import IsType
		def h2s( ang ) : 
			h = int(ang[i])
			m = int((ang[i]-h)*60)
			s = ((ang[i]-h)*60-m)*60
			hms = str(h)+'h'+str(m)+'m'+('%.2f'%s)+'s'
			return hms
		def s2h( ang ) : 
			ang = str(ang)
			nh, nm = ang.find('h'), ang.find('m')
			h = float(ang[:nh])
			m = float(ang[nh+1:nm])
			s = float(ang[nm+1:-1])
			hms = h + m/60 + s/3600
			return hms
		#--------------------------------------------------
		if (IsType.isint(angle) or IsType.isfloat(angle)) : 
			ang, islist = [angle], False
		else : ang, islist = angle, True
		ret = []
		for i in range(len(ang)) : 
			if   (what == 'h2d') : ret.append(ang[i]*15)
			elif (what == 'h2s') : ret.append(h2s(ang[i]))
			elif (what == 'd2h') : ret.append(ang[i]/15.)
			elif (what == 'd2s') : ret.append(h2s(ang[i]/15.))
			elif (what == 's2h') : ret.append(s2h(ang[i]))
			elif (what == 's2d') : ret.append(s2h(ang[i])*15)
		return ret





	def FluxDensity( self, sourcename, frequency=None ) : 
		'''
		sourcename: 
			str, one sourcename or list
			Must be the same as self.regname.keys()
	
		frequency: 
			in MHz, scale or list or ndarray
			if ==None: return measured flux

		sourcename and frequency can broadcast
	
		Return: 
			flux density
			Nrow == sourcename.size
			Ncol == frequency.size
			** On sourcename, 1D

		Data from:
			http://www.gb.nrao.edu/electronics/edir/
			http://www.gb.nrao.edu/electronics/edir/edir35.pdf
			Electronics Division Internal Report No.35
			"The flux density values of standard sources used for antenna calibrations"
				J.W.M. Baars, P.G. Mezger and H. Wendker
				1964

		@ARTICLE{1964calibrator,
		   author = {J.W.M. Baars, P.G. Mezger and H. Wendker},
		    title = "{The flux density values of standard sources used for antenna calibrations}",
		  journal = {NCRA Technical ReportElectronics Division Internal Report No.35},
		 keywords = {calibrat},
		     year = 1964,
		    month = Aug,
		   adsurl = {http://www.gb.nrao.edu/electronics/edir/, http://www.gb.nrao.edu/electronics/edir/edir35.pdf}
		}
		'''
		if (self.doinit): self.__init__()
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import IsType
		from jizhipy.Optimize import Interp1d
		# freq = 300MHz ~ 85GHz

		onesource = False
		if (IsType.isstr(sourcename)) : 
			sourcename = [sourcename]
			onesource = True
		freqbcast = np.zeros(len(sourcename))

		freq1d = False
		onefreq = True if(IsType.isnum(frequency))else False
		if (frequency is None) : frequency = self.regfreq
		frequency = npfmt(frequency)
		if (len(frequency.shape) == 1) : 
			if (frequency.size == freqbcast.size) : 
				frequency = frequency[:,None]
				freq1d = True
			else: frequency=freqbcast[:,None]+frequency[None,:]
		elif (len(frequency.shape) == 2) : 
			frequency = freqbcast[:,None] + frequency

		reflux = []
		for i in range(len(sourcename)) : 
			flux = self.regflux[sourcename[i]]
			# power-law
			reflux.append( 10.**Interp1d( np.log10(self.regfreq), np.log10(flux), np.log10(frequency[i]), 'linear' ))
		reflux = np.array(reflux)
		if (onesource and not onefreq) : reflux = reflux[0]
		elif (not onesource and onefreq): reflux = reflux[:,0]
		elif (onesource and onefreq): reflux = reflux[0,0]
		if (len(reflux.shape)==2 and reflux.shape[1]==1 and freq1d) : reflux = reflux[:,0]
		return reflux





BrightSource = BrightSource()
