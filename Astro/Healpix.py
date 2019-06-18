
class HealpixMap( object ) :


	def __init__( self, inputwhat=None, which=(1,0), ordering=None, coordsys=None, unit=None, nside=None, freq=None, dtype=None, Nprocess=None, verbose=False ) : 
		'''
		Input map must be healpix map

		inputwhat, which:
			(1) Path of healpix map with ".fits": 
				which = (hdu, field)

			(2) Path of healpix map with ".hdf5": 
				hpmap = h5py.open(inputwhat)[which]
				which is the key (str)

			(3) np.ndarray with healpix pixelization (size=12*2**n)
				which is invalid here

		ordering:
			'RING' | 'NESTED'
			the output ordering of self.hpmap

		coordsys:
			'Equatorial' | 'Galactic'
			the output coordsys of self.hpmap

		unit:
			'K' | 'mK' | 'uK'
			the output unit of self.hpmap
		
		nside:
			the output nside of self.hpmap

		dtype:
			'float32' | 'float64' | np.float32 | np.float64
			the output self.hpmap.dtype
		'''
		import os, pyfits
		import healpy as hp
		import numpy as np
		from jizhipy.Process import NprocessCPU
		from jizhipy.Basic import IsType
		self.verbose = bool(verbose)
		if (self.verbose) : print('HealpixMap.__init__:')
		self.Nprocess = NprocessCPU(Nprocess)[0]
		self.ordering, self.nest = self._OrderingCheck(None, ordering)[2:]
		self.coordsys = self._CoordsysCheck(None, coordsys)[1]
		self.nside = self._NsideCheck(nside)
		self.unit = self._UnitCheck(None, unit)[1]
		if (dtype is not None) : 
			try    : dtype = np.dtype(dtype)
			except : dtype = None
		self.dtype = dtype
		self.freq = freq
		if (self.verbose) : 
			try : freqstr = ('%.1fMHz' % self.freq)
			except : freqstr = str(self.freq)
			print("    ordering="+str(self.ordering)+"(nest="+str(self.nest)+"), coordsys="+str(self.coordsys)+", nside="+str(self.nside)+", unit="+str(self.unit)+", dtype="+str(self.dtype)+', freq='+freqstr)
		#--------------------------------------------------

		if (not IsType.isstr(inputwhat)) : 
			self.hpmap   = np.array(inputwhat)
			self.nsidein = hp.get_nside(self.hpmap)
			self.dtypein = self.hpmap.dtype
			self.hpmap   = np.array(self.hpmap, self.dtype)
			self.dtype   = self.hpmap.dtype
			self.inputwhat  = None
			self.orderingin = None
			self.coordsysin = None
			self.unitin     = None
			self.freqin     = None
			printstrNone = ['orderingin', 'coordsysin', 'unitin', 'freqin']
			if (self.verbose) : 
				print('    inputwhat is np.ndarray')
				print('    orderingin = coordsysin = unitin = freqin = None. You can use HealpixMap.InProperty() to set them')
		#--------------------------------------------------

		elif ('.fits' == str(inputwhat).lower()[-5:]) : 
			printstrNone = []
			self.inputwhat = os.path.abspath(os.path.expanduser(inputwhat))
			fo = pyfits.open(self.inputwhat)
			hdr = fo[0].header
			for i in range(1, which[0]+1): hdr.update(fo[i].header)
			fo = fo[which[0]]
			self.dtypein=np.dtype(fo.data.dtype[which[1]].name)
			self.hpmap = np.array(fo.data[fo.data.names[which[1]]], self.dtype)  # Read data directly
			self.dtype = self.hpmap.dtype
			self.nsidein = hp.get_nside(self.hpmap)
			#--------------------------------------------------
			try : 
				orderingin = str(hdr['ORDERING']).upper()
				orderingin,nestin=self._OrderingCheck(orderingin)[:2]
			except : 
				printstrNone.append('orderingin')
				orderingin, nestin = None, None
			self.orderingin, self.nestin = orderingin, nestin
			#--------------------------------------------------
			try : 
				try : coordsysin = str(hdr['COORDSYS']).lower()
				except : coordsysin = str(hdr['SKYCOORD']).lower()
				coordsysin = self._CoordsysCheck(coordsysin)[0]
			except : 
				printstrNone.append('coordsysin')
				coordsysin = None
			self.coordsysin = coordsysin
			#--------------------------------------------------
			unitin = 'None'
			try : 
				unitin = str(hdr['TUNIT'+str(which[1]+1)])
				unitin = self._UnitCheck(unitin)[0]
			except : 
				if (self.verbose) : print("    Warning: unitin = "+unitin+" not in ['K', 'mK', 'uK']")
				printstrNone.append('unitin')
				unitin = None
			self.unitin = unitin
			#--------------------------------------------------
			printstr = ''
			try : 
				n = hdr.keys().index('FREQ')
				freqin, com = str(hdr['FREQ']).lower(), hdr.comments[n].lower()
				if   ('ghz' in freqin or 'ghz' in com) : freqin = 1e3 * float(freqin.split('ghz')[0])
				elif ('mhz' in freqin or 'mhz' in com) : freqin = float(freqin.split('mhz')[0])
				elif ( 'hz' in freqin or  'hz' in com) : freqin = float(freqin.split('hz')[0])
				else : printstr = 'FREQ : '+freqin+' / '+com+", NOT with unit ['Hz', 'MHz', 'GHz']"
			except : 
				printstrNone.append('freqin')
				freqin = None
			self.freqin = freqin
			if (self.verbose) : 
				try : freqstrin = ('%.1fMHz' % self.freqin)
				except : freqstrin = str(self.freqin)
				print('    inputwhat="'+inputwhat+'"')
				print("    orderingin="+str(self.orderingin)+"(nestin="+str(self.nestin)+"), coordsysin="+str(self.coordsysin)+", nsidein="+str(self.nsidein)+", unitin="+str(self.unitin)+", dtypein="+str(self.dtypein)+', freqin='+freqstrin)
		#--------------------------------------------------

		# Do
		if (self.orderingin is not None) : 
			# coordsys
			if (self.coordsysin is None and self.coordsys is not None) : 
				if (self.verbose) : print("    Warning: coordsysin = None, coordsys = '"+self.coordsys+"', can NOT change")
				self.coordsys = None
			elif (self.coordsysin is None and self.coordsys is None) : pass
			elif (self.coordsysin is not None and self.coordsys is None) : self.coordsys = self.coordsysin
			elif (self.coordsysin != self.coordsys) : self.CoordsysConvert() 
			#--------------------------------------------------

			# ordering
			if (self.ordering is None) : self.ordering, self.nest = self.orderingin, self.nestin
			if (self.orderingin != self.ordering) : self.OrderingConvert()
			#--------------------------------------------------

			# nside
			if (self.nside is None) : self.nside = self.nsidein
			if (self.nsidein != self.nside): self.NsideConvert(None, self.ordering)
		#--------------------------------------------------

		try : 
			self.unit+self.unitin
			self.UnitConvert()
		except : pass

		try : 
			self.freq+self.freqin
			self.FreqConvert()
		except : pass





	def _CoordsysCheck( self, coordsysin=None, coordsysout=None ): 
		''' '''
		from jizhipy.Basic import Raise
		doraise = []
		if (coordsysin is None) : 
			try : coordsysin = self.coordsysin
			except : pass
		else : 
			coordsysin = str(coordsysin)
			if (coordsysin.lower() not in ['galactic','equatorial']) : doraise.append('in')
			else : coordsysin = coordsysin.lower()
		#--------------------------------------------------
		if (coordsysout is None) : 
			try : coordsysout = self.coordsys
			except : pass
		else : 
			coordsysout = str(coordsysout)
			if (coordsysout.lower() not in ['galactic','equatorial']) : doraise.append('out')
			else : coordsysout = coordsysout.lower()
		#--------------------------------------------------
		printstr = ''
		if ('in' in doraise) : printstr += "coordsysin = '"+coordsysin+"', "
		if ('out' in doraise) : printstr += "coordsysout = '"+coordsysout+"', "
		if (printstr != '') : Raise(Exception, printstr[:-2]+" not in ['galactic', 'equatorial']")
		else : return [coordsysin, coordsysout]





	def _OrderingCheck( self, orderingin=None, orderingout=None):
		''' '''
		from jizhipy.Basic import Raise
		doraise = []
		if (orderingin is None) : 
			try : orderingin = self.orderingin
			except : pass
		else : 
			orderingin = str(orderingin)
			if (orderingin.upper() not in ['RING', 'NESTED', 'NEST']) : doraise.append('in')
			else : 
				orderingin = orderingin.upper()
				if ('NEST' in orderingin) : orderingin = 'NESTED'
		#--------------------------------------------------
		if (orderingout is None) : 
			try : orderingout = self.ordering
			except : pass
		else : 
			orderingout = str(orderingout)
			if (orderingout.upper() not in ['RING', 'NESTED', 'NEST']) : doraise.append('out')
			else : 
				orderingout = orderingout.upper()
				if ('NEST' in orderingout) : orderingout = 'NESTED'
		#--------------------------------------------------
		printstr = ''
		if ('in' in doraise) : printstr += "orderingin = '"+orderingin+"', "
		if ('out' in doraise) : printstr += "orderingout = '"+orderingout+"', "
		if (printstr != '') : Raise(Exception, printstr[:-2]+" not in ['RING', 'NESTED']")
		#--------------------------------------------------
		if (orderingin is None) : nestin = None
		elif (orderingin == 'RING') : nestin = False
		else : nestin = True
		if (orderingout is None) : nestout = None
		elif (orderingout == 'RING') : nestout = False
		else : nestout = True
		return [orderingin, nestin, orderingout, nestout]





	def _NsideCheck( self, nsideout=None ) : 
		if (nsideout is None) : 
			try : nsideout = self.nside
			except : pass
		else : 
			n = np.log(nsideout) / np.log(2)
			reset = True if(n != int(n))else False
			if (reset and self.verbose) : print('   Warning: nsideout = 2**'+('%.3f' % n)+' != 2**int, ',)
			n = int(round(n))
			nsideout = 2**n
			if (reset and self.verbose) : print('reset nsideout = 2**'+str(n)+' = '+str(nsideout))
		return nsideout





	def _UnitCheck( self, unitin=None, unitout=None ) : 
		''' return [unitin, unitout, k_in2out] '''
		from jizhipy.Basic import Raise
		doraise = []
		if (unitin is None) : 
			try : unitin = self.unitin
			except : pass
		else : 
			unitin = str(unitin)
			if   (unitin.lower() ==  'k') : unitin =  'K'
			elif (unitin.lower() == 'mk') : unitin = 'mK'
			elif (unitin.lower() == 'uk') : unitin = 'uK'
			else : doraise.append('in')
		#--------------------------------------------------
		if (unitout is None) : 
			try : unitout = self.unit
			except : pass
		else : 
			unitout = str(unitout)
			if   (unitout.lower() ==  'k') : unitout =  'K'
			elif (unitout.lower() == 'mk') : unitout = 'mK'
			elif (unitout.lower() == 'uk') : unitout = 'uK'
			else : doraise.append('out')
		#--------------------------------------------------
		printstr = ''
		if ('in' in doraise): printstr+="unitin = '"+unitinin+"', "
		if('out' in doraise): printstr+="unitout = '"+unitout+"', "
		if (printstr != '') : Raise(Exception, printstr[:-2]+" not in ['K', 'mK', 'uK']")
		#--------------------------------------------------
		if (unitin is None or unitout is None): k_in2out = 1
		elif (unitin== 'K' and unitout== 'K') : k_in2out = 1
		elif (unitin== 'K' and unitout=='mK') : k_in2out = 1e3
		elif (unitin== 'K' and unitout=='uK') : k_in2out = 1e6
		elif (unitin=='mK' and unitout== 'K') : k_in2out = 1e-3
		elif (unitin=='mK' and unitout=='mK') : k_in2out = 1
		elif (unitin=='mK' and unitout=='uK') : k_in2out = 1e3
		elif (unitin=='uK' and unitout== 'K') : k_in2out = 1e-6
		elif (unitin=='uK' and unitout=='mK') : k_in2out = 1e-3
		elif (unitin=='uK' and unitout=='uK') : k_in2out = 1
		return [unitin, unitout, k_in2out]





	def CoordsysConvert( self, coordsysin=None, coordsysout=None, orderingin=None ) : 
		''' '''
		from jizhipy.Transform import CoordTrans
		coordsysin, coordsysout = self._CoordsysCheck(coordsysin, coordsysout)
		if (coordsysin == coordsysout) : return
		orderingin = self._OrderingCheck(orderingin)[0]
		if (self.verbose) : 
			print('HealpixMap.CoordsysConvert:')
			print("    coordsysin='"+coordsysin+"', coordsysout='"+coordsysout+"', ordering='"+orderingin+"'")
		CoordTrans.Nprocess = self.Nprocess
		self.hpmap = CoordTrans.CelestialHealpix(self.hpmap, orderingin, coordsysin, coordsysout)[1]




	def OrderingConvert( self, orderingin=None, orderingout=None):
		''' '''
		import healpy as hp
		orderingin, nestin, orderingout = self._OrderingCheck(orderingin, orderingout)[:3]
		if (orderingin == orderingout) : return
		if (self.verbose) : 
			print('HealpixMap.OrderingConvert:')
			print("    orderingin='"+orderingin+"', orderingout='"+orderingout+"'")
		self.hpmap = hp.ud_grade(self.hpmap, self.nsidein, order_in=orderingin, order_out=orderingout)





	def NsideConvert( self, nsideout=None, orderingin=None ) :
		''' '''
		import healpy as hp
		nsideout = self._NsideCheck(nsideout)
		if (nsideout == hp.get_nside(self.hpmap)) : return
		orderingin = self._OrderingCheck(orderingin)[0]
		if (self.verbose) : 
			print('HealpixMap.NsideConvert:')
			print("    nsidein="+str(hp.get_nside(self.hpmap))+", nsideout="+str(nsideout)+", ordering='"+orderingin+"'")
		self.hpmap = hp.ud_grade(self.hpmap, nsideout, order_in=orderingin)





	def UnitConvert( self, unitin=None, unitout=None ) : 
		''' '''
		unitin,unitout, k_in2out = self._UnitCheck(unitin,unitout)
		if (unitin == unitout) : return
		if (self.verbose) : 
			print('HealpixMap.UnitConvert:')
			print("    unitin='"+unitin+"', unitout='"+unitout+"'")
		self.hpmap = self.hpmap * k_in2out





	def FreqConvert( self, freqin=None, freqout=None ) : 
		if (freqin is None) : freqin = self.freqin
		if (freqout is None) : freqout = self.freq
		try : 
			freqin+freqout
			if (freqin == freqout) : raise
			if (self.verbose) : 
				print('HealpixMap.FreqConvert:')
				print('    freqin=%.3f MHz, freqout=%.3f MHz' % (freqin, freqout))
			self.hpmap *= (1.*freqin/freqout)**2.8
		except : pass





	def InProperty( self, orderingin=None, coordsysin=None, unitin=None, freqin=None ) : 
		if (orderingin is not None) : self.orderingin, self.nestin = self._OrderingCheck(orderingin, None)[:2]
		if (coordsysin is not None) : self.coordsysin= self._CoordsysCheck(coordsysin,None)[0]
		if (unitin is not None) : self.unitin = self._UnitCheck(unitin, None)[0]
		if (freqin is not None) : self.freqin = freqin
		if (self.verbose) : 
			try : freqstrin = ('%.1fMHz' % self.freqin)
			except : freqstrin = str(self.freqin)
			print('HealpixMap.InProperty:')
			print("    orderingin="+str(self.orderingin)+"(nestin="+str(self.nestin)+"), coordsysin="+str(self.coordsysin)+", unitin="+str(self.unitin)+', freqin='+freqstrin)
		if (orderingin is not None) : 
			try : 
				self.orderingin+self.ordering
				self.OrderingConvert()
			except : pass
			try : 
				self.nsidein+self.nside
				self.NsideConvert(None, self.ordering)
			except : pass
		if (coordsysin is not None) : 
			try : 
				self.coordsysin+self.coordsys
				self.CoordsysConvert()
			except : pass
		if (unitin is not None) : 
			try : 
				self.unitin+self.unit
				self.UnitConvert()
			except : pass
		if (freqin is not None) : 
			try : 
				self.freqin+self.freq
				self.FreqConvert()
			except : pass




	def SkyRegion( self, lon=None, lat=None, fwhm=None ) : 
		'''
		fwhm:
			[degree]
			fwhm==None: don't smooth the map

		lon, lat:
			[degree]
			lon: RA  | l
			lat: Dec | b
			Must can be broadcast
		'''
		import healpy as hp
		from jizhipy.Basic import Raise
		if (lon is not None and lat is not None) : 
			if (np.sum((lat<-90)+(lat>90)) > 0) : Raise(Exception, 'lat out of [-90, 90] degree')
		self.lonsky, self.latsky = lon, lat
		if (fwhm not in [0, None]) : self.hpmap = np.array(hp.smoothing(self.hpmap, fwhm*np.pi/180, verbose=False), self.hpmap.dtype)
		nside = hp.get_nside(self.hpmap)
		if (lon is not None and lat is not None) : 
			islist = False if(IsType.isnum(lon+lat))else True
			lon, lat = lon+lat*0, lon*0+lat  # same shape
			lon %= 360
			npix = hp.ang2pix(nside, np.pi/2-lat*np.pi/180, lon*np.pi/180)
			self.sky = self.hpmap[npix]
			if (not islist) : self.sky = self.sky[0]





class Healpix( object ) : 


	def HealpixHeader( self, nside, ordering, coordsys, freq=None, unit=None, epoch=None, beamsize=None ) : 
		'''
		nside: 2**n
		ordering: 'RING', 'NESTED'
		coordsys: 'EQUATORIAL', 'GALACTIC'
		freq: in MHz
		unit: Unit of the healpix map pixel value
		beamsize: Observation FWHM of the healpix map in arcmin
	
		return: [key, value, comment]
		'''
		from jizhipy.Basic import Time
		key=['PIXTYPE', 'DATECREA', 'ORDERING', 'NSIDE', 'COORDSYS']
		value = ['HEALPIX', Time(1), ordering.upper(), nside, coordsys.upper()]
		comment = ['HEALPIX pixelisation', 'Creation date of this file', 'Pixel ordering scheme, RING or NESTED', 'Healpix resolution parameter', 'Coordinate system, EQUATORIAL or GALACTIC']
		if (freq is not None) : 
			key, value, comment = key+['FREQ'], value+[freq], comment+['MHz']
		if (unit is not None) : 
			key, value, comment = key+['UNIT'], value+[unit], comment+["Unit of the healpix map's values"]
		if (epoch is not None) : 
			key, value, comment = key+['EPOCH'], value+[str(epoch)], comment+['Epoch of coordinate system']
		if (beamsize is not None) : 
			key, value, comment = key+['BEAMSIZE'], value+[beamsize], comment+['arcmin']
		return [key, value, comment]





	def Smoothing( self, hpmap, fwhm, verbose=False ) : 
		'''
		hpmap:
			np.array(), with or without np.nan/np.inf
			np.ma.MaskedArray()
		'''
		from jizhipy.Array import Invalid
		import healpy as hp
		import numpy as np
		hpmap = Invalid(hpmap, True)
		hpmap, mask = hpmap.data, hpmap.mask
		hpmap = hp.smoothing(hpmap, fwhm, verbose=verbose)
		hpmap[mask] = np.nan
		return hpmap


Healpix = Healpix()
