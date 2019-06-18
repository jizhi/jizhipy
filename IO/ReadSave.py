
def Hdf52Fits( inputwhat, hduname='', verbose=False ) : 
	'''
	inputwhat:
		(1) str, path of .hdf5
		(2) one of the dataset: inputwhat = fo[keys[i]]
	'''
	from jizhipy.Basic import Path, IsType
	import numpy as np
	import h5py, pyfits
	def _Hdf52Fits( dataset, hduname='' ) : 
		if (hduname in ['', None]) : 
			name = str(dataset.name)
			name = name[name.rfind('/')+1:]
		else : name = str(hduname)
		attrs = dataset.attrs.items()
		data  = dataset.value
		hdulist = []
		try : 
			if (data.dtype.name[:7] == 'complex') : 
				hdu = pyfits.ImageHDU( data.real )
				hdu.name = name+'REAL'
				hdr = hdu.header
				for i in range(len(attrs)) : 
					hdr[str(attrs[i][0])] = attrs[i][1]
				hdulist.append(hdu)  #@#@
				hdu = pyfits.ImageHDU( data.imag )
				hdu.name = name+'IMAG'
				hdr = hdu.header
				for i in range(len(attrs)) : 
					hdr[str(attrs[i][0])] = attrs[i][1]
				hdulist.append(hdu)  #@#@
			else : raise
		except : 
			if (IsType.isndarray(data) or IsType.islist(data) or IsType.istuple(data)) : 
				try : hdu = pyfits.ImageHDU( data )
				except : hdu = pyfits.ImageHDU( np.array(data, np.float32) )
				hdu.name = name
				hdr = hdu.header
				for i in range(len(attrs)) : 
					hdr[str(attrs[i][0])[:8]] = attrs[i][1]
				hdulist.append(hdu)
			else : other.header[name[:8]] = data
		return hdulist
	#--------------------------------------------------
	hdulist = pyfits.HDUList()
	other = pyfits.ImageHDU([])
	other.name = 'other'
	hdrother = other.header.keys()
	if (IsType.isstr(inputwhat)) : 
		inname = Path.AbsPath(inputwhat)
		outname = inname[:-5] + '.fits'
		if (verbose) : print('Saving  "'+outname+'"')
		fo = h5py.File(inname, 'r')
		keys = fo.keys()
		for i in range(len(keys)) : 
			hdu = _Hdf52Fits( fo[keys[i]], hduname )
			for j in range(len(hdu)) : hdulist.append( hdu[j] )
	#--------------------------------------------------
	else : 
		if (hduname in ['', None]) : 
			name = str(inputwhat.name)
			name = name[name.rfind('/')+1:]
		else : name = str(hduname)
		outname = name + '.fits'
		if (verbose) : print('Saving  "'+outname+'"')
		hdu = _Hdf52Fits( inputwhat, hduname )
		for j in range(len(hdu)) : hdulist.append( hdu[j] )
	#--------------------------------------------------
	hdr = other.header.keys()
	if (hdr != hdrother) : hdulist.append(other)
	Path.ExistsPath(outname, 'True')
	hdulist.writeto( outname )
	hdulist.close()





def Fits2Hdf5( inputwhat, hduname=None, verbose=False ) : 
	'''
	inputwhat:
		(1) str, path of .fits
		(2) one of the HDU: inputwhat = fo[i]

	hduname:
		if exists hdu.header['extname'] or inputwhat.name, use them, otherwise, use hduname+str(i)
	'''
	from jizhipy.Basic import Path, IsType
	import h5py, pyfits
	def _Fits2Hdf5( hname, hdu, outhdf5 ) : 
		try : name = hdu.header['extname']
		except : name = hdu.name
		if (name == '') : 
			if (hduname in ['', None]) : name = str(hname)
			else : name = str(hduname)+str(hname)[3:]
		outhdf5[name] = hdu.data  #@#@
		hdr = hdu.header
		keys, values, comments = hdr.keys(), hdr.values(), hdr.comments
		for i in range(len(keys)) : 
			outhdf5[name].attrs[keys[i]] = values[i]
			if (comments[i] != '') : 
				outhdf5[name].attrs[keys[i]+'_comment'] = comments[i]
	#--------------------------------------------------
	if (IsType.isstr(inputwhat)) : 
		inname = Path.AbsPath(inputwhat)
		outname = inname[:-5] + '.hdf5'
		if (verbose) : print('Saving  "'+outname+'"')
		Path.ExistsPath(outname, 'True')
		h = h5py.File(outname, 'w')
		f = pyfits.open(inname)
		for i in range(len(f.info(0))) : 
			_Fits2Hdf5( 'hdu'+str(i), f[i], h )
	#--------------------------------------------------
	else : 
		if (hduname not in ['', None]) : name = str(hduname)
		else : 
			name = inputwhat.name
			if (name == '') : name = 'hdu'
		outname = name + '.hdf5'
		if (verbose) : print('Saving  "'+outname+'"')
		Path.ExistsPath(outname, 'True')
		h = h5py.File(outname, 'w')
		_Fits2Hdf5( hduname, inputwhat, h )
	#--------------------------------------------------
	h.close()





class ClassHdf5( object ) : 

	def __init__( self, classinstance, outname, verbose=True ):
		'''
		classinstance:
			Instance of class: a = A()

		outname:
			end with '.hdf5'

		Usave:
			a = A()
			a.set(1)
			classhdf5 = jp.ClassHdf5(a, 'a.hdf5')
			classhdf5.Save()
			
			b = A()
			classhdf5 = jp.ClassHdf5(b, 'a.hdf5')
			classhdf5.Read()
		'''
		self.classinstance = classinstance
		self.outname = outname
		self.verbose = bool(verbose)
		self.keys, self.value = None, None



	def Keys( self ) : 
		'''
		return:
			all keys in a
		'''
		from jizhipy.Basic import IsType
		keys = self.classinstance.__dict__.keys()
		dowhile, times = True, 0
		while(dowhile) : 
			dowhile, times = False, times+1
			for i in range(len(keys)) : 
				k = keys[i].split('.')
				v = self.classinstance
				for j in range(len(k)) : v = v.__dict__[k[j]]
				if (IsType.isfunc(v) or IsType.isclass(v)) : 
					keys[i] = ''
				elif (IsType.isinstance(v) and not IsType.isspecialinstance(v)) : 
					k = v.__dict__.keys()
					for j in range(len(k)) : k[j] = keys[i] + '.'+k[j]
					keys[i] = k
					dowhile = True
			keytmp = []
			for i in range(len(keys)) : 
				if (keys[i] == '') : continue
				elif (type(keys[i])==list): keytmp += keys[i]
				else : keytmp.append(keys[i])
			keys = keytmp
		self.keys = keys



	def Values( self ) : 
		'''
		return:
			All values corresponds to keys
		'''
		if (self.keys is None) : self.Keys()
		self.values = []
		for i in range(len(self.keys)) : 
			k = self.keys[i].split('.')
			v = self.classinstance
			for j in range(len(k)) : 
				v = v.__dict__[k[j]]
			self.values.append( v )



	def _OperaList( self, inlist ) : 
		'''
		(1) None in inlist, replaced by '__None_'
		(2) '__None_' in inlist, replaced by None
		'''
		tf = (type(inlist) is tuple)
		inlist = list(inlist)
		for i in range(len(inlist)) : 
			if (inlist[i] is None) : inlist[i] = -12321
			elif (type(inlist[i]) is int and inlist[i]==-12321) : inlist[i] = None 
			elif (type(inlist[i]) in [tuple, list]) : 
				inlist[i] = self._OperaList(inlist[i])
			elif (type(inlist[i]) is dict) : 
				self._OperaDict(inlist[i])
		if (tf) : inlist = tuple(inlist)
		return inlist



	def _OperaDict( self, indict ) : 
		'''
		(1) None in indict, replaced by '__None_'
		(2) '__None_' in indict, replaced by None
		'''
		key, value = indict.keys(), indict.values()
		for i in range(len(key)) : 
			if (value[i] is None) : indict[key[i]] = -12321
			elif (type(value[i]) is int and value[i]==-12321) : indict[key[i]] = None
			elif (type(value[i]) in [list, tuple]) : 
				indict[key[i]] = self._OperaList(value[i])
			elif (type(value[i]) is dict) : 
				self._OperaDict(indict[key[i]])
		


	def Save( self ) : 
		'''
		Save instance to .hdf5
		'''
		from jizhipy.Basic import Path, IsType, Raise
		import numpy as np
		import h5py
		self.outname = str(self.outname)
		if (self.outname[-5:] !='.hdf5') : self.outname += '.hdf5'
		Path.ExistsPath(self.outname, old=True)
		if (self.verbose) : print('jizhipy.ClassHdf5: Saveing to  "'+self.outname+'"')
		fo = h5py.File(self.outname, 'w')
		if (self.value is None or self.keys is None): self.Values()
		fo['classinstance'] = str(type(self.classinstance))
		for i in range(len(self.keys)) : 
			if (self.values[i] is None) : 
				fo[self.keys[i]] = '__None__'
				fo[self.keys[i]].attrs['type'] = 'is None, not str'
			#--------------------------------------------------
			elif (IsType.isdtype(self.values[i])) : 
				fo[self.keys[i]]=np.dtype(self.values[i]).name
				fo[self.keys[i]].attrs['type'] = 'numpy.dtype'
			#--------------------------------------------------
			elif (IsType.ismatrix(self.values[i])) : 
				fo[self.keys[i]] = np.array(self.values[i])
				fo[self.keys[i]].attrs['type'] = 'numpy.matrix'
			#--------------------------------------------------
			elif (IsType.ismaskedarray(self.values[i])) : 
				fo[self.keys[i]] = self.values[i].data
				fo[self.keys[i]+'.mask'] = self.values[i].mask
				fo[self.keys[i]].attrs['type'] = 'numpy.MaskedArray .data'
				fo[self.keys[i]+'.mask'].attrs['type'] = 'numpy.MaskedArray .mask'
			#--------------------------------------------------
			elif (IsType.isdict(self.values[i])) : 
				self._OperaDict(self.values[i])
				try : fo[self.keys[i]] =self.values[i].values() 
				except : fo[self.keys[i]] = str(self.values[i].values())
				fo[self.keys[i]+'.keys'] = self.values[i].keys()
				fo[self.keys[i]].attrs['type']='dict .values()'
				fo[self.keys[i]+'.keys'].attrs['type']='dict .keys()'
				self._OperaDict(self.values[i])
			#--------------------------------------------------
			else : 
				# inside list and tuple, must not contain dict
				try : fo[self.keys[i]] = self.values[i]  # convert to np.ndarray with same dtype, if contains string, will conver to string array
				except : 
					strv = str([self.values[i]])[1:-1]
					Raise(Exception, "fo['"+self.keys[i]+"'] = "+strv)
				if   (IsType.islist(self.values[i])) : 
					fo[self.keys[i]].attrs['type'] = 'list'
				elif (IsType.istuple(self.values[i])) : 
					fo[self.keys[i]].attrs['type'] = 'tuple'
				else : fo[self.keys[i]].attrs['type'] = 'numpy.array'
			#--------------------------------------------------
		fo.flush()
		fo.close()



	def Read( self ) : 
		'''
		When read .hdf5 to a class:instance, it needs this class:instance already has all necessary sub-instance !
		'''
		import numpy as np
		import h5py
		from jizhipy.Basic import Str, IsType
		self.outname = str(self.outname)
		if (self.outname[-5:] !='.hdf5') : self.outname += '.hdf5'
		if (self.verbose) : print('jizhipy.ClassHdf5: Reading from  "'+self.outname+'"')
		fo = h5py.File(self.outname, 'r')
		keys = fo.keys()
		for i in range(len(keys)) : 
			keys[i] = str(keys[i])
			if (keys[i]=='classinstance' or keys[i][-5:]=='.mask' or keys[i][-5:]=='.keys') : continue
			Type = fo[keys[i]].attrs['type']
			k = keys[i].split('.')
			v = self.classinstance
			for j in range(len(k)-1) : v = v.__dict__[k[j]]
			#--------------------------------------------------
			if ('is None' in Type) : 
				v.__dict__[k[-1]] = None
			#--------------------------------------------------
			elif ('numpy.dtype' in Type) : 
				v.__dict__[k[-1]] = np.dtype(fo[keys[i]].value)
			#--------------------------------------------------
			elif ('numpy.matrix' in Type) : 
				v.__dict__[k[-1]] = np.matrix(fo[keys[i]].value)
			#--------------------------------------------------
			elif ('numpy.MaskedArray' in Type) : 
				value = np.ma.MaskedArray(fo[keys[i]].value)
				value.mask = fo[keys[i]+'.mask'].value
				v.__dict__[k[-1]] = value 
			#--------------------------------------------------
			elif ('dict' in Type) : 
				fokey = fo[keys[i]+'.keys'].value
				fovalue = fo[keys[i]].value
				if (IsType.isstr(fovalue)) : 
					fovalue = fovalue[1:-1]
					#------------------------------
					def split( w, u ) :
						strw, fnew = [], ''
						n1 = Str.StrFind(u,w[0])
						if (len(n1) == 0) : 
							return [strw, u]
						n1=np.array(n1)[:,0]
						n2=np.array(Str.StrFind(u,w[1]))[:,0]
						for i in range(len(n1)) : 
							strw.append(u[n1[i]+1:n2[i]])
						n3 = np.array([np.append([0],n2), np.append(n1+1,[len(fovalue)])]).T
						for i in range(len(n3)) : 
							fnew += u[n3[i][0]:n3[i][1]]
						return [strw, fnew]
					#------------------------------
					def conv( u ) : 
						if ('__None__' in u or '-12321' in u):
							u = None
						elif ('True' in u) : u = True
						elif ('False' in u) : u = False
						else : 
							try : u = int(u)
							except : 
								try : u = float(u)
								except : 
									try : 
										n =Str.StrFind(u,"'")
										u =u[n[0][1]:n[1][0]]
									except : pass
						return u
					def convlist( strxxx, t='list' ) : 
						for i in range(len(strxxx)) : 
							s = strxxx[i].split(',')
							for j in range(len(s)) : 
								s[j] = conv(s[j])
							if (t == 'tuple') : s = tuple(s)
							strxxx[i] = s
					#------------------------------
					strlist, fovalue =split(['[',']'],fovalue)
					strtuple, fovalue=split(['(',')'],fovalue)
					strdict, fovalue =split(['{','}'],fovalue)
					fovalue = fovalue.split(',')
					#------------------------------
					for i in range(len(fovalue)) : 
						fovalue[i] = conv(fovalue[i])
					convlist(strlist, 'list')
					convlist(strtuple, 'tuple')
					for i in range(len(strdict)) : 
						s, d = strdict[i].split(','), {}
						for j in range(len(s)) : 
							s[j] = s[j].split(':')
							n = Str.StrFind(s[j][0], "'")
							m = s[j][0][n[0][1]:n[1][0]]
							d[m] = conv(s[j][1])
						strdict[i] = d
					#------------------------------
					n1, n2, n3 = 0, 0, 0
					for i in range(len(fovalue)) : 
						if (not IsType.isstr(fovalue[i])) : continue 
						if   ('[]' in fovalue[i]) : 
							fovalue[i] = strlist[n1]
							n1 += 1
						elif ('()' in fovalue[i]) : 
							fovalue[i] = strtuple[n2]
							n2 += 1
						elif ('{}' in fovalue[i]) : 
							fovalue[i] = strdict[n3]
							n3 += 1
				value = {}
				for j in range(len(fokey)) : 
					value[fokey[j]] = fovalue[j]
				v.__dict__[k[-1]] = value
			#--------------------------------------------------
			elif (Type == 'list') : 
				v.__dict__[k[-1]] = list(fo[keys[i]].value)
			#--------------------------------------------------
			elif (Type == 'tuple') : 
				v.__dict__[k[-1]] = tuple(fo[keys[i]].value)
			#--------------------------------------------------
			else : v.__dict__[k[-1]] = fo[keys[i]].value
		#--------------------------------------------------
		fo.close()
		return self.classinstance





def Array2FitsImage( arraylist, outname, names=None, keys=None, values=None, comments=None, verbose=False ) : 
	'''
	arraylist:
		(1) isndarray
		(2) list/tuple of ndarray: [array1, array2, ...]
		Otherwise, raise error

		Examples:
			Array2FitsImage( [1,2,3] ), means there are 3 arrays, array1=1, array2=2, array3=3
				FITS can save int/float number to ImageHDU, but will raise error "IOError: Header missing END card." when pyfits.open() it.
			You must convert to
					Array2FitsImage( np.array([1,2,3]), 'test.fits' )
			OR		Array2FitsImage(         [[1,2,3]], 'test.fits' )

	names:
		None | Same shape as arraylist
		Name of each ImageHDU in fo.info()

	keys:
		None | Same shape as arraylist
		(1) arraylist isndarray: keys=['key1', 'key2', ...], list of str
		(2) arraylist is list/tuple of ndarray: [array1, array2, ...], then keys=[arraykey1, arraykey2, ...], arraykey1=['key1', 'key2', ...] (list of list)

	values:
		Same shape as keys

	comments:
		None | same shape as keys even though is ''
	'''
	from jizhipy.Basic import Path, IsType
	import pyfits
	outname = Path.AbsPath(outname)
	if (outname[-5:].lower() != '.fits') : outname += '.fits'
	Path.ExistsPath(outname, True)
	if (verbose) : print('Saving  "'+outname+'"')
	#--------------------------------------------------
	if (IsType.isstr(keys)) :
		keys, values = [keys], [values]
		if (comments is not None) : comments = [comments]
		else : comments = ['']
	#--------------------------------------------------
	if (not IsType.islist(arraylist) and not IsType.istuple(arraylist)) : 
		arraylist = [arraylist]
		if (names is not None) : names = [names]
		keys = [keys for i in range(len(arraylist))]
		values = [values for i in range(len(arraylist))]
		comments = [comments for i in range(len(arraylist))]
	#--------------------------------------------------
	hdulist = pyfits.HDUList()
	for i in range(len(arraylist)) : 
		data = arraylist[i]
		try : name = str(names[i])
		except : name = 'hdu'+str(i)
		if (data.dtype.name[:7] == 'complex') : 
			hdu = pyfits.ImageHDU( data.real )
			hdu.name = name+'REAL'
			hdr = hdu.header
			for j in range(len(keys[i])) : 
				key, value = keys[i][j], values[i][j]
				try : comment = comments[i][j]
				except : comment = ''
				hdr.set(key, value, comment)
			hdulist.append(hdu)  # Real part
			hdu = pyfits.ImageHDU( data.imag )
			hdu.name = name+'IMAG'
			hdr = hdu.header
			try : 
				for j in range(len(keys[i])) : 
					key, value = keys[i][j], values[i][j]
					try : comment = comments[i][j]
					except : comment = ''
					hdr.set(key, value, comment)
			except : pass
			hdulist.append(hdu)  # Imag part
		#--------------------------------------------------
		else : 
			hdu = pyfits.ImageHDU( data )
			hdu.name = name
			hdr = hdu.header
			try : 
				for j in range(len(keys[i])) : 
					key, value = keys[i][j], values[i][j]
					try : comment = comments[i][j]
					except : comment = ''
					hdr.set(key, value, comment)
			except : pass
			hdulist.append(hdu)
	#--------------------------------------------------
	hdulist.writeto( outname )





def HealpixHeader( nside, ordering, coordsys, freq=None, unit=None, beamsize=None ) : 
	'''
	nside: 2**n
	ordering: 'RING', 'NESTED'
	coordsys: 'EQUATORIAL', 'GALACTIC'
	freq: in MHz
	unit: Unit of the healpix map pixel value
	beamsize: Observation FWHM of the healpix map in arcmin
	'''
	key = ['PIXTYPE', 'DATECREA', 'ORDERING', 'NSIDE', 'COORDSYS']
	value = ['HEALPIX', Time()[0], ordering.upper(), nside, coordsys.upper()]
	comment = ['HEALPIX pixelisation', 'Creation date of this FITS', 'Pixel ordering scheme, RING or NESTED', 'Healpix resolution parameter', 'Coordinate system, EQUATORIAL or GALACTIC']
	if (freq is not None) : 
		key, value, comment = key+['FREQ'], value+[freq], comment+['MHz']
	if (unit is not None) : 
		key, value, comment = key+['UNIT'], value+[unit], comment+['Unit of the healpix map pixel value']
	if (freq is not None) : 
		key, value, comment = key+['BEAMSIZE'], value+[beamsize], comment+['arcmin']
	return [key, value, comment]





def Ass2Txt( pos, path, fmt=None ) : 
	#'''
	#Read and write Chinese

	#pos: 
	#	str to judge what are the real texts
	#  For example:
	#	pos = [('0000,0000,0000,,', '\N{'), ('shad1}', '\n')]

	#path:
	#	Must be one
	#	If path is one file, handle it
	#	If path is directory, use fmt to judge which files will be handled

	#fmt:
	#	with or without '.' is OK: 'ass' | '.ass'
	#'''
	from jizhipy.Basic import ShellCmd, Path, IsType
	import chardet, codecs
	path = Path.AbsPath(path)
	if (IsType.isfile(path)) : path = [path]
	else : 
		indir = path
		if (fmt is None) : path = ShellCmd('ls '+indir)
		else : 
			fmt = str(fmt).lower()
			if (fmt[0] != '.') : fmt = '.' + fmt
			path = ShellCmd('ls '+indir+'*'+fmt)
			fmt = fmt.upper()
			path += ShellCmd('ls '+indir+'*'+fmt)
		for i in range(len(path)) : path[i] = indir+path[i]
	#---------------------------------------------
	outnamelist = []
	for i in range(len(path)) : 
		filename = path[i]
		try : a = open(filename).read()
		except : continue
		code = chardet.detect(a)['encoding']
		a = a.decode(code)
		txt = ''
		while ('\n' in a) : 
			n = a.index('\n')
			b = a[:n+1]
			for i in range(len(pos)) : 
				n1 = n2 = -1
				if (pos[i][0] in b) : 
					n1 = b.rfind(pos[i][0])+len(pos[i][0])
				if (pos[i][1] in b) : n2 = b.rfind(pos[i][1])
				if (n1<0 or n2<0) : break
				c = b[n1:n2]
				if (c[0]  == '\r') : c = c[1:]
				if (c[-1] == '\r') : c = c[:-1]
				txt += c + '\n'
			if (txt != '') : txt += '\n'
			a = a[n+1:]
		outname = filename + '.txt'
		b = codecs.open(outname, 'w', code)
		b.write(txt)
		b.close()
		outnamelist.append(outname)
	return outnamelist






#
#class A(object) : 
#	def __init__(self) : 
#		self.a1, self.a2 = None, (1,2)
#
#class B(object) : 
#	def __init__(self) : 
#		self.b1, self.b2, self.a = ['b1','b2'], np.arange(2), A()
#
#class C(object) : 
#	def __init__(self) : 
#		self.c1, self.c2, self.a, self.b = np.matrix([[1,2],[3,4]]), np.ma.MaskedArray(np.arange(2), [True,False]), A(), B()
#
#class D(object) : 
#	def __init__(self) : 
#		self.d1, self.d2, self.a, self.b, self.c = 'whoami', 789, A(), B(), C()
#
#
#
#outname = 'test.hdf5'
#
#a = D()
#classhdf5 = jp.ClassHdf5(a, outname)
#classhdf5.Save()
#print a.a.a1, a.a.a2
#print a.b.b1, a.b.b2, a.a.a1, a.a.a2
#print a.c.c1, a.c.c2, a.c.b.b1, a.c.b.b2, a.c.b.a.a1, a.c.b.a.a2
#print a.d1, a.d2
#print
#print '---------------'
#print
#
#
#b = D()
#b.a.a1, b.a.a2 = None, None
#b.b.b1, b.b.b2, b.a.a1, b.a.a2 = None, None, None, None
#b.c.c1, b.c.c2, b.c.b.b1, b.c.b.b2, b.c.b.a.a1, b.c.b.a.a2 = None, None, None, None, None, None
#b.d1, b.d2 = None, None
#
#classhdf5 = jp.ClassHdf5(b, outname)
#classhdf5.Read()
#
#print b.a.a1, b.a.a2
#print b.b.b1, b.b.b2, b.a.a1, b.a.a2
#print b.c.c1, b.c.c2, b.c.b.b1, b.c.b.b2, b.c.b.a.a1, b.c.b.a.a2
#print b.d1, b.d2
#


