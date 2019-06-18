
class Path( object ) : 


	def EscPath( self, path, which='rm', abspath=False ) : 
		''' 
		Remove/Add escape characters from/to the path
	
		In Linux:
			in shell and GUI
			(1) Filename can't contain "/" but can contain ":"
			(2) ":" will be shown as ":"
		In Mac OSX:
			in shell: 
				(1) Filename can't contain "/" but can contain ":"
				(2) ":" will be shown as ":", like in Linux
			in GUI:
				(1) Filename can't contain ":" but can contain "/"
				(2) "/" will be shown as "/" in GUI, but will be shown as ":" in shell
	
		path:
			Relative or absolute paths are OK
	
		which:
			=='rm'/'remove': remove "\" from the escape character
			=='add': add "\" to the escape character
		'''
		import os
		escchar = '()[]{}<>`!@$^&*=|;,? :'
		path = list(path)
		while ('\\' in path) : path.remove('\\')
		if (which.lower() == 'add') : 
			for i in range(len(path)) : 
				if(path[i] in escchar or path[i] in ['"',"'"]):
					path[i] = '\\' + path[i]
		if (path == []) : path = ''
		else : 
			for i in range(1, len(path)) : path[0] += path[i]
			path = path[0]
			if (path[-1] == '/') : path = path[:-1]
		if (abspath) : path = os.path.abspath(os.path.expanduser(path))
		return path

	def Esc( self, path, which='rm', abspath=False ) : 
		return self.EscPath(path, which, abspath)





	def FindDir( self, dirpath, exclude=[], name='' ) : 
		'''
		Shell command: find dirpath -name name, plus+ exclude
		'''
		import os
		from jizhipy.Basic import IsType, ShellCmd
		if (not IsType.isstr(name)) : name = ''
		name = EscPath(name, 'add')
		#--------------------------------------------------
		dirpath = EscPath(dirpath, 'add', True)
		if (name) : walk = ShellCmd('find '+dirpath+' -name '+name)
		else : walk = ShellCmd('find '+dirpath)
		#--------------------------------------------------
		exctmp = []
		if (IsType.isstr(exclude)) : exclude = [exclude]
		for i in range(len(exclude)) : 
			exclude[i] = EscPath(exclude[i], 'add')
			if ('*' not in exclude[i]) : 
				dirname=os.path.abspath(os.path.expanduser(exclude[i]))
				exctmp += ShellCmd('find '+dirname)
			else : 
				dirname, basename = os.path.split(exclude[i])
				dirname = os.path.abspath(os.path.expanduser(dirname))
				exctmp += ShellCmd('find '+dirname+' -name '+basename)
		exclude = exctmp
		#--------------------------------------------------
		for i in range(len(exclude)) : 
			try : walk.remove(exclude[i])
			except : pass
		walk.remove(dirpath)
		dirs, files = [], []
		for i in range(len(walk)) : 
			if (os.path.isdir(walk[i])) : dirs.append(walk[i][len(dirpath)+1:])
			else : files.append(walk[i][len(dirpath)+1:])
		#--------------------------------------------------
		return [dirpath, dirs, files]





	def AbsPath( self, path, tilde=False ) : 
		'''
		Return the absolute path

		path:
			[str] | list of [str]

		tilde:
			=True: use '~'
			=False: expanduser '~'
		'''
		import os
		from jizhipy.Basic import Raise, IsType
		home = os.path.expanduser('~')
		if (IsType.isstr(path)): islist, path = False, [path]
		else: islist, path = True, list(path) # change address
		for i in range(len(path)): 
			tail = '/' if(path[i][-1]=='/')else ''
			path[i] = os.path.expanduser(path[i])
			path[i] = os.path.abspath(path[i])+tail
			if (tilde) : 
				if (path[i][:len(home)] == home) : 
					path[i] = '~'+path[i][len(home):]
		if (not islist): path = path[0]
		return path

	def Abs( self, path, tilde=False ) : 
		return self.AbsPath(path, tilde)





	def ExistsPath( self, path, old=False, stop=False, mkdir=False, touch=False, delete=False ) : 
		'''
		Parameters
		----------
		path:
			str | list/tuple of str

		old:
			=True: rename the exists file with '_old'
			=False | None: don't rename
		
		stop:
			=True: if not exists, raise error

		mkdir:
			True | False
			If not exists, mkdir it

		touch:
			True | False
			If not exists, touch it

		delete:
			True | False
			If exists, delete it


		Returns
		----------
		If path is str: return True/False
		If path is list/tuple of str: return list of True/False
		'''
		import os
		from jizhipy.Basic import Raise, IsType
		if (IsType.isstr(path)): islist, path = False, [path]
		else: islist, path = True, list(path)
		info, delname, trfa, pathabs = '', '', [], []
		for i in range(len(path)): 
			pathabs.append(self.AbsPath(path[i]))
			trfa.append(os.path.exists(pathabs[-1]))
			if (trfa[-1] and IsType.isdir(pathabs[-1])):
				pathabs[-1] = self.CheckDir(pathabs[-1])
		#----------------------------------------
		# stop
		if (stop):
			for i in range(len(trfa)):
				if (trfa[i]): continue
				info += 'Not exists: '+path[i]+'\n'
			if (len(info)>0): Raise(Exception, info[:-1])
		# old
		if (old):
			for i in range(len(trfa)):
				if (not trfa[i]): continue
				if (pathabs[i][-1] == '/'):
					pathold = pathabs[i][:-1]+'_old/'
				else:
					n = pathabs[i].rfind('.')
					if (n < 0) : pathold = pathabs[i] + '_old'
					else : pathold = pathabs[i][:n] + '_old' + pathabs[i][n:]
				os.system('mv '+pathabs[i]+' '+pathold)
		# mkdir or touch
		if (mkdir): 
			for i in range(len(trfa)):
				if (trfa[i] is False): os.makedirs(path[i])
		if (touch): 
			for i in range(len(trfa)):
				if (trfa[i] is True): continue 
				if (path[i][-1] == '/'): 
					print('Warning: '+path[i]+' is a directory, can NOT touch')
					continue
				dirname = os.path.dirname(path[i])
				if (not os.path.exists(dirname)): 
					os.makedirs(dirname)
				os.system('touch '+path[i])
		# delete
		if (delete):
			for i in range(len(trfa)):
				if (not trfa[i]): continue
				delname += pathabs[i]+' '
			if (len(delname)>0): 
				os.system('/bin/rm -r '+delname)
		if (not islist): trfa = trfa[0]
		return trfa

	def Exists( self, path, old=False, stop=False, mkdir=False, touch=False, delete=False ) : 
		return self.ExistsPath(path, old, stop, mkdir, touch, delete)





	def OutnamePath( self, outname ) : 
		'''
		return [outdir, basename]
			and (1) Mkdir(outdir) if not exists
			    (2) mv to '_old' if outname is exists
		'''
		import os
		outname = self.AbsPath(outname)
		home = os.path.expanduser('~/')
		if ('/' not in outname) : outdir = self.AbsPath('./')
		else : 
			n = outname.rfind('/')+1
			outdir, outname = outname[:n], outname[n:]
		if (outdir[:len(home)] == home) : outdir = '~/'+outdir[len(home):]
		os.makedirs(outdir)
		self.ExistsPath(outdir+outname, True)
		return [outdir, outname]

	def Outname( self, outname ) : 
		return self.OutnamePath(outname)





	def jizhipyPath( self, which='' ) : 
		'''
		return: 
			dir of jizhipy + which
		For example: 
			(1) which='', return:
				'~/.mysoftware/python-packages/jizhipy/'
			(2) which='jizhipy_tool/', return:
				'~/.mysoftware/python-packages/jizhipy/jizhipy_tool/'
		'''
		from jizhipy.Basic import SysFrame
		path = SysFrame()[1][-1]
		n = path.rfind('/')
		path = path[:n+1]+str(which)
		return path

	def jizhipy( self, which='' ) : 
		return self.jizhipyPath(which)




	def DirPath( self, path, up=0 ) : 
		'''
		Return the path of current or upper directory

		Parameters
		----------
		up:
			[int], 0 to ...
			up=0: current directory
			up=1: father directory
			up=2: grandfather directory
			......
		'''
		import os
		if (path[-1] =='/'): path = path[:-1]
		for i in range(int(up)):
			path = os.path.dirname(path)
		return path





	def ModulePath( self, module, syspath=None ):
		'''
		Return the path of a module

		Parameters
		----------
		module:
			[str], for example:
				module = 'numpy'
				module = 'uniface.module.UniFace'

		syspath:
			None | [str] 
			syspath = '~/uniface', means sys.path.append('~/uniface')
			syspath = '~/uniface:/usr/package' (use : to separate), means sys.path.append('~/uniface'), sys.path.append('/usr/package')

		Returns
		----------
			return moduleDir: [str], directory of this module/class/function
		'''
		import os
		heredir = os.path.dirname(__file__)
		option = ' -f --notvim '
		if (syspath is not None): option += '--syspath '+syspath+' '
		os.system('python '+heredir+'/helpy'+option+module)
		filename = self.AbsPath('~/helpy_doc/help_'+module)
		if (not os.path.exists(filename)): return None
		fo = open(filename)
		txt = fo.readline()[:-1]
		fo.close()
		return txt

	def Module( self, module, syspath=None ):
		return self.ModulePath(module, syspath)





	def CheckDir(self, path):
		'''
		path:
			[str] | list of [str]

		if (path[-1] != '/'): path += '/'
		(don't chech its existance)
		'''
		from jizhipy.Basic import Raise, IsType
		if (IsType.isstr(path)): islist, path = False, [path]
		else: islist, path = True, list(path)
		for i in range(len(path)):
			if (path[i][-1] != '/'): path[i] += '/'
		if (not islist): path = path[0]
		return path





	def Property(self, path):
		'''
		return the property of the file/directory

		Parameters
		----------
		path:
			[str] | [list of str]

		Returns
		----------
		return property

		property:
			[dict] | [list of dict]
			property[i] = {'permision':'?', 'group':'?', 'size':'?', 'date':'?', 'path':path[i]}
		'''
		from jizhipy.Basic import IsType, ShellCmd, Time
		import os
		year = Time(0).split('/')[0]
		if (IsType.isstr(path)): islist, path = False, [path]
		else: islist = True
		prop, path = [], self.Abs(path)
		for i in range(len(path)):
			d = {'permision':'', 'group':'', 'size':'', 'date':'', 'path':path[i]}
			if (os.path.exists(path[i]) is False): 
				prop.append(d)
				continue
			if (IsType.isfile(path[i])):
				p, ds = ShellCmd('ls -lh '+path[i])[0], None
			else:
				if (path[i][-1] !='/'): path[i] += '/'
				n = path[i][:-1].rfind('/')
				dn, bn = path[i][:n], path[i][n+1:-1]
				q = ShellCmd('ls -lh '+dn+' | grep '+bn)
				for p in q:
					if (p[-len(bn)-1:] == ' '+bn): break
				ds = ShellCmd('du -sh '+path[i])[0].split('\t')[0]
			p = p.split(' ')
			while ('' in p): p.remove('')
			d['path'] = path[i]
			d['permision'] = p[0]
			d['group'] = p[2]+':'+p[3]
			d['size'] = p[4] if(ds is None)else ds
			m, day = p[5], p[6]
			try: int(m)
			except: m = m[:-1]
			if (':' in p[7]): y, h = year, p[7]+':00'
			else: y, h = p[7], '12:34:56'
			d['date'] = y+'/'+m+'/'+day+' '+h
			prop.append(d)
		if (not islist): prop = prop[0]
		return prop





	def DBMFname(self, path):
		'''
		'''
		import os
		from jizhipy.Basic import IsType
		if (IsType.isstr(path)): islist, path = False, [path]
		else: islist, path = True, list(path)
		dirname, basename, mainname, formatname =[],[],[],[]
		for f in path:
			n = f.rfind('/')
			if (n >=0): 
				dirname.append(f[:n+1])
				basename.append(f[n+1:])
			else: 
				dirname.append('')
				basename.append(f)
			n = basename[-1].rfind('.')
			if (n >=0):
				mainname.append(basename[-1][:n])
				formatname.append(basename[-1][n:])
			else:
				mainname.append(basename[-1])
				formatname.append('')
		if (not islist):
			dirname, basename = dirname[0], basename[0]
			mainname, formatname = mainname[0], formatname[0]
		return (dirname, basename, mainname, formatname)

	def Dirname(self, path):
		return self.DBMFname(path)[0]

	def Basename(self, path):
		return self.DBMFname(path)[1]

	def Mainname(self, path):
		return self.DBMFname(path)[2]

	def Formatname(self, path):
		return self.DBMFname(path)[3]





Path = Path()

