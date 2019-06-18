
class Cd( object ) : 

	def __doinit__(self):
		self.doinit = True

	def __init__( self ) : 
		import os
		self.doinit = False
		self.originalpath = os.getcwd()

	def Go( self, path=None ) : 
		import os
		if (self.doinit): self.__init__()
		path = os.path.abspath(os.path.expanduser(path))
		if (path is not None) : os.chdir(path)
		return path

	def Back( self ) : 
		import os
		if (self.doinit): self.__init__()
		os.chdir(self.originalpath)
		return self.originalpath
Cd = Cd()





def Mkdir( path ) : 
	import os
	if (path == '') : path = './'
	if (path[-1] != '/') : path += '/'
	if (not os.path.exists(path)) : os.makedirs(path)
	return path





def ShellCmd( cmd, islist=True ) : 
	'''
	Absolutely use shell command, and return the strings that should be printed on the screen
	return: list of str

	def ShellCmd( cmd ) : 
		import os
		strlist = os.popen(cmd).read().split('\n')[:-1]
		return strlist

	** get the size of shell-window:  stty size
	'''
	import os
	if (cmd.lower() == 'pdf') : 
		from jizhipy.Basic import SysFrame
		path = SysFrame()[1][-1]
		n = path.rfind('/')
		path = path[:n+1]
		uname = os.popen('uname -a').readlines()[0][:5]
		if (uname == 'Darwi') : which = 'open'
		elif (uname == 'Linux') : which = 'evince' # 'xdg-open'
		os.system(which+' '+path+'python-shellcmd.pdf')
		return
	#---------------------------------------------
	elif (cmd.lower()[:3] == 'cd ') : 
		print("Please use jp.Cd.go(...) instead of ShellCmd('cd ...')")
		return
	#---------------------------------------------
	strlist = os.popen(cmd).read()
	if (islist): strlist = strlist.split('\n')[:-1]
	return strlist



