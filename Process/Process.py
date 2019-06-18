
class Process( object ):

	def __init__(self):
		'''
		'''
		self.process, self.pid, self.name = [], [], []



	def _pidname2idx(self, pidname):
		'''
		Parameters
		----------
		pidname:
			(1) [int] pidname is pid
			(2) [str] pidname is name
		'''
		from jizhipy.Basic import IsType
		if (IsType.isnum(pidname)):
			pidname = int(pidname)
			which = self.pid
		elif (IsType.isstr(pidname)):
			which = self.name
		for i in range(len(which)):
			if (which[i] == pidname): return i





	def ShareList(self):
		from multiprocessing import Manager
		return Manager().list()


	def ShareDict(self):
		from multiprocessing import Manager
		return Manager().dict()





	def Add(self, func, args=(), kwargs={}, name=None, **add_kwargs):
		'''
		Pass args to func, and start a process
		'''
		from multiprocessing import Process
		if (name is None):
			name = 'Process-'+str(len(self.process)+1)
		add_kwargs['name'] = name
		p = Process(target=func, args=args, kwargs=kwargs, **add_kwargs)
		self.process.append(p)
		self.pid.append(None)
		self.name.append(p.name)





	def Get(self, *args):
		'''
		Parameters
		----------
		*args:
			=(pidname1, pidname2, ...)
				pidname:
					(1) [int] pidname is pid
					(2) [str] pidname is name
			(3) =None or =(): all processes
		'''
		p = []
		islist = False if(len(args)<=1)else True
		if (None in args or len(args) ==0): 
			islist = True
			n = range(len(self.process))
		else: 
			n = []
			for a in args: n.append(self._pidname2idx(a))
		for i in n:
			p.append(self.process[i])
		if (not islist): p = p[0]
		return p





	def Start(self, *args):
		'''
		Parameters
		----------
		*args:
			=(pidname1, pidname2, ...)
				pidname:
					(1) [int] pidname is pid
					(2) [str] pidname is name
			(3) =None: all processes
		'''
		if (None in args or len(args) ==0): 
			n = range(len(self.process))
		else: 
			n = []
			for a in args: n.append(self._pidname2idx(a))
		for i in n:
			self.process[i].start()
			self.pid[i] = self.process[i].pid





	def Stop(self, *args):
		'''
		Parameters
		----------
		*args:
			=(pidname1, pidname2, ...)
				pidname:
					(1) [int] pidname is pid
					(2) [str] pidname is name
			(3) =None: all processes
		'''
		if (None in args or len(args) ==0): 
			n = range(len(self.process))
		else: 
			n = []
			for a in args: n.append(self._pidname2idx(a))
		for i in n:
			self.process[i].terminate()
			self.pid[i] = None





	def Remove(self, *args):
		'''
		Parameters
		----------
		*args:
			=(pidname1, pidname2, ...)
				pidname:
					(1) [int] pidname is pid
					(2) [str] pidname is name
			(3) =None: all processes
		'''
		self.Stop(*args)
		p, m = [], range(len(self.process))
		if (None in args or len(args) ==0): 
			n = range(len(self.process))
		else: 
			n = []
			for a in args: n.append(self._pidname2idx(a))
		for i in m:
			if (i not in n): p.append(self.process[i])
		self.process = p





	def IsAlive(self, *args):
		'''
		Parameters
		----------
		*args:
			=(pidname1, pidname2, ...)
				pidname:
					(1) [int] pidname is pid
					(2) [str] pidname is name
			(3) =None: all processes
		'''
		is_alive = []
		islist = False if(len(args)<=1)else True
		if (None in args or len(args) ==0): 
			islist = True
			n = range(len(self.process))
		else: 
			n = []
			for a in args: n.append(self._pidname2idx(a))
		for i in n:
			is_alive.append(self.process[i].is_alive())
		if (not islist): is_alive = is_alive[0]
		return is_alive





	def Join(self, *args, **kwargs):
		'''
		Wait until child process terminates

		Parameters
		----------
		*args:
			=(pidname1, pidname2, ...)
				pidname:
					(1) [int] pidname is pid
					(2) [str] pidname is name
			(3) =None: all processes

		**kwargs:
			key1: 'timeout', default timeout=None
		'''
		if (None in args or len(args) ==0): 
			n = range(len(self.process))
		else: 
			n = []
			for a in args: n.append(self._pidname2idx(a))
		for i in n:
			self.process[i].join(**kwargs)

