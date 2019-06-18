
'''
def _DoMultiprocess_xxx( iterable ) : 
'''

def NprocessCPU( Nprocess=None ) : 
	'''
	return [Nprocess, NprocessTot, threads, cores, Ncpu, hyper, cpuinfo]
	'''
	from jizhipy.Basic import ShellCmd
	try : 
		uname = ShellCmd('uname')[0]
		#--------------------------------------------------
		if (uname == 'Linux') : 
			threads = ShellCmd('cat /proc/cpuinfo | grep siblings')[-1]
			cores = ShellCmd('cat /proc/cpuinfo | grep "cpu cores"')[-1]
		elif (uname == 'Darwin') : 
			threads =ShellCmd('sysctl machdep.cpu.thread_count')[-1]
			cores = ShellCmd('sysctl machdep.cpu.core_count')[-1]
		threads = int(threads.split(':')[-1])
		cores   = int(cores.split(':')[-1])
		#--------------------------------------------------
		if (uname == 'Linux') : 
			NprocessTot = len(ShellCmd('cat /proc/cpuinfo | grep processor'))
		elif (uname == 'Darwin') : NprocessTot = threads
		Ncpu = NprocessTot / threads
		hyper = False if(threads==cores)else True
		#--------------------------------------------------
		if (Nprocess is None) : Nprocess = NprocessTot
		elif (type(Nprocess) == str) : 
			Nprocess = Nprocess.lower()
			Nprocess = float(Nprocess.split('*none')[0])
			Nprocess = int(Nprocess * NprocessTot)
		rest = NprocessTot / 8
		if (rest < 1) : rest = 1
		if (Nprocess>NprocessTot-rest): Nprocess = NprocessTot-rest
		if (Nprocess < 1) : Nprocess = 1
		#--------------------------------------------------
		strs = '' if(Ncpu==1)else 's'
		cpuinfo = ('CPU INFO: %i cpu'+strs+', each cpu has %i cores %i threads (hyper='+str(hyper)+'), total %i processors') % (Ncpu, cores, threads, NprocessTot)
		return [Nprocess, NprocessTot, threads, cores, Ncpu, hyper, cpuinfo]
	except : return [4, 4, 2, 2, 1, True, '']





class PoolFor( object ) : 
	'''
	def _DoMultiprocess( iterable ) : 
		return a
	
	pool = PoolFor(Nstart, Nend, Nprocess)
	data = pool.map_async(_DoMultiprocess, send, cast)
	
	data = np.concatenate(data, 0)
	'''

	def __init__( self, Nstart=None, Nend=None, Nprocess=None, nsplit=None, thread=False, verbose=False ) :
		'''
		(1) PoolFor( Nstart, Nend, Nprocess )
				use Nstart, Nend, Nprocess to calculate nsplit
				split send in self.map_async()
		(2) PoolFor( nsplit )
				use this nsplit
				split send in self.map_async()
		(3) PoolFor()
			don't split send in self.map_async(), send has been splitted when gived to self.map_async(splitted_send, bcast)
				send[0] for process-1
				send[1] for process-2
				......
				send[n] for process-n+1
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		self.verbose, self.thread = bool(verbose), bool(thread)
		self.zero, self.splitsend = False, True
		if((Nstart is None or Nend is None)and nsplit is None):
			self.splitsend = False
			return
		#--------------------------------------------------
		if (nsplit is not None) : 
			Nprocess, NprocessTot, threads, cores, Ncpu, hyper, cpuinfo = NprocessCPU(len(nsplit))
		#--------------------------------------------------
		else : 
			if (Nend-Nstart <= 0) : 
				Raise(Warning, 'Nend-Nstart='+str(Nend-Nstart)+'<=0, return None')
				self.zero = True
				return
			Nprocess, NprocessTot, threads, cores, Ncpu, hyper, cpuinfo = NprocessCPU(Nprocess)
			if (Nend-Nstart < Nprocess) : Nprocess = Nend-Nstart
			# nsplit
			nsplit=np.linspace(Nstart,Nend, Nprocess+1).astype(int)
			nsplit = np.array([nsplit[:-1], nsplit[1:]]).T
		#--------------------------------------------------
		self.Nprocess, self.nsplit = Nprocess, Asarray(nsplit)
		self.Nmax = (self.nsplit[:,1] - self.nsplit[:,0]).max()





	def map_async( self, func, send=None, bcast=None ) : 
		'''
		If use apply() or map(), we can stop the pool program with Ctrl-c. However, if use apply_async().get(xxx) or map_async().get(xxx), we can use Ctrl-c to stop the program at any time.

		iterable:
			iterable[0] = [n1, n2, PoolWorder-i], i=1,2,..,Nprocess

		func:
			_DoMultiprocess()

		send:
			None or tuple or 2D-ndarray: send[i] for ranks
				a, b = iterable[1] | a = iterable[1][0] (one array, must give[0])
			If is tuple/list, means each element is one array (note that must be 2D, and split along axis=0/row)
			If not tuple/list, means the send is an unity: send = Asarray(send)
			If is 2D array, will split along axis-0 (row)

		bcast:
			None or tuple or others/as a unity
			If is others, may bcast = (bcast,)
			Some data which will be broadcasted to each processes, such as the x and p0 in test_multiprocess_poolfor_class-func.py
			Must be tuple: (x, p0, ...)
		'''
		if (self.zero) : return
		import numpy as np
		from jizhipy.Array import Asarray
		if (self.splitsend) :   # Main here
			istuple = True
			if (type(send) != tuple and type(send) != list) : 
				send, istuple = (Asarray(send),), False
			iterable, nsl = list(self.nsplit), self.nsplit
			for i in range(len(nsl)) : 
				iterable[i] = [[tuple(iterable[i]), self.Nmax]]
				sendtmp = ()
				for j in range(len(send)) : 
					if (send[j] is None) : sendtmp += (None,)
					else : sendtmp += (send[j][nsl[i][0]:nsl[i][1]],)
				if (not istuple) : sendtmp = sendtmp[0]
				iterable[i] += [sendtmp, bcast]
		#--------------------------------------------------
		else : 
			if (send is not None) : 
				self.Nprocess = len(send)
				iterable = []
				Nmax = 0
				for i in range(len(send)) : 
					try : 
						if (len(send[i]) > Nmax) : Nmax = len(send[i])
					except : pass
				for i in range(len(send)) : iterable.append([[(None,None),Nmax], send[i], bcast])
			else : 
				self.Nprocess = 1
				iterable = [[(None,None,1), None, bcast]]
		#--------------------------------------------------
		cpuinfo = NprocessCPU()[-1]
		if (self.verbose) : print('Open '+str(self.Nprocess)+' processes \n'+cpuinfo)
	#	if (not self.thread) : 
	#		import multiprocessing
	#		pool = multiprocessing.Pool(self.Nprocess)
	#	else : 
	#		from multiprocessing.pool import ThreadPool
	#		pool = ThreadPool(self.Nprocess)
		import multiprocessing
		pool = multiprocessing.Pool(self.Nprocess)
		self.data=pool.map_async(func,iterable).get(10**9)
		pool.close()
		pool.join()
		return self.data



