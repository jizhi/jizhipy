
'''
comm.Send(a), comm.Recv(a), comm.Bcast(a)
	a array must be 1D(flatten) : (1) fastest (2) definitely correct
	a is NOT 1D: (1) slower (2) maybe wrong, mismatch shape

# Use MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
root = 0
rank, Nprocess, host, info = jp.MPI.Comm(comm)
verbose = True if(rank==root)else False

# Get sys.argv, parope={str: str, ...}
paropt = jp.ParserOption()
if (verbose) : print(jp.Time(1))
'''


class MPI( object ) : 


	def Comm( self, comm ) : 
		'''
		return (rank, Nprocess, host, info)

		if __name__ == '__main__' : 
			from mpi4py import MPI
			comm = MPI.COMM_WORLD
			rank, Nprocess, host, info = jp.MPI.Comm(comm)
			if (verbose) : print(info)
		'''
		if (comm is None) : 
			return (0, 1, 'localhost', 'NOT use MPI')
		else : 
			try : rank = comm.Get_rank()
			except : return (0, 1, 'localhost', 'NOT use MPI, BAD comm')
		#---------------------------------------------
		from mpi4py import MPI as Mpi
		host = Mpi.Get_processor_name()
	#	comm = Mpi.COMM_WORLD
		rank = comm.Get_rank()
		Nprocess = comm.Get_size()
		#---------------------------------------------
		rh = self.Gather((rank, host), comm, None)
		rh2 = len(rh)*[0]
		for i in range(len(rh)) : rh2[rh[i][0]] = rh[i]
		h, n, info = rh2[0][1], 0, rh2[0][1]+':'
		while (n != len(rh2)) : 
			if (rh2[n][1] == h) : info += str(rh2[n][0])+','
			else : 
				h = rh2[n][1]
				info =info[:-1]+'\n'+h+':'+str(rh2[n][0])+','
			n += 1
		info = info[:-1].split('\n')
		#---------------------------------------------
		n = []
		for i in range(len(info)) : 
			if (host not in info[i]) : continue
			m = info[i].split(':')[1].split(',')
			for j in range(len(m)) : n.append(int(m[j]))
			break
		host = (host, tuple(n))
		#---------------------------------------------
		for i in range(len(info)) : 
			ifi = info[i].split(',')
			if (len(ifi) > 1) : info[i] = ifi[0]+'~'+ifi[-1]
			info[i] = info[i].split(':')
			info[i] = info[i][0] + ':('+str(len(ifi))+')' + info[i][1]
			if (i == 0) : continue
			info[0] += '  '+info[i]
		info = info[0]
		return (rank, Nprocess, host, info)





	def Scatter( self, array, comm, root, axis=None, order=None ) :
		''' 
		split array into Nprocess pieces, then sent to each rank of MPI if root!=None
	
		array:
			np.ndarray with any shape, will Send() and Recv()
	
		root:
			int | None
			==int: use rank=root to scatter and send, other ranks receive
			==None: split local, don't send and receive
	
		axis:
			which axis to split
	

		order:
			None/False | True | 1Darray/list
			(1) ==None/False: scatter as normal:
				rank0 gets 1-10, rank1 gets 11-20, ......
			(2) ==True: scatter randomly:
				Maybe rank0 gets [4,9,12,15,20], rank1 gets [1,5,7,8,17], ......
			(3) 1Darray/list: Use this order to scatter
				input order.size == array.shape[axis]
	
		return:
			[Splited_array,  order], list
			return order.size == Splited_array.shape[axis] <= input order.size

		order: 
			IS return_array = original_array[order]
				!!! NOT original_array = return_array[order]
	
		Send/Recv are faster than send/recv
		'''
		from jizhipy.Basic import IsType
		import numpy as np
		case = 0  # ndarray
		if (IsType.isdataframe(array)) : case = 1
		elif (IsType.isseries(array)) : case = 2
		if (case == 0) : array = np.array(array)
		if (len(array.shape) == 1) : axis = 0
		if (comm is None) : 
			order = np.arange(array.shape[axis])
			return [array, order]
		#---------------------------------------------
		from jizhipy.Array import ArrayAxis
		from jizhipy.Basic import Raise
		from jizhipy.Math import Random
		try: rank, Nprocess=comm.Get_rank(), comm.Get_size()  
		except : rank, Nprocess, comm = 0, 1, None
		#---------------------------------------------
		if (Nprocess == 1) : 
			order = np.arange(array.shape[axis])
			return [array, order]
		#---------------------------------------------
		if (root is None) : root, local = 0, True
		else : local = False
		if (root < 0) : root = Nprocess + root
		#---------------------------------------------
		if (len(array.shape) == 1) : axis = 0
		elif (axis is None) : 
			if (rank == root) : Raise(Exception, 'axis=None, please set which axis')
			else : exit()
		#---------------------------------------------
		err = False
		if (rank == root) : 
			if (axis < 0) : axis = len(array.shape) + axis
			Naxis = array.shape[axis]
			#---------------------------------------------
			# Check order
			if (order is None or order is False) : 
				order = np.arange(Naxis)
			elif (order is True) : 
				order = Random.RandomChoice(Naxis)
			else : 
				order = np.array(order, int)
				if (order.size != Naxis) : err = True
		#---------------------------------------------
		err = comm.bcast(err, root=root)
		#---------------------------------------------
		if (rank == root) : 
			if (err) : 
				if (local) : 
					if (rank == 0) : Raise(Exception, 'jizhipy.MPI.Scatter(): order.size='+str(order.size)+' != array.shape[axis]='+str(Naxis))
					else : exit()
				else:Raise(Exception,'jizhipy.MPI.Scatter(): order.size='+str(order.size)+' != array.shape[axis]='+str(Naxis))
			#---------------------------------------------
			orders, arrays = [], []
			# Split order
			n = np.linspace(0, Naxis, Nprocess+1).astype(int)
			if (n[1] == n[0]) : n[1], n[2] = n[1]+1, n[2]+1  # ensure rank0 gets !=0
			n = np.sort(n)
			for i in range(len(n)-1) : 
				orders.append(order[n[i]:n[i+1]])  # orders
			#---------------------------------------------
			# Split array
			if (case == 0) :  # ndarray
				array = ArrayAxis(array, axis, 0, 'move')
				for i in range(len(orders)) : 
					a = array[orders[i]]
					a = ArrayAxis(a, 0, axis, 'move')
					arrays.append(a)  # arrays
				#---------------------------------------------
				# Send to other ranks
				for i in range(len(arrays)) : 
					c = arrays[i]
					if (i == root) : b, order = c, orders[i]
					else : 
						comm.send((c.shape, c.dtype.name), dest=i, tag=10010)
						# comm.Send(), Recv(), Bcast(), use 1D, fastest and no any problem
						comm.Send(orders[i],dest=i,tag=10000)
						comm.Send(c.flatten(), dest=i, tag=10086)
		#---------------------------------------------
		else : 
			if (err) : exit()
			if (case == 0) :  # ndarray
				shape, dtype =comm.recv(source=root, tag=10010)
				b = np.empty(int(np.prod(shape)), dtype)
				order = np.empty(shape[axis], int)
				comm.Recv(b, source=root, tag=10086)
				comm.Recv(order, source=root, tag=10000)
				b = b.reshape(shape)
		#---------------------------------------------
		return [b, order]





	def range( self, n, comm ) : 
		'''
		n: 
			int (one number)
			return tuple(n1,n2), index range (include n1, exclude n2) when use MPI
		'''
		if (comm is None) : return range(n)
		import numpy as np
		rank, Nprocess = comm.Get_rank(), comm.Get_size()  
		n = np.linspace(0, n, Nprocess+1).astype(int)
		return range(n[rank], n[rank+1])

	def range( self, n, comm ) : 
		return self.range(n, comm)





	def Gather( self, a, comm, root, axis=None, order=None ) : 
		'''
		NOTE THAT jp.MPI.Gather() must be put in the main body, NOT in if(rank=0), otherwise, other ranks doesn't have jp.MPI.Gather() operation, will wait forever !!!!!

		Likes np.concatenate([a1, a2, ...], axis)

		root:
			==None: Allgather

		a:
			(1) np.ndarray | np.matrix
			(2) python object, in this case, don't use parameters axis and order

		order:
			if a is ndarray, order.size == a.shape[axis] == return order.size in Scatter()

		** NOTE THAT, a in all ranks must have the same dtype, otherwise, raise
			mpi4py.MPI.Exception: MPI_ERR_TRUNCATE: message truncated

		return:
			rank==root: np.ndarray
			else: None
		'''
		if (comm is None) : return a
		from jizhipy.Basic import Raise, IsType
		import numpy as np
		from jizhipy.Array import Asarray
		try : rank, Nprocess =comm.Get_rank(), comm.Get_size()
		except : rank, Nprocess = 0, 1
		#---------------------------------------------
		array, allg = a, False
		if (root is None) : root, allg = 0, True
		if (root < 0) : root = Nprocess + root
		#---------------------------------------------
		if   (IsType.isndarray(array)) : which = 'ndarray'
		elif (IsType.ismatrix( array)) : which = 'matrix'
		elif (IsType.isnum(    array)) : which = 'num'
		else : which = ''
		if (which in ['', 'num']) : 
			if (not allg) : b = comm.gather(array, root=root)
			else : b = comm.allgather(array)
			return b
		#---------------------------------------------
		if (Nprocess == 1) : return array
		if (axis is None) : axis = 0
		if (array is None) : array = np.array([])  # 1D
		# Check shapeall, same shape
		shapeall = comm.allgather(array.shape)
		s, strs = [], ''
		for i in range(len(shapeall)) :  
			s.append(len(shapeall[i]))
			strs += 'rank'+str(i)+'='+str(shapeall[i])+', '
		s = np.array(s)
		s = abs(s[1:] - s[:-1]).sum()
		if (s != 0) : 
			if (rank == 0) : Raise(Exception, 'shapes are NOT the same: '+strs[:-2])
			else : exit()
		shapeall = np.array(shapeall)
		#---------------------------------------------
		# Check dtypeall, same dtype
		dtypeall = comm.allgather(array.dtype.name)
		n, dtype = shapeall.prod(1), []
		for i in range(n.size) : 
			if (n[i] > 0.1) : dtype.append(dtypeall[i])
		dt1, dt2, d1, d2 = '', 0, '', 0
		for i in range(len(dtype)) : 
			if ('int' in dtype[i]) : 
				d1, d2 = 'int', int(dtype[i][3:])
			elif ('float' in dtype[i]) : 
				d1, d2 = 'float', int(dtype[i][5:])
			elif ('complex' in dtype[i]) : 
				d1, d2 = 'complex', int(dtype[i][7:])
			if (len(dt1) < len(d1)) : dt1 = d1
			if (dt2 < d2) : dt2 = d2
		if (dt1 == '') : dtype = dtype[0]
		else : dtype = dt1+str(dt2)
		if (array.dtype.name != dtype) : 
			array = array.astype(np.dtype(dtype))
		#---------------------------------------------
	
		# Judge which axis to merge  =>  axis
		n = shapeall.prod(1)  #@#@#@
		s = shapeall[n!=0]
		s = abs(s - s[:1]).sum(0)
		s = np.where(s!=0)[0]
		if   (len(s) == 0) : pass
		elif (len(s) == 1) : axis = s[0]
		else : 
			if (rank == 0) : Raise(Exception, '>= 2 axes are different shape: '+strs[:-2])
			else : exit()
		if (axis is not None and axis < 0) : axis = len(array.shape) + axis
	#	if (shapeall.shape[1] == 1) : axis = None  # all arrays are 1D (N1,), use flatten method
		if (shapeall.shape[1] == 1) : axis = 0  # all arrays are 1D (N1,), use flatten method
		if (axis is not None and axis >= len(array.shape)) : 
			if (rank==0): Raise(Exception,'jizhipy.MPI.Gather(): axis='+str(axis)+' out of '+str(len(array.shape))+'D array') 
			else : exit()
		#---------------------------------------------
	
		#----- when all array from each rank has the same shape, comm.Gather is fastest !!!!! -----
		array = array.flatten()
		array = np.append(array, np.empty(n.max()-array.size, array.dtype))
		if (order is not None) : 
			order = np.append(order, np.empty(shapeall[:,axis].max()-order.size))
		#---------------------------------------------
	
		# Gather flatten, easier
		# Gather array to b
		if (rank == root) : 
			b = np.empty((Nprocess, array.size), array.dtype)
		#	b = np.empty((Nprocess, array.size))
			if (order is not None) : 
				d = np.empty((Nprocess, order.size), int)
		else : b, d = None, None
	#	comm.Gather(array.astype(float), b, root=root)  #@#@#@
		comm.Gather(array, b, root=root)  #@#@#@
		if (order is not None) : 
			comm.Gather(order, d, root=root)  #@#@#@
		#---------------------------------------------

		if (rank == root) :  
			c, e = [], []
			for i in range(Nprocess) : 
				if (n[i] != 0) : 
					c.append(b[i][:n[i]].reshape(shapeall[i]))
					if (order is not None) : 
						e.append(d[i][:shapeall[i,axis]])
			b = np.concatenate(c, axis)

			# re-order !!!
			if (order is not None) : 
				order = np.concatenate(e)
				order = order + 1j*np.arange(b.shape[axis])
				order = np.sort(order).imag.astype(int)
				if   (axis == 0) : b = b[order]
				elif (axis == 1) : b = b[:,order]
				elif (axis == 2) : b = b[:,:,order]
				elif (axis == 3) : b = b[:,:,:,order]
				elif (axis == 4) : b = b[:,:,:,:,order]
				elif (axis == 5) : b = b[:,:,:,:,:,order]
		#---------------------------------------------

		if (allg) : 
			if (rank != root) : b = np.array([])
			shape = comm.bcast(b.shape, root=root)
			if (rank != root): b =np.empty(shape, array.dtype)
			b = b.flatten()
			comm.Bcast(b, root=root)
			b = b.reshape(shape)
		return b





	def Sum( self, array, comm, root ) : 
		'''
		array is distributed in ranks
		sum array in all ranks to root

		in root    : a = a0+a1+a2+a3+...+a_rank
		other ranks: a = None
			a.shape = a0.shape

		if root=None: Bcast(a, root)
		'''
		import numpy as np
		a = np.array(array)
		if (comm is None) : return a
		bcast = True if(root is None)else False
		if (root is None) : root = 0
		rank, Nprocess, host, info = self.Comm(comm)
		if (rank == root) : 
			shape, dtype = a.shape, a.dtype
			a = 1.*a.flatten()
		n = range(Nprocess)
		n.pop(root)
		for i in n : 
			if (rank == i) : comm.Send(a.flatten(), dest=root, tag=10086)
			elif (rank == root) : 
				b = np.empty(int(np.prod(shape)), dtype)
				comm.Recv(b, source=i, tag=10086)
				a = a + b
		if (rank == root) : a = a.reshape(shape)
		if (bcast) : a = self.Bcast(a, comm, root)
		elif (rank != root) : a = None
		return a





	def Prod( self, array, comm, root ) : 
		'''
		array is distributed in ranks
		prod array in all ranks to root

		in root    : a = a0*a1*a2*a3*...*a_rank
		other ranks: a = None
			a.shape = a0.shape

		if root=None: Bcast(a, root)
		'''
		import numpy as np
		a = np.array(array)
		if (comm is None) : return a
		bcast = True if(root is None)else False
		if (root is None) : root = 0
		rank, Nprocess, host, info = self.Comm(comm)
		if (rank == root) : 
			shape, dtype = a.shape, a.dtype
			a = 1.*a.flatten()
		n = range(Nprocess)
		n.pop(root)
		for i in n : 
			if (rank == i) : comm.Send(a.flatten(), dest=root, tag=10086)
			elif (rank == root) : 
				b = np.empty(int(np.prod(shape)), dtype)
				comm.Recv(b, source=i, tag=10086)
				a = a * b
		if (rank == root) : a = a.reshape(shape)
		if (bcast) : a = self.Bcast(a, comm, root)
		elif (rank != root) : a = None
		return a





	def Dot( self, a, b, comm, root, axis=None, order=None ) : 
		'''
		a, b are distributed in ranks
		The effect like this:
			Gather all a and b to root:
				a0 = Gather(a, root)
				b0 = Gather(b, root)
			Then dot like matrix:
				np.dot(a0, b0)

		if root==None: Bcast the result

		a, b:
			in local rank, a.T.shape == b.shape

		axis:
			Gather along which axis of a (NOT b)
		'''
		if (comm is None) : 
			from jizhipy.Math import MatrixDot
			return MatrixDot(a, b)
		import numpy as np
		rank, Nprocess, host, info = self.Comm(comm)
		if (root is None) : root = 0
		elif (root < 0) : root += Nprocess
		if (axis is None) : axis = 0
		elif (axis < 0) : axis += 2
		if (order is not None) : 
			n = self.Gather(order, comm, None)
			n = n + 1j*np.arange(n.size)
			n = np.sort(n).imag.astype(int)
		#----------------------------------------
		if (axis == 0) : 
			AB = []
			for i in range(Nprocess) : 
				B = self.Bcast(b, comm, i)
				AB.append( np.dot(a, B) )
			AB = np.concatenate(AB, 1)
			if (order is not None) : AB = AB[:,n]
			AB = self.Gather(AB, comm, root, axis, order)
		#----------------------------------------
		elif (axis == 1) : 
			AB = np.dot(a, b)
			AB = self.Sum(AB, comm, root)
		#----------------------------------------
		if (rank == root) : return AB
		else : return
		




	def ArrayDot( self, a, b, comm, root, axis=None ) : 
		'''
		a, b are large arrays in root
		use MPI with comm to compute a*b, faster
		a.shape == b.shape

		axis:
			split a and b along this axis
		'''
		if (comm is None) : 
			import numpy as np
			return np.prod(a, b)
		if (axis is None) : axis = 0
		if (axis < 0) : axis += 2
		a = self.Scatter(a, comm, root, axis)[0]
		b = self.Scatter(b, comm, root, axis)[0]
		return self.Prod(a, b, comm, root, axis)





	def MatrixDot( self, a, b, comm, root, axis=None ) : 
		'''
		a, b are large arrays in root
		use MPI with comm to compute np.dot(a,b), faster
		a.T.shape == b.shape

		axis:
			split a along this axis
			for b, along 1-axis
		'''
		if (comm is None) : 
			from jizhipy.Math import MatrixDot
			return MatrixDot(a, b)
		if (axis is None) : axis = 0
		if (axis < 0) : axis += 2
		a = self.Scatter(a, comm, root, axis)[0]
		b = self.Scatter(b, comm, root, 1-axis)[0]
		return self.Dot(a, b, comm, root, axis)





	def GatherScatter( self, array, comm, gaxis, saxis ) : 
		''' 
		2 steps:
			(1) Gather array along gaxis
			(2) Scatter the whole array along saxis
		'''
		import numpy as np
		if (comm is None) : return np.array(array)
		try : rank, Nprocess =comm.Get_rank(), comm.Get_size()
		except : rank, Nprocess = 0, 1
		if (Nprocess == 1) : return array
		b = []
		for i in range(Nprocess) : 
			c = self.Scatter(array, comm, i, saxis)
			b.append(c)
		b = np.concatenate(b, gaxis)
		return b





	def Bcast( self, a, comm, root ) : 
		'''
		a:
			(1) a is np.array() or np.matrix(), use Bcast()
			(2) a is other type, use bcast()
		'''
		if (comm is None) : return a
		from jizhipy.Basic import IsType
		import numpy as np
		try : rank, Nprocess =comm.Get_rank(), comm.Get_size()
		except : rank, Nprocess = 0, 1
		if (Nprocess == 1) : return a
		if (root < 0) : root = Nprocess + root
		if   (IsType.isndarray(a)) : which = 'ndarray'
		elif (IsType.ismatrix(a))  : which = 'matrix'
		else : which = ''
		#---------------------------------------------
		which = comm.bcast(which, root=root)
		if (which == '') : return comm.bcast(a, root=root)
		#---------------------------------------------
		if (rank == root) : 
			shape, dtype = a.shape, a.dtype
			a = np.array(a).flatten()  # 1D, fastest, exact
		else : shape, dtype = None, None
		shape, dtype = comm.bcast((shape, dtype), root=root)
		if (rank != root) : a = np.empty(int(np.prod(shape)), dtype)
		comm.Bcast(a, root=root)
		a = a.reshape(shape)
		if (which == 'matrix') : a = np.matrix(a)
		return a





	def _scalapy_block( self, array, comm, root ) : 
		'''
		array:
			2D array/matrix with any dtype
			array must be the "whole" array/matrix, gathered from all rank
	
		comm:
			must be <mpi4py.MPI.Intracomm>, otherwise use spla.pinv2(array)
	
		*** In root, array is the whole array, in other ranks, array can be anything, because will be Bcast() from root
		*** jizhipy.MPI.Pinv2() must be called in all ranks, NOT just root
	
	
		scalapy.core.initmpi(gridshape, block_shape)
			gridshape:
				devide input 2D matrix "a" with this shape
				for example, a.shape = (7, 5)
					gridshape = (3,1) : 
						devide a into [(3,5), (2,5), (2,5)]
					gridshape = (1,3) : 
						devide a into [(7,2), (7,1), (7,2)]
		'''
		import numpy as np
		from jizhipy.Basic import IsType, Raise
		from jizhipy.Math import DecomInt
		if (comm is None) : 
			from mpi4py import MPI as Mpi
			comm = Mpi.COMM_WORLD
		rank, Nprocess, host, info = self.Comm(comm)
		if (rank == root) : 
			ismatrix = IsType.ismatrix(array)
			array = np.array(array)
			shape = array.shape
		else : shape = None
		shape = self.Bcast(shape, comm, root)
		if (len(shape) != 2) : 
			if (rank == root) : Raise(Exception, 'array.shape='+str(shape)+' != 2D')
			else : exit()
		#----------------------------------------
		if (rank == root) : 
			# set gridshape
			shape = array.shape
			n = DecomInt(Nprocess)
			k = abs(n[1]/n[0] - max(shape)/min(shape))
			k = np.where(k==k.min())[0][0]
			gridshape = tuple(n[:,k])
			if(shape[0]>=shape[1]): gridshape =gridshape[::-1]
			#----------------------------------------
			# set block_shape
			n=min(shape[0]/gridshape[0],shape[1]/gridshape[1])
			if (n != 1) : n /= 2 
			if (n > 32) : n = 32
			block_shape = (n, n)
		else : gridshape, block_shape = None, None
		gridshape = self.Bcast(gridshape, comm, root)
		block_shape = self.Bcast(block_shape, comm, root)
		if (rank == root) : array = np.asfortranarray(array)
		else : array = None
		return [gridshape, block_shape, array]



	def _ArrAmp( self, array, comm, root, indexlim=4 ) : 
		'''
		scale array to 1e-4 to 1e+4
		'''
		rank, Nprocess, host, info = self.Comm(comm)
		if (rank == root) : 
			import numpy as np
			array = np.array(array)
			vmax = abs(array).max()
			n = int(np.log10(vmax))
			if (n > indexlim) : 
				scale = 10.**(n-indexlim)
				array = array / scale
			else : scale = 1.
			return [array, scale]
		else : return [None, None]



	def Pinv2( self, array, comm, root ) : 
		'''
		rank==root: return pinv2(array)
		other ranks: return None

		use scalapack to perform pinv2() 
		array:
			the large/whole array in root
		'''
	#	array, scale = self._ArrAmp(array, comm, root)
		scale = 1
		if (comm is None) : 
			import scipy.linalg as spla
			Apinv = spla.pinv2(array)
			return Apinv*scale
		import scalapy
		rank, Nprocess, host, info = self.Comm(comm)
		gridshape, block_shape, array = self._scalapy_block(array, comm, root)
		scalapy.core.initmpi(gridshape, block_shape)  
		A = scalapy.core.DistributedMatrix.from_global_array(array, rank=root)
		Apinv = scalapy.routines.pinv2(A)
		Apinv = Apinv.to_global_array(rank=root)
		if (rank == root) : return Apinv*scale
		else : return





	def Inv( self, array, comm, root ) : 
		'''
		use scalapack to perform inv() 
		array:
			the large/whole array in root

		ipiv in LU(): piv=pivoting
		'''
	#	array, scale = self._ArrAmp(array, comm, root)
		scale = 1
		if (comm is None) : 
			import scipy.linalg as spla
			Ainv = spla.inv(array)
			return Ainv*scale
		import scalapy
		rank, Nprocess, host, info = self.Comm(comm)
		gridshape, block_shape, array = self._scalapy_block(array, comm, root)
		scalapy.core.initmpi(gridshape, block_shape)  
		A = scalapy.core.DistributedMatrix.from_global_array(array, rank=root)
		Ainv = scalapy.routines.inv(A)[0]
		Ainv = Ainv.to_global_array(rank=root)
		if (rank == root) : return Ainv*scale
		else : return





	def Svd( self, array, comm, root ) : 
		'''
		use scalapack to perform svd() 
		array:
			the large/whole array in root
		'''
		if (comm is None) : 
			import scipy.linalg as spla
			return spla.svd(array)
		import scalapy
		rank, Nprocess, host, info = self.Comm(comm)
		gridshape, block_shape, array = self._scalapy_block(array, comm, root)
		scalapy.core.initmpi(gridshape, block_shape)  
		A = scalapy.core.DistributedMatrix.from_global_array(array, rank=root)
		u, s, vt = scalapy.routines.svd(A)
		u  =  u.to_global_array(rank=root)
		vt = vt.to_global_array(rank=root)
		if (rank == root) : return (u, s, vt)
		else : return (None, None, None)





	def Eigh( self, array, comm, root ) : 
		'''
		use scalapack to perform eigh() 
		array:
			the large/whole array in root
		'''
		if (comm is None) : 
			import scipy.linalg as spla
			return spla.eig(array)
		import scalapy
		rank, Nprocess, host, info = self.Comm(comm)
		gridshape, block_shape, array = self._scalapy_block(array, comm, root)
		scalapy.core.initmpi(gridshape, block_shape)  
		A = scalapy.core.DistributedMatrix.from_global_array(array, rank=root)
		s, v = scalapy.routines.svd(A)
		v = v.to_global_array(rank=root)
		if (rank == root) : return (s, v)
		else : return (None, None)





	def Covariance( self, a, b, comm, root ) : 
		'''
		Covariance( a, None, comm, root, mean ):
			<a a.T>
		Covariance( a, b, comm, root, mean ):
			<a b>

		Covariance(a, b, comm, root, mean=False)
			== MatrixDot(a, b, comm, root)

		mean:
			True: return <a a.T> = (a a.T)/N
			False: return (a a.T)
		'''
		from jizhipy.Basic import Raise
		rank, Nprocess, host, info = self.Comm(comm)
		if (rank == root) : 
			import numpy as np
			a = np.array(a)
			if (b is None) : 
				if (len(a.shape) == 1) : a = a[:,None]
				b = a.T
			b = np.array(b)
			if (len(a.shape)!=2 or len(b.shape)!=2) : Raise(Exception, 'a.shape='+str(a.shape)+', b.shape='+str(b.shape)+', NOT 2D')
			if (a.shape[1] != b.shape[0]) : Raise(Exception, 'a.shape='+str(a.shape)+', b.shape='+str(b.shape)+', can NOT MatrixDot')
			N = a.shape[1]
			a = a / N**0.5
			b = b / N**0.5
		else : a, b = None, None
		return self.MatrixDot(a, b, comm, root)










MPI = MPI()










if (__name__ == '__main__') : 
	#! /usr/bin/env python
	'''
	nohup runmpi fat,32 node2,16 node3,16 node4,16 node5,16 node6,16 python pinv2test.py --log &
	'''
	import numpy as np
	from jizhipy.Process.Log import *
	import jizhipy as jp
	islaohu = jp.IsType.islaohu()
	
	# Use MPI
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	root = 0
	rank, Nprocess, host, info = jp.MPI.Comm(comm)
	verbose = True if(rank==root)else False
	
	for n in range(1000, 30001, 1000) : 
	#for n in range(1000, 2001, 1000) : 
		if (rank == root) : 
			a = np.random.random((n,n))+1j*np.random.random((n,n))
			if (n == 1000) : print()
			print(jp.Time(1))
			print(a.shape, (' %.3f GB' % (a.size*2/1e8*0.8)))
		else : a = None
		jp.MPI.Pinv2(a, comm, root)
		if (verbose) : 
			print(jp.Time(1))
			if (n != 30000) : print()
	





# (1) 1D, same length
#a = (np.random.random(3)*100).astype(int)
# (2) 1D, different length, without 0
#a = (np.random.random(rank+2)*100).astype(int)
# (3) 1D, different length, with 0
#if (rank == 1) : a = np.array([], int)
#else : a = (np.random.random(rank+2)*100).astype(int)

# (4) 2D, same shape
#a = (np.random.random((1,4))*100).astype(int)
# (4) 2D, different shape, without 0
#a = (np.random.random((rank+1,4))*100).astype(int)
# (4) 2D, different shape, with 0
#if (rank == 1) : a = np.zeros((0,4), int)
#else : a = (np.random.random((rank+1,4))*100).astype(int)




#	def Pinv2( self, array, comm, root ) : 
#		'''
#		array:
#			2D array/matrix with any dtype
#			array must be the "whole" array/matrix, gathered from all rank
#	
#		comm:
#			must be <mpi4py.MPI.Intracomm>, otherwise use spla.pinv2(array)
#	
#		*** In root, array is the whole array, in other ranks, array can be anything, because will be Bcast() from root
#		*** jizhipy.MPI.Pinv2() must be called in all ranks, NOT just root
#	
#	
#		scalapy.core.initmpi(gridshape, block_shape)
#			gridshape:
#				devide input 2D matrix "a" with this shape
#				for example, a.shape = (7, 5)
#					gridshape = (3,1) : 
#						devide a into [(3,5), (2,5), (2,5)]
#					gridshape = (1,3) : 
#						devide a into [(7,2), (7,1), (7,2)]
#		'''
#		import scalapy
#		import numpy as np
#		from Funcs import Funcs
#		from Raise import Raise
#		from IsType import IsType
#		if (comm is None) : 
#			from mpi4py import MPI as Mpi
#			comm = Mpi.COMM_WORLD
#		rank, Nprocess, host, info = self.Comm(comm)
#		if (rank == root) : 
#			ismatrix = IsType.ismatrix(array)
#			array = np.array(array)
#			shape = array.shape
#		else : shape = None
#		shape = self.Bcast(shape, comm, root)
#		if (len(shape) != 2) : 
#			if (rank == root) : Raise(Exception, 'array.shape='+str(shape)+' != 2D')
#			else : exit()
#		#----------------------------------------
#		if (rank == root) : 
#			# set gridshape
#			shape = array.shape
#			n = Funcs.DecomInt(Nprocess)
#			k = abs(n[1]/n[0] - max(shape)/min(shape))
#			k = np.where(k==k.min())[0][0]
#			gridshape = tuple(n[:,k])
#			if(shape[0]>=shape[1]): gridshape =gridshape[::-1]
#			#----------------------------------------
#			# set block_shape
#			n=min(shape[0]/gridshape[0],shape[1]/gridshape[1])
#			if (n != 1) : n /= 2 
#			if (n > 32) : n = 32
#			block_shape = (n, n)
#		else : gridshape, block_shape = None, None
#		gridshape = self.Bcast(gridshape, comm, root)
#		block_shape = self.Bcast(block_shape, comm, root)
#		#----------------------------------------
#		scalapy.core.initmpi(gridshape, block_shape)  
#		if (rank == root) : array = np.asfortranarray(array)
#		else : array = None
#		A = scalapy.core.DistributedMatrix.from_global_array(array, rank=root)
#		Apinv = scalapy.routines.pinv2(A)
#		Apinv = Apinv.to_global_array()
#		if (rank == root) : return Apinv
#		else : return
#
