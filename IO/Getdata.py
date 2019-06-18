
def Getdata( dataaddress, n0=None, n1=None, n2=None ) : 
	'''
	dataaddress:
		(1) HDF5: 
			fo = h5py.File(path, 'r')
			dataaddress = fo['xxx']
		(2) FITS:
			fo = pyfits.open(path)
			dataaddress = fo[i].data

	n0, n1, n2:
		Can be  :,  single int,  int array with any size
	
	==================================================
	
	FITS file
	
	fo = pyfits.open(path)
	fo[0].data[n0,n1,n2]
	
	(1) NO limit to use ":", any axis can use ":" at any time, it won't raise "broadcast" error.
	
	(2) Use single int, fo[0].data[:,1,2], its shape=(1000,), 1D array
	
	(3) Use int array: fo[0].data[:,array1,array2]
		1. fo[0].data[:,np.array([1]),2], shape=(1000,1)
		2. fo[0].data[:,1,np.array([2])], shape=(1000,1), same shape as above
		3. fo[0].data[:,np.array([1]),np.array([2])], shape=(1000,1), same shape as above
	
	(4)
		1. fo[0].data[:,np.array([1,2]),np.array([3,4])], shape=(1000,2), NOT (1000,2,2)
		2. fo[0].data[np.array([1,2]),:,np.array([3,4])], shape=(2,:)
	
	(5) fo[0].data[:,np.array([1,2]),np.array([3,4,5])], raise ValueError: shape mismatch: objects cannot be broadcast to a single shape
	
	Rule to get data from FITS file:
		*   Concatenate all axis' indices to one array:
			fo[0].data[np.array([1,2,3]), :, np.array([4,5,6])]
				==> indices = np.array([[1,:,4],
				                        [2,:,5],
				                        [3,:,6]])
		** Then get [1,:,4], and [2,:,5], and [3,:,6], and combine them. Finally, we get shape=(1000,3), NOT (1000,3,3). NOTE THAT it won't broadcast !
	
		*** That's why (3.1), (3.2), (3.3) above have the same shape, and shape of (4) is (1000,2), and (5) has "broadcast" error => can't make an array with different size [np.array([1,2]), np.array([3,4,5])]
	
	==================================================
	
	HDF5 file
	
	fo = h5py.File(path, 'r')
	fo['vis']
	
	(1) NO limit to use ":", any axis can use ":" at any time.
	
	(2) Use single int, fo['vis'][:,1,2], its shape=(1000,), 1D array
	
	(3) Use int array: fo['vis'][:,array1,array2]
		1. fo['vis'][:,np.array([1]),2], shape=(1000,), 1D array, same as above
		2. fo['vis'][:,1,np.array([2])], shape=(1000,), same shape as above
		3. fo['vis'][:,np.array([1]),np.array([2])], shape=(1000,), same shape as above
	
	(4) fo['vis'][:,np.array([1,2]),np.array([3,4])], raise TypeError: Only one indexing vector or array is currently allowed for advanced selection
	
	Rule to get data from HDF5 file:
		*   NO limit to use ":"
		**  Use <= 1 int array. So (4) raises error, it uses two int arrays. FITS can use any number of int array as long as their size are the same
		*** Reduce all axes with size=1. So (3.1), (3.2), (3.3) are just 1D, are different from FITS
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	shapein, dtype = dataaddress.shape, dataaddress.dtype
	shapeout = ()
	n = [n0, n1, n2][:len(shapein)]
	for i in range(len(n)) : 
		if (n[i] is not None) : 
			n[i] = Asarray(n[i]).round().astype(int)
			shapeout += (n[i].size,)
			if (shapeout[i] == shapein[i]) : 
				if ((np.arange(shapein[i])-n[i]).sum() == 0) : 
					n[i] = None
		else : shapeout += (shapein[i],)
	#--------------------------------------------------
	if (n0 is None and n1 is None and n2 is None) : return dataaddress[:]
	#--------------------------------------------------
	if (n0 is None and n1 is None and n2 is not None) : 
		data = dataaddress[:,:,n2]
		if (len(data.shape) != 3) : data = data[:,:,None]
		return data
	#--------------------------------------------------
	if (n0 is None and n1 is not None and n2 is None) : 
		data = dataaddress[:,n1,:]
		if (len(data.shape) != 3) : data = data[:,None]
		return data
	#--------------------------------------------------
	if (n0 is None and n1 is not None and n2 is not None) : 
		data = np.zeros(shapeout, dtype)
		if (n1.size >= n2.size) : 
			for i in range(n2.size) : 
				try   : data[:,:,i] = dataaddress[:,n1,n2[i]]
				except: data[:,:,i] = dataaddress[:,n1,n2[i]][:,None]
			return data
		else : 
			for i in range(n1.size) : 
				try   : data[:,i,:] = dataaddress[:,n1[i],n2]
				except: data[:,i,:] = dataaddress[:,n1[i],n2][:,None]
			return data
	#--------------------------------------------------
	if (n0 is not None and n1 is None and n2 is None) : 
		data = dataaddress[n0,:,:]
		if (len(data.shape) != 3) : data = data[None,:]
		return data
	#--------------------------------------------------
	if (n0 is not None and n1 is not None and n2 is None) : 
		data = np.zeros(shapeout, dtype)
		if (n0.size >= n1.size) : 
			for i in range(n1.size) : 
				try   : data[:,i,:] = dataaddress[n0,n1[i],:]
				except: data[:,i,:] = dataaddress[n0,n1[i],:][None,:]
			return data
		else : 
			for i in range(n0.size) : 
				try   : data[i,:,:] = dataaddress[n0[i],n1,:]
				except: data[i,:,:] = dataaddress[n0[i],n1,:][None,:]
			return data
	#--------------------------------------------------
	if (n0 is not None and n1 is None and n2 is not None) : 
		data = np.zeros(shapeout, dtype)
		if (n0.size >= n2.size) : 
			for i in range(n2.size) : 
				try   : data[:,:,i] = dataaddress[n0,:,n2[i]]
				except: data[:,:,i] = dataaddress[n0,:,n2[i]][None,:]
			return data
		else : 
			for i in range(n0.size) : 
				try   : data[i,:,:] = dataaddress[n0[i],:,n2]
				except: data[i,:,:] = dataaddress[n0[i],:,n2][:,None]
			return data
	#--------------------------------------------------
	if (n0 is not None and n1 is not None and n2 is not None) : 
		data = np.zeros(shapeout, dtype)
		for i in range(n1.size) : 
			for j in range(n2.size) : 
				data[:,i,j] = dataaddress[n0,n1[i],n2[j]]
		return data



