
def _ArrayAxis( array, axis1, axis2, act='move' ) : 
	'''
	array:
		Any dimension array (real, complex, masked)

	axis1, axis2, act:
	  act='move' : old a[axis1] becomes new b[axis2]
		a.shape=(2,3,4,5,6)
		b = ArrayAxis(a, 3, 1, 'move')
		b.shape=(2,5,3,4,6)
	  act='exchange' : exchange a[axis1] with a[axis2]
		a.shape=(2,3,4,5,6)
		b = ArrayAxis(a, 3, 1, 'exchange')
		b.shape=(2,5,4,3,6)
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise, IsType
	ismatrix = IsType.ismatrix(array)
	array = Asarray(array)
	shapeo = array.shape
	if (len(shapeo) <= 1) : return array
	if (axis1 < 0) : axis1 = len(shapeo) + axis1
	if (axis2 < 0) : axis2 = len(shapeo) + axis2
	if (axis1 == axis2) : return array
	if (axis1>=len(shapeo) or axis2>=len(shapeo)) : Raise(Exception, 'axis1='+str(axis1)+', axis2='+str(axis2)+' out of array.shape='+str(shapeo)+'=>'+str(len(shapeo))+'D')
	if (len(shapeo) == 2) : return array.T
	if (len(shapeo)==3 and ((axis1==0 and axis2==2) or (axis1==2 and axis2==0)) and act.lower()=='exchange') : return array.T
	#--------------------------------------------------
	def __Move( array, axis1, axis2 ) : 
		''' My old function, insteaded by np.moveaxis() '''
		shape = list(array.shape)
		s1, s2 = shape[axis1], shape[axis2]
		shape.pop(axis1)
		shape = shape[:axis2] + [s1] + shape[axis2:]
		a = np.zeros(shape, array.dtype)
		for i in range(s1) : 
			if (axis1 == 0) : 
				if  (axis2==1): a[:,i]        =array[i]
				elif(axis2==2): a[:,:,i]      =array[i]
				elif(axis2==3): a[:,:,:,i]    =array[i]
				elif(axis2==4): a[:,:,:,:,i]  =array[i]
				elif(axis2==5): a[:,:,:,:,:,i]=array[i]
			elif (axis1 == 1) :
				if  (axis2==0): a[i]          =array[:,i]
				elif(axis2==2): a[:,:,i]      =array[:,i]
				elif(axis2==3): a[:,:,:,i]    =array[:,i]
				elif(axis2==4): a[:,:,:,:,i]  =array[:,i]
				elif(axis2==5): a[:,:,:,:,:,i]=array[:,i]
			elif (axis1 == 2) :
				if  (axis2==0): a[i]          =array[:,:,i]
				elif(axis2==1): a[:,i]        =array[:,:,i]
				elif(axis2==3): a[:,:,:,i]    =array[:,:,i]
				elif(axis2==4): a[:,:,:,:,i]  =array[:,:,i]
				elif(axis2==5): a[:,:,:,:,:,i]=array[:,:,i]
			elif (axis1 == 3) :
				if  (axis2==0): a[i]          =array[:,:,:,i]
				elif(axis2==1): a[:,i]        =array[:,:,:,i]
				elif(axis2==2): a[:,:,i]      =array[:,:,:,i]
				elif(axis2==4): a[:,:,:,:,i]  =array[:,:,:,i]
				elif(axis2==5): a[:,:,:,:,:,i]=array[:,:,:,i]
			elif (axis1 == 4) :
				if  (axis2==0):a[i]          =array[:,:,:,:,i]
				elif(axis2==1):a[:,i]        =array[:,:,:,:,i]
				elif(axis2==2):a[:,:,i]      =array[:,:,:,:,i]
				elif(axis2==3):a[:,:,:,i]    =array[:,:,:,:,i]
				elif(axis2==5):a[:,:,:,:,:,i]=array[:,:,:,:,i]
			elif (axis1 == 5) :
				if  (axis2==0):a[i]        =array[:,:,:,:,:,i]
				elif(axis2==1):a[:,i]      =array[:,:,:,:,:,i]
				elif(axis2==2):a[:,:,i]    =array[:,:,:,:,:,i]
				elif(axis2==3):a[:,:,:,i]  =array[:,:,:,:,:,i]
				elif(axis2==4):a[:,:,:,:,i]=array[:,:,:,:,:,i]
			else : Raise(Exception, 'ArrayAxis() can just handel 6-D array. For >= 7-D array, you can modify this function by yourself.')
		return a

	def _Move( array, axis1, axis2 ) : 
		return np.moveaxis(array, axis1, axis2)
	#--------------------------------------------------
	array = _Move(array, axis1, axis2)
	if (abs(axis1-axis2)!=1 and act.lower()=='exchange') : 
		if (axis1 < axis2) : axis1, axis2 = [axis2-1, axis1]
		else : axis1, axis2 = [axis2+1, axis1]
		array = _Move(array, axis1, axis2)
	if (ismatrix) : array = np.matrix(array)
	return array





def ArrayAxis( array, axis1, axis2, act='move' ) : 
	'''
	array:
		Any dimension array (real, complex, masked)

	axis1, axis2, act:
	  act='move': old a[axis1] becomes new b[axis2]
		a.shape=(2,3,4,5,6)
		b = ArrayAxis(a, 3, 1, 'move')
		b.shape=(2,5,3,4,6)
	  act='exchange' : exchange a[axis1] with a[axis2]
		a.shape=(2,3,4,5,6)
		b = ArrayAxis(a, 3, 1, 'exchange')
		b.shape=(2,5,4,3,6)
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise, IsType
	ismatrix = IsType.ismatrix(array)
	array = Asarray(array)
	shapeo = array.shape
	if (len(shapeo) <= 1) : return array
	if (axis1 < 0) : axis1 = len(shapeo) + axis1
	if (axis2 < 0) : axis2 = len(shapeo) + axis2
	if (axis1 == axis2) : return array
	if (axis1>=len(shapeo) or axis2>=len(shapeo)) : Raise(Exception, 'axis1='+str(axis1)+', axis2='+str(axis2)+' out of array.shape='+str(shapeo)+'=>'+str(len(shapeo))+'D')
	if (len(shapeo) == 2) : return array.T
	if (len(shapeo)==3 and ((axis1==0 and axis2==2) or (axis1==2 and axis2==0)) and act.lower()=='exchange') : return array.T
	#--------------------------------------------------
	axis = list(range(len(shapeo)))
	if (act.lower() =='move'): 
		axis.remove(axis1)
		axis = axis[:axis2]+[axis1]+axis[axis2:]
	elif (act.lower() =='exchange'): 
		axis[axis1] = axis2
		axis[axis2] = axis1
	array = np.transpose(array, axis)
	if (ismatrix) : array = np.matrix(array)
	return array





def Transpose(array, axes=None):
	'''
	== np.transpose(array, axes)
	'''
	import numpy as np
	return np.transpose(array, axes)





def ArrayRotate( array, rot ) : 
	'''
	array:
		np.array | np.matrix

	rot:
		(1) degree
			rotate anti-clock, like phi angle
			can be one or 1D list
		(2) True : flip horizontal (left-right)
			False: flip vertical (up-down)
		(3) None: rotate/flip randomly, maybe rotate any angle, maybe flip
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Raise, IsType
	ismatrix = IsType.ismatrix(array)
	array = np.array(array)
	if (len(array.shape) == 1) : 
		if (ismatrix) : return np.matrix(array)
		else : return array
	if (len(array.shape) >= 3) : Raise(Exception, 'jizhipy.ArrayRotate() can just accept 2D array/matrix now, but input array.shape='+str(array.shape))
	#----------------------------------------
	isnum = False
	if (rot is None) : 
		n = int(np.random.random()*4.99)
		rot = [[True, False, 90, 180, 270][n]]
		isnum = True
	elif (rot is True or rot is False) : rot = [rot]
	else : 
		isnum = IsType.isnum(rot)
		rot = Asarray(rot).flatten().round().astype(int) % 360
		for i in range(len(rot)) : 
			if (rot[i] not in [0,90,180,270]): jp.Raise(Exception, 'jizhipy.ArrayRotate() can just accept rot in [0, 90, 180, 270, True, False, None] now, but rot['+str(i)+']='+str(rot[i]))
	#----------------------------------------
	def rotate( a, r ) : 
		if   (r is  True) : return a[:,::-1]
		elif (r is False) : return a[::-1]
		elif (r ==   0)   : return a
		elif (r ==  90)   : return a.T[::-1]
		elif (r == 180)   : return a[::-1][:,::-1]
		elif (r == 270)   : return a.T[:,::-1]
	#----------------------------------------
	b = []
	for i in range(len(rot)) : 
		b.append( rotate(array, rot[i]) )
	#----------------------------------------
	if (ismatrix) : 
		for i in range(len(b)) : b[i] = np.matrix(b[i])
	if (isnum) : b = b[0]
	if (ismatrix) : return b
	else : return np.array(b)

