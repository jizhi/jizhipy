
class Angle( object ) : 


	def Angle( self, cosa, sina=None ) : 
		'''
		return the phase
	
		(case 1) 
			cosa = A*np.cos(a)
			sina = A*np.sin(a)
			return a
	
		(case 2)
			cosa = np.complex  (real + 1j*imag)
			sina = None
		'''
		import numpy as np
		if (sina is not None) : cosa = cosa + 1j*sina
		return np.angle(cosa) % (2*np.pi)





	def AngleMean( self, angle1, angle2=None, axis=0 ) : 
		'''
		return: 
			angle.mean(axis)

		angle1:
			[rad]
			Any shape array

		angle2:
			(1) angle2=None:
					return the mean of angle1
			(2) Can broadcast: angle1-angle2
					return the mean of the difference of two angles (angle1-angle2)

		axis:
			average the angle along which axis
		'''
		import numpy as np
		from jizhipy.Array import Asarray
		from jizhipy.Basic import Raise
		angle, axis = npfmt(angle1), int(axis)
		if (axis < 0) : axis += len(angle.shape)
		if (axis > len(angle.shape)) : Raise(Exception, 'axis='+str(axis)+' out of angle.shape='+str(angle.shape))
		if (angle2 is not None) : 
			angle2 = npfmt(angle2)
			try : angle2 = npfmt(angle2) + 0*angle
			except: Raise(Exception,'angle1.shape='+str(angle.shape)+', angle2.shape='+str(angle2.shape)+', can NOT broadcast')
			angle = np.exp(1j*angle) * np.exp(-1j*angle2)
		else : angle = np.exp(1j*angle)
		angle = angle.mean(axis)
		angle = np.angle(angle)
		angle = angle % (2*np.pi)
		return angle





	def Unwrap( self, angle, tf=True, degree=False ) : 
		'''
		angle:
			[rad]
	
		tf:
			True | False | None
			(1) ==True, use np.unwrap()
			(2) ==False, do %(2*np.pi)
			(3) ==None, return input angle
	
		degree:
			angle and return are degree or not?
	
		return:
			[rad]
		'''
		import numpy as np
		if (angle is None) : return None
		if (tf is None) : return angle
		angle = np.array(angle)
		if (degree) : angle *= np.pi/180
		angle %= (2*np.pi)
		if (tf) : 
			try : angle = np.unwrap(angle)
			except : pass
		if (degree) : angle *= 180/np.pi
		return angle





Angle = Angle()
