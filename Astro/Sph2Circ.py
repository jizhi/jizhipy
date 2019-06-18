
def Sph2Circ( angle, Dec ) : 
	'''
	Spherical2CircularAngle()
	Convert between spherical and circular angles

	angle:
		scale of an angle between two points

	All angles are in rad
	'''
	import numpy as np
#	dang = np.pi/2-Dec - angle/2.
#	if (abs(dang) < 1e-6) : return np.pi
#	elif (dang < 0) : 
#		raise Exception('Maximum Sph at Dec='+('%.3f' % (Dec*180/np.pi))+'deg is 2*(90-Dec)='+('%.3f' % (2*(90-Dec*180/np.pi)))+'deg, now angle='+('%.3f' % (angle*180/np.pi))+'deg')
#	else : 
#		return 2*np.arcsin( np.sin(angle/2.) /np.cos(Dec) )
	return np.arcsin( np.sin(angle) / np.cos(Dec) )



def Circ2Sph( angle, Dec ) : 
	import numpy as np
#	return 2*np.arcsin( np.sin(angle/2.) *np.cos(Dec) )
	return np.arcsin( np.sin(angle) * np.cos(Dec) )


