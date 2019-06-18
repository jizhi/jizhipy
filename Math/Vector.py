
class Vector(object):

	def Angle(self, aStart, aEnd, bStart=None, bEnd=None):
		'''
		(1) Return the angle between two vectors
		(2) Return the angle between two angles

		Parameters
		----------
		(1) if the returned angle is the angle between two vectors, then aStart, aEnd, bStart, bEnd are not None !!!
			aStart: The start point(coordinate) of vector a
			aEnd  : The end point of vector a
			     => aEend-aStart is the aVector

			bStart, bEnd: the same as aVector

			aStart, aEnd, bStart, bEnd:
				can be any shape, but should can broadcast !!!
				the last axis is coordinate:
					e.g. a.shape=(3,4,5)
					then for each point, coordinate is a[i,j,:]

		(2) if the returned angle is the angle between two angles, then bStart=bEnd=None !!!
			aStart, aEnd:
				two angles in [rad] !!!


		Returns
		----------
		return angle in [rad]

		angle:
			.shape = aStart.shape[:-1]
		'''
		import numpy as np
		aStart, aEnd = np.array(aStart), np.array(aEnd)
		if (bStart is not None and bEnd is not None): 
			bStart, bEnd = np.array(bStart), np.array(bEnd)
		else:
			ang1, ang2 = aStart, aEnd
			aStart = bStart = 0
			aEnd = np.array([np.cos(ang1), np.sin(ang1)])
			axis = list(range(1, len(aEnd.shape)))+[0]
			aEnd = np.transpose(aEnd, axis)
			bEnd = np.array([np.cos(ang2), np.sin(ang2)])
			axis = list(range(1, len(bEnd.shape)))+[0]
			bEnd = np.transpose(bEnd, axis)
		a, b = aEnd-aStart, bEnd-bStart
		na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
		nab, ab = na*nb, 1.*(a*b).sum(-1)
		ab[nab==0] = 0
		nab[nab==0] = 1
		ang = np.arccos(ab / nab)
		return ang





	def Distance(self, aPoint, bStart, bEnd):
		'''
		Return the vertical distance from aPoint to bVector

		Parameters
		----------
		aPoint:
			The coordinate of this point

		bStart, bEnd:
			See jizhipy.Vector.Angle

		aPoint, bStart, bEnd:
			can be any shape, but should can broadcast !!!
			the last axis is coordinate:
				e.g. a.shape=(3,4,5)
				then for each point, coordinate is a[i,j,:]


		Returns
		----------
		return angle in rad

		angle:
			.shape = aStart.shape[:-1]
		'''
		import numpy as np
		a = aPoint-bStart  # bStart as the original point
		ang = self.Angle(bStart, aPoint, bStart, bEnd)
		d = np.linalg.norm(a, axis=-1)*np.sin(ang)
		return d





Vector = Vector()

