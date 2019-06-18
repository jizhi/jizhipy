
def Quad( func, ranges, args=(), err=False ) : 
	'''
	integrate

	func:
		<type 'function'>
		def func(x, y, z, a, b, c) : 
			return (a*x + b*y**2) * z + c

	ranges:
		* Must list | tuple
		* len(ranges) == number of variables
		* ranges[i]:
			Must list | tuple and must len(ranges[i]) == 2
		* ranges = [first, second, third, ......] parameter of func from left to right
			Here first-x, second-y, third-z

	args:
		Must tuple, if len==1: args=(a,) => don't forget `,`
		arguments of the constants in func, from left to right
		Here args=(a,b,c)

	err:
		True | False
		Return the error the result or not?
	'''
	from scipy.integrate import dblquad
	from scipy.integrate import tplquad
	from scipy.integrate import nquad
	from jizhipy.Basic import IsType, Raise
	constranges = True
	for i in range(len(ranges)) : 
		if (IsType.isfunc(ranges[i][0])) : constranges = False
		if (IsType.isfunc(ranges[i][1])) : constranges = False
		if (not constranges) : break
	#--------------------------------------------------
	if (constranges) : res = nquad(func, ranges, args)
	#--------------------------------------------------
	elif (len(ranges) == 2) : 
		x1, x2 = ranges[0]
		if (not IsType.isfunc(x1)) : 
			def xlow(y) : return x1
		else : xlow = x1
		if (not IsType.isfunc(x2)) : 
			def xup(y) : return x2
		else : xup = x2
		res = dblquad(func, ranges[1][0], ranges[1][1], xlow, xup, args)
	#--------------------------------------------------
	elif (len(ranges) == 3) : 
		y1, y2 = ranges[1]
		if (not IsType.isfunc(y1)) : 
			def ylow(z) : return y1
		else : ylow = y1
		if (not IsType.isfunc(y2)) : 
			def yup(z) : return y2
		else : yup = y2
		#--------------------
		x1, x2 = ranges[0]
		if (not IsType.isfunc(x1)) : 
			def xlow(y,z) : return x1
		else : xlow = x1
		if (not IsType.isfunc(x2)) : 
			def xup(y,z) : return x2
		else : xup = x2
		res = tplquad(func, ranges[2][0], ranges[2][1], ylow, yup, xlow, xup)
	else : Raise(Exception, 'constranges = False and len(ranges) = '+str(len(ranges))+' > 3')
	#--------------------------------------------------
	if (err) : return res
	else : return res[0]


