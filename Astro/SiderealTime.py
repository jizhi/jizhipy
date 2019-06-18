
def SiderealTime( lon, RA=None, date='' ) : 
	'''
	Return the local sidereal time
	sidereal time : RA that lon is pointing to now !

	lon:
		Must in unit "degree". Both str and float are OK.
		Must be one, not list
		longitude of the local geography of the observer.

	RA:
		Must in unit "degree". Both str and float are OK.
		Can be one or list (must be 1D, shape=(N,))
		RA of the target (source/object).
		If given, also return when/after how long time will this target/source reaches the local merdian.
		If not given, just return the sidereal time.

	date:
		local time (NOT UTC) at this longitude/timezone.
		Must be str, date=jp.Time(0)
		Format of the date must be "2015/3/12 10:55:00"  (ephem)

	return:
		Return the sidereal time.
		(1) SiderealTime(lon, date=): return LAST
		(2) SiderealTime(lon, RA, date=): return [LAST, aftertime]
	'''
	import ephem
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import Time, IsType
	lon = str(lon)
	if (date is '' or date is None) : date = Time(0) # Now
	here = ephem.Observer()
	here.lon = lon
	here.date = date
	last = here.sidereal_time()
	if (RA is None) : return last
	else : 
		if (IsType.isnum(RA) or IsType.isstr(RA)) : islist = False
		else : islist = True
		RA = npfmt(RA, float) *np.pi/180
		after = (RA - last) % (2*np.pi)  # rad
		after = after.flatten()
		afterh = []
		for a in after : afterh.append( ephem.hours(a) )
		if (not islist) : afterh = afterh[0]
		return [last, afterh]
