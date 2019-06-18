
def _Time( a, b=None, c=None ) : 
	'''
	sec1970:
		second starts from '1970/01/01 00:00:00'
		It is an unity time all over the world (all timezones). No matter what timezone is, time.time() are the same !
	ephemtime:
		different "labels" of sec1970 at different timezones

	The time here is basing on the timezone, don't take local-law into account (Daylight Saving _Time)!

	(1) Return local/system timezone (int)
	** _Time('timezone')
	timezone >=0:east, <0:west

	(2) Convert ephemtime to sec1970 with timezone
	** _Time(ephemtime, timezone)
	** _Time('2016/10/25 16:39:24.68', 8)

	(3) Convert ephemtime from timezonein to timezoneout
	** _Time(ephemtime, timezonein, timezoneout)
	** _Time('2016/10/25 16:39:24.68', 8, 5)

	(4) Convert sec1970 to ephemtime with timezone
	** _Time(sec1970, timezone)
	** _Time(1477411200.729252, 8)

	(5) Calculate time interval between two ephemtimes
	** _Time(ephemtime1, ephemtime2)
	** _Time('2016/10/25 16:39:24.68', '2016/11/25 18:49:54.37')
	return (ephemtime2 - ephemtime1)

	(6) Return now-time with timezone WITH/WITHOUT Daylight Saving _Time
	a in [0,1,2], b=timezone/None
	c=None/'daylight': use daylight, c=False: don't use daylight
	** _Time(0/1/2, timezone, c)
	'''
	import time
	from jizhipy.Basic import IsType
	#--------------------------------------------------
	def _Timezone( a, b ) : 
		'''
		Return local/system timezone (int)
		_Time('timezone')
		>=0:east, <0:west
		'''
		return -time.timezone/3600
	#--------------------------------------------------
	def ephemTOsec1970( a, b ) : 
		'''
		ephemtime to sec1970 with timezone
		_Time(ephemtime, timezone),  timezone can =None
		_Time('2016/10/25 16:39:24.68', 8)
		_Time('2016/10/25 PM 04:39:24.68', 8)
		_Time('2016.10.25.PM.04.39.24.68', 8)
		'''
		if (b is None) : b, offset = _Timezone('timezone',None), 0
		else : offset = b*3600 + time.timezone  # second
		# Calibrate the timezone without local-law (Daylight Saving _Time)
		offset += time.mktime(time.strptime('1970/01/01 00:00:00', '%Y/%m/%d %H:%M:%S')) + _Timezone('timezone',None)*3600
		n = None
		if (len(a.split('.'))>=7):
			a2 = a.split('M.')[1].split('.')
			if (len(a2)==3): n = -1
		if (n is None): n = a.rfind('.')
		if (n < 0) : dot = 0
		else : dot, a = float(a[n:]), a[:n]

		try : sec1970 = time.mktime(time.strptime(a, '%Y/%m/%d %H:%M:%S')) + dot - offset
		except : 
			try: sec1970 = time.mktime(time.strptime(a, '%Y/%m/%d %p %I:%M:%S')) + dot - offset
			except: sec1970 = time.mktime(time.strptime(a, '%Y.%m.%d.%p.%I.%M.%S')) + dot - offset
		return sec1970
	#--------------------------------------------------
	def Interval( a, b, c ) : 
		'''
		time interval between two ephemtimes
		_Time(ephemtime1, ephemtime2)
		ephemtime2 - ephemtime1
		_Time('2016/10/25 16:39:24.68', '2016/11/25 18:49:54.37')
		c:
			c=0: return  02:13:45  format
			c=1: return second
		'''
		sec19701 = ephemTOsec1970(a, None)
		sec19702 = ephemTOsec1970(b, None)
		if (c==1): return sec19702-sec19701
		d = (sec19702 - sec19701)/3600.  # hour
		h, d = int(d), (d-int(d))*60  # min
		m, s = int(d), abs(d-int(d))*60  # sec
		h = str(h) if(h>=10)else '0'+str(h)
		m = str(m) if(m>=10)else '0'+str(m)
		s = ('%.1f' % s) if(s>=10)else ('0%.1f' % s)
		hms = h+':'+m+':'+s
		return hms
	#--------------------------------------------------
	def sec1970TOephem( a, b, c=0 ) : 
		'''
		sec1970 to ephemtime with timezone
		_Time(sec1970, timezone),  timezone can =None
		_Time(1477411200.729252, 8)
		c=0: 2019/02/15 21:54:29
		c=1: 2019/02/15 PM 09:54:29
		c=2: 2019.02.15.PM.09.54.29
		'''
		if (b is None) : b, offset = _Timezone('timezone',None), 0
		else : offset = b*3600 + time.timezone  # second
		offset += time.mktime(time.strptime('1970/01/01 00:00:00', '%Y/%m/%d %H:%M:%S')) + _Timezone('timezone',None)*3600
		dot, a = a-int(a), int(a)
		if (c is None or c==0) : ephemtime = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(a+offset)) + str(dot)[1:3]
		elif (c==1) : ephemtime = time.strftime('%Y/%m/%d %p %I:%M:%S', time.localtime(a+offset)) + str(dot)[1:3]
		elif (c==2) : ephemtime = time.strftime('%Y.%m.%d.%p.%I.%M.%S', time.localtime(a+offset)) + str(dot)[1:3]
		return ephemtime
	#--------------------------------------------------
	def Nowtime( a, b, c ) : 
		'''
		Return now-time with timezone
		a in [0,1,2], b=timezone/None
		WITH 

		timestr = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
		'''
		if (b is None) : b, offset = _Timezone('timezone',None), 0
		else : offset = b*3600 + time.timezone  # second
		if (c is False) : offset += time.mktime(time.strptime('1970/01/01 00:00:00', '%Y/%m/%d %H:%M:%S')) + _Timezone('timezone',None)*3600
		sec1970 = time.time()+offset
		dot = ('%.4f' % (sec1970-int(sec1970)))[1:]
		localtime = time.localtime(sec1970)
		a = int(a)
		if   (a==0) : fmt, dot='%Y/%m/%d %H:%M:%S', dot[:2]
		elif (a==1) : fmt, dot='%Y/%m/%d %p %I:%M:%S', dot[:2]
		elif (a==2) : fmt, dot='%Y.%m.%d.%p.%I.%M.%S', dot[:2]
		if   (a==3) : fmt, dot='%Y%m%d%H%M%S', dot[1:]
		elif (a==4) : fmt, dot='%Y-%m-%d', ''  # line:153
		timestr = time.strftime(fmt, localtime)+dot
		return timestr
	#--------------------------------------------------
	# (1) Return local/system timezone (int)
	# ** _Time('timezone')
	# timezone >=0:east, <0:west
	if (str(a).lower() == 'timezone') : 
		return _Timezone(a, b)
	#--------------------------------------------------
	elif (IsType.isstr(a) and (b is None or IsType.isint(b))) : 
	# (2) Convert ephemtime to sec1970 with timezone
	# ** _Time(ephemtime, timezone)
	# ** _Time('2016/10/25 16:39:24.68', 8)
		if (c is None) : return ephemTOsec1970(a, b)
	# (3) Convert ephemtime from timezonein to timezoneout
	# ** _Time(ephemtime, timezonein, timezoneout)
	# ** _Time('2016/10/25 16:39:24.68', 8, 5)
		else : return sec1970TOephem( ephemTOsec1970(a, b), c )
	#--------------------------------------------------
	# (5) Calculate time interval between two ephemtimes
	# ** _Time(ephemtime1, ephemtime2)
	# ** _Time('2016/10/25 16:39:24.6','2016/11/25 18:49:54.3')
	# return (ephemtime2 - ephemtime1)
	elif (IsType.isstr(a) and IsType.isstr(b)) : 
		return Interval(a, b, c)
	#--------------------------------------------------
	# (4) Convert sec1970 to ephemtime with timezone
	# ** _Time(sec1970, timezone)
	# ** _Time(1477411200.729252, 8)
	elif (IsType.isfloat(a) or a > 3) : 
		return sec1970TOephem(a, b, c)
	#--------------------------------------------------
	#  (6) Return now-time with timezone WITH/WITHOUT Daylight Saving _Time
	#  a in [0,1,2,3,4,5], b=timezone/None
	#  c = None/'daylight': use daylight, c=False: don't use daylight
	# ** _Time(0/1/2/3/4/5, timezone, c)
	# Time(0): 2019/02/23 15:32:09.3
	# Time(1): 2019/02/23 PM 03:32:09.3
	# Time(2): 2019.02.23.PM.03.32.09.3
	# Time(3): 2019022315320937
	# Time(4): 2019-02-23
	# Time(5): 1970/01/01 AM 12:00:04
	else : return Nowtime(a, b, c)





def Time( current=None, ephemtime=None, sec1970=None, timezone=None, which=None, datelist=None ) : 
	'''
	(1) Return local/system timezone (int)
		** Time( timezone=True )
		timezone >=0:east, <0:west

	(2) Convert ephemtime to sec1970 with timezone
		** Time(ephemtime='2016/10/25 16:39:24.68', timezone=8)

	(3) Convert ephemtime from timezonein to timezoneout
		** Time( ephemtime='2016/10/25 16:39:24.68', timezone=[8, 5] )

	(4) Convert sec1970 to ephemtime with timezone
		** Time( sec1970=1477411200.729252, timezone=8, which=)

	(5) Calculate time interval between two ephemtimes
		** Time( ephemtime=['2016/10/25 16:39:24.68', '2016/11/25 18:49:54.37'], which=0/1 )
			which=0: return h:m:s     which=1: return second
		return (ephemtime2 - ephemtime1)

	(6) Return current-time with timezone WITH/WITHOUT Daylight Saving Time
		** Time( current=0/1/2/3/4/5, timezone=8, daylight=True/False )

	(7) Date list ['2010-10-01', '2010-11-01', ...]
		** Time( datelist=['2010-10-01', how, length] )
		how: 2 case: '+3d'/'-3d' | '+3D'/'-3D'
			'+3d': add 3 days
			'+3D': each months' date 3: 2018-01-03, 2018-02-03, 2018-03-03, 2018-04-03, ...
			** Similar: 
				'+3m', '-3m': month
				'+3y', '-3y': year
				'+3M', '-3M': each years' March
		how must be str, not list, default '+1d'
		length: int, the length of return list
		** Time( datelist=['2004-02-25', '+1d', 10] )
		** Time( datelist=['2004-02-25', '+1D', 10] )

	(8) Time() = time.time()
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import IsType
	# (1) Time(timezone=True)
	if (timezone is True) : return _Time('timezone')
	#----------------------------------------
	elif (IsType.isstr(ephemtime)) : 
		# (2) Time(ephemtime='2019/02/15 18:54:29.7')
		if (timezone is None) : return _Time(ephemtime)
		timezone = Asarray(timezone, int).flatten()
		if (timezone.size == 1) : return _Time(ephemtime, timezone[0])
		#----------------------------------------
		# (3) Time(ephemtime='2019/02/15 18:54:29.7', timezone=[0,8])
		else: return _Time(ephemtime, timezone[0],timezone[1])
	#----------------------------------------
	# (4) Time(sec1970=1550228069.8, timezone=?, which=0/1)
#	elif (sec1970 is not None and IsType.isnum(timezone)) : return _Time(sec1970, timezone, which)
	elif (sec1970 is not None) : return _Time(sec1970, timezone, which)
	#----------------------------------------
	# (5)
	elif (ephemtime is not None) : return _Time(ephemtime[0], ephemtime[1], which)
	#----------------------------------------
	# (6)
	elif (IsType.isnum(current)) : 
		daylight = False if(which is True)else True
		return _Time(current, timezone, daylight)
	#----------------------------------------
	# (7)
	elif (datelist is not None) : 
		date, how, N = datelist[:3]
		date, how, N = str(date), str(how), int(N)
		datein = date
		if ('/' in date) : s, date = '/', date.split('/')
		elif ('-' in date) : s, date = '-', date.split('-')
		#----------------------------------------
		def Case1( date, how, N ) : 
			datelist = []
			if ('y' in how) : 
				year = int(date[0]) +np.arange(1, N) *int(how[:-1])
				for y in year : datelist.append( str(y)+s+date[1]+s+date[2] )
			elif ('m' in how) : 
				m = int(date[1])+np.arange(1,N)*int(how[:-1])
				year = m / 12
				month = m % 12
				tf = month==0
				month[tf] = 12
				year[tf] -= 1
				for i in range(len(m)) : 
					m = str(month[i])
					if (len(m) == 1) : m = '0'+m
					datelist.append( str(int(date[0])+year[i])+s+m+s+date[2] )
			elif ('d' in how) : 
				ephemtime = date[0]+'/'+date[1]+'/'+date[2]
				if (':' not in ephemtime) : ephemtime += ' 12:00:00'
				second = 24*3600 *np.arange(1,N)*int(how[:-1])
				sec1970 = _Time(ephemtime) + second
				for i in range(len(sec1970)) : 
					ephemtime=_Time(sec1970[i]).split(' ')[0]
					if (s == '/') : datelist.append(ephemtime)
					else : 
						ephemtime = ephemtime.split('/')
						datelist.append( ephemtime[0]+s+ephemtime[1]+s+ephemtime[2] )
			return datelist
		#----------------------------------------
		if (how[-1] in 'dmy') : 
			return [datein] + Case1(date, how, N)
		else : 
			if ('M' in how) : 
				date[1] = how[1:-1]
				if (len(date[1])==1) : date[1] = '0'+date[1]
				return [datein] + Case1(date, how[0]+'1y', N)
			elif ('D' in how) : 
				if (':' not in date[2]) : date[2] = how[1:-1]
				else : date[2] =how[1:-1]+date[2].split(' ')[1]
				if (len(date[2])==1) : date[2] = '0'+date[2]
				return [datein] + Case1(date, how[0]+'1m', N)
	#----------------------------------------
	else: 
		import time
		return time.time()





def TimeConvert( tin, infmt, outfmt ) : 
	'''
	infmt, outfmt: 
		(1) 's' | 'sec' | 'second'   => int/float
		(2) 'm' | 'min' | 'minute'   => int/float
		(3) 'h' | 'hour'             => int/float
		(4) 'hms'                    => str as '03:14:07.56'

	tin:
		isnum | ndarray/list

	return:
		tout same shape as tin
	'''
	from jizhipy.Basic import IsType
	import numpy as np
	infmt, outfmt = str(infmt).lower(), str(outfmt).lower()
	if (infmt == 'hms') : pass
	elif ('s' in infmt ) : infmt  = 's'
	elif ('m' in infmt ) : infmt  = 'm'
	elif ('h' in infmt ) : infmt  = 'h'
	if (outfmt == 'hms') : pass
	elif ('s' in outfmt) : outfmt = 's'
	elif ('m' in outfmt) : outfmt = 'm'
	elif ('h' in outfmt) : outfmt = 'h'
	if (infmt == outfmt) : return tin
	#--------------------------------------------------
	if (IsType.isnum(tin) or IsType.isstr(tin)) : 
		islist, tin = False, [tin]
	else : islist = True
	if (IsType.isstr(tin[0])) : shape = (len(tin),)
	else : 
		tin = np.array(tin, float)
		shape = tin.shape
		tin = tin.flatten()
	#--------------------------------------------------
	if (infmt == 'hms') : 
		tinlist = []
		for i in range(len(tin)) : 
			h, m, s = np.array(tin[i].split(':'), float)
			tinlist.append(h*3600. + m*60 + s)  # second
		tin = np.array(tinlist)
	elif (infmt == 's') : pass
	elif (infmt == 'm') : tin = tin * 60.
	elif (infmt == 'h') : tin = tin * 3600.
	#--------------------------------------------------
	if (outfmt == 'hms') : 
		tout = []
		for i in range(len(tin)) : 
			h = int(tin[i] / 3600)
			m = int((tin[i] - h*3600) / 60)
			s = tin[i] - h*3600 - m*60
			h = str(h) if(h>=10)else '0'+str(h)
			m = str(m) if(m>=10)else '0'+str(m)
			s = ('%.2f' % s) if(s>=10)else ('0%.2f' % s)
			tout.append( h + ':' + m + ':' + s )
	#--------------------------------------------------
	tin = tin.reshape(shape)
	if   (outfmt == 's') : tout = tin
	elif (outfmt == 'm') : tout = tin / 60
	elif (outfmt == 'h') : tout = tin / 3600
	return tout

