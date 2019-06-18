
def DopplerEffect( f0, df=None, dv=None ) : 
	'''
	In astronomy, when the object goes away/recede from us, it has a "positive" relative velocipy, while closing to use has a "negative" velocity
		+df => -dv
		+dv => -df

	(1) dv: 
		[km/s]
		dv = vobs - vs
		Relative velocity of observer to the source

	(2) f0:
		[MHz]
		Original frequency of the source

	(3) f1:
		[MHz]
		Observed frequency by observer

	(4) df:
		[MHz]
		df = fobs - f0
		Frequency differece of observed frequency to original frequency

	Case 1: 
		DopplerEffect( f0, dv ), return df
	Case 2: 
		DopplerEffect( f0, df ), return dv
	'''
	import numpy as np
	from jizhipy.Array import Asarray
	from jizhipy.Basic import IsType, Raise
	from jizhipy.Astro import Const
	c = Const.c/1000.
	if   (df is not None) : 
		islist = False if(IsType.isnum(f0+df))else True
		f0, df = npfmt(f0), npfmt(df)
		f0 = f0 + df*0
		df = f0*0 + df
		dv = -df * c / f0
		if (not islist) : dv = dv[0]
		return dv
	elif (dv is not None) : 
		islist = False if(IsType.isnum(f0+dv))else True
		f0, dv = npfmt(f0), npfmt(dv)
		f0 = f0 + dv*0
		dv = f0*0 + dv
		df = -dv / c * f0	
		if (not islist) : df = df[0]
		return df
	else : Raise(Exception, 'df=None and dv=None')

