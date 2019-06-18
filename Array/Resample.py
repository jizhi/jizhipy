
def Resample( dataframe, rule, how ) : 
	'''
	Like: pandas.resample()

	dataframe:
		pandas.DataFrame | pandas.Series()

	rule:
		'A'/'AS'		year
		'M'/'MS'		month
		'W'/'WS'		week
		'D'/'DS'		day
		'H'/'HS'		hour
		'T'/'TS'		minute
		'S'/'SS'		second

	how:
		'mean', 'sum', 'min', 'max'
	'''
	import numpy as np
	from jizhipy.Basic import IsType
	how = str(how).lower()
	shape = dataframe.shape
	y = np.arange(1900, 2201).astype(str)
	m = np.arange(1, 13).astype(str)
	d = np.arange(1, 32).astype(str)
	for i in range(9) : 
		m[i] = '0'+m[i]
		d[i] = '0'+d[i]
	#--------------------------------------------------
	# Find date 
	timecol = []
	for i in range(shape[1]) : 
		i = 1
		a = dataframe.iloc[0,i]
		tf = False
		if (IsType.isstr(a) and '-' in a) : 
			a = a.split('-')
			if (len(a) == 3 and a[0] in y and a[1] in m and a[2] in d) : tf = True 
		if (tf) : 
			timecol = [i]
			break
	#--------------------------------------------------
	# Separate str and float
	numcol, strcol = [], []
	a = dataframe.iloc[0].values
	for i in range(shape[1]) : 
		if (not IsType.isstr(a[i])) : numcol.append(i) 
		elif (i not in timecol) : strcol.append(i)
	numcol = np.array(numcol)
	frontcol = numcol[numcol<timecol[0]]
	numcol = numcol[numcol>timecol[0]]
	#--------------------------------------------------
	# Separate different str
	diffrow = [0]
	if (len(strcol)==1) : strarr =dataframe.iloc[:,strcol[0]]
	else : 
		strarr = dataframe.iloc[:,strcol[0]] + dataframe.iloc[:,strcol[1]]
		for i in range(2, len(strcol)) : 
			strarr += dataframe.iloc[:,strcol[i]]
	strnow = strarr[0]
	for i in range(1, shape[0]) : 
		if (strarr[i] != strnow) : 
			strnow = strarr[i]
			diffrow.append(i)
	diffrow.append(shape[0])
	#--------------------------------------------------
	redataframe = []
	isbool = True
	for i in range(len(numcol)) : 
		if (not IsType.isbool(dataframe.iloc[0,numcol[i]])) : 
			isbool = False
	if (isbool) : how = 'sum'
	for i in range(len(diffrow)-1) : 
		n1, n2 = diffrow[i:i+2]
		a = dataframe.iloc[n1:n2, numcol]
		index=pd.to_datetime(dataframe.iloc[n1:n2,timecol[0]])
		a.index = index
		if   (how =='mean') : a = a.resample(rule).mean()
		elif (how == 'sum') : a = a.resample(rule).sum()
		elif (how == 'min') : a = a.resample(rule).min()
		elif (how == 'max') : a = a.resample(rule).max()
		if (isbool) : a.astype(bool)
		#----------------------------------------
		b = dataframe.iloc[n1:n2, frontcol]
		bdtype = b.values.dtype
		b.index = index
		b.to_csv('b.csv', encoding='utf-8')
		b = b.resample(rule).mean()
		#----------------------------------------
		c = dataframe.iloc[[n1]]
		c = pd.concat(len(a)*[c], axis=0)
		c[b.columns] = b.values
		c[a.columns] = a.values
		c.iloc[:,timecol] = a.index.format()
		#----------------------------------------
		c.dropna(axis=0, how='any', inplace=True)
		c[b.columns] = c[b.columns].astype(bdtype)
		redataframe.append(c)
	redataframe = pd.concat(redataframe, axis=0)
	return redataframe
