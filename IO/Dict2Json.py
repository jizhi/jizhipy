
def Dict2Json( indict ):
	'''
	Parameters
	----------
	indict:
		(1) one dict: {"a":1, "b":2}
		(2) list of dicts: [{"a":1, "b":2}, {"c":3, "d":4}, ...]


	Returns
	----------
	Return json

	json:
		[str]
		
	'''
	from jizhipy.Basic import IsType
	if (not IsType.islistuple(indict)): indict = [indict]
	json = '['
	


