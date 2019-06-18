
def Sha( a, algorithm=None, multi=True ):
	'''
	algorithm:
		in [None, 'md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512']

	multi:
		True: multiple Encryption
	'''
	import hashlib
	from jizhipy.Basic import IsType
	names = ['sha512','sha384','sha256','sha224','sha1','md5']
	algorithms = [hashlib.sha512, hashlib.sha384, hashlib.sha256, hashlib.sha224, hashlib.sha1, hashlib.md5]
	ret, alg, algorithm = [], [], str(algorithm).lower()
	if (algorithm in names): n = names.index(algorithm)
	else: n = 3  # sha224
	if (multi): alg = [hashlib.sha512, hashlib.sha512]
	alg += [algorithms[n]]
	if (IsType.isstr(a)): a, islist = [a], False
	else: islist = True
	for i in range(len(a)):
		b = a[i]
		for j in range(len(alg)):
			try: b = bytes(b, 'utf-8')  # python 3
			except: pass
			b = alg[j](b).hexdigest()
		ret.append(b)
	if (not islist): ret = ret[0]
	return ret

