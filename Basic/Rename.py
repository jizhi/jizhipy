
def Rename( *args ) : 
	'''
	(1) Rename('01.txt', 'test.txt')
			rename file '01.txt' to 'test.txt'
	(2) Rename('haha_01.txt', 'test_%n.txt')
			'%n' means that splits inname into 2 parts, left part is before number, right part is after number. number is any number in [0,1,2,3,4,5,6,7,8,9]
			** return 'test_01.txt'
	(3) Rename('ha_01.txt', 'en_2.dat', ..., 'test_%n.txt')
			args[:-1] are inname, args[-1] is outname
			See (2)
			** return (test_01.txt, test_2.dat)
	(4) Rename('ha_12.txt', 'en_230.dat', ..., 'test_%m.txt')
			'%m': find number in the inname automatically, get the longest length of the numbers, when rename, all number is retained the same length, adding '0' in front for short number
			** return ('test_012.txt', test_230.dat)
	(5) Rename('~/figure/', '~/data/', ..., 'test_%m.txt')
			For the directory, do for all files in this directory
	'''
	print('jizhipy.Rename:')
	if (len(args) == 1) : print("  Rename('"+args[0]+"')")
	else : print('  Rename'+str(args))
	if (len(args) < 2) : return
	import os
	infile, indir, outname = [], [], args[-1]
	# Split outname
	if (outname[-1] == '/') : outname = outname[:-1]
	outnamebak = outname
	outname = os.path.basename(outname)
	#for s in '0123456789nm' : 
	for s in 'nm' : 
		if ('%'+s not in outname) : continue
		outname = outname.split('%'+s)
		outname = (outname[0], '%'+s, outname[-1])
	if (type(outname)!=tuple): outname = (outnamebak, '','')
	if ('%' in outname[0]) : 
		print('    Error: outname = '+outname[0]+outname[1]+outname[2])
		return
	if (outname[1] == '') : 
		if (len(args[:-1]) != 1) : print('    Error: rename >1 files to the same name')
		else : 
			a=os.path.abspath(os.path.expanduser(args[0]))
			b=os.path.abspath(os.path.expanduser(outname[0]))
			print('    Rename:  '+args[0]+'  =>  '+outname[0])
			if (os.path.exists(b)) : print('    Error: Exists '+outname[0]+', forbid to rename')
			print('  **** Do you really want to rename them?')
			print('  **** Type "yes" to continue, others to stop')
			opt = raw_input().lower()
			if (opt != 'yes') : return
			os.rename(a, b)
		return
	#---------------------------------------------
	# Check infile, indir
	indir2, args, n = [], list(args[:-1]), 0
	try : 
		while(n < len(args)) : 
			if ('*' in args[n]) : 
				dirname = os.path.dirname(args[n])
				N = len(dirname)
				if (dirname != '') : N += 1
				a = os.popen('ls '+args[n]).readlines()
				for i in range(len(a)) : 
					a[i] = a[i][N:-1]
				indir2.append( [os.path.abspath(os.path.expanduser(dirname))+'/', a] )
				args.pop(n)
			else : n += 1
	except : print('    Warning: Fail to "ls *"')
	#---------------------------------------------
	for i in range(len(args)) : 
		if (not os.path.exists(args[i])) : continue
		a = os.path.abspath(os.path.expanduser(args[i]))
		if (os.path.isfile(a)) : infile.append(a)
		elif (os.path.isdir(a)) : indir.append(a+'/')
	#---------------------------------------------
	if (len(infile) != 0) : 
		infiledir = os.path.dirname(infile[0])+'/'
		n = len(infiledir)
		for i in range(len(infile)) : 
			dirname = os.path.dirname(infile[i])+'/'
			if (dirname != infiledir) : 
				print('    Error: NOT all files are in the same directory')
				return
			else : infile[i] = infile[i][n:]
		infile = [infiledir, infile]
	#---------------------------------------------
	for i in range(len(indir)) : 
		indir[i] = [indir[i], os.listdir(indir[i])]
	if (len(infile) != 0) : indir.append(infile)
	if (len(indir2) != 0) : indir += indir2
	if (len(indir) == 0) : 
		print('    Error: Total number of valid files and directories:  %i, %i' % (len(infile), len(indir)))
		return
	#---------------------------------------------
	for i in range(len(indir)) : 
		n, num = 0, []
		while(n < len(indir[i][1])) : 
			f, pop = indir[i][1][n], False
			if (f[0] == '.') : pop = True
			else : 
				n1, n2, strn = -1, -1, '0123456789'
				for j in range(len(f)) : 
					if (f[j] in strn) : 
						n1 = j
						break
				if (n1 >= 0) : 
					for j in range(n1+1, len(f)) : 
						if (j == len(f)-1) : 
							n2 = j+1 if(f[j] in strn)else j
							break
						elif (f[j] in strn) : continue
						elif (f[j]=='.' and f[j+1] in strn) : continue
						else : 
							n2 = j
							break
				if (n1 < 0) : pop = True
				else : num.append( f[n1:n2] )
			if (pop) : indir[i][1].pop(n)
			else : n += 1
		indir[i].append(num)
	n = 0
	while(n < len(indir)) : 
		if (len(indir[n][1]) != 0) : n += 1
		else : indir.pop(n)
	home = os.path.expanduser('~/')
	for i in range(len(indir)) : 
		d = indir[i][0]
		if (d[:len(home)] == home) : d = '~/'+d[len(home):]
		print('    '+d+' : '+str(len(indir[i][1])))
	print('     =>  '+outname[0]+outname[1]+outname[2])
	print('  **** Do you really want to rename them?')
	print('  **** Type "yes" to continue, others to stop')
	opt = raw_input().lower()
	if (opt != 'yes') : return
	#---------------------------------------------
	tf = False
	for i in range(len(indir)) : 
		N = -1
		if (outname[1] == '%m') : 
			for j in range(len(indir[i][2])) : 
				if (len(indir[i][2][j]) > N) : N = len(indir[i][2][j])
		for j in range(len(indir[i][2])) : 
			inname = indir[i][0]+indir[i][1][j]
			num = indir[i][2][j]
			if (outname[1] == '%m' and len(num) < N) : num = (N-len(num))*'0' + num
			outn = indir[i][0]+outname[0] +num +outname[2]
			if (os.path.exists(outn)) : 
				tf = True
				continue
			os.rename(inname, outn)
	if (tf) : print('    Some files are exists, can NOT rename')


