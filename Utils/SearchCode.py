#! /usr/bin/env python
import os, sys
'''
Usage: 
	(1) find_code.py --dir ~/jizhipy --key 'import numpy' 'Asarray.py'
	(2) ......
'''

if ('--dir' not in sys.argv or '--key' not in sys.argv):
	print('Error: must include both --dir and --key')
	exit()

n1 = sys.argv.index('--dir')
n2 = sys.argv.index('--key')
indir = sys.argv[n1+1]
if (indir[-1] =='/'): indir = indir[:-1]
if (n2 > n1): keys = sys.argv[n2+1:]
else: keys = sys.argv[n2+1:n1]
values = [[] for i in range(len(keys))]

pyfile = os.popen('find '+indir+' -name "*.py"').readlines()
home = os.path.expanduser('~')

for f in pyfile:
	fo = open(f[:-1])
	txt = fo.read()
	fo.close()
	for i in range(len(keys)):
		if (keys[i] in f): values[i].append(f[:-1])
		if (keys[i] in txt): values[i].append(f[:-1])

for i in range(len(keys)):
	print('keyword: "'+keys[i]+'"')
	for j in range(len(values[i])):
		s = values[i][j]
		if (s[:len(home)] == home): s = '~'+s[len(home):]
		s = '  '+s
		if (i !=len(keys)-1 and j ==len(values[i])-1): 
			s += '\n'
		print(s)

