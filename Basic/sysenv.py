#! /usr/bin/env python
'''
Function:
	Provide 1 command: 
		sysenv: is used to "add" or "remove" the environment in .bashrc

Usage, examples:
	sysenv -add alias 'cpnot="~/bin/cmdnot.py cp"'
	sysenv -rm  alias 'cpnot="~/bin/cmdnot.py cp"'

NOTE THAT, can NOT write as:
	sysenv -add alias cpnot="~/bin/cmdnot.py cp"
		--> cpnot="~/bin/cmdnot.py cp" is not an valid argument, will raise error: syntax error near unexpected token `('
		--> use '' or "" to make it to be an unity

	sysenv -add export "PYTHONPATH=/python2.7/site-packages/"
	sysenv -rm  export 'PYTHONPATH="/python2.7/site-packages/"'
'''


import os, sys

def ShellCmd( cmd ) : 
	strlist = os.popen(cmd).readlines()
	for i in range(len(strlist)): strlist[i] = strlist[i][:-1]
	return strlist



# Environment
try : 
	pm = sys.argv[1].lower()
	which = sys.argv[2]
	env = sys.argv[3]
except : 
	print('Error, must be:    sysenv  -add/-rm  export/alias  envvalue')
	exit()

envkey = env[:env.index('=')]
envvalue = env[env.index('=')+1:]
if (envvalue[0] == "'") : envvalue = '"'+envvalue[1:-1]+'"'
if (envvalue[0] != '"') : envvalue = '"'+envvalue+'"'

if (which=='export' and envvalue[-2]=='/') : envvalue = envvalue[:-2]+'"' # rm /



# List
font = which+' '+envkey+'='
env1 = font+ envvalue[:-1]+'/"' +'\n'  # +" +/
env2 = font+ envvalue +'\n'  # +" -/
env3 = font+ envvalue[1:-1]+'/' +'\n'  # -" +/
env4 = font+ envvalue[1:-1] +'\n'  # -" -/
if (which == 'export') : 
	back = ':$'+envkey
	env5 = font+ envvalue[:-1]+'/'+back+'"' +'\n'  # +" +/
	env6 = font+ envvalue[:-1]+back+'"' +'\n'  # +" -/
	env7 = font+ envvalue[1:-1]+'/'+back +'\n'  # -" +/
	env8 = font+ envvalue[1:-1]+back +'\n'  # -" -/
	envlist = [env1, env2, env3, env4, env5, env6, env7, env8]
else : envlist = [env2, env4]



# Read .bash
bash = os.path.expanduser('~/.bashrc')
if (not os.path.exists(bash)) : os.system('touch '+bash)

txt = open(bash).readlines()
while (txt[-1] == '\n') : txt.pop(-1)



# Remove
if (pm.lower() in ['-rm', '-remove', '--rm', '--remove']) : 
	for i in range(len(envlist)) : 
		while (envlist[i] in txt) : txt.remove(envlist[i])
	while (txt[-1] == '\n') : txt.pop(-1)
	open(bash, 'w').writelines(txt)
	os.system('source '+bash)
	exit()



# Add
envexist = False
for i in range(len(envlist)) : 
	if (envlist[i] in txt) : envexist = True
if (envexist) : exit()

if   (which == 'alias' ) : txt.append('\n'+env2)
elif (which == 'export') : txt.append('\n'+env5)  # or env6

open(bash, 'w').writelines(txt)
os.system('source '+bash)

