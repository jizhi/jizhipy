#! /usr/bin/env python
##################################################

# directory_name must be the same as module_name
module_name = 'jizhipy'

exclude = ['.DS_Store', '*.pyc', '*.swp']

##################################################


import os, sys
def ShellCmd( cmd ) : 
	return os.popen(cmd).read().split('\n')[:-1]

pwd = os.getcwd()
dir_name = pwd[pwd.rfind('/')+1:]
if (dir_name != module_name) : 
	raise Exception('directory_name must be the same as module_name, but now dir_name="'+dir_name+'", module_name="'+module_name+'"')

# dest, without / at the end
whichpython = ShellCmd('which python')[0].split('/')
whichpython = '/'+whichpython[1]+'/'

# syspath, usrpath
syspath, isstr = sys.path[:], False
for i in range(len(syspath)) : 
	if (syspath[i][:len(whichpython)]==whichpython and syspath[i][-9:]=='-packages') : 
		syspath, isstr = syspath[i]+'/', True
		break
if (not isstr):
	for i in range(len(syspath)) : 
		if (syspath[i][-9:]=='-packages') : 
			syspath, isstr = syspath[i]+'/', True
			break

key = '/lib64/' if('/lib64/' in syspath)else '/lib/'
usrpath = '~/.local/lib/'+syspath.split(key)[1]
#usrpath = '~/Library/Python/2.7/lib/python/site-packages/'
usrpathpwd = os.path.abspath(os.path.expanduser(usrpath))

home = os.path.expanduser('~')

fixpath = None
for argv in sys.argv[1:] : 
	if (argv[:9] == '--prefix=') : fixpath = argv[9:]
	else : continue
	fixpathpwd = os.path.abspath(os.path.expanduser(fixpath))
	fixpath = fixpathpwd
	if (fixpath[:len(home)] == home) : fixpath = '~'+fixpath[len(home):]
	if (fixpath[-1] != '/') : fixpath += '/'

syspath += module_name
usrpath += module_name
if (fixpath is not None) : fixpath += module_name

whoami = ShellCmd('whoami')[0]
if (whoami == 'root') : 
	dest = syspath
	envlist = []
elif (fixpath is None) : 
	dest = usrpath
	if (usrpathpwd[:len(home)] == home) : usrpathpwd = '$HOME'+usrpathpwd[len(home):]
	envlist = ['export PYTHONPATH="'+usrpathpwd+'"']
else : 
	dest = fixpath
	if (fixpathpwd[:len(home)] == home) : fixpathpwd = '$HOME'+fixpathpwd[len(home):]
	envlist = ['export PYTHONPATH="'+fixpathpwd+'"']


##################################################


def HELP() : 
	print('--------------- Path: ---------------')
	print('DefaultPath     : '+syspath)
	print('UsrPath         : '+usrpath)
	print('Destination     : '+dest)
	if (envlist) : print('Add environment : '+envlist[0])
	print('--------------- Usage: ---------------')
	print('./setup.py install [--user]  : install '+module_name)
	print('./setup.py install --prefix= : install to a specific destination')
	print('./setyp.py unstall           : uninstall '+module_name)
	exit()

if ((len(sys.argv) == 1) or ('-h' in sys.argv) or ('--help' in sys.argv) or ('install' not in sys.argv and 'uninstall' not in sys.argv)) : HELP()


if ('install' in sys.argv) : which = 'install'
elif ('uninstall' in sys.argv) : which = 'uninstall'


##################################################


# uninstall
if (which == 'uninstall') : 
	print('Uninstall from  '+dest)
	opt = '-rm'  # remove from the PYTHONPATH
	dest = os.path.abspath(os.path.expanduser(dest))
	if (os.path.exists(dest)) : 
		os.system('/bin/rm -rf '+dest)



elif (which == 'install') : 
	print('Install to  '+dest)
	# Check pwd
	if (pwd[-len(module_name):] != module_name) : 
		print('Error: please run  ./setup.py install  inside the directory "'+module_name+'/"')
		exit()
	opt = '-add'  # add to the PYTHONPATH
	dest = os.path.abspath(os.path.expanduser(dest))
	if (not os.path.exists(dest[:-8])): os.makedirs(dest[:-8])
	# rm old
	if (os.path.exists(dest)) : 
		os.system('/bin/rm -rf '+dest)
	# install
	os.system('/bin/cp -r ../'+module_name+' '+dest)
	#----------------------------------------
	# Generate tags
#	outdir = home + 'tagsdir/'
#	if (not os.path.exists(outdir)) : os.mkdir(outdir)
#	os.chdir(dest)
#	os.system('python ptags.py '+dest+'/*.py')
#	os.system('mv tags '+module_name+'.tags')
#	os.system('cp '+module_name+'.tags '+outdir)
#	outname = home+'tags'
#	if (os.path.exists(outname)): os.system('/bin/rm '+outname)
#	fn = os.popen('ls '+outdir+'*.tags').readlines()
#	tags = []
#	for i in range(len(fn)) : 
#		tags += open(fn[i][:-1]).readlines()
#	tags.sort()  # Must sort, otherwaise useless
#	outname = open(outname, 'w')
#	for tag in tags : outname.write(tag)
#	outname.close()
	#----------------------------------------
	os.system('cp Vision/OpencvInstaller.py ~/bin/OpencvInstaller')
	os.system('chmod +x ~/bin/OpencvInstaller')

	

# Add/Remove environment
for i in range(len(envlist)) : 
	env = envlist[i]
	n1 = env.find(' ')
	for n2 in range(n1+1, len(env)) : 
		if (env[n2] != ' ') : break
	which = env[:n1]
	env = env[n2:]
	os.system('python Basic/sysenv.py '+opt+' '+which+" '"+env+"'")

