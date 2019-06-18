#! /usr/bin/env python

def OpencvInstaller():
	import os, jizhipy
	location = jizhipy.Vision.__path__[0]+'/'
	os.system('bash '+location+'OpencvInstaller.sh')


if __name__=='__main__':
	OpencvInstaller()
