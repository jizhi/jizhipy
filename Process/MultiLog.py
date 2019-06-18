import sys
'''
Import this module if you want to write a log file with argument "--log": 
from jizhipy.Log import *
	$ python test.py --log
	$ python test.py nside 512 --log haha.log
'''


class LogStdout( object ) : 


	def __init__( self, logname=None ) : 
		from jizhipy.Basic import Time
		if (logname is None) : logname = sys.argv[0].split('/')[-1]
		else : logname = str(logname)
		if (logname[-4:] != '.log') : logname += '.log'
		self.logf = open(logname, 'a')
		self.rank = 0
		try : 
			from mpi4py import MPI
			comm = MPI.COMM_WORLD
			self.rank = comm.Get_rank()
		except : pass
		if (self.rank == 0) : self.logf.write('\n\n\n=============== '+Time(1)+' ===============\n')


	def write( self, outstream ) : 
		sys.__stdout__.write(outstream)  # print(to screen)
		# for jizhipy.ProgressBar()
		if(outstream == '\r'+' '*(len(outstream)-1)): return
		if(outstream[-1]=='\r'): outstream=outstream[:-1]+'\n'
		if (outstream[0]=='\r') : 
			if (outstream[-1]=='\n'): outstream=outstream[1:]
			else : outstream = outstream[1:]+'\n'
		self.logf.write(outstream)  # print(to log file)
		self.logf.flush()


	def flush( self ) : 
		sys.__stdout__.flush()





if ('--log' in sys.argv[1:]) : 
	try : 
		n = sys.argv[1:].index('--log')
		logname = sys.argv[1:][n+1]
	except : logname = None
	sysstd = LogStdout(logname)
	sys.stdout = sysstd
	sys.stderr = sysstd



