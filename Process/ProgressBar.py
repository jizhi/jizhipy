import sys


'''
In jp.PoolFor():
	if (self.verbose) : 
		progressbar = jp.ProgressBar('    completed:', Nmax, False)
		progressbar.starttime = mapmaking._starttime
		progressbar.count -= Nmax-len(obsRAlist)

# progressbar is jp.Time(1)
'''



class ProgressBar( object ) : 
	'''
	progressbar = ProgressBar()
	progressbar.__init__('RemoveRFI.RemoveLoop():', len(array))
	for i in range(len(a)) : progressbar.Progress()
	'''

	def __init__( self, string=None, Ntot=None, precent=True, endnewline=True, showend=False, verbose=True ) : 
		'''
		string:
			(1) String shown on the screen WITHOUT '|'
			(2) String WITH '|'  => self.sub=False

		Ntot:
			Total iterables will be done

		precent:
			[True | False] show in precentage or not?

		endnewline:
			True | False
			When reach the end (self.count==0), whether go to new/next line

		showend:
			True | False
			Whether go to new line and print some strings

		verbose:
			True: always print
			False: always don't print
			Logger(): use this logger
		'''
		from jizhipy.Basic import Time
		try: 
			verbose = verbose._level
			if (verbose <= 20): self.verbose = True
			else: self.verbose = False
		except: self.verbose = bool(verbose)
		if (Ntot is None) : Ntot = 1
		self.string, self.Ntot, self.count = string, Ntot,Ntot
		self.starttime = Time(1)
		self.sec0 = Time(ephemtime=self.starttime)
		self.endnewline, self.showend = endnewline, showend
		if (self.showend) : self.endnewline = True
		self.precent = bool(precent)
		#---------------------------------------------
		self.fmt1, n = '%'+str(len(str(self.Ntot)))+'i', 0
		try : 
			while (self.string[n] == ' ') : n += 1
		except : n = len(self.string)
		self.font = self.string[:n]
		#---------------------------------------------
		self.lastend = ''
		if ('|' not in self.string) : self.sub = True
		else : 
			self.sub, self.m = False, 0
			self.string = self.string[n:].split('|')
			self.fmt2 ='%'+str(len(str(len(self.string))))+'i'

 

	def Progress( self, newline=False, end=False ) : 
		'''
		**kwargs:
			Acceptable keys: 'end'=True/False
				==> usage: progressbar.Progress(end=True)
		'''
		if (not self.verbose): return
		if (end):
			self._End()
			return
		from jizhipy.Basic import Time
		if (self.sub) : 
			self.count -= 1
			nowtime = Time(1)
			sec = Time(ephemtime=nowtime)
			dtime = Time(ephemtime=[self.starttime, nowtime])
			if (not self.precent):
				up = (self.fmt1 % (self.Ntot-self.count))
				down = '/'+str(self.Ntot)
			else:
				p = 100.*(self.Ntot-self.count)/self.Ntot
				up = '%5.1f' % p
				down = ' %'
			string = '\r'+self.string+'  '+up+down+'  '+dtime
			blank = '\r'+len(string)*' '
			if (self.endnewline and self.count==0) : 
				string += '\n'
			elif (newline): string += '\n'
			if (self.count != self.Ntot-1 and not newline) : 
				if (self.verbose):
					sys.stdout.write(blank)
					sys.stdout.flush()
			if (self.verbose):
				sys.stdout.write(string)
				sys.stdout.flush()
			if (self.showend and self.count==0) : 
				try : 
					sec = Time(ephemtime=nowtime)
					sec = sec +(sec-self.sec0)/(self.Ntot-1.)
					nowtime = Time(sec1970=sec, which=1)
				except : pass
				if (self.verbose): print(self.font+'    '+self.starttime+' => '+nowtime)
			self.lastend = '\n' if(string[-1] =='\n')else ''
		#---------------------------------------------
		else : 
			n2 = len(self.string)
			if (self.showend and self.count==0 and self.m==n2): 
				if (self.verbose): print(self.font+'    '+self.starttime+' => '+Time(1))
				return
			self.m = self.m % n2 + 1
			if (self.m == 1) : self.count -= 1
			n = self.Ntot - self.count
			str1 = (self.fmt1 % n) + '/' + str(self.Ntot)
			str2 = (self.fmt2 % self.m) + '/' + str(n2)
			#------------------------------
			string = ''
			for i in range(self.m) : 
				string += self.string[i] + '|'
			if (self.m == n2) : string = string[:-1]
			#------------------------------
			string = '\r'+self.font+str1+', '+str2+':  '+string
			blank = '\r'+len(string)*' '
			if ('blank' not in self.__dict__.keys()) : 
				if (self.m == n2) : self.blank = blank
			if (self.endnewline and self.count==0 and self.m==n2) : string += '\n'
			#------------------------------
			if (self.count != self.Ntot-1) : 
				if (self.verbose):
					if ('blank' not in self.__dict__.keys()): 
						sys.stdout.write(blank)
					else : sys.stdout.write(self.blank)
					sys.stdout.flush()
			if (self.verbose):
				sys.stdout.write(string)
				sys.stdout.flush()
			self.lastend = '\n' if(string[-1] =='\n')else ''



	def _End(self):
		if (self.count >1):
			self.count = 1
			self.Progress()


