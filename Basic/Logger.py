
class Logger(object):
	__name__ = 'jizhipy.Basic.Logger'

	def __init__(self, level=None, streamHandler=False, fileHandler=False, fileMode='a', srMode='a'):
		'''
		Logger(level): change the logLevel of current environment to @level

		try: 
			if (logger.__name__ == 'jizhipy.Basic.Logger'):
				self.logger = logger
			else: exit+1
		except: self.logger = jp.Basic.Logger(logger, True)


		Parameters
		----------
		level:
			='notset'   : logging.NOTSET,   level=0
			='debug'    : logging.DEBUG,    level=10
			='info'     : logging.INFO,     level=20
			='warning'  : logging.WARNING,  level=30
			='error'    : logging.ERROR,    level=40
			='critical' : logging.CRITICAL, level=50
			=False      :                   level=60

		streamHandler:
			True | False
			Whether print the log on the screen?

		fileHandler:
			[str] | False
			Whether save the log to a file?
			(1) [str]: (abs)path of the output log file

		fileMode:
			this parameter is only for fileHandler()
			'w' | 'a'

		srMode:
			sr=SendRecv, this parameter is only for self.Send() and self.Recv()
			'w' | 'a'
		'''
		import os, logging
		msgdir = os.path.expanduser('~/Log/SendRecv/')
		if (not os.path.exists(msgdir)): os.makedirs(msgdir)
		self._msgfile = msgdir+'SendRecv.msg'
		if (str(srMode).lower()=='w'):
			fo = open(self._msgfile, 'w')
		else: fo = open(self._msgfile, 'a')
		fo.close()
		#----------------------------------------
		self.handlers = []
		self.SetLevel(level)
		if (self.level is False):
			fileHandler, streamHandler = False, False
		#----------------------------------------
		if (type(fileHandler)==str or streamHandler is True): 
			# not give name, set root logger
			self.logger = logging.getLogger()
		#	self.logger.setLevel(0) # only when player >= logger.level, logger.level can muit handlers
		else: logging.getLogger().setLevel(self._level)
		#----------------------------------------
		if (streamHandler is True):
			# StreamHandler to print on the screen
			sh = logging.StreamHandler()
			sh.setLevel(5678)
			self.handlers.append(len(self.logger.handlers))
			self.logger.addHandler(sh)
		#----------------------------------------
		if (type(fileHandler) == str): 
			# FileHandler to save log file
			fileHandler = os.path.abspath(os.path.expanduser(fileHandler))
			fileMode='w' if(str(fileMode).lower()=='w')else 'a'
			fh = logging.FileHandler(fileHandler, fileMode)
			fh.setLevel(5678)
			self.handlers.append(len(self.logger.handlers))
			self.logger.addHandler(fh)





	def SetLevel(self, level, **kwargs):
		'''
		self.level = 'DEBUG/INFO/...'
		self._level = 10/20/...

		Parameters
		----------
		level:
			[str] or [int]
		'''
		import logging
		from jizhipy.Basic import Raise, IsType
		if   (level is True): level ='INFO'
		elif (level is None): level =logging.getLogger().level
		elif (level ==0): level = False
		elif (level ==1): level = True
		if (not IsType.isnum(level)): 
			if (level is not False):
				level, level0 = str(level).upper(), level
				if   (level =='CRITICAL'): 
					_level = logging.CRITICAL
				elif (level =='ERROR'): 
					_level = logging.ERROR
				elif (level =='WARNING'): 
					_level = logging.WARNING
				elif (level =='DEBUG'): 
					_level = logging.DEBUG
				elif (level =='INFO'): 
					_level = logging.INFO
				elif (level =='NOTSET'): 
					_level = logging.NOTSET
				else: Raise(Exception, "level='"+str(level0)+"' not in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']")
			else: _level = False
		#----------------------------------------
		else: 
			if   (   level<= 0): level, _level= 'NOTSET' , 0
			elif ( 0<level<=10): level, _level= 'DEBUG'  , 10
			elif (10<level<=20): level, _level= 'INFO'   , 20
			elif (20<level<=30): level, _level= 'WARNING', 30
			elif (30<level<=40): level, _level= 'ERROR'  , 40
			elif (40<level<=50): level, _level= 'CRITICAL',50
			elif (50<level    ): level, _level= False, False
		#----------------------------------------
		if ('verbose' in kwargs.keys() and kwargs['verbose'] is True): pass
		else: self.level, self._level = level, _level
		return (level, _level)





	def AddHandler(self, level=False, streamHandler=False, fileHandler=False, formatter=False, fileMode='a', srMode='a'): 
		self.__init__(level=level, streamHandler=streamHandler, fileHandler=fileHandler, formatter=formatter, fileMode=fileMode, srMode=srMode)





	def Verbose(self, level):
		'''
		Compare the level with self.level
		if level <= self.level, return True: will print
		else, return False: won't print
		'''
		_, _level = self.SetLevel(level, verbose=True)
		if (self._level >= _level): return True
		else: return False





	def Out(self, level, message):
		'''
		Out put the message, if level in ['ERROR', 'CRITICAL'], raise and stop

		Parameters
		----------
		level:
			level of this message, only when self.player >= level can print this message
			[str]: 'notset' | 'debug' | 'info' | 'warning' | 'error' | 'critical'

		message:
			[str] | list of [str]
			Example: message = 'running OK'
			         message = ['running OK', 'Loading...']
		'''
		if (self.level is False or level is False): return
		from jizhipy.Basic import IsType, Raise
		_, level = self.SetLevel(level, verbose=True)
		if (level < self._level): level = 5678
		# Retset levels of handlers
		for i in range(len(self.logger.handlers)):
			self.logger.handlers[i].setLevel(5678)
		for i in self.handlers:
			self.logger.handlers[i].setLevel(level)
		self.logger.setLevel(level)
		#--------------------------------------------------
		if (IsType.isstr(message)): message = [message]
		for i in range(1, len(message)):
			message[0] += '\n'+message[i]
		message = message[0]
		if(self._level >=level): self.logger.critical(message)





	def Send(self, message):
		'''
		Send a message

		Parameters
		----------
		message:
			must be [str]

		Returns
		----------
			True | False
				True: send the message successfully
				False: fail to send the message
		'''
		import os
		message = str(message)
		while True:
			if (message[-1] == '\n'): message = message[:-1]
			else: break
		fo = open(self._msgfile, 'a')
		fo.write(message+'\n')
		fo.close()
		fo = open(self._msgfile, 'r')
		msg = fo.read()
		if (msg[-len(message)-1:-1] == message): return True
		else: return False





	def Recv(self, message, row=False, wait=False):
		'''
		Judge whether can receive this message

		Parameters
		----------
		message:
			must be [str]

		row:
			[int] | list of [int] | False
			(1) row=False: not return this row
			(2) row=3: also return the content of the 3-th row below current row(current row is 0)
			(3) row=-3: also return the content of the 3-th row above current row
			(4) row=[-3,-2,-1,1,2]

		wait:
			False | True | [int]: second
			(1) wait=False: don't wait
			(2) wait=True: wait forever until receive
			(3) wait=[int]: wait ? seconds

		Returns
		----------
			(1) True | False
					True: receive successfully
					False: fail to receive
			(2) (True/False,  content)
		'''
		import os, time
		from jizhipy.Basic import IsType
		message = str(message)
		while True:
			if (message[-1] == '\n'): message = message[:-1]
			else: break
		message += '\n'
		if (IsType.isnum(row)): 
			row, isint = [int(row)], True
		else: isint = False
		startTime = time.time()
		result, isbreak = None, False
		while (not isbreak):
			fo = open(self._msgfile, 'r')
			msg = fo.readlines()
			fo.close()
			got = True if(message in msg)else False
			if (row is False or row is None): 
				result = got
			elif (got is False): result = (got, None)
			else:
				content = []
				n = msg.index(message)
				try: 
					for r in row:
						content.append(msg[n+r][:-1])
					if (isint): content = content[0]
				except: got = False
				result = (got, content)
			if (wait is False or wait is None): isbreak =True
			elif (wait is True): isbreak = got
			else:
				if (time.time()-startTime >= wait):
					isbreak = True
				else: isbreak = got
		return result


