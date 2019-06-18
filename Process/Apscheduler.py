# -*- coding:utf-8 -*-

class Apscheduler(object):

	def __init__(self, scheduler):
		'''
		https://apscheduler.readthedocs.io/en/latest/userguide.html?highlight=add_job

		Parameters
		----------
		scheduler:
			[str] 调度器，根据开发需求选择相应的调度器
			'BlockingScheduler' 阻塞式调度器：
				适用于只跑调度器的程序
			'BackgroundScheduler' 后台调度器：
				适用于非阻塞的情况，调度器会在后台独立运行
			'AsyncIOScheduler' AsyncIO调度器：
				适用于应用使用AsnycIO的情况
			'GeventScheduler' Gevent调度器：
				适用于应用通过Gevent的情况
			'TornadoScheduler' Tornado调度器：
				适用于构建Tornado应用
			'TwistedScheduler' Twisted调度器：
				适用于构建Twisted应用
			'QtScheduler' Qt调度器：
				适用于构建Qt应用
		'''
		import logging
		logging.basicConfig()
		scheduler = str(scheduler).lower()
		if ('blocking' in scheduler):
			from apscheduler.schedulers.blocking import BlockingScheduler
			self.scheduler = BlockingScheduler()
		elif ('background' in scheduler):
			from apscheduler.schedulers.background import BackgroundScheduler
			self.scheduler = BackgroundScheduler()
		elif ('asyncio' in scheduler): 
			from apscheduler.schedulers.asyncio import AsyncIOScheduler
			self.scheduler = AsyncIOScheduler()
		elif ('gevent' in scheduler): 
			from apscheduler.schedulers.gevent import GeventScheduler
			self.scheduler = GeventScheduler()
		elif ('tornado' in scheduler):
			from apscheduler.schedulers.tornado import TornadoScheduler
			self.scheduler = TornadoScheduler()
		elif ('twisted' in scheduler):
			from apscheduler.schedulers.twisted import TwistedScheduler
			self.scheduler = TwistedScheduler()
		elif ('qt' in scheduler):
			from apscheduler.schedulers.qt import QtScheduler
			self.scheduler = QtScheduler()





	def Add(self, trigger, func, args=(), kwargs={}, date=None, period=None, jobid=None, name=None, **add_kwargs):
		'''
		Parameters
		----------
		trigger:
			[str] 'cron': 某一定时时刻执行
			      'interval': 每隔多长时间执行
			      'date': 只有某一该执行一次（一次性任务）

		date:
			[str] 'year/month/day hour:minute:second week'
			week:
				(1) trigger='cron': week=星期几 [str]:
					'mon','tue','wed','thu','fri','sat','sun'
					(1.1) 每分钟的第5秒：date='// ::5'
					(1.2) 每小时的第5分钟：date='// :5:'
					(1.3) 每个星期五：date='// :: fri'
				(2) trigger='interval': week=每隔多少个星期[int]
			注意，data中最多只允许有2个空格，第1个空格是年月日与时分秒之间，第2个空格是时分秒与星期之间，其他地方的空格会造成时间解释错误，从而运行错误

		period:
			[str] period='start_date--end_date--timezone'
			start_date, end_date:
				[str] 'year/month/day'
			timezone:
				[str of int]
			(1) period='2019/5/21--2020/7/16--8'  => 开始日期2019/5/21，结束日期2020/7/16，时区8
			(2) period='2019/5/21--//--8'  => 开始日期2019/5/21，没有结束日期，时区8
			(3) period='//--2020/7/16--8'  => 没有开始日期，结束日期2020/7/16，时区8
			(4) period='2019/5/21--2020/7/16--'  => 开始日期2019/5/21，结束日期2020/7/16，local电脑系统默认时区
		'''
		import numpy as np
		def toint(x): 
			for i in range(len(x)):
				try: x[i] = int(x[i])
				except: pass
			return x
		date = date.split(' ')
		year, month, day = toint(date[0].split('/'))
		hour, minute, second = toint(date[1].split(':'))
		week = '' if(len(date)==2)else date[2]
		date_kwargs = {}
		#----------------------------------------
		if (trigger == 'cron'):
			if (year !=''): date_kwargs['year'] = year
			if (month !=''): date_kwargs['month'] = month
			if (day !=''): date_kwargs['day'] = day
			if (hour !=''): date_kwargs['hour'] = hour
			if (minute !=''): date_kwargs['minute'] = minute
			if (second !=''): date_kwargs['second'] = second 
			if (week !=''): date_kwargs['day_of_week'] = week 
		#----------------------------------------
		elif (trigger == 'interval'):
			if (week !=''): date_kwargs['weeks'] = int(week)
			if (day !=''): date_kwargs['days'] = day
			if (hour !=''): date_kwargs['hours'] = hour
			if (minute !=''): date_kwargs['minutes'] = minute
			if (second !=''): date_kwargs['seconds'] = second
		#----------------------------------------
		elif (trigger == 'date'):
			run_date = '%i-%i-%i %i:%i:%i' % (year, month, day, hour, minute, second)
			date_kwargs['run_date'] = run_date
		#----------------------------------------
		if (period is not None):
			period = period.split('--')
			if (period[0] !='//'): date_kwargs['start_date'] = '%s-%s-%s' % tuple(period[0].split('/'))
			if (period[1] !='//'): date_kwargs['end_date'] = '%s-%s-%s' % tuple(period[1].split('/'))
			if (len(period) ==3): date_kwargs['timezone'] = period[2]
		#----------------------------------------
		add_kwargs.update(date_kwargs)
		self.scheduler.add_job(func=func, trigger=trigger, args=args, kwargs=kwargs, id=jobid, name=name, **add_kwargs)





	def Get(self, jobid):
		'''
		Parameters
		----------
		jobid:
			(1) =None: return all jobs in a list
			(2) =[str]: return this job
		'''
		if (jobid is None): return self.scheduler.get_jobs()
		else: return self.scheduler.get_job(jobid)


	def Remove(self, jobid):
		'''
		'''
		if (jobid is None): self.scheduler.remove()
		else: self.scheduler.remove_job(jobid)


	def Pause(self, jobid):
		'''
		'''
		if (jobid is None): self.scheduler.pause()
		else: self.scheduler.pause_job(jobid)


	def Resume(self, jobid):
		'''
		'''
		if (jobid is None): self.scheduler.resume()
		else: self.scheduler.resume_job(jobid)





	def Start(self, *args, **kwargs):
		'''
		'''
		self.scheduler.start(*args, **kwargs)


	def Shutdown(self, *args, **kwargs):
		'''
		'''
		self.scheduler.shutdown(*args, **kwargs)

