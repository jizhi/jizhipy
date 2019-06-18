# -*- coding: utf-8 -*-

class Sporttery( object ) : 


	def Web( self, filename='sporttery_web.txt', match=None ) : 
		'''
		http://info.sporttery.cn/football/hhad_list.php
	
		http://info.sporttery.cn/football/match_result.php

		Return the probability of each case
		'''
		import numpy as np
		txt = open(filename).read().decode('utf-8')
		#txt =unicode(open('sporttery_web.txt').read(),'utf-8')
		txt = txt.split('\n')
		case = None
		for i in xrange(len(txt)) : 
			if ('VS' in txt[i] and txt[i][:2] in [' 0',' 1',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9']) : 
				if(txt[i+1]=='0' and txt[i+2] in ['-1','+1']):
					case = 1
				else : case = 4
				break
			if ('VS' in txt[i] and txt[i][:4] in np.arange(2017, 2117, 1).astype(str)) : 
				case = 2
				break
			if ('VS' in txt[i] and u'周'==txt[i][0]) : 
				case = 3
				break
			if (txt[i][:13] == u'竞彩网 > 联赛资料 > ') : 
				case = 5
				league = txt[i+5]
				break
		#----------------------------------------
		if (case == 1) : 
			a, i = [], 0
			while (i < len(txt)) : 
				if ('VS' in txt[i] and txt[i][:2] in [' 0',' 1',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9']) : 
					a1 = txt[i].find('\t')
					a1 = txt[i][a1+1:]
					if (a1[-1] == '\t') : a1 = a1[:-1]
					a2 = txt[i-2].find('\t')
					a2 = txt[i-2][a2+1:].split('\t')
					a3 = ' '+txt[i+1]+': '+txt[i+3][:4]+', '+txt[i+3][4:8]+', '+txt[i+3][8:12]
					a4 = txt[i+2]+': '+txt[i+4][:4]+', '+txt[i+4][4:8]+', '+txt[i+4][8:12]
					a.append(a2[0]+' '+a2[1]+'  '+a1)
					a.append(u'让:   胜,   平,   负')
					a.append(a3)
					a.append(a4)
					a.append('')
				i += 1
			txt = a[:-1]
		#----------------------------------------
		elif (case == 2) : 
			a = []
			for i in xrange(len(txt)) : 
				if ('VS' in txt[i] and txt[i][:4] in np.arange(2017, 2117, 1).astype(str)) : 
					n = txt[i].rfind(':')
					m = txt[i][n:].find('\t') + n
					c = txt[i][:m].split('\t')
					b = txt[i][m+1:m+1+14].split('\t')
					for j in xrange(1,len(c)): c[0]+='  '+c[j]
					for j in xrange(1,len(b)): b[0]+=', '+b[j]
					a.append(c[0])
					a.append(u'让:   胜,   平,   负')
					a.append( ' 0: '+b[0] )
				#	s = '-1' if('(-1)' in c[0])else '+1'
				#	a.append( s+': 0, 0, 0' )
					a.append('')
					a.append('')
			txt = a[:-1]
		#----------------------------------------
		elif (case == 3) : 
			n = []
			for i in xrange(len(txt)) : 
				if ('VS' in txt[i] and u'周'==txt[i][0]) : 
					n.append(i)
			a = ['' for i in xrange(len(n))]
			b = ['' for i in xrange(len(n))]
			c = ['' for i in xrange(len(n))]
			n += [len(txt)]
			i = 0
			while (i < len(txt)) : 
				for w in xrange(len(n)-1) :  # where is i?
					if (n[w] <= i < n[w+1]) : break
				if ('VS' in txt[i] and u'周'==txt[i][0]) : 
					a1 = txt[i].find(':')
					a1 = txt[i][:a1+3].split('\t')
					for j in xrange(1,len(a1)): a1[0]+=' '+a1[j]
					a2 = txt[i][len(a1[0])+3:].split('\t')[1]
					a[w] = a1[0]+'  '+a2
				if (txt[i][:2] in ['0:','1:','2:','3:','4:','5:','6:','7:','8:','9:'] or txt[i] in [u'胜其他',u'平其他',u'负其他']) : 
					if (u'其他' in txt[i]): b[w] +=txt[i]+'; '
					else : b[w] += txt[i]+', '
				if ('.' in txt[i] and len(txt[i])<=6) : 
					if (txt[i+2]==u'0:0' or txt[i+2]==u'0:1') : c[w] += txt[i]+'; '
					else : c[w] += txt[i]+', '
				i += 1
			txt = []
			for i in xrange(len(a)) : 
				txt.append(a[i])
				txt.append(u'比分: '+b[i][:-2])
				txt.append(u'比分: '+c[i][:-2])
				txt.append('')
				txt.append('')
			txt = txt[:-1]
		#----------------------------------------
		elif (case == 4) : 
			a, i = [], 0
			while (i < len(txt)) : 
				if ('VS' in txt[i] and txt[i][:2] in [' 0',' 1',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9']) : 
					a1 = txt[i].find('\t')
					a1, a5 = txt[i][a1+1:].split('\t')[:2]
					a2 = txt[i-2].find('\t')
					a2 = txt[i-2][a2+1:].split('\t')
					a5 = a5.split('.')
					for j in xrange(1, len(a5)) : 
					#	a5[0] += '.'+a5[j][:2]+',  '+a5[j][2:]
						a5[0] += '.'+a5[j][:2]+','+a5[j][2:]
					a.append(a2[0]+' '+a2[1]+'  '+a1)
					a1 = a5[0].split(',')
					for j in xrange(len(a1)) : 
						if (len(a1[j])==4) : a1[j] = ' '+a1[j]
					for j in xrange(1, len(a1)) : 
						a1[0] += ', '+a1[j]
					a1 = u'  赔率:  ' + a1[0][:-2]
					a2 = u'总进球:    0球,   1球,   2球,   3球,   4球,   5球,   6球,  7+球'
					a.append(a2)
					a.append(a1)
					a.append('')
					a.append('')
				i += 1
			txt = a[:-1]
		#----------------------------------------
		elif (case == 5) : 
			a = []
			for i in xrange(len(txt)) : 
				if (txt[i][:2] == u'时间' and len(a) == 0) : 
					a.append(league+': '+(' '.join(txt[i][3:].split('\t'))))
				if (txt[i][:2] == u'20' and u':' in txt[i]) : 
					n1 = txt[i].rfind('.')
					n1 = txt[i][:n1].rfind('\t')
					n2 = txt[i][:n1].find(':')
					n3 = txt[i][n2:].find('\t')
					b = txt[i][n2+n3+1:n1].split('\t')
					c = b[-1].split('.')
					for j in xrange(1, len(c)) : 
						c[0] += '.'+c[j][:2]+', '+c[j][2:]
					if (u'一' in c[0]) : c[0] = '0.00, 0.00, 0.00, '
					b = b[:-1] + [c[0][:-2]]
					a.append(' '.join(b))
			txt = a
		#----------------------------------------
		#----------------------------------------
		if (match is not None) : 
			a, N, i = [], len(txt), 0
			while (i < N) : 
				if (match in txt[i]) : 
					if (u'总进球' in txt[i+1]) : dn = 5
					a += txt[i:i+dn]
					i += dn
				else : i += 1
			txt = a[:-1]
		return txt




	def Sporttery( self, odds, nshow=5, which=[], order=[], chinese=True, N=6 ) : 
		from Print import Print
		import numpy as np
		total_number0 = np.array([N, N, N], int)
		if (len(order) == 0) : order = ['mean', 'max', 'min']
		if (len(which) == 0) : which = ['ms', 'lm', 'ls']  # ms=MiddleSmall, lm=LargeMiddle, ls=LargeSmall
		odds_name = ['win', 'draw', 'loss']
		odds = np.array(odds, float)
		if (odds.sum() < 1e-6) : 
			print 'NO cheat ball odds'
			return
		#----------------------------------------
	
		#----------------------------------------
		# number0, earn0: 无论什么结果，均净赚
		total_number = [100, 100, 100]
		total_number = np.array(total_number, int)
	
		number = np.zeros([total_number.prod(), total_number.size])
		for i in xrange(total_number.size) : 
			N = total_number[i+1:].prod()
			n = n1 = n2 = m = 0
			while (n2 < len(number)) : 
				n1, n2 = n*N, (n+1)*N
				number[n1:n2,i] = m % total_number[i]
				n, m = n+1, m+1
		number = number[1:]
	
		back = np.zeros(number.shape)
		for i in xrange(odds.size) : back[:,i] = number[:,i] * odds[i]
		earn = back - number.sum(1)[:,None]
		
		sign = np.sign(earn).sum(1)
		n = np.arange(sign.size)[sign>2.5]
		number0, earn0 = number[n], earn[n]
		#----------------------------------------
	
		#----------------------------------------
		# number: 买注组合
		total_number = total_number0
		number = np.zeros([total_number.prod(), total_number.size])
		for i in xrange(total_number.size) : 
			N = total_number[i+1:].prod()
			n = n1 = n2 = m = 0
			while (n2 < len(number)) : 
				n1, n2 = n*N, (n+1)*N
				number[n1:n2,i] = m % total_number[i]
				n, m = n+1, m+1
		number = number[1:]
		
		n = np.arange(total_number.max()-1, 1, -1)
		for i in xrange(len(number)) : 
			m = number[i:i+1] / n[:,None]
			m = abs(m - m.astype(int)).sum(1)
			m = n[m<1e-6]
			if (m.size > 0) : number[i] /= m[0]
		
		scale = list((total_number-1).astype(str))
		for i in xrange(len(scale)) : scale[i] = len(scale[i])
		scale = np.append(np.cumsum(scale[::-1])[::-1][1:], [0])
		n = (number * 10**scale).sum(1) + 1j*np.arange(len(number))
		
		m = []
		while (len(n) > 0) : 
			m.append(n[0].imag)
			n = n[n.real!=n[0].real]
		number = number[np.array(m,int)]
		#----------------------------------------
	
		#----------------------------------------
		# earn: 每付出1元净赚
		pay = number.sum(1)[:,None]
		
		back = np.zeros(number.shape)
		for i in xrange(odds.size) : back[:,i]=number[:,i]*odds[i]
		
		earn = (back - pay)/pay  # earn per yuan
		#----------------------------------------
	
		#----------------------------------------
		# number1, earn1: 每个组合，均有两种结果能净赚，一种结果净亏。而两种净赚结果，是赔率最低的两个，即概率最高的两个。这里“拒绝爆冷”。
		# sign= 3: 3 True
		# sign= 1: 2 True 1 False
		# sign=-1: 1 True 2 False
		# sign=-3: 3 False
		n = [0, 1, 2]
		m = np.where(odds==odds.max())[0][0]
		n.remove(m)
		sign = np.sign(earn)[:,np.array(n)].sum(1)
		n = np.arange(sign.size)[sign>0.5]
		number1, earn1 = number[n], earn[n]
		earn1 /= abs(earn1[:,m:m+1])
	
		# number2, earn2: 每个组合，均有两种结果能净赚，一种结果净亏。而两种净赚结果，是赔率最高的两个，即概率最低的两个。这里“期待爆冷”。
		n = [0, 1, 2]
		m = np.where(odds==odds.min())[0][0]
		n.remove(m)
		sign = np.sign(earn)[:,np.array(n)].sum(1)
		n = np.arange(sign.size)[sign>0.5]
		number2, earn2 = number[n], earn[n]
		earn2 /= abs(earn2[:,m:m+1])
	
		# number3, earn3: 每个组合，均有两种结果能净赚，一种结果净亏。而两种净赚结果，是赔率最高和最低的两个，“兼顾爆冷和保底”。
		m1 = np.where(odds==odds.min())[0][0]
		m2 = np.where(odds==odds.max())[0][0]
		n = [m1, m2]
		m = [0, 1, 2]
		m.remove(m1)
		m.remove(m2)
		m = m[0]
		sign = np.sign(earn)[:,np.array(n)].sum(1)
		n = np.arange(sign.size)[sign>0.5]
		number3, earn3 = number[n], earn[n]
		earn3 /= abs(earn3[:,m:m+1])
		#----------------------------------------
	
		#----------------------------------------
		if (len(earn1) > 0) : 
			e1 = earn1.copy()
			e1[e1<-0.5] = e1.max()*2
			nmin1 = e1.min(1) + 1j*np.arange(len(earn1))
			nmin1 = np.sort(nmin1)[::-1].imag.astype(int)
			nmax1 = earn1.max(1) + 1j*np.arange(len(earn1))
			nmax1 = np.sort(nmax1)[::-1].imag.astype(int)
			nmean1=e1.min(1)+earn1.max(1)+1j*np.arange(len(earn1))
			nmean1 = np.sort(nmean1)[::-1].imag.astype(int)
		else : nmin1 = nmax1 = nmean1 = np.array([])
	
		if (len(earn2) > 0) : 
			e2 = earn2.copy()
			e2[e2<-0.5] = e2.max()*2
			nmin2 = e2.min(1) + 1j*np.arange(len(earn2))
			nmin2 = np.sort(nmin2)[::-1].imag.astype(int)
			nmax2 = earn2.max(1) + 1j*np.arange(len(earn2))
			nmax2 = np.sort(nmax2)[::-1].imag.astype(int)
			nmean2=e2.min(1)+earn2.max(1)+1j*np.arange(len(earn2))
			nmean2 = np.sort(nmean2)[::-1].imag.astype(int)
		else : nmin2 = nmax2 = nmean2 = np.array([])
	
		if (len(earn3) > 0) : 
			e3 = earn3.copy()
			e3[e3<-0.5] = e3.max()*2
			nmin3 = e3.min(1) + 1j*np.arange(len(earn3))
			nmin3 = np.sort(nmin3)[::-1].imag.astype(int)
			nmax3 = earn3.max(1) + 1j*np.arange(len(earn3))
			nmax3 = np.sort(nmax3)[::-1].imag.astype(int)
			nmean3=e3.min(1)+earn3.max(1)+1j*np.arange(len(earn3))
			nmean3 = np.sort(nmean3)[::-1].imag.astype(int)
		else : nmin3 = nmax3 = nmean3 = np.array([])
	
		n1 = np.array([nmean1, nmax1, nmin1])
		n2 = np.array([nmean2, nmax2, nmin2])
		n3 = np.array([nmean3, nmax3, nmin3])

		N = total_number[0]-1.
		a = number1.max(1)
		number1[a==1] *= N
		number1[a==2] *= round(N/2)
		number1[a==3] *= round(N/3)
		a = number2.max(1)
		number2[a==1] *= N
		number2[a==2] *= round(N/2)
		number2[a==3] *= round(N/3)
		a = number3.max(1)
		number3[a==1] *= N
		number3[a==2] *= round(N/2)
		number3[a==3] *= round(N/3)
	
		earn0 = np.concatenate([number0, earn0], 1)
		earn1 = np.concatenate([number1, earn1], 1)
		earn2 = np.concatenate([number2, earn2], 1)
		earn3 = np.concatenate([number3, earn3], 1)
		#----------------------------------------
	
	
		Print('***** 大中小 *****', precision=2)
		print earn0
		if ('ms' in which) : 
			if (chinese) : print '***** 中小 *****'
			if ('mean' in order) : 
				if (chinese) : print '平均'
				if (n1[0].size > 0) : print earn1[n1[0]][:nshow]
				else : print earn1
			if ('max' in order) : 
				if (chinese) : print '最大'
				if (n1[1].size > 0) : print earn1[n1[1]][:nshow]
				else : print earn1
			if ('min' in order) : 
				if (chinese) : print '最小'
				if (n1[2].size > 0) : print earn1[n1[2]][:nshow]
				else : print earn1
	
		if ('lm' in which) : 
			if (chinese) : print '***** 大中 *****'
			if ('mean' in order) : 
				if (chinese) : print '平均'
				if (n2[0].size > 0) : print earn2[n2[0]][:nshow]
				else : print earn2
			if ('max' in order) : 
				if (chinese) : print '最大'
				if (n2[1].size > 0) : print earn2[n2[1]][:nshow]
				else : print earn2
			if ('min' in order) : 
				if (chinese) : print '最小'
				if (n2[2].size > 0) : print earn2[n2[2]][:nshow]
				else : print earn2
	
		if ('ls' in which) : 
			if (chinese) : print '***** 大小 *****'
			if ('mean' in order) : 
				if (chinese) : print '平均'
				if (n3[0].size > 0) : print earn3[n3[0]][:nshow]
				else : print earn3
			if ('max' in order) : 
				if (chinese) : print '最大'
				if (n3[1].size > 0) : print earn3[n3[1]][:nshow]
				else : print earn3
			if ('min' in order) : 
				if (chinese) : print '最小'
				if (n3[2].size > 0) : print earn3[n3[2]][:nshow]
				else : print earn3





	def Goal( self, match, update=None, goalname=None, odds=None ) : 
		'''
		match:
			必须为 unicode
			u'国冠杯', u'公开赛杯', u'解放者杯', u'俱乐部杯', u'墨西哥杯', u'澳杯', u'欧冠', u'英冠' 等等
	
		update:
			[int, int, int, int, int, int, int, int]
			更新该联赛最新的 0~7+球 数据
			或直接修改文件：Sporttery/total_goal.txt
	
		goalname:
			total_goal.txt文件路径，可以用默认值

		return:
			(1) odds is None: 返回当前概率
			(2) odds = jp.Sporttery.Web(): 返回[结果, 当前赔率]
		'''
		from Path import Path
		import numpy as np
		if (goalname is None) : goalname = Path.jizhipyPath('Sporttery/total_goal.txt')
		txt = open(goalname).read().decode('utf-8')
		n1 = txt.find(match)
		if (txt[n1-2] == '#') : 
			n2 = txt[n1:].find('\n')
			print txt[n1-2:n1+n2]
			n1 = n1+n2+1
		n2 = txt.rfind(match)
		n3 = txt[n2:].find('\n')
		txt = txt[n1:n2+n3].split('\n')
		#--------------------------------------------------
		name, goal = [], []
		for i in xrange(len(txt)) : 
			n = txt[i].find(':')
			name.append( txt[i][:n] )
			goal.append(np.array(txt[i][n+1:].split(','),int))
		goal = np.array(goal)
		#--------------------------------------------------
		Pexpect = goal.sum(0)
		Pexpect = 100. * Pexpect / Pexpect.sum()
		Nexpect = goal[1:].sum() / len(goal[1:])
		Gexpect = Nexpect * Pexpect/100
		#--------------------------------------------------
		Pnow = Gexpect - goal[0]
		Pnow = 100. * Pnow / Pnow.sum()
		#--------------------------------------------------
		# print
		a, b, c, d = match, u'总场数', u'总进球数', u'概率'
		n = max(len(a), len(b), len(c), len(d))
		a = 2*(n-len(a))*' '+a+':  '
		b = 2*(n-len(b))*' '+b+':  '
		c = 2*(n-len(c))*' '+c+':  '
		d = 2*(n-len(d)-1)*' '+d+' %:  '
		#----------
		for i in xrange(len(name)) : 
			a += name[i][len(match):]+',  '
			b += '%4i,  ' % goal[i].sum()
		c += u' 0球,   1球,   2球,   3球,   4球,   5球,   6球,   7+球,  '
		for i in xrange(len(Pnow)-1) : d += '%4.1f,  ' % Pnow[i]
		d += '%5.1f,  ' % Pnow[-1]
		print a[:-3]+'\n'+b[:-3]+'\n'+c[:-3]+'\n'+d[:-3]
		if (odds is None) : return Pnow
		#--------------------------------------------------
		#--------------------------------------------------
		#--------------------------------------------------
		result, odds, txt = [], [], odds
		for t in txt : 
			if (u'赔率' in t) : odds.append( np.array(t.split(':')[-1].split(','), float) )
		odds = np.array(odds)
		#--------------------
		for n in xrange(1, 9) : 
			a = self.Select(n, 8)
			for i in xrange(len(a)) : 
				Pnowi = Pnow[a[i]]
				oddi  = odds[:,a[i]]
				buy   = (oddi.prod(1)[:,None] / oddi)
				buy   = buy / buy.sum(1)[:,None] * 100
				cost  = buy.sum(1)
				earn  = buy[:,0] * oddi[:,0] - cost
				raw = np.zeros(odds.shape, int)
				raw[:,a[i]] = buy.round()
				#--------------------
				buystr = len(oddi)*['    买:  ']
				for j in xrange(len(buystr)) : 
					for k in xrange(raw.shape[1]) : 
						buystr[j] += '%5i, ' % raw[j,k]
				#--------------------
				earnstr = len(oddi)*['  结果:  ']
				resulti = []
				for j in xrange(len(earnstr)) : 
					earn1 = int(round(earn[j]))
					buy1 = int(round(buy[j].sum()))
					p = Pnowi.sum().round(1)
					if (p > 100) : p = 100
					pure = int(round((earn[j]*Pnowi.sum() - 100*(100-Pnowi.sum()))/100))
					purestr = str(pure) if(pure<=0)else '+'+str(pure)
					earnstr[j] += ('%4.1f' % p)+' % 赚 '+('%3i' % earn1)+' 元,  '+('%4.1f' % (100-p))+' % 亏 '+str(buy1)+' 元,  净 '+purestr+' 元  '+('(%i)' % len(result))
					resulti.append( (buystr[j][:-2]+'\n'+earnstr[j], [p, earn1, buy1, pure]) )
				result.append( resulti )
		return [result, Pnow]





	def Select( self, n, Ntot ) : 
		'''
		C_Ntot^n
		'''
		import numpy as np
		if   (n == 0) : return np.array([])
		elif (n == 1) : return np.arange(Ntot)[:,None]
		elif (n == Ntot) : return np.arange(Ntot)[None,:]
		Na = np.arange(Ntot+1-n,Ntot+1).prod() / np.arange(2,n+1).prod()
		a = np.random.randint(0, Ntot, (n,100*Na))
		for i in xrange(n-1) : 
			tf = (a[i:i+1] - a[i+1:]).prod(0)
			a = a[:,tf!=0]
		m = (a**3).sum(0)
		tf = np.ones(m.size, int)
		for i in xrange(len(m)-1) : 
			tf[i+1:] *= np.sign(abs(m[i] - m[i+1:]))
		a = a[:,tf.astype(bool)]
		for i in xrange(a.shape[1]) : a[:,i] = np.sort(a[:,i])
		m = a.sum(0) + 1j*np.arange(a.shape[1])
		m = np.sort(m).imag.astype(int)
		a = a[:,m]
		m = a[0] + 1j*np.arange(a.shape[1])
		m = np.sort(m).imag.astype(int)
		a = a[:,m].T
		return a





Sporttery = Sporttery()





if __name__ == '__main__' : 
	# -*- coding: utf-8 -*-
	import jizhipy as jp
	for i in xrange(int(round(len(txt)/4.))) : 
		nshow = 1
		order = ['min']
		which = ['lm']
		chinese = False
		print '----------------------------------------'
		info = txt[4*i:4*i+3]
		print info[0], '\n', info[1]
		odds = info[1].split(': ')[-1].split(', ')
		jp.Sporttery.Sporttery(odds,nshow,which,order,chinese)
		print '\n', info[0], '\n', info[2]
		odds = info[2].split(': ')[-1].split(', ')
		jp.Sporttery.Sporttery(odds,nshow,which,order,chinese)
		print '----------------------------------------\n'



