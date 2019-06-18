from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement


def CrawlerBaiduImage(keyword, number, outdir, outname, whichurl='thumbURL', verbose=2, num_each=30): 
	'''
	On my Mac, 1 minute can download 500 pictures
	
	Parameters
	----------
	keyword:
		[str] keyword which is searched by baidu.com

	number:
		[int] total number of images would be downloaded
	
	outdir:
		[str] path of the output directory
	outname:
		[str] main_name of the picture without format
		outname[-1] != '_'
	
	verbose:
		[int] 0: not print | 1: print count | 2: print all
	
	whichurl:
		'thumbURL': small size
		'middleURL': medium size
		'objURL': original size
		'hoverURL': float version on mouse

	num_each:
		[int] number of images in one page
		for baidu.com is 30


	Example
	----------
	CrawlerBaiduImage('Trump', 123, 'Trump/', 'Trump', 'thumbURL', 2)
	'''
	import requests, os
	import jizhipy as jp
	verbose = int(verbose)
	if (verbose >=1): 
		print('keyword  =', keyword)
		print('number   = '+str(number))
		print('outdir   = '+outdir)
		print('outname  = '+outname)
		print('whichurl = '+whichurl)
	if (verbose == 1): progressbar = jp.Process.ProgressBar('***** Done:', number)
	figfmtlist = ['.jpg', '.png', '.bmp', '.gif']
	if (outdir[-1] != '/'): outdir += '/'
	outdir2 = os.path.expanduser(outdir)
	if (not os.path.exists(outdir2)): os.makedirs(outdir2)
	#----------------------------------------
	paramseach = {'tn':'resultjson_com', 'ipn':'rj', 
		'ct':201326592, 'is':'', 'fp':'result',
		'queryWord':'change_keyword', 'cl':2, 'lm':-1, 
		'ie':'utf-8', 'oe':'utf-8', 'adpicid':'', 'st':-1, 
		'z':'', 'ic':0, 'word':'change_keyword', 's':'', 
		'se':'', 'tab':'', 'width':'', 'height':'', 
		'face':0, 'istype':2, 'qc':'', 'nc':1, 'fr':'', 
		'pn':'change_i', 'gsm':'1e', '1488942260214':'', 
		'rn':num_each}
	params, num_tot = [], int(1.3*number)+num_each
	for i in range(num_each, num_tot, num_each):
		paramsi = paramseach.copy()
		paramsi['pn'] = i
		paramsi['queryWord'] = keyword
		paramsi['word'] = keyword
		params.append(paramsi)
	url = 'https://image.baidu.com/search/acjson'
	urllist = []
	for i in params: 
		try: urllist.append(requests.get(url, params=i).json().get('data'))
		except: pass
	#----------------------------------------
	strlen = len(str(number))
	count, isbreak = 0, False
	for urls in urllist:
		if (isbreak): break
		for url in urls:
			if (isbreak): break
			u = url.get(whichurl)
			if (u is not None):
				count += 1
				figfmt = u[u.rfind('.'):].lower()
				if (figfmt not in figfmtlist):
					for f in figfmtlist:
						if (f in figfmt):
							figfmt = f
							break
				if (figfmt not in figfmtlist): figfmt = '.jpg'
				if (verbose == 1): progressbar.Progress()
				elif (verbose == 2): print('['+str(count)+'] download from '+u)
				strcount = str(count)
				strcount=(strlen-len(strcount))*'0'+strcount
				figname=outdir2+outname+'_'+strcount+figfmt
				try: 
					pic = requests.get(u).content
					fo = open(figname, 'wb')
					fo.write(pic)
					fo.close()
				except: count -= 1
				if (count == number): isbreak = True
			else: 
			#	if (verbose ==0): print("["+str(count+1)+"] fail to download this image")
				pass
	if (verbose ==2): print('********** Images are saved to  '+outdir) # only for verbose=2, because verbose=2 prints a lot of information, hard to find the outdir


