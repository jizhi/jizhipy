# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement


def CrawlerBaiduWenku(website, page50, outdir, outname, browserDriverPath, logger='INFO', timeout=5): 
	'''
	browserDriverPath:
		How to download and use browserDriver ?
		Using Chrome for example. 
		(1) Check the Chrome version: 
			input  chrome://version  on the website bar
			You can see something like:
			Google Chrome	69.0.3497.92
			It means that the version is 69
		(2) Open website  http://chromedriver.storage.googleapis.com/index.html
			There are some documents from 2.0 to 2.9
			Open one of them, for example 2.46
			Then open notes.txt, then you will see:
				Supports Chrome v......
			Find that matches your Chrome, here 2.40 to 2.44 are OK for Chrome 69
			Then go the document 2.44, download the chromedriver for your system (linux, mac, win)
		(3) Unzip the downloaded file, then you will obtain an application named "chromedriver"
		(4) @browserDriverPath is the path of this "chromedriver"

	website:
		The url/website/link of the document on the wenku.baidu.com
		e.g. the website of document "人脸检测与识别关键技术研究" is "https://wenku.baidu.com/view/f267c923dd36a32d737581df.html?sxts=1552644845367"

	page50:
		For a thesis or some document which are very long: exceed 50 pages, some times default website is only for the front 50 page, for the next 50 pages, the website is 
		https://wenku.baidu.com/view/f267c923dd36a32d737581df.html?sxts=1552644845367&pn=51
			(added "&pn=51" behind the normal website)
		The parameter @page50 here is [int]:
			page50=False: only crawl @website is enough
			page50=51: crawl @website and @website+'&pn=51'
			page50=101: crawl @website, @website+'&pn=51', @website+'&pn=101'
		NOTE THAT &pn= is only 51, 101, 151, 201, ...

	outdir:
		[str] path of the output directory
	outname:
		[str] main_name of the picture without format
		NOTE THAT outname[-1] != '_'

	logger:
		=False (not print) | ='DEBUG' (print all) | 'INFo' (print count)

	timeout:
		How many seconds to be timeout (Default timeout=3)
	'''
	from selenium import  webdriver
	from selenium.webdriver.common.by import By
	from selenium.webdriver.support import expected_conditions as EC
	from selenium.webdriver.support.wait import WebDriverWait
	import time, re, requests
	from jizhipy.Basic import Logger, Path
	#----------------------------------------
	# Check arguments
	browserDriverPath = Path.AbsPath(browserDriverPath)
	Path.ExistsPath(browserDriverPath, stop=True)
	try: page50 = int(page50)
	except: page50 = False
	if (page50 not in range(51, 10000, 50)): page50 = ['']
	else: page50 = ['']+range(51, page50+2, 50)
	if (outdir[-1] !='/'): outdir += '/'
	outdirAbs = Path.AbsPath(outdir)
	Path.ExistsPath(outdirAbs, mkdir=True)
	outname = str(outname)
	if (outname[-1] =='_'): outname = outname[:-1]
	try: timeout = float(timeout)
	except: timeout = 5
	logger = Logger(logger, True)
	#----------------------------------------
	# __init__
	browser = webdriver.Chrome(browserDriverPath)
	wait = WebDriverWait(browser, timeout)
	# 匹配url(网址)信息的模板，url\("(.*?)"\) 相当于url("...")，其中"..."就是用来打开图片的网址
	pattern = re.compile('.*?url\("(.*?)"\)',re.S) 
	#----------------------------------------
	count = 0
	for p5 in page50:
		# Get the website
		if (p5 !=''): p5 = '&pn='+str(p5)
		browser.get(website+p5)
		# 定位到:鼠标'banner-more-btn'
		submit = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'banner-more-btn'))) 
		# 滑轮滑倒看得到鼠标的位置，不然鼠标会点不到
		browser.execute_script("arguments[0].scrollIntoViewIfNeeded(true);", submit) 
		# 让爬虫点击一下鼠标
		submit.click() 
		# 定位到第一个节点
		elem = wait.until(EC.presence_of_element_located((By.ID, 'reader-container-inner-1')))
		# 获取图片link
		for i in elem.find_elements_by_class_name('bd'):
			count += 1
			try: 
				browser.execute_script("arguments[0].scrollIntoViewIfNeeded(true);", i)
			#	time.sleep(0.6)	#@#@
				# 定位到存放图片信息的节点
				n = i.find_element_by_class_name('reader-pic-item')
				js = n.get_attribute('style')
				# findall，用模板匹配url(网址)
				href = pattern.findall(js)	
				link = href[0]
				# 保存图片
				html = requests.get(link).content
				stri = (4-len(str(count)))*'0'+str(count)
				imageName = outdirAbs+outname+'_'+stri+'.jpg'
				fo = open(imageName, 'wb')
				fo.write(html)
				fo.close()
				print(count)
			except: pass

