
def CrawlerBingImage(keyword, number, outdir, outname, verbose=2, num_each=35): 
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
	CrawlerImage.Baidu('Trump', 123, 'Trump/', 'Trump', 'thumbURL', 2)
	'''
	import os
	import urllib.request
	import urllib.parse
	from bs4 import BeautifulSoup
	print('Running ...')
	verbose = int(verbose)
	figfmtlist = ['.jpg', '.png', '.bmp', '.gif']
	if (outdir[-1] != '/'): outdir += '/'
	outdir2 = os.path.expanduser(outdir)
	if (not os.path.exists(outdir2)): os.makedirs(outdir2)
	PageNum = int(1.3*(number/num_each+1))
	InputData = urllib.parse.quote(keyword)
	#----------------------------------------
	url = 'http://cn.bing.com/images/async?q={0}&first={1}&count=35&relp=35&lostate=r&mmasync=1&dgState=x*175_y*848_h*199_c*1_i*106_r*0'
	agent = {'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.165063 Safari/537.36 AppEngine-Google."}
	strlen = len(str(number))
	count, isbreak = 0, False
	for i in range(PageNum):
		if (isbreak): break
		try: 
			page1 = urllib.request.Request(url.format(InputData, i*num_each+1), headers=agent)
			page = urllib.request.urlopen(page1)
			soup =BeautifulSoup(page.read(),'html.parser')
			for StepOne in soup.select('.mimg'):
				if (isbreak): break
				link = StepOne.attrs['src']
				count += 1
				try:
					strcount = str(count)
					strcount = (strlen-len(strcount))*'0'+strcount
					figfmt = '.jpg'
					figname = outdir2+outname+'_'+strcount+figfmt
					urllib.request.urlretrieve(link, figname)
					if (count == number): isbreak = True
				except: 
					count -= 1
					if (verbose != 0): print("["+str(count+1)+"] fail to download this image")
		except:
			if (verbose != 0): print("["+str(count+1)+"] fail to open this page")
	print('\nImages are saved to  '+outdir)

