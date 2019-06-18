
def Savefig( *args, **kwargs ) : 
	'''
	For pcolor:
		figsize = (6.1, 5)
		axsize  = (0.66, 0.8)
		imgsize = (4, 4) inch
		*_temp.png to *.pdf, need dpi >= 300 for good quality, then size of *.pdf is 200 KB, 1200x1200 pixels
		however, if plt.savefig(*.pdf) for 1200x1200 image, size ~10 MB
	*** NOTE THAT some times, array is small like 100x100, jp.Plt.Savefig(*.pdf, vector=True) is smaller than jp.Plt.Savefig(*.pdf, vector=False)

	plt.figure(figsize=(8,6))
	plt.savefig('fig.png')
		fig.png's size = (800, 600)
			figsize=(8,6): 8inch x 6inch
		savefig(dpi=100): each inch 100 pixel
		so, fig.png's size = 800pixel x 600pixel

	Therefore, 
		plt.figure(figsize=(8,6))
		plt.savefig('fig.png', dpi=300)
			== (only the image equal, the text/label become small !!!)
		plt.figure(figsize=(24,18))
		plt.savefig('fig.png', dpi=100)

	kwargs:
		vector: 
			==True: plt.savefig(*.pdf)
			==False: save to *_temp.png, then convert to *.pdf
		png:
			be used only when vector=False
			==False: delete *_temp.png after converted to *.pdf
			==True: save the *_temp.png

	for hp.mollview()
		(1) plt.savefig('test.pdf')
		(2) plt.savefig('test.png') + magick test.png ./test.pdf
		method (1) and (2) have the same size of pdf
	'''
	import matplotlib.pyplot as plt
	import os
	if (not (args[0][-4:]=='.pdf' and len(plt.gca().collections)>0)) : 
		return plt.savefig(*args, **kwargs)
	#----------------------------------------
	figname = args[0]
	outdir = '' if('/' in figname)else './' 
	if ('vector' in kwargs.keys()) : 
		vector = bool(kwargs['vector'])
		kwargs.pop('vector')
	else : vector = False
	if ('png' in kwargs.keys()) : 
		png = bool(kwargs['png'])
		kwargs.pop('png')
	else : png = False
	if (not vector) : 
		try : 
			dpi = kwargs['dpi']
			if (dpi is None) : raise
		except : 
		#	im = plt.gca().collections[-1]
		#	w, h = im._coordinates.shape[:2]
		#	figsize,axsize = GetSize(plt.gcf(), plt.gca())
		#	wfig, hfig=figsize['width'],figsize['height']
		#	wax = axsize['right'] - axsize['left']
		#	hax = axsize['top'] - axsize['bottom']
		#	wpix, hpix = wfig*wax, hfig*hax
		#	dpi = int(max(w/wpix, h/hpix))
		#	if (dpi < 200) : dpi = 200
		#	if (dpi > 300) : dpi = 300
			dpi = 300  # fix 300
		kwargs['dpi'] = dpi
		#----------------------------------------
		args = list(args)
		args[0] = args[0][:-4] + '_temp.png'
		refig = plt.savefig(*args, **kwargs)
		os.system('magick '+args[0]+' '+outdir+figname+' verbose=False')
		if (not png) : os.system('/bin/rm '+args[0])
		return refig
	#----------------------------------------
	else : 
		refig = plt.savefig(*args, **kwargs)
		os.system('magick '+figname+' '+outdir+figname+' verbose=False')
		return refig
