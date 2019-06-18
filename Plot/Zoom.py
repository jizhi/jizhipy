
def Zoom( box1=None, box2=None, link1=None, link2=None, labeloff1=False, labeloff2=False, frameoff1=False, frameoff2=False, edgecolor1='k', edgecolor2='k', facecolor1='none', facecolor2='none' ) : 
	''' 
	Create ax on current fig
	NOT THAT on gcf(), NOT on gca()

	return:
		[ax1, lrbt1]  |  [ax1, ax2, lrbt1, lrbt2]

	box1, box2:
		box1={'left':, 'right':, 'bottom':, 'left':,}
		lrbt relate to the edge of figure

	link1, link2:
		tuple, any size
		link1 for box1, link2 for box2
			link box1 and box2
		if link2=None: link2=link1
		Example: 
			link1=(1,2), link2=None
				link box1-1 and box2=1, box1-2 and box2=2
			link1=(1,2), link2=(3,4)
				link box1-1 and box2=3, box1-2 and box2=4

	color=='none' mean transparent

	if link, lrbt1 is the box to the original data, lrbt2 is the zoom in

	Example 1:
		a = np.arange(20).reshape(5,4)
		x = np.arange(1000)-500
		y = np.exp(-x**2)
		x0 = np.linspace(100, 100+5*np.pi, 100)
		y0 = np.sin(x0)

		plt_color.face((1,0,0), alpha=0.1)
		plt.plot(x0, y0, 'b-')
		
		ax1, ax2, lrbt1, lrbt2 = plt_zoom([0.1, 0.4, 0.1, 0.4], [0.3, 0.9, 0.5, 0.95], (1,2), (4, 3))
		
		plt.sca(ax1)
		plt_axes.frameoff()
		lrbt1[1] *= 1.19
		ax1 = plt_zoom(lrbt1)[0]
		plt.pcolormesh(a)
		plt.colorbar()
		
		plt.sca(ax2)
		plt.plot([110, 230], [300, 370], 'g-')
		
		plt.show()


	Example 2:
		nside = 256
		hpmap = np.zeros(12*nside**2)
		npix = (jp.Random.Random(hpmap.size/20)*hpmap.size).astype(int)
		A = jp.Random.Random(npix.size)*2000+1000
		hpmap[npix] = A
		hpmap =hp.smoothing(hpmap, np.pi/180*2, verbose=False)

		lon, lat, dlon, dlat, Nlon, Nlat = 68, -2.5, 25, 25, 250, None 
		a = jp.CoordTrans.Healpix2Flat(lon, lat, dlon, dlat, Nlon, Nlat, hpmap)[-1]
		b = jp.CoordTrans.Healpix2Flat(lon, lat, dlon, dlat, Nlon, Nlat, hpmap)[-1]
	
		fig = plt.figure(figsize=(4,8))
		plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
		cmap = plt_color.cmap('jet')
	
		plt.annotate('21cm + noise', (0.1, 0.95), rotation=90)
		plt.annotate('extracted 21cm', (0.1, 0.715), rotation=90)
		plt.annotate('input 21cm', (0.1, 0.48), rotation=90)
		
		# 4 planes, each 0.25
		lrbt1 = [0.1, 1, 0.79, 0.99]
		ax = plt_zoom(lrbt1)[0]
		hp.mollview(hpmap, title='1', fig=fig.number, hold=True, cmap=cmap)
		
		lrbt2 = [0.1, 1, 0.555, 0.755]
		plt_zoom(lrbt2)
		hp.mollview(hpmap, title='2', fig=fig.number, hold=True, cmap=cmap)
		
		lrbt3 = [0.1, 1, 0.32, 0.52]
		plt_zoom(lrbt3)
		hp.mollview(hpmap, title='3', fig=fig.number, hold=True, cmap=cmap)
		
		# zoom in
		lrbt1 = [0.4, 0.45, 0.655, 0.68]
		lrbt2 = [0.02, 0.49, 0.01, 0.26]
		lrbt2 = [0.02, 0.48, 0.01, 0.25]
		plt_zoom(lrbt1, lrbt2, (3,4), (2,1), labeloff1=True, edgecolor='r')
		plt.pcolormesh(a, cmap=cmap)
		
		lrbt1 = [0.4, 0.45, 0.42, 0.445]
		lrbt2 = [0.52, 0.98, 0.01, 0.25]
		bg = (0, 0.7843, 0)
		plt_zoom(lrbt1, lrbt2, (3,4), (2,1), labeloff2=True, edgecolor='m')
		plt.pcolormesh(b, cmap=cmap)
		plt.show()

	hp.mollview():
        figsize = (8.5, 5.4) => w/h=1.57 | h/w=0.64
        lbwh = (0.02,0.05,0.96,0.9)  => 
		lrbt = (0.02,0.9,0.05,0.95)
	'''
	import numpy as np
	import matplotlib.pyplot as plt
	from jizhipy.Plot import Color, Axes
	if (facecolor1 is None) : facecolor1 = 'none'
	if (facecolor2 is None) : facecolor2 = 'none'
	if (edgecolor1 is None) : edgecolor1 = 'none'
	if (edgecolor2 is None) : edgecolor2 = 'none'
	#----------------------------------------
	def box2lrbt( box ) : return (box['left'], box['right'], box['bottom'], box['top'])
	def lrbt2box( lrbt ) : return {'left':lrbt[0], 'right':lrbt[1], 'bottom':lrbt[2], 'top':lrbt[3]}
	#----------------------------------------
	def LBWH( lrbt ) :  # conver lrbt to lbwh and xy
		l, r, b, t = lrbt
		w, h = r-l, t-b
	#	pars = FigureFrame()
		pars = {'left':0, 'right':1, 'bottom':0, 'top':1}
		w0 = pars['right'] - pars['left']
		h0 = pars['top']   - pars['bottom']
		l, w = pars['left']+l*w0, w*w0
		b, h = pars['bottom']+b*h0, h*h0
		lbwh = [l, b, w, h]
		r, t = l+w, b+h
		xy = np.array([(r,t), (l,t), (l,b), (r,b)])
		return lbwh, xy
	#----------------------------------------
	lrbt1, lrbt2 = None, None
	if (box1 is not None) : lrbt1 = box2lrbt(box1)
	if (box2 is not None) : lrbt2 = box2lrbt(box2)
	#----------------------------------------
	if (link1 is None) :  # NOT link
		ec0, fc0 = Color.Edgecolor(edgecolor1), Color.Facecolor(facecolor1)
		axlist, ax1, ax2, lbwh1, lbwh2 = [], None, None, None, None
		if (lrbt1 is not None) : 
			Color.Edgecolor(edgecolor1)
			Color.Facecolor(facecolor1)
			lbwh1 = LBWH( lrbt1)[0]
			ax1 = plt.axes(lbwh1)
			axlist.append(ax1)
			if (labeloff1) : 
				Axes.Label('both', left=False, right=False, top=False, bottom=False)
				Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
			if (frameoff1) : Axes.Frameoff(ax1)
		#----------------------------------------
		if (lrbt2 is not None) : 
			Color.Edgecolor(edgecolor2)
			Color.Facecolor(facecolor2)
			lbwh2 = LBWH(lrbt2)[0]
			ax2 = plt.axes(lbwh2)
			axlist.append(ax2)
			if (labeloff2) : 
				Axes.Label('both', left=False, right=False, top=False, bottom=False)
				Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
			if (frameoff2) : Axes.Frameoff(ax2)
		#----------------------------------------
		Color.Edgecolor(ec0), Color.Facecolor(fc0)
		if (lrbt1 is not None and lrbt2 is None) : 
			axlist = [ax1, lrbt2box(lrbt1)]
		elif (lrbt1 is not None and lrbt2 is not None) : 
			axlist = [ax1, ax2, lrbt2box(lrbt1), lrbt2box(lrbt2)]
		return axlist
	#----------------------------------------
	#----------------------------------------
	else :  # link
		from jizhipy.Basic import IsType
		from jizhipy.Array import Asarray
		from mpl_toolkits.axes_grid1.inset_locator import mark_inset
		if (link2 is None) : link2 = link1
		link1, link2 = Asarray(link1), Asarray(link2) 
		n = min(len(link1), len(link2))
		link1, link2 = link1[:n], link2[:n]
		#----------------------------------------
		ec0, fc0 = Color.Edgecolor('none'), Color.Facecolor('none')
		lbwh1 = LBWH(lrbt1)[0]
		ax1 = plt.axes(lbwh1)
		Axes.Frameoff(ax1)
		#----------------------------------------
		lbwh2, xy2 = LBWH(lrbt2)
		w0, h0 = 0.0001, 0.0001
		for i in range(n) : 
			l1, l2 = link1[i], link2[i]
			x, y = xy2[l2-1]
			if   (l1 == 1) : lbwh = [x-w0, y-h0, w0, h0]
			elif (l1 == 2) : lbwh = [x, y-h0, w0, h0]
			elif (l1 == 3) : lbwh = [x, y, w0, h0]
			elif (l1 == 4) : lbwh = [x-w0, y, w0, h0]
			ax2 = plt.axes(lbwh)
			Axes.Frameoff(ax2)
			mark_inset(ax1, ax2, l1, l1, fc='none', ec=edgecolor2) # when use mark_inset() to link, frame of ax1 can NOT be remove while ax2 can. Therefore, use axlist[0] to be original image, axlist[1] to be the zoom region.
		#----------------------------------------
		# Re-plot ax1, ax2
		Color.Edgecolor(edgecolor1)
		Color.Facecolor(facecolor1)
		h1, h2 = lbwh1[-1]*0.9999, lbwh2[-1]*0.9999
		d1, d2 = lbwh1[-1]*0.0001, lbwh2[-1]*0.0001
		lbwh1[-1] = h1
		lbwh2[-1] = h2
		ax1 = plt.axes(lbwh1)  #@#@
		if (labeloff1) : 
			Axes.Label('both', left=False, right=False, top=False, bottom=False, ax=ax1)
			Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
		if (frameoff1) : Axes.Frameoff(ax1)
		Color.Edgecolor(edgecolor2)
		Color.Facecolor(facecolor2)
		ax2 = plt.axes(lbwh2)  #@#@
		if (labeloff2) : 
			Axes.Label('both', left=False, right=False, top=False, bottom=False, ax=ax2)
			Axes.Tick('both', 'both', left=False, right=False, top=False, bottom=False)
		if (frameoff2) : Axes.Frameoff(ax2)
		Color.Edgecolor(ec0), Color.Facecolor(fc0)
		lrbt1, lrbt2 = list(lrbt1), list(lrbt2)
		lrbt1[-1] -= d1
		lrbt2[-1] -= d2
		axlist =[ax1,ax2,lrbt2box(lrbt1),lrbt2box(lrbt2)]
		return axlist

