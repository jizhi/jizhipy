
def Graticule( hpmapORdlat=None, dlon=None, **kwargs ) : 
	'''
	(1) Graticule( dlat, dlon ):
			dlat, dlon in degree

	(2) Graticule( hpmap ):
			hpmap is healpix map
		Example:
			a = np.log10(jp.GSM2016(example=True))
			nside = hp.get_nside(a)
			hp.mollview(a, title='diffuse')
			b = np.zeros(12*nside**2)
			theta, phi = np.array(hp.pix2ang(nside, np.arange(12*nside**2)))*180/np.pi
			b[(89.5<theta)*(theta<90.5)] = 1
			b=jp.CoordTrans.HealpixRotation(b, ax=40)[-1]
			jp.Plt.Graticule(b, color='r', ls='--', lw=3)
			plt.show(), exit()

	(3) Graticule( border=True, **kwargs )
			plot border
	'''
	import matplotlib.pyplot as plt
	import numpy as np
	import healpy as hp
	from jizhipy.Basic import IsType
	from jizhipy.Array import Invalid
	from jizhipy.Plot import GetSize, Scf, Sca, Zoom, Axes
	fig0, ax0 = plt.gcf(), plt.gca()
	if ('border' in kwargs.keys()) : 
		theta_border = np.arange(0, 181)*np.pi/180
		border = bool(kwargs['border'])
		kwargs.pop('border')
	else : border = False
	#----------------------------------------
	if(IsType.isnum(hpmapORdlat) and IsType.isnum(dlon)):
		kwargs['verbose'] = False
		hp.graticule(hpmapORdlat, dlon, **kwargs)
		if (border) : 
			ax0.projplot(theta_border, theta_border*0-np.pi, direct=True, color='k', ls='-', lw=2)
			ax0.projplot(theta_border, theta_border*0+0.9999*np.pi, direct=True, color='k', ls='-', lw=2)
		return ax0
	#----------------------------------------
	elif (hpmapORdlat is not None) : 
		figsize, axsize = GetSize(fig0, ax0)
		fig = plt.figure(figsize=(figsize['width'], figsize['height']))
		hpmap = hp.mollview(hpmapORdlat, fig=fig.number, return_projected_map=True)
		plt.close(fig)
		del fig
		hpmap = Invalid(hpmap)
		hpmap.data[hpmap.mask] = 0
		hpmap = hpmap.data
		#----------------------------------------
		drow =(1.*axsize['top'] -axsize['bottom']) /hpmap.shape[0]
		dcol =(1.*axsize['right'] -axsize['left']) /hpmap.shape[1]
		x, y = [], []
		for i in range(1, hpmap.shape[1]) : 
			b = hpmap[:,i]
			n = np.arange(b.size)[b>0.5]
			if (n.size > 0) : 
				x.append(axsize['left'] + dcol*i)
				y.append(axsize['bottom'] + drow*n.mean())
		Scf(fig0)
		Sca(ax0)
		Zoom(axsize)
		plt.plot(x, y, **kwargs)
		plt.xlim(axsize['left'], axsize['right'])
		plt.ylim(axsize['bottom'], axsize['top'])
		Axes.Frameoff()
		plt.sca(ax0)
		if (border) : 
			ax0.projplot(theta_border, theta_border*0-np.pi, direct=True, color='k', ls='-', lw=2)
			ax0.projplot(theta_border, theta_border*0+0.9999*np.pi, direct=True, color='k', ls='-', lw=2)
		return ax0
	#----------------------------------------
	elif (border) : 
		key = kwargs.keys()
		if ('color' not in key) : kwargs['color'] = 'k'
		if ('lw' not in key and 'linewidth' not in key) : kwargs['lw'] = 2
		ax0.projplot(theta_border, theta_border*0-np.pi, direct=True, **kwargs)
		ax0.projplot(theta_border, theta_border*0+0.9999*np.pi, direct=True, **kwargs)
		return ax0
