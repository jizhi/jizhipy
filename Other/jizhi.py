import multiprocessing
import traceback
import warnings
import gc
from copy import deepcopy as dcopy
from types import *
import scipy.signal as spsn
import scipy.linalg as spla
from scipy.linalg import det
from scipy.signal import argrelextrema as spsaa
import scipy.special as spsp
from scipy.optimize import leastsq, curve_fit
#from mpl_toolkits.mplot3d import Axes3D
#from iminuit import Minuit, describe
#from scipy.interpolate import interp1d
#from scipy.interpolate import interp2d
#from scipy.ndimage import map_coordinages
from scipy import interpolate
from scipy.integrate import quad
from scipy.integrate import dblquad
#import healpy as hp
import ephem
import pyfits
import aipy
from optparse import OptionParser 
import h5py

# Label of colorbar:
# cbar = plt.colorbar()
# cbar.set_label('K')

# a.size=1e8, 64bit, folat/int/complex, 800MB
# a.size=1e8, 32bit, folat/int, 400MB
# a.size=1e8, 16bit, int, 200MB
# a.size=1e8, 8bit , bool, 100MB

# class test( object ) : 
# 	dtype = 'class:'+sys._getframe().f_code.co_name

# When use plt.pcolormesh(), plt.figure(figsize=(a,b)), if want to plot a square (NxN) and plt.colorbar(), must a=1.28*b

# Plot 3D: 
# 	ax = plt.figure(figsize=()).gca(projection='3d')
# 	ax.plot(x, y, z, 'r-', lw=2, label='')
# 	ax.set_xlim3d(0,1)
# 	ax.set_ylim3d(0,2)
# 	ax.set_zlim3d(0,3)
# 	ax.legend()
# 	ax.view_init(30, -60)  # ax.view_init(0,-0.1)
# 	plt.show()

# Scientific notation:
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

# matplotlib, plot marker without line:
# plt.plot(x, y, ls='', marker='o')

# fig = plt.gcf()
# ax  = plt.gca()

# hp.graticule(interval_latitude, interval_longitude, verbose=False, color='w'), set verbose=False to avoid "ValueError: Unknown format code 'd' for object of type 'float'"

# hp.mollview(map, coord='gc'), coord='gc' means convert from galactic to equatorial

# a = hp.mollview(np.log10(h), return_projected_map=True, xsize=800, coord='gc', rot=(0,90,0))


##################################################
##################################################
##################################################


# list of the function

# ReNameLog()
# Showtime()
# ShowProcess()
# PlotXY()
# CalRoot() np.roots()
# EquatorialGalactic()
# RADecxyz()
# ConjugateSymmetrize()
# SequenceArray()
# CalCosmologyDistance()
# CalRedshift()
# CAMB()
# GSM()
# Gaussian()
# Precision()
# ArcAngleRADec()
# T21_average()
# GetMatch()
# Taken()
# PixelCoordiante()
# LocalTime2UTC()
# GaussianValue()
# LogNormalValue()
# Fit()
# Histogram()
# ProbabilityDensity()
# CMBtemperature()
# GaussianBeam()
# EllipticGaussianBeam()
# ThetaPhiMatrix()
# BeamModel()
# Jy2K()
# PCA()
# Convolve()
# CoordConvert()
# hp_Alm_getidx_lm()
# Legendre()
# SphericHarmonic()
# healpix2xy()
# xy2healpix()
# HealpixRegion2Flat()
# k_FFT_smoothing()
# RMS()
# IndexA1A2()
# IndexValue()
# arcsinh2log()
# npfmt()
# ConvertPhaseLinear( phase ) : 
# RemoveRFI( array, filtersize=7, smoothtime=0, smoothsize=3 ) : 
# plt_scinot( axis ) : 
# Compile( code=None, machine='bao' ) : 
# SelectLeastsq( a, axis, average=False, usefor=True ) : 

##################################################
##################################################
##################################################


"""
    Use HealpixRegion2Flat() to select a sky region from GSM, then fit the Gaussian beam of GSM, we get the result: FWHMfit.
    But actually we need resolution of Flat map to be infinited to get the accurate FWHMgsm. Also FWHMfit relates to the nside of GSM map.
    If we the fitting result is FWHMfit, and the real FWHM of GSM is FWHMgsm, then they have a relation:
			FWHMgsm = FWHMfit / (1 + 0.0012 * 1024/nside)
    So, for nside=1024 GSM map, FWHMgsm=FWHMfit/1.0012, and for nside=512, FWHMgsm = FWHMfit/1.0024
"""


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################



##################################################
##################################################
##################################################



##################################################
##################################################
##################################################


##################################################
##################################################
##################################################



##################################################
##################################################
##################################################


##################################################
##################################################
##################################################

	
''' Please use numpy.roots() '''
def SolvePolynomial( p, optimize=True ) : 
	'''
	Calculate the root of n=len(p) order equation.
	p[0]*x^n + p[1]*x^(n-1) + ... + p[-2]*x + p[-1] = 0

	Can calculate any order of polynomial equation !
	'''
	ro = np.roots(p)
	if (optimize is True) : 
		re, im = [], []
		for i in range(len(ro)) : 
			if (abs(ro[i].imag) < 1e-10) : re = re + [ro[i].real]
			else : im = im + [ro[i]]
		re, im = list(Sort(re)), list(Sort(im))
		ro = re + im
	return ro


#def RootFunc( func, rootrange=None, N=1e7, err=1 ) : 
def Solve( func, rootrange=None, N=1e7, err=1, twice=False ) : 
	'''
	When we solve the non-linear equation with scipy.optimize.fsolve(), we need to set the initial guess.

	func:
		func(x) or func(x,p)
		We let func(x)=0 to get x(root)
		func is from:
			def func(x) : 
				return ......

	rootrange:
		list [], rootrange = [xmin, xmax]
		Root may be in this range.
		If None, set a very large range.

	N:
		Number of the points/bins.
		The larger the N is, the more precise the result is!

	err:
		error of the result, err>=0.
		Ideally, func(x)=0. Now, if func(x)<err, we assume this x is correct.

	twice:
		True or False. If True, do xr again, the result will be better.

	Return:
		Maybe return some roots, and the reliability decreases
	'''
	if (rootrange is None) : rootrange = [-1e8, 1e8]
	if (N > 1e8) : N = 1e8
	err = abs(err)
	x = np.linspace(rootrange[0], rootrange[1], N)
	y = abs(func(x)) + 1j*np.arange(N)
	y = Sort(Invalid(y,False))[:10]
	xr = np.zeros([10,])
	for i in range(10) : 
		ni = int(y[i].imag)
		xi = np.linspace(x[ni-10], x[ni+10], N)
		yi = abs(func(xi)) + 1j*np.arange(N)
		yi = Sort(yi)[0]
		xr[i] = xi[int(yi.imag)]
	x = y = xi = yi = 0 #@
	if (twice is True) : 
		xr2 = Sort(xr)
		for i in range(10) : 
			if (i == 0) : n1, n2 = i+1, i+1
			elif (i == 9) : n1, n2 = i-1, i-1
			else : n1, n2 = i-1, i+1
			x1 = xr2[i] - (xr2[i]+xr2[n1])/2.
			x2 = xr2[i] + (xr2[i]+xr2[n2])/2.
			xi = np.linspace(x1, x2, N)
			yi = abs(func(xi)) + 1j*np.arange(N)
			yi = Sort(yi)[0]
			xr[i] = xi[int(yi.imag)]
	yr = abs(func(xr)) + 1j*np.arange(xr.size)
	yr = Sort(yr)
	yr = yr[yr.real<err]
	xr = xr[(yr.imag).astype(int)]
	x = [xr[0]]
	for i in range(len(xr)) : 
		n = 1
		for j in range(len(x)) : 
			y = abs((xr[i]-x[j])/x[j])
			if (y < 1e-3) : n = 0
		if (n == 1) : x = x + [xr[i]]
	return np.array(x)
		

#def CalRoot( a, b, c=0, d=0, e=0 ) : 
#	'''
#	Calculate the root of n order equation.
#
#	y = a + b*x + c*x^2 + d*x^3 + e*x^4 = 0
#
#	Parameter:
#		n: highest order, x^n
#		a, b, c, d, e: parameter of the function, a + b*x + c*x^2 + d*x^3 + e*x^4
#
#	Return: list [x1, x2, ...]
#	'''
#	# set n
#	if (e != 0) : n = 4
#	elif (d != 0) : n = 3
#	elif (c != 0) : n = 2
#	elif (b != 0) : return -1.*a/b
#	else : raise Exception("Error: CalRoot(), must at least b!=0")
##	a, b, c, d, e = [e, d, c, b, a]
#
#	def CalRoot2( a, b, c ) : 
#		delta = b**2 - 4*a*c
#		if ( type(delta) != complex ) : 
#			if ( delta < 0 ) : rd = (abs(delta))**0.5 * 1j
#			else : rd = delta**0.5
#		else : rd = delta**0.5
#		y1 = (-b + rd) / 2.0 / a
#		y2 = (-b - rd) / 2.0 / a
#		if ( -1e-10 < y1.imag < 1e-10 ) : y1 = y1.real
#		if ( -1e-10 < y2.imag < 1e-10 ) : y2 = y2.real
#		return [y1, y2]
#	
#	def CalRoot3( a, b, c, d ) : 
#		p1 = -b/3./a
#		p2 = b*c/6./a**2 - b**3/27./a**3 - d/2./a
#		p3 = c/3./a - b**2/9./a**2
#		delta = p2**2 + p3**3
#		##
#		if (delta < -1e-20) : d2 = (-delta)**0.5*1j
#		else : d2 = abs(delta)**0.5
#		##
#		ppd = p2 + d2
#		pmd = p2 - d2
#		if (ppd < -1e-20) : ppd3 = -abs(ppd)**(1/3.)
#		else : ppd3 = abs(ppd)**(1/3.)
#		if (pmd < -1e-20) : pmd3 = -abs(pmd)**(1/3.)
#		else : pmd3 = abs(pmd)**(1/3.)
#		##
#		x1 = p1 + ppd3 + pmd3
#		x2 = p1 + (-1+3**0.5*1j)/2*ppd3 + (-1-3**0.5*1j)/2*pmd3
#		x3 = p1 + (-1-3**0.5*1j)/2*ppd3 + (-1+3**0.5*1j)/2*pmd3
#		if (abs(x1.imag) < 1e-20) : x1 = x1.real
#		if (abs(x2.imag) < 1e-20) : x2 = x2.real
#		if (abs(x3.imag) < 1e-20) : x3 = x2.real
#		return [x1, x2, x3]
#	
#	def CalRoot4( a, b, c, d, e ) : 
#		# convert a to 1, then the equation becomes x^4 + a*x^3 + b*x^2 + c*x + d = 0
#		a0 = a
#		a = 1.0 * b / a0
#		b = 1.0 * c / a0
#		c = 1.0 * d / a0
#		d = 1.0 * e / a0
#		# first calculate y from CalRoot3()
#		y = CalRoot3( 8, -4*b, 2*a*c-8*d, 4*b*d-a**2*d-c**2 )
#		for i in np.arange( 3 ) : 
#			if ( type(y[i]) != complex ) : 
#				y = y[i]
#				break
#		m = (8*y + a**2 - 4*b)**0.5
#		if ( -1e-10 < m < 1e-10 ) : 
#			Error('m = 0')
#		n = a*y - c
#		y1, y2 = CalRoot2( 2, a+m, 2*(y+n/m) )
#		y3, y4 = CalRoot2( 2, a-m, 2*(y-n/m) )
#		if ( -1e-10 < y1.imag < 1e-10 ) : y1 = y1.real
#		if ( -1e-10 < y2.imag < 1e-10 ) : y2 = y2.real
#		if ( -1e-10 < y3.imag < 1e-10 ) : y3 = y3.real
#		if ( -1e-10 < y4.imag < 1e-10 ) : y4 = y4.real
#		return [y1, y2, y3, y4]
#
#	if   ( n == 2 ) : return CalRoot2( c, b, a )
#	elif ( n == 3 ) : return CalRoot3( d, c, b, a )
#	elif ( n == 4 ) : return CalRoot4( e, d, c, b, a )


##################################################
##################################################
##################################################



#	Mathematical formulas
#	RAgp = 192.85948 * np.pi / 180.0
#	Decgp = 27.12825 * np.pi / 180.0
#	lcp = 122.932 * np.pi / 180.0
#
#	def equatorial2galactic( RA, Dec ) :
#
#		if ( type(RA) != list and type(RA) != np.ndarray ) : RA, Dec = np.array([ RA ]), np.array([ Dec ])
#		elif ( type(RA) == list ) : RA, Dec = np.array( RA ), np.array( Dec )
#	
#		sinb = np.sin( Dec ) * np.sin( Decgp ) + np.cos( Dec ) * np.cos( Decgp ) * np.cos( RA - RAgp )
#		cosbsinlcp_l = np.cos( Dec ) * np.sin( RA - RAgp )
#		cosbcoslcp_l = np.sin( Dec ) * np.cos( Decgp ) - np.cos( Dec ) * np.sin( Decgp ) * np.cos( RA - RAgp )
#		
#		b = np.arcsin( sinb )
#		sinlcp_l = cosbsinlcp_l / np.cos( b )
#		coslcp_l = cosbcoslcp_l / np.cos( b )
#
#		sincoslcp_l = sinlcp_l + 1j*coslcp_l
#		sinlcp_l = coslcp_l = 0 #@
#		a = sincoslcp_l[sincoslcp_l.real>0]
#		c = sincoslcp_l[sincoslcp_l.real<=0]
#		d = c[c.imag<0]
#		e = c[c.imag>=0]
#
#		a = np.arccos( a.imag )
#		d = 2 * np.pi - np.arccos( d.imag )
#		e = np.arcsin( e.real )
#
#		sincoslcp_l[sincoslcp_l.real>0] = a
#		c[c.imag>=0] = e
#		c[c.imag<0] = d
#		sincoslcp_l[sincoslcp_l.real<=0] = c
#		a = c = d = e = 0 #@
#
#		lcp_l = sincoslcp_l.real
#		l = lcp - lcp_l
#		l[l<0] = 2 * np.pi + l[l<0]
#		
#		if ( l.size == 1 ) : return np.array( [ l[0], b[0] ] )
#		else : return np.array( [ l, b ] )
#			
#	##############################
#	
#	def galactic2equatorial( l, b, In, Out, Print ) : 
#	 
#		if ( In == 1 ) : 
#			l = l * np.pi / 180
#			b = b * np.pi / 180
#	
#		sinDec = np.sin( Decgp ) * np.sin( b ) + np.cos( Decgp ) * np.cos( b ) * np.cos( lcp - l )
#		cosDecsinRA_RAgp = np.cos( b ) * np.sin( lcp - l )
#		cosDeccosRA_RAgp = np.cos( Decgp ) * np.sin( b ) - np.sin( Decgp ) * np.cos( b ) * np.cos( lcp - l )
#	
#		Dec = np.arcsin( sinDec )
#		sinRA_RAgp = cosDecsinRA_RAgp / np.cos( Dec )
#		cosRA_RAgp = cosDeccosRA_RAgp / np.cos( Dec )
#	
#		if ( sinRA_RAgp * cosRA_RAgp > 0 ) : RA_RAgp = np.arcsin( sinRA_RAgp )
#		elif ( sinRA_RAgp * cosRA_RAgp < 0 ) : 
#			if ( sinRA_RAgp > 0 ) : RA_RAgp = np.arccos( cosRA_RAgp )
#			else : RA_RAgp = np.pi + abs( np.arcsin( sinRA_RAgp ) )
#		elif ( sinRA_RAgp * cosRA_RAgp == 0 ) : 
#			if ( cosRA_RAgp > 0 ) : RA_RAgp = 0
#			elif ( cosRA_RAgp < 0 ) : RA_RAgp = np.pi
#			elif ( sinRA_RAgp > 0 ) : RA_RAgp = np.pi / 2
#			elif ( sinRA_RAgp < 0 ) : RA_RAgp = 3 * np.pi / 2
#		RA = RAgp + RA_RAgp
#		if ( RA > 2 * np.pi ) : RA = RA - 2 * np.pi
#		elif ( RA < 0 ) : RA = RA + 2 * np.pi
#	
#		if ( Out == 1 ) : 
#			RA = RA * 180 / np.pi / 15
#			h = int( RA )
#			remin = ( RA - h ) * 60
#			min = int( remin )
#			sec = ( remin - min ) * 60
#			if ( Dec < 0 ) : sign = -1
#			else : sign = 1
#			Dec = Dec * 180 / np.pi
#			deg = int( abs(Dec) )
#			rearcmin = ( abs(Dec) - deg ) * 60
#			arcmin = int( rearcmin )
#			arcsec = ( rearcmin - arcmin ) * 60
#			deg = sign * deg
#			RA = ('%2i' % h)+'h '+('%2i' % min)+'m '+('%5.2f' % sec)+'s'
#	
#		return np.array( [ RA, Dec ] )
#
#	if   ( mode == 0 ) : return equatorial2galactic( para1, para2 )
#	elif ( mode == 1 ) : return galactic2equatorial( para1, para2 )


##################################################
##################################################
##################################################


def RADecxyz( func, dc, RA0=0, Dec0=0, RA_range=0, Dec_range=0, RA=0, Dec=0, Lx=0, Ly=0, x=0, y=0 ) : 
	'''
	for default, center of RADec sky region is (RA0,Dec0)=(0,0), center of the xy plane is (Lx/2,Ly/2)=(0,0)
	and note that center of the plane (Lx/2,Ly/2) and center of the sky region (RA0,Dec0) are at the same line of sight, and this line of sight is perpendicular to the xy plane

	func=0 -> run RADec2boxpoint, give (RA,Dec) and return (x,y) in the plane. if center (RA0,Dec0)=(0,0)(default), RA and Dec are the coordinate related to the center of the sky region, if not, give a center (RA0,Dec0)!=(0,0), then RA and Dec are the absolute coordinate in the Equatorial coordinate system

	func=1 -> run boxpoint2RADec, give (x,y) in the plane and return (RA,Dec). if center (Lx/2,Ly/2)=(0,0)(default), x and y are the coordinate related to the center of the plane (x0,y0)=(Lx/2,Ly/2), if not, give the edge length of the plane (Lx,Ly) (so that the center is (Lx/2,Ly/2)), then x and y are the absolute coordinate in the xy coordinate system

	func=2 -> give dc, Lx, Ly, RA0, Dec0(RA and Dec of the line of sight of the sky region center), return the capable observable area of the sky region (RA_range, Dec_range, centered at Dec0)

	dc is the comoving distance of the box plane which you want to calculate (one of the slices of the box with index k->frequency) aparting from the Earth in h^{-1}Mpc, get it from the corresponding frequency
	for different frequency, we must use different dc

	Lx is the length of the edge of this box slice in h^{-1}Mpc
	also Ly

	RA0, Dec0 are the equatorial coordinate of the line of sigh of the box center

	RA_range is the RA np.arange, Dec_range is the Dec np.arange, they give the area of the sky region which cutted from the box at RA, Dec in rad 

	(RA,Dec) and (x,y), can just choose and set one pair
	if set (RA,Dec), that means you give the RA, Dec, and want to get the corrosponding xy coordinate in the box
	if set (x,y), it means you give xy coordinate in the box, and want to get its corrosponding RADec coordinate

	dc can be a number or a list or an array, but the their sizes must be same
	if dc is an array, then the form of the return : 
	RADec2xyz( RA, Dec ) : first index->x,y, second index->dc sequency, third index->different x or y obtained by different RA,Dec

	RA, Dec, can be a number or a list or an array, but the their sizes must be same
	x, y, can be a number or a list or an array, but the their sizes must be same
	'''	
	# RA,Dec, x,y must be convert to array first
	# each pair (RA,Dec) and (x,y) must havs same size
	if ( type(RA) != np.ndarray ) : 
		if ( type(RA) != list ) : 
			RA, Dec = np.array( [ RA ] ), np.array( [ Dec ] )
		else : RA, Dec = np.array( RA ), np.array( Dec )
	if ( type(x) != np.ndarray ) : 
		if ( type(x) != list ) : 
			x, y = np.array( [ x ] ), np.array( [ y ] )
		else : x, y = np.array( x ), np.array( y )

	Type = 0
	if ( type(dc)==np.ndarray or type(dc)==list ) : 
		if ( type(dc) == list ) : dc = np.array( dc )
		Type = 1
#		dc = dc[:,None]

	# a is an array
	def GetSign( a, a0 ) : 
		a = a - a0
		a[a<0] = -1
		a[a>=0] = 1
		return a

	# at Dec circle with RA angle -> RA_range, call it curve. calculate the central angel of this curve
	def RArange2CentralAngle( Dec, RA_range ) : 
		central_angle = 2 * np.arcsin( np.sin( RA_range/2.0 ) * np.cos( Dec ) )
		return central_angle
	def CenteralAngle2RArange( Dec, centeral_angle ) : 
		RArange = 2 * np.arcsin( np.sin( centeral_angle/2.0 ) / np.cos(Dec) )
		return RArange

	# for simple, we place the center axis of the trapezium box at RA0 circle
	# then the lower Dec is Dec0-Dec_range/2, the upper Dec is Dec0+Dec_range/2
	# for any point (RA,Dec), calculate the corresponding point (x,y) in box
	def RADec2xyz( RA, Dec ) : 
		central_angle = RArange2CentralAngle( Dec, abs(RA-RA0) )
		if ( Type == 0 ) : 
			lr = dc * np.tan( central_angle )   # distance apart from the center (x0,y0)
			ld = dc * np.tan( abs(Dec-Dec0) )
		elif ( Type == 1 ) : 
			lr = dc[:,None] * np.tan( central_angle )[None,:]
			ld = dc[:,None] * np.tan( abs(Dec-Dec0) )[None,:]
		# treat RA and Dec to be array
		# and their sizes must be same
		signr = GetSign( RA, RA0 )
		signd = GetSign( Dec, Dec0 )
		x, y = Lx/2.0+signr*lr, Ly/2.0+signd*ld
		if ( x.size == 1 ) : x, y = x[0], y[0]
		return np.array( [ x, y ] )

	def xyz2RADec( x, y ) : 
		if ( Type == 0 ) : 
			absDec_Dec0 = np.arctan( abs(y-Ly/2.0) / dc )
		elif ( Type == 1 ) : 
			absDec_Dec0 = np.arctan( abs(y-Ly/2.0)[None,:] / dc[:,None] )
		signy = GetSign( y, Ly/2.0 )
		Dec = Dec0 + signy*absDec_Dec0
		if ( Type == 0 ) : 
			centeral_angle = np.arctan( abs(x-Lx/2.0) / dc )
		elif ( Type == 1 ) : 
			centeral_angle = np.arctan( abs(x-Lx/2.0)[None,:] / dc[:,None] )
		absRA_RA0 = CenteralAngle2RArange( Dec, centeral_angle )  # no problem
		signx = GetSign( x, Lx/2.0 )
		RA = RA0 + signx*absRA_RA0
		if ( RA.size == 1 ) : RA, Dec = RA[0], Dec[0]
		return np.array( [ RA, Dec ] )

	def SkyArea() : 
		Dec_range = 2 * np.arctan( Ly / 2.0 / dc )
		center_angle = 2 * np.arctan( Lx / 2.0 / dc )
		RA_range = CenteralAngle2RArange( Dec0, center_angle )
		return np.array( [ [ RA0-RA_range/2.0, RA0+RA_range/2.0 ], [ Dec0-Dec_range/2.0, Dec0+Dec_range/2.0 ] ] )

	if ( func == 0 ) : 
		if ( Lx !=0 and Ly !=0 ) : 
			Range = SkyArea()
			RAmin, RAmax, Decmin, Decmax = Range[0,0].max(), Range[0,1].min(), Range[1,0].max(), Range[1,1].min()
			RAamin, RAamax = RA.min(), RA.max()
			Decamin, Decamax = Dec.min(), Dec.max()
			if ( not ( RAmin<=RAamin and RAmax>=RAamax and Decmin<=Decamin and Decmax>=Decamax ) ) : 
				if ( RAmin.size == 1 ) : 
					print ' RA arange is : '+('%.4f' % (RAmin*180/np.pi))+' ~ '+('%.4f' % (RAmax*180/np.pi))+' degree'
					print 'Dec arange is : '+('%.4f' % (Decmin*180/np.pi))+' ~ '+('%.4f' % (Decmax*180/np.pi))+' degree'
				Error('some RA or Dec out of range, please check by yourself')
		return RADec2xyz( RA, Dec )
	elif ( func == 1 ) : return xyz2RADec( x, y ) 
		

##################################################
##################################################
##################################################


##################################################
##################################################
##################################################



#def CalCosmologyDistance( redshift, h=0, Omega_m=0.2865, Omega_l=0.7135, Omega_k=0, Omega_r=0 ) : 
def CosmologyDistance( redshift, h=0, Omega_m=0.049+0.2685, Omega_l=0.6825, Omega_k=0, Omega_r=0 ) : 
	'''
	Calculate the cosmology distance with redshift.
	
	Note that, the formulas here are for the flat Lambda_CDM model
	Unit of the distance: 
		defult h=0 -> if ( h == 0 ), unit is h^{-1}Mpc; 
		elif ( h != 0 ), unit is Mpc. Here h=H0/100
	for Plank, h=0.6711
	
	This code matchs webtool : 
	http://www.astro.ucla.edu/~wright/CosmoCalc.html
	
	redshift, can be a number, a list (1D), an array (1D)

	Return [dc, dA, dL]
	'''
	c = 3e+5 # km/s
	A = c / 100.0     # c/H0, H0=h*100km/s/Mpc, note that c in km/s
#	h=0.6932
	# judge type of redshift
	if ( type(redshift) == type([]) or type(redshift) == type(np.array(0)) ) : 
		zlist = redshift
		# list to store the result
		dlist = [ 0 for i in np.arange( len( zlist ) ) ]
		zn = 0
	else : 
		zlist = [redshift]
		zn = -1
	# defind function using lambda
	func = lambda x : 1/( Omega_l + Omega_m*(1+x)**3 )**0.5
	for z in zlist : 
		# note that quad return two element, first is the result, second is the error
		dc = quad( func, 0, z )[0]
		dc = A * dc
		if ( h != 0 ) : dc = dc / h
		dL = dc * (1+z)
		dA = dc / (1+z)
		# '%.3f' is enough
		dc = float( '%.3f' % dc )
		dA = float( '%.3f' % dA )
		dL = float( '%.3f' % dL )
		if ( zn >= 0 ) : 
			dlist[zn] = [ dc, dA, dL ]
			zn = zn + 1
		else : dlist = [ dc, dA, dL ]
	dlist = np.array( dlist )
	return dlist


def CosmologyDistanceApprox( z, h=0, Omega_m=0.049+0.2685, Omega_l=0.6825, Omega_k=0, Omega_r=0 ) : 
	'''
	This function is just for z<<1, save to z^5
	Actually, we don't use this function, use CosmologyDistance() above.
	Note, 
		z=1,0, err=1.8% (error of the result compared with the theory)
		z=1.1, err=2.8%
		z=1.2, err=4.2%
		z=1.3, err=6.0%
		z=1.4, err=8.3%
		z=1.5, err=11.2%
	Return:
		h==0, return h^(-1)Mpc
		h!=0, return Mpc
	'''
	if ( type(z) == list ) : z = np.array(z)
	elif ( type(z) != np.ndarray ) : z = np.array([z])
	c = 3*10**5
	if ( h != 0 ) : c = c/h
#	q0 = 1./2*Omega_m - Omega_l
#	d1 = c/100 * ( z + 1./2*(1-q0)*z**2 - 1./6*(1-2*q0-3*q0**2+3./2*Omega_m)*z**3 )
	B = 3. * Omega_m
#	r = z - B/4*z**2 + (B**2/8-B/6)*z**3 + (-5./64*B**3+3./16*B**2-B/24)*z**4
	dL = z + (1-B/4)*z**2 + (B**2/8-5./12*B)*z**3 + (-5./64*B**3+5./16*B**2-5./24*B)*z**4 + (7./128*B**4-17./64*B**3+5./16*B**2-B/24)*z**5
	a3 = np.array([(1.+z)**(-1), (1.+z)**(-2), 1+z*0]).reshape(z.size,3)
	d = c/100 * dL[:,None] * a3
	if ( z.size == 1 ) : d = d[0]
	return d

##############################


def CalRedshift( distance, zdpath='/usr/bin/redshift-distance.dat', typed=0, h=0, Omega_m=0.2865, Omega_l=0.7135, Omega_k=0, Omega_r=0 ) : 
	'''
	Give cosmology distance, calculate corresponding redshift.

	Parameter:
	distance: can be a number (int or float), or a list->[100,200,300](1D), or an array->np.array([100,200,300])(1D)
	
	typed: type of the distance, 
		0 -> commoving distance, 
		1 -> angular diameter distance, 
		2 -> luminosity distance

	zdpath, path of the redshift-distance.dat file, defult is '/usr/bin/redshift-distance.dat'
	'''
#	h=0.6932
	# for k=0, dA = dp = dL/(1+z)^2
	zd = np.loadtxt( zdpath )
	if ( h != 0 ) : zd[:,1:] = zd[:,1:] * h
	# typed=0->dc, typed=1->d=dA, typed=2->d=dL
	if ( typed == 1 ) : 
		zr = []
		# judge the type of distance
		if ( type(distance) == list or type(distance) == np.ndarray ) : 
			dlist = distance
			zn = 0
		else : 
			dlist = [distance]
			zn = -1
		for d in dlist : 
		# find the dA.max() -> z
		# note that, for one angular diameter distance, it may have one or two redshift
			for i in np.arange( len( zd )-1 ) : 
				if ( zd[i-1][2] < zd[i][2] > zd[i+1][2] ) : break   # find the max point
			x, y = zd[:,2], zd[:,0]
			x1, y1 = x[:i], y[:i]
			x2, y2 = x[i:], y[i:]
			z1 = z2 = -1
			# linear interpolate
			if ( x1.min() <= d <= x1.max() ) : 
				f = interpolate.interp1d( x1, y1 )
				z1 = f( d )
				z1 = float( '%.4f' % z1 )
			if ( x2.min() <= d <= x2.max() ) : 
				# note that x must be increasing in interp1d
				f = interpolate.interp1d( x2[::-1], y2[::-1] )
				z2 = f( d )
				z2 = float( '%.4f' % z2 )
			if ( z1 >= 0 and z2 < 0 ) : z = [ z1 ]
			elif ( z1 < 0 and z2 >= 0 ) : z = [ z2 ]
			elif ( z1 >= 0 and z2 >= 0 ) : z = [ z1, z2 ]
			else : print 'angular diameter distance out of np.arange( dA.min(), dA.max() )'
			zr = zr + z
		zr = np.around( np.array( zr ), 4 )
		if ( zn < 0 ) : return zr[0]
		else : return zr
	else : 
		if ( typed == 0 ) : x = zd[:,1]
		elif ( typed == 2 ) : x = zd[:,3]
		y = zd[:,0]
		f = interpolate.interp1d( x, y )
		z = f( distance )   # if dc is an array, then z also is an array
		z = np.around( z, 4 )
		return z


##################################################
##################################################
##################################################


def CAMB( redshift, cambpath='/usr/bin/camb' ) : 
	'''
	Produce dark matter 1D power spectrum with CAMB at redshift z.
	Redshift can be a number, a list (1D), an array (1D)
	
	cambpath: where is the camb program folder, default is '/usr/bin/camb'
	
	First step, create a directory named 'output' (not include '') at current directory.
	Second step, create a directory named 'camb_output' (not include '') in 'output' directory
	
	Note that os.system() can not use cd : os.system( 'cd file1' ), so for the path not in the current folder, use the absolute path
	'''
	os.system( 'cp -r '+cambpath+' cambpy' )
	os.system( 'mkdir camb_output' )
	
	# open file params.ini, np.loadtxt( 'params.ini', str ) doesn't work, don't know why
	paramsfile = open( 'cambpy/params.ini' )
	parafile = paramsfile.readlines()
	
	# find the lines which will be modified
	# output file's path
	# redshift setting
	for line in np.arange( len( parafile ) ) : 
		if ( parafile[line][:11] == 'output_root' ) : line1 = line
		if ( parafile[line][:17] == 'transfer_redshift' ) : 
			line2 = line
			break
	
	# modify params.ini and calculate camb several times

	# judge type of redshift
	if ( type(redshift) == type([]) or type(redshift) == type(np.array(0)) ) : 
		zlist = redshift
	else : zlist = [redshift]

	for z in zlist : 

		# modify params.ini with different redshifts
		parafile[line1] = 'output_root = camb_output/z'+('%.4f' % z)+'\n'
		parafile[line2] = 'transfer_redshift(1)    = '+str(z)+'\n'
		outfile = open( 'cambpy/params.ini', 'w' )
		for i in np.arange( len( parafile ) ) : 
			print >> outfile, parafile[i],
		outfile.close()

		# calculate with camb with different params.ini
		os.system( './cambpy/camb cambpy/params.ini' )
		# modify and calculate several times OK !

	paramsfile.close()
	# delete the temp file
	os.system( 'rm -r cambpy' )


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


def LAB( freq, theta, phi, coordsys, labpath='labh_3D.fits' ) : 
	'''
	freq:
		in MHz
		(1) Scale/One value: freq=1420.40575
		(2) "1D" list/tuple/ndarray: freq=np.array([1420.3, 1420.406, 1420.5])

	theta:
		in degree
		from +90 (North pole) to -90 (South pole)
		Must have the same shape as phi: theta.shape==phi.shape
		theta can be any shape (N-D)
	
	phi:
		in degree
		from 0 to 360
		Must have the same shape as theta: theta.shape==phi.shape
		phi can be any shape (N-D)

	coordsys:
		Must be 'Galactic' or 'Equatorial'
		If coordsys=='Galactic',   theta=>b,   phi=>l
		If coordsys=='Equatorial', theta=>Dec, phi=>RA

	labpath:
		Path of LAB FITS file 'labh_3D.fits'

	return:
		data
		data.shape = (freq.size, theta.shape)


	#*****************************************************


	if __name__=='__main__' : 
	
		if (len(sys.argv) > 1) : 
			
			parser = OptionParser() 
			parser.add_option('-f', '--freq', dest='freq', type='float', help='frequency in MHz') 
			parser.add_option('-t', '--theta', dest='theta', type='string', help='theta angle in degree, from +90 (North pole) to -90 (South pole). Usage: --theta -30,45,5  => start -30deg, end +45deg, step 5deg')
			parser.add_option('-p', '--phi', dest='phi', type='string', help='phi angle in degree, from 0 to 360. Usage: --phi 50,300,10  => start 50deg, end 300deg, step 10deg')
			parser.add_option('-c', '--coordsys', dest='coordsys', type='string', help="Coordinate system, must be 'Galactic' or 'Equatorial'") 
			parser.add_option('-o', '--outname', dest='outname', type='string', default='', help='The name of output file') 
			parser.add_option('--labpath', dest='labpath', type='string', default='labh_3D.fits', help='Path of the LAB FITS file') 
			options, args = parser.parse_args() 
	
			# 1D theta
			theta = np.array(options.theta.split(','), float)
			if (theta[2] == 0) : theta[2] = 1e-4
			theta = np.arange(theta[0], theta[1]+theta[2]/100, theta[2])
			
			# 1D phi
			phi = np.array(options.phi.split(','), float)
			if (phi[2] == 0) : phi[2] = 1e-4
			phi = np.arange(phi[0], phi[1]+phi[2]/100., phi[2])
			
			# 2D theta, phi
			theta = theta[:,None] + np.zeros([1,phi.size])
			phi = phi[None,:] + np.zeros(theta.shape)
			
			freq = options.freq
			coordsys = options.coordsys
			outname = options.outname
			labpath = options.labpath
		
		else : 
			# Please see LAB() function above for the parameters below
			freq  = 
			theta = 
			phi   = 
			coordsys =
			outname  = 
			labpath  = 'labh_3D.fits'

		##################################################
	
		if (outname == '') : outname = 'lab_slice.npy'
		if (outname[-4:] != '.npy') : outname += '.npy'
		data = LAB(freq, theta, phi, coordsys, labpath)
		np.save(outname, data)
		print outname+'  -->  saved'
	'''
	coordsys = str(coordsys)
	if (coordsys.lower() not in ['galactic', 'equatorial']) : raise Exception("coordsys='"+coordsys+"' not in ['Galactic', 'Equatorial']")
	
	# freq must be 1D
	freq = np.array(freq).flatten()
	
	# theta, phi must have the same shape
	theta, phi = np.array(theta), np.array(phi)
	if (theta.shape != phi.shape) : raise Exception('thets.shape != phi.shape')
	shapethetaphi = theta.shape
	if (shapethetaphi == ()) : shapethetaphi = (1,)
	theta, phi = theta.flatten(), phi.flatten()
	
	# Open LAB.fits
	fo  = pyfits.open(labpath)
	hdr = fo[0].header
	bzero  = hdr['BZERO_']
	bscale = hdr['BSCALE_']
	blank  = hdr['BLANK_']
	rfreq = hdr['RFREQ']
	dfreq = hdr['DFREQ']
	Nfreq = hdr['NAXIS3']
	rtheta = hdr['CRVAL2']
	dtheta = hdr['CDELT2']
	Ntheta = hdr['NAXIS2']
	rphi = hdr['CRVAL1']
	dphi = hdr['CDELT1']
	Nphi = hdr['NAXIS1']
	
	freqmax = rfreq + (Nfreq-1)*dfreq
	if (freq.min()<rfreq or freq.max()>freqmax) : raise Exception('freq.min()='+str(freq.min())+', freq.max()='+str(freq.max())+' out of LAB frequency range: '+str(rfreq)+' ~ '+str(freqmax)+' MHz')
	
	# phi from 180 to -180
	phi[phi>rphi] -= 360
	
	# index of theta and phi
	ntheta = ((theta-rtheta)/dtheta).astype(int)
	nphi   = ((phi-rphi)/dphi).astype(int)
	ntheta[ntheta>=Ntheta] = Ntheta-1
	nphi[nphi>=Nphi] = Nphi-1
	theta = phi = 0 #@
	
	# index of freq
	nfreq = ((freq-rfreq)/dfreq).round().astype(int)
	nfreq[nfreq>=Nfreq] = Nfreq-1
	freq = 0 #@
	
	# theta-phi 2D index
	nthetaphi = np.array([ntheta, nphi]).T
	ntheta = nphi = 0 #@
	
	def N2One( indexnd, shapend ) : 
		indexnd = np.array(indexnd)
		if (len(indexnd.shape) == 1) : indexnd = indexnd[None,:]
		shapendprod = np.cumprod(shapend[::-1])[:-1][::-1]
		shapendprod = np.concatenate([shapendprod, [1]])[None,:]
		indexnd *= shapendprod
		return indexnd.sum(1)
	
	# LAB data
	data = fo[0].data[nfreq]
	shapedata = data.shape
	
	# Reshape to 2D
	# First axis is freq, second axis is 1D theta-phi
	data = data.reshape(shapedata[0], np.prod(shapedata[1:]))
	
	# 2D theta-phi to 1D
	n1d = N2One(nthetaphi, shapedata[1:])
	
	if (coordsys.lower() == 'equatorial' ) : 
		nDecb, nRAl = fo[1].data
		nDecb, nRAl = nDecb.flatten()[n1d], nRAl.flatten()[n1d]
		nbl = np.array([nDecb, nRAl], int).T
		n1d = N2One(nbl, shapedata[1:])
	
	data = data[:,n1d]
	
	# Reshape
	shape = (shapedata[0],) + shapethetaphi
	data = data.reshape( shape )
	
	# BLANK
	data[data<=blank] = 0
	# Rescale
	data = np.float32(data *bscale +bzero)
	return data


##################################################
##################################################
##################################################


def GaussianRandom( mean=0, stdev=1, shape=(), Type='real' ) : 
	'''
	Generate real/complex Gaussian random number/field

	Parameter:
		mean: mean of the Gaussian
		stdev: standard deviation (variance**0.5) of the Gaussian
		shape: shape of the output Gaussian field, default () means a number
	'''
	G = np.random.standard_normal( shape )
	if ( Type == 'complex' ) : 
		G = G + 1j*np.random.standard_normal( shape )
	G = mean + stdev * G
	if (G.size == 1) : G = float(G)
	return G

#	# generate uniformly distributed U and V in the interva (0,1) or [0,1)
#	# for the np.arange [0,1) case
#	a = np.random.random()
#	b = np.random.random()
#	# Box-Muller transform
#	X = np.sqrt( -2 * np.log(a) ) * np.cos( 2*np.pi * b )
#	Y = np.sqrt( -2 * np.log(a) ) * np.sin( 2*np.pi * b )
#	# complex space Gaussian random field is builded by to independent Gaussian random field
#	Gc = mean + variance**0.5 * complex( X, Y )
#	return Gc


##################################################
##################################################
##################################################


#def LeastSqFit( x, y, p0 ) : 
#	def residuals( p, y, x ) :
#	    err = y - ( p[0] + p[1]*x )
#	    return err
#	plsq = leastsq( residuals, p0, args=( y, x ), maxfev=0 )
#	return plsq[0]


##################################################
##################################################
##################################################


def Savetxt( outname, array ) : 
	'''
	Use this function Savetxt() to instead of np.savetxt() because Savetxt() will set fmt=[] automatically but np.savetxt() set by hand.

	When use np.savetxt(outname, array, fmt=[]), in order to make outfile look better, we use fmt to contral the output format.
	Usually we set fmt by hand, but we need to know the up and down limit of the array. However we don't know them at most of cases, so we use this function Savetxt() to set fmt=[] automatically.

	Precision: save to 6 decimal places.

	float to str: a = 1.234, b = str(a). 
		significant digit(figure): 12(without unvalid 0) + sign(+-). 
		abs(a)>1, integer part >= 12, then will convert to scientific notation.
		abs(a)<1, decimal part <0.0001, then will convert to scientific notation.
		For example:
			str( 12345678901.12345) =  '1234567890.1'
			str(-12345678901.12345) = '-1234567890.1'
			str(123456789012.12345) =  '1.23456789012e+12'
			str(0.0001)   = '0.0001'
			str(0.000099) = '9.9e-05'

	int to str: no matter how large the integer is, it will completely convert to str:
		str(123456789012345678901234567890123456789) = '123456789012345678901234567890123456789'
	'''
	# check the data type of array
	typeid = array.dtype.name
	# shape of array
	shapea = array.shape
	#
	interval = 5
	if (len(shapea) <= 2) : 
		if (len(shapea) == 1) : shapea = (shapea[0], 1)
		for i in range(shapea[1]) : 
			a = array[:,i]
			if (typeid[:3] == 'int') : 
				n = len(str(abs(a).max())) + 1 # +1 -> sign
				fmti = '%'+str(n)+'i'
			if (typeid[:3] == 'flo') : 
				mina = abs(a).min()
				maxa = abs(a).max()
				strmax = ('%e' % maxa)
				strmin = ('%e' % mina)
				imax = int(strmax[-3:])
				imin = int(strmin[-3:])
				if (imax>=0 and imin>=0) : 
					num = imax+6+3
					if (i > 0) : num = num + interval
					fmt = '%'+str(num)+'.6f'
				if (imax>=0 and imin<0) : 
					num = imax+imin+3
					if (i > 0) : num = num + interval
					fmt = '%'+str(num)+'.'+str(imin)+'f'
				if (imax<0 and imin<0) : 
					num = imin+3
					if (i > 0) : num = num + interval
					fmt = '%'+str(num)+'.'+str(imin)+'f'
			fmt = fmt + [fmti]
		if (len(fmt) == 1) : fmt = fmt[0]
		np.savetxt(outname, array, fmt)
	else : 
		np.savetxt(outname, array)
	print outname+'   --->   saved.   Savetxt()'


##################################################
##################################################
##################################################


def CentralAngleRADec( RA1, Dec1, RA2, Dec2 ) : 
	'''
	Calculate circular central angule of two points with RA and Dec (equatorial coordinate, rad)
	'''
	Decu = max( abs(Dec1), abs(Dec2) )
	dRA2 = abs( RA1 - RA2 ) / 2.0
	dDec2 = abs( Dec1 - Dec2 ) / 2.0
	h1 = np.sin( dRA2 )
	l1 = 1 - np.cos( dRA2 )
	l2 = 2 * np.sin( dDec2 ) / np.cos( Decu )
	a = np.pi/2 - dRA2 + Decu
	h22 = l1**2 + l2**2 - 2*l1*l2*np.cos( a )
	l = ( h1**2 + h22 )**0.5
	ac = 2 * np.arcsin( l/2 * np.cos( Decu ) )
	return ac


##################################################
##################################################
##################################################


def T21_average( z, Omega_HI=0.5e-3, Omega_m=0.04628+0.2402, Omega_l=0.7135 ) : 
	'''
	Calculate the average brightness temperature of the 21cm signal as a function of redshift.
	
	bias that HI traces the dark matter delta_HI = bHI * delta_x (Masui et al. 2013, ApJL, 763,20)
		bHI = 0.8
	
	Mass fraction of HI with critical density unit (Masui et al. 2013, ApJL, 763,20)
		Omega_HI = 0.5e-3
	
	Hubble constant h = H0 / (100 km/s/Mpc)
	h = 0.6932
	
	Other cosmological parameters
		Omega_b  = 0.04628
		Omega_dm = 0.2402
		Omega_m = Omega_dm + Omega_b
		Omega_l  = 0.7135
	
	Richard code
	Omega_HI = 0.62e-3
	omega_b = 0.0483
	omega_c = 0.2589
	omega_m = omega_b + omega_c
	omega_l = 0.6914

	Notes: the constant used to be 0.3 mK, but Tzu-Ching pointed out that this was wrong in 2008PhRvL.100i1303C, Eric recalculated this to be 0.39 mK (agrees with 0.4 mK quoted over phone from Tzu-Ching)
	'''

#	T21_ave = 0.29 * Omega_HI / 1e-3 * ( ( Omega_m + (1+z)**(-3)*Omega_l ) / 0.37 )**(-0.5) * ( (1+z) / 1.8 )**0.5
	T21_ave = 0.39 * Omega_HI / 1e-3 * ( ( Omega_m + (1+z)**(-3)*Omega_l ) / 0.29 )**(-0.5) * ( (1+z) / 2.5 )**0.5
	return T21_ave


##################################################
##################################################
##################################################



def Take( array, index=[()] ) : 
	'''
	One tuple() for one element. For example, for 3D, (n1,n2,n3) for one element. 

	Take the value of elements or array listed in index=[()]
	'''
	result = []
	for i in range(len(index)) : 
		result = result + [array[index[i]]]
	return np.array(result)
	

##################################################
##################################################
##################################################


def Localtime2UTC( time_ymdhms='', timezone='', l2u=True ) : 
	'''
	Give a local time at timezone, return the UTC
	UTC is useful in pyephem.
	
	Parameter:
		time_ymdhms: 
			(1) format like "2015/2/28 20:13:57"
			(2) list or np.array = [year, month, day, hour, minute, second]
			(3) if not set (default []), it will get the computer/system time

		timezone: 
			int
			set the time zone of the time_ymdhms, default, if will be set to as your computer setting (time.timezone will get the computer system date setting)
			If set the timezone by hand:
				east of UK, + (Beijing, timezone=+8=8)
				west of UK, - (Washington, timezone=-5)

		l2u:
			This function can do "Localtime2UTC"(l2u=True) and "UTC2Localtime"(l2u=False) 
	'''
	if (type(time_ymdhms) == str) : 
		ymdhms, n1, n2, n3 = [], [], [], 0
		for i in range(len(time_ymdhms)) : 
			if (time_ymdhms[i] == '/') : n1 = n1 + [i]
			if (time_ymdhms[i] == ':') : n2 = n2 + [i]
			if (time_ymdhms[i] == ' ') : n3 = i
		if (len(n1) != 2) : raise Exception(efn()+'Format of the year,month,day is not correct.')
		if (len(n2) == 0) : raise Exception(efn()+'Please set the minute')
		if (n3 == 0) : raise Exception(efn()+'Please use space=" " to separate the day and hour.')
		n = [-1] + n1 + [n3] + n2
		n = n + [len(time_ymdhms)]
		for i in range(len(n)-1) : 
			ymdhms = ymdhms + [int(time_ymdhms[n[i]+1:n[i+1]])]
		time_ymdhms = ymdhms + [0 for i in range(6-len(ymdhms))]
	if ( time_ymdhms != [] ) :  
	#	time_9 = np.array( list( time_ymdhms ) + [ 0, 0, 0 ] ).astype( int )
		time_9 = list( time_ymdhms ) + [ 0, 0, 0 ]
		time1 = time.struct_time( time_9 )
	else : time1 = time.localtime()
	time_second = time.mktime( time1 )
	if (timezone == '') : 
		seconddev = time.timezone
	#	seconddev = time.altzone
	else: 
		if (not(-12<=timezone<12)) : raise Exception("Error: Time2UTC(), timezone must in [-12, +12]")
		if (l2u) : seconddev = -3600 * timezone
		else : seconddev = +3600 * timezone
#	UTC = time.gmtime( time_second )
	UTC = time.localtime( time_second + seconddev )
	UTC = list( UTC )[:6]
	UTCstr = str(UTC[0])+'/'+str(UTC[1])+'/'+str(UTC[2])+' '+str(UTC[3])+':'+str(UTC[4])+':'+str(UTC[5])
	return [UTCstr, UTC]


def UTC2Localtime( time_ymdhms='', timezone='' ) : 
	return Localtime2UTC( time_ymdhms, timezone, False )


def Timezone12( time_ymdhms='', timezone1='', timezone2='' ) : 
	'''
	Give the time on timezone1, return the time on timezone2
	'''
	return UTC2Localtime( Localtime2UTC(time_ymdhms,timezone1)[1], timezone2 )


def SiderealTime( lon, RAs=None, date='' ) : 
	'''
	Return the local sidereal time

	lon:
		Must in unit "degree". Both str and float are OK.
		longitude of the local geography.

	RAs:
		Must in unit "degree". Both str and float are OK.
		RA of the source/object.
		If given, also return when/after how long time will this source reaches the local merdian.
		If not given, just return the sidereal time.

	date:
		local time (realte to UTC) at this longitude/timezone.
		Must be str, date=Time()
		Format of the date must be "2015/3/12 10:55:00"  (ephem)

	Return:
		Must return the sidereal time.
		If RAs is given, also return when/after how long time will this source reaches the local merdian.
	'''
	if (type(lon) != str) : lon = str(lon)
	if (date == '') : date = Time() # Now
	here = ephem.Observer()
	here.lon = lon
	here.date = date
	last = here.sidereal_time()
	if (RAs is None) : return last
	else : 
		if (type(RAs) == str) : RAs = float(RAs)
		after = RAs*np.pi/180 - last  # rad
		after = ephem.hours(after)
	#	after = (RAs - last*180/np.pi)/15 # hour
	#	h = int(after)
	#	m = int((after-h)*60)
	#	s = round((after-h-m/60.)*3600, 2)
	#	after = str(h)+':'+str(m)+':'+str(s)
		return [last, after]


##################################################
##################################################
##################################################


def GaussianValue( x, mean, std ) : 
	y = 1/(2*np.pi)**0.5 / std * np.e**( -(x-mean)**2 / (2*std**2) )
	return y

 
def LogNormalValue( x, mean, std ) : 
	if ( x.min() < 0 ) : 
		Error('x.min() < 0')
	y = 1/(2*np.pi)**0.5 / std / x * np.e**( -(np.log(x)-mean)**2 / (2*std**2) )
	return y


##################################################
##################################################
##################################################

#
#def LeastsqPolynomial( x, y, index=3 ) : 
#	''' x and y can not be complex (must real)
#	Use matrix method '''
#	if (len(x.shape)!=1 or len(y.shape)!=1) : Raise(Exception, 'x and y must be 1D')
#	if (((x*y).dtype.name)[:3]=='com') : Raise(Exception, 'x and y must be real, not be complex')
#	X = np.zeros([y.size, index+1])
#	for i in xrange(index+1) : X[:,i] = x**i
#	X = np.matrix(X)
#	Y = np.matrix(y[:,None])
#	p = (X.T * X).I * X.T * Y
#	p = np.array(p)[:,0]
#	return p
#
#
#def LeastsqMatrix( x, y, func='', index=3 ) : 
#	'''
#	func: 
#		(1) "polynomial":
#			index is the highest order of x
#			index=0: y = a
#			index=1: y = a + b*x
#			index=2: y = a + b*x + c*x^2
#			return [a, b, c]
#		(2) "gaussian":
#			y = 1/(sigma*sqrt(2pi) * exp(-(x-mean)^2/(2sigma^2))
#			return [mean, sigma]
#		(3) This function doesn't fit 'power-law', use LeqstsqSP() instead
#	'''
#	if (len(x.shape)!=1 or len(y.shape)!=1) : Raise(Exception, 'x and y must be 1D')
#	if (((x*y).dtype.name)[:3]=='com') : Raise(Exception, 'x and y must be real, not be complex')
#	if (func.lower() == 'polynomial') : return LeastsqPolynomial(x, y, index)
#	elif (func.lower() == 'gaussian') : 
#		xy = np.append(x[:,None], y[:,None], 1)
#		xy = xy[xy[:,1]>0]
#		x, y = xy[:,0], xy[:,1]
#		p = LeastsqPolynomial(x, np.log(y), 2)
#		stdev = (-0.5/p[2])**0.5
#		mean = -0.5*p[1]/p[2]
#		if (np.isnan(stdev) or np.isinf(stdev)) : 
#			# fit Gaussian fail (not a Gaussian)
#			mean = stdev = 0
#		return np.array([mean, stdev])
#
#
#def LeastsqSP( x, y, func='', p0index=3, normalized=True ) : 
#	'''
#	Use scipy.optimize.leastsq
#
#	func: 
#		(1) "polynomial":
#			index is the highest order of x
#			index=0: y = a + const*x
#			index=1: y = a + b*x
#			index=2: y = a + b*x + c*x^2
#			return [a, b, c]
#		(2) "gaussian":
#			y = 1/(sigma*sqrt(2pi) * exp(-(x-mean)^2/(2sigma^2))
#			return [mean, sigma]
#		(3) "power-law":
#			y = a*x^b
#			return [a, b]
#
#	p0index: initial values of the fitting, p0index may be one number/index, array/list basing on the func
#
#	func=='polynomial': p0index=index
#	func=='gaussian'  : p0index=[mean, stdev]
#	func=='power-law' : p0index=[a,b]
#
#	normalized:
#		Just be used to 'gaussian', set whether normalized
#	'''
#	if (func.lower() == 'gaussian') : 
#		if (-1e-8 < p0index[1] < 1e-8) : 
#			Raise(Exception, 'stdev=p0index[1] can not be 0')
#		p0index[1] = p0index[1]**2
#		if (normalized) : 
#			def function( p, x ) : 
#				yf = 1/(2*np.pi)**0.5 / p[1]**0.5 * np.e**( -(x-p[0])**2 / (2*p[1]) )
#				return yf
#		else : 
#			def function( p, x ) : 
#				yf = np.e**( -(x-p[0])**2 / (2*p[1]) )
#				return yf
#	elif (func.lower() == 'power-law') : 
#		def function( p, x ) : 
#			yf = p[0] * x**p[1]
#			return yf
#	elif (func.lower() == 'polynomial') : 
#		def function( p, x ) : 
#			if   (len(p) == 1) : yf = p[0]
#			elif (len(p) == 2) : yf = p[0] + p[1]*x
#			elif (len(p) == 3) : yf = p[0] + p[1]*x + p[2]*x**2
#			elif (len(p) == 4) : yf = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3
#			elif (len(p) == 5) : yf = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4
#			elif (len(p) == 6) : yf = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 + p[5]*x**5
#			else : Raise(Exception, 'func="polynomial", now can just to len(p)<=6, you can add higher order by yourself.')
#			return yf
#	def residuals( p, y, x ) : 
#		if (func.lower() == 'gaussian') : 
#			if (p[1] < 0) : p[1] = 1e10
#		err = y - function( p, x )
#		return err
#	plsq = leastsq(residuals, p0index, args=(y,x), maxfev=20000)
#	p = plsq[0]
#	if (func.lower() == 'gaussian') : p[1] = p[1]**0.5
#	return p
#
#
##################################################
##################################################
##################################################


def CompactDimension( array ) : 
	'''
	array.shape = (n1, n2, n3, ...)
	Remove the extra axis with n=1

	For example, array.shape=(1,2,3,1,4,1,5)
	CompactAllDimensions(array).shape=(1,2,3,4,5)
	'''
	shape = np.array(array.shape)
	shape = shape[shape>1]
	return np.array(array).reshape(shape)


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


def CMBtemperature( frequency ) : 
	'''
	CMB thermal temperature is 2.726K, a constant for all frequencies, but it is not the brightness from Rayleigh-Jeans law.
	If convert to brightness temperature, it is frequency dependent. 

	frequency in MHz is the observable frequency
	'''
	hk = 6.626*10**(-34) / (1.381*10**(-23))
	hkv = hk * frequency * 10**6
	Tcmbb = hkv / (np.e**(hkv/2.726) - 1)
	return Tcmbb


##################################################
##################################################
##################################################


class ThetaPhi( object ) : 


	def Theta1D( self, thetasize, Ntheta, project=True ) :
		'''
		thetasize:
			Total angle size of theta in rad

		Ntheta:
			Total number of theta points

		project:
			False: uniform theta
			True: sin speed theta
		'''
		N, resol = Ntheta, 1.*thetasize / Ntheta
		if (N%2 == 0) : N, thetasize = N+1, thetasize+resol
		t = np.linspace(-thetasize/2., thetasize/2., N)
		if (project) : 
			t = np.sin(t)
			t = t[1:] - t[:-1]
			t = np.cumsum((t / t.sum() * thetasize)[N/2:])
			t = np.concatenate([-t[::-1], [0], t])
		t = t[-Ntheta:]
		return t


	def Theta2D( self, thetarow, thetacol, mask=False ) : 
		'''
		thetarow, thetacol:
			ndarray/list of theta of row/column in rad

		theta: theta[Nrow/2,Ncol/2]=0, center=0
		'''
		thetarow, thetacol = npfmt(thetarow).flatten(), npfmt(thetacol).flatten()
		theta2d = (thetarow[:,None]**2+thetacol[None,:]**2)**0.5
		if (mask) : 
			# Must thetarow, thetacol in [-180->0->180]
			nl = thetarow[thetarow>np.pi].size
			ns = thetarow[thetarow<-np.pi].size
			if (nl != 0 or ns != 0) : 
				thetarow %= 2*np.pi
				thetarow[thetarow>np.pi] -= 2*np.pi
			nrl = thetacol[thetacol>np.pi].size
			nrs = thetacol[thetacol<-np.pi].size
			if (nl != 0 or ns != 0) : 
				thetacol %= 2*np.pi
				thetacol[thetacol>np.pi] -= 2*np.pi
			one = thetarow[:,None]**2/thetarow.max()**2 + thetacol[None,:]**2/thetacol.max()**2
			theta2d = np.ma.MaskedArray(theta2d, one>1)
		return theta2d


	def Phi2D( self, thetarow, thetacol, mask=False ) : 
		'''
		thetarow, thetacol:
			ndarray/list of theta of row/column in rad
	
		phi:      90                  270    
		     180      0     or    180      0 
		         270                   90    
		'''
		thetarow, thetacol = npfmt(thetarow).flatten(), npfmt(thetacol).flatten()
		theta2d = self.Theta2D(thetarow, thetacol, mask=mask)
		if (mask) : 
			theta2dmask = theta2d.mask
			theta2d = theta2d.data
		theta2d[theta2d==0] = 1e-20
		Nr, Nc = thetarow.size, thetacol.size
		phi2d = np.angle(thetacol[None,:]/theta2d + 1j*thetarow[:,None]/theta2d) %(2*np.pi)
		if (mask) : phi2d = np.ma.MaskedArray(phi2d, theta2dmask)
		return phi2d


def GaussianBeam( theta, FWHM, normalized=False ) : 
	'''
	GaussianBeam = exp{-(theta/FWHM)^2/2/sigma^2}, its max=1, not normalized.

	Set theta=0.5*FWHM, compute exp{-(1/2)^2/2/sigma^2}=0.5(because the max=1, theta=0.5*FWHM will decreate to half power) and get sigma^2=1/(8ln2)

	\int{e^(-a*x^2)dx} = (pi/a)^0.5
	So, for the normalized GaussianBeam_normalized 
	= 1/FWHM * (4ln2/pi)^0.5 * exp(-4ln2 * theta^2 / FWHM^2)
	'''
	b = np.exp( -4*np.log(2) * theta**2 / FWHM**2 )
	a = 4*np.log(2) / FWHM**2
	if (normalized == False) : return b
	else : return b/(np.pi/a)**0.5


def EllipticGaussianBeam( theta, phi, FWHM1, FWHM2, normalized=False ) : 
	# theta, phi, FWHM1, FWHM2 all are in rad
	b = np.exp( -4*np.log(2) * theta**2 * ((np.cos(phi)/FWHM1)**2 + (np.sin(phi)/FWHM2)**2) )
	a = 4*np.log(2) * ((np.cos(phi)/FWHM1)**2 + (np.sin(phi)/FWHM2)**2)
	if (normalized == False) : return b
	else : return b/(np.pi/a)**0.5


#def ThetaPhiMatrix(fieldofview, pixelnumber, project=False) :
#	'''
#	fieldofview:
#		total field of view (from left to rignt) in rad.
#
#	pixelnumber:
#		pixelnumber must be odd.
#		theta and phi matrix shape = (pixelnumber, pixelnumber)
#
#	project:
#		True: Spherical projects to flat.
#		False: Spherical angle.
#
#	# At the center (pixelnumber/2, pixelnumber/2), theta = 0
#	# phi = 0 at the [pixelnumber/2:, pixelnumber/2], it means:
#			-x
#		-y		+y
#			+x
#
#	return:
#		[theta, phi] in rad.
#	'''
#	# pixelnumber must be odd
#	if (project is not False) : 
#		if (fieldofview > np.pi) : fieldofview = np.pi
#	N0 = pixelnumber = int(pixelnumber)
#	if (pixelnumber%2 == 0) : pixelnumber = pixelnumber + 1
#	N = pixelnumber / 2
#	# Matrix
#	theta = np.zeros([pixelnumber, pixelnumber]) -1
#	phi = theta.copy()
#	# Length of each pixel
#	if (project is False) : 
#		dx = fieldofview / (pixelnumber-1)
#	else : 
#		dx = np.sin(fieldofview/2.) / N
#	for i in range(N+1) : 
#		for j in range(N) : 
#			if (project is False) : 
#				theta[i,j] = dx*((N-i)**2+(N-j)**2)**0.5 
#			else : 
#				d = dx*((N-i)**2+(N-j)**2)**0.5
#				if (abs(d) > 1) : pass
#				else : theta[i,j] = np.arcsin(d)
#			phi[i,j] = np.pi - np.arctan(1.*(N-i)/(N-j))
#	theta[N+1:] = theta[:N][::-1]
#	theta[:,N+1:] = theta[:,:N][:,::-1]
#	theta[N,N] = 0
#	theta[:,N] = theta[N]
#	phi[:N+1,N+1:] = np.pi - phi[:N+1,:N][:,::-1]
#	phi[:N,N] = np.pi/2
#	phi[N+1:] = 2*np.pi - phi[:N][::-1]
#	phi[N,N:] = 0
#	theta[theta==-1] = np.nan
#	theta = np.ma.masked_invalid(theta)
#	phi = phi[::-1].T
#	if (theta.mask.sum() == 0) : 
#		return np.array([theta.data[:N0,:N0], phi[:N0,:N0]])
#	else : 
#		return [theta[:N0,:N0], phi[:N0,:N0]]


def BeamModel( beamtype, antennatype, diameter, frequency, fovORtpm, pixelnumber, project=False, power=True, dim='2D', kbeam=1.03 ) :
	'''
	This function is used to model the primary beam of a single dish or cylinder.
	
	beamtype: 
		str, 'gaussian'/'Gaussian' or 'sinc2'(sinc^2) or 'sinc'
		Primary beam, Gaussian e^(-x^2/2/var) (where var=1/(8*ln2)) or sinc^2=(sinx/x)^2
	
	antennatype: 
		str, 'dish' or 'cylinder'/'ellipse'
	
	diameter: 
	    For dish, diameter is an value, diameter=5 (meter)
	    For cylinder, diameter is list or array or tuple, diameter=[cylinder_width, feed_size], feed_size decides the field of view of length direction
	
	frequency: 
		Observable frequency (MHz), one value, list, np.array
	
	fovORtpm:
		Field of view OR ThetaPhiMatrix
      If fovORtpm is the field of view (one angle in rad), then it will use ThetaPhiMatrix() function to get theta and phi matrixs.
		If fovORtpm is the ThetePhiMatrix -> [theta_matrix, phi_matrix], then it use this matrix directly, and we don't need to set pixelnumber parameter below

	pixelnumber:
		Use to obtain theta and phi matrixs.
		See parameter fovORtpm above.

	project:
		If =False, we deal with plane angle.
		If =True,  we consider as spherical angle and project it into plane, in this case, field of view can't be larger than 180 degree.

	power:
		True or False.
		Return the power beam(real) or the electromagnetic beam(complex).

	kbeam:
		 =1.22 or =1.03
	
	return: 
		   return [Beam, theta_matrix, phi_matrix]
		OR return Beam
	'''
	if (dim != '2D' and dim != '1D') : 
		raise Exception('dim= should be "2D" or "1D"')
	if ( antennatype == 'dish' ) : 
		diameter0 = diameter	
	elif ( antennatype=='cylinder' or antennatype=='ellipse' ) : 
		diameter0 = diameter[0]
	else : raise ValueError(efn()+'Antenna type wrong, not "dish" or "cylinder"/"ellipse"')
	# Check fovORtpm
#	fovORtpm = npfmt(fovORtpm)
	if (type(fovORtpm)==np.array or type(fovORtpm)==list) : 
		if (len(fovORtpm) > 1) : 
			fm = 1
			theta, phi = fovORtpm
		else : 
		#	if (pixelnumber is None) : raise Exception('pixelnumber must be an valid number')
			fovORtpm = 1.*npfmt(fovORtpm)
			fm, fovORtpm, pixelnumber = 0, fovORtpm[0], int(pixelnumber)
	else : 
	#	if (pixelnumber is None) : raise Exception(efn()+'pixelnumber must be an valid number')
		fm, pixelnumber = 0, int(pixelnumber)
		if (dim == '2D') : 
			theta, phi = ThetaPhiMatrix(fovORtpm, pixelnumber, project)
		else : 
			theta = np.linspace(-fovORtpm/2., fovORtpm/2., pixelnumber)
			phi = np.pi/2 + theta*0
	frequency = npfmt(frequency).flatten()
	wl = 3.*10**8 / (frequency*10**6) # wavelength

	if (dim == '2D') : 
		wl = wl[:,None,None]
		theta = theta[None,:,:]
		phi   = phi[None,:,:]
	elif (dim == '1D') : 
		wl = wl[:,None]
		theta = theta[None,:]
		phi   = phi[None,:]

	theta0 = kbeam * wl / diameter0  #
	if ( beamtype=='gaussian' or beamtype=='Gaussian' ) : 
		def Beamshape( x ) : 
			var = 1./(8*np.log(2))
			if (power is True) : 
				return np.e**(-x**2/(2*var))
			else : 
				return np.e**(-x**2/(2*var)/2)
	elif ( beamtype == 'sinc' ) : 
		def Beamshape( x ) : 
			if (power is True) : 
				return np.sinc( 2.783*x / np.pi )**2
			else : 
				return np.sinc( 2.783*x / np.pi )
	else : raise ValueError(efn()+'beamtype wrong, must be "gaussian"/"Gaussian" or "sinc"')
	if ( antennatype == 'dish' ) : 
		A = Beamshape( theta / theta0 )
	elif ( antennatype=='cylinder' or antennatype=='ellipse' ) : 
		# Check phi
		phi = npfmt(phi)
		if (phi.shape != theta.shape) : raise Exception(efn()+'phi.shape != theta.shape')
		theta1 = kbeam * wl / diameter[1]  #
		A = Beamshape( theta/theta0 * np.cos(phi) ) * Beamshape( theta/theta1 * np.sin(phi) )
	else : raise ValueError(efn()+'antennatype wrong, nust be "dish" or "cylinder"/"ellipse"')
	for i in range(len(A)) : A[i] = A[i] / A[i].max()
	if (len(A) == 1) : A = A[0]
	if (fm == 0) : 
		if (dim == '2D') : return [A, theta[0], phi[0]]
		else : return [A, theta[0]]
	else : return A


##################################################
##################################################
##################################################


'''
Gaussian Beam:
	A = exp( -x^2 / (2*std^2) )
	FWHM: std = x_fwhm / (8*ln2)^0.5
	Rayleigh criterion: std = 1.03/1.22 * x_ray / (8*ln2)^0.5
'''

def Resolution( kind='FWHM', wavelength='', D='' ) : 
	'''
	Return the resolutions of 
		(1) Rayleigh criterion: 1.22*l/D, I=0.3675*I0 (first dark ring)
		(2) FWHM: 1.03*l/D, I=0.5*I0

	kind:
		Return which type of resolution.
		(1) name = 'FWHM'
		(2) name = 'Rayleigh_criterion'

	wavelength, D:
		Both in meter (m)
		(1) If don't give, return the factor 1.22 or 1.03
		(2) If give, return 1.22*l/D or 1.03*l/D

	All angle in rad.
	'''
	if (name == 'FWHM') : k = 1.03
	elif (name == 'Rayleigh_criterion') : k = 1.22
	elif (name == '') : k = np.array([1.03, 1.22])
	else : raise Exception(efn()+'name must = "FWHM" or "Rayleigh_criterion"')
	if (wavelength == '') : return k
	else : return k*wavelength/D


def SolidAngle( theta_0='' ) : 
	'''
	The solid angle of Gaussian beam 
		k0 = 1.1244(from NVSS) (or 1.133(from book))
		(1) Rayleigh criterion: k0*(1.22*l/D)**2, I=0.3675*I0 (first dark ring)
		(2) FWHM: k0*(1.03*l/D)**2, I=0.5*I0

	theta_0:
		Resolution from Resolution() function
	'''
	k0 = 1.1244
	if (theta_0 == '') : return k0
	else : return k0 * theta_0**2


##################################################
##################################################
##################################################


def ArrayLeftRightInterp( array, axis, num, back=False ) : 
	''' Mostly be used in the function Smooth() '''
	if (type(array) == np.ma.core.MaskedArray) : return array
	nalr, array = num, npfmt(array)
	# move axis to axis=0
	array = ArrayAxis(array, axis, 0, 'move')
	if (back == False) : 
		aleft = array[:nalr]
		aright = array[-nalr:]
		aleft = (aleft - aleft[-1:] + aleft[:1])[:-1]
		array = np.append(aleft, array, axis)
		aright = (aright - aright[:1] + aright[-1:])[1:]
		array = np.append(array, aright, axis)
	else : array = array[nalr-1:-nalr+1]
	array = ArrayAxis(array, 0, axis, 'move')
	return array


def _DoMultiprocess_Smooth( iterable ) : 
	mp1, mp2 = iterable[0]
	sigma, reduceshape, per, N, arrayfont, arrayend, array0font, array0end, nsplit, dosigma = iterable[2]
	#--------------------------------------------------
	if (sigma is not False) : 
		arrayloc = iterable[1]
		arrayloc = ArrayAxis(arrayloc, -1, 0, 'move')
		nr = len(arrayloc)/2
		if (dosigma) : arrayloc0 = arrayloc[nr:]
		arrayloc  = arrayloc[:nr]
		arrayloc  = ArrayAxis(arrayloc,  0, -1, 'move')
		if (dosigma) : 
			arrayloc0 = ArrayAxis(arrayloc0, 0, -1, 'move')
			sa = arrayloc*0.
		else : sa = None
	else : 
		arrayloc = iterable[1]
		sa = None
	#--------------------------------------------------
	result = arrayloc*0.
	n = nsplit.index(iterable[0])
	nf = len(arrayfont[n])
	ne = nf + len(arrayloc)
	arrayloc=np.concatenate([arrayfont[n],arrayloc,arrayend[n]])
	if (sigma is not False and dosigma) : 
		arrayloc0 = np.concatenate([array0font[n], arrayloc0, array0end[n]])
	N = len(arrayloc)
	#--------------------------------------------------
	for i in xrange(nf, ne) : 
		if (i < per/2) : n1, n2 = 0, per-per/2+i
		elif (i > N-(per+1)/2) : n1, n2 = (per+1)/2-per+i, N
		else : n1, n2 = i-per/2, per-per/2+i
		#--------------------------------------------------
		result[i-nf] = arrayloc[n1:n2].mean(0)  # main !
		#--------------------------------------------------
		if (sigma is not False and dosigma) : 
			if (sigma is True) : sa[i-nf] = RMS(arrayloc0[n1:n2]-result[i-nf:i-nf+1], 0) /(n2-n1)
			else : 
				if (sigma.size==1) : error = (n2-n1)**0.5*sigma
				else : error = ((sigma[n1-nf:n2-nf]**2).sum())**0.5
				sa[i-nf] = error /(n2-n1)
	#--------------------------------------------------
	return [result, sa]


def Smooth( array, axis, per, times=1, sigma=False, reduceshape=False, applr=False, Nprocess=None ) : 
	'''
	Smooth/Average/Mean array along one axis.
	we can also use spsn.convolve() to do this, but spsn.convolve() will spend much more memory and time, so, the function written here is the best and fastest.

	Weighted average, sigma will reduce to 1/sqrt{per**times}
	Equivalent to average over per**times

	axis:
		array will be smoothed/averaged along which axis.

	per:
		How many bins/elements to average.

	times:
		How many times to smooth.
		For random noise, times=4 is OK
		Note that this parameter is just for reduceshape=False

	sigma:
		False, True, int, np.array
		If False, don't return the error.
		If True, calculate the error from input array.
		If int or np.array, use this sigma to calculate the error of the result.

	reduceshape:
		False, return.shape = array.shape
		True, return.shape < array.shape

	# Also, we can use Convolve to do it:
	#    w = array*0
	#    w = w[len(array)/2-per/2:len(array)/2+per/2+1] = 1
	#    return Convolve(array, w/w.sum())

	Note that:
		Wide of 2 windows: w1 < w2
		a1  = Convolve(a,  w1)
		a2  = Convolve(a,  w2)
		a12 = Convolve(a1, w2)
		=> a12 = Smooth(a2)
		But the beam sizes of the result maps are similar (roughly the same), Beamsize(a12) >= Beamsize(a2).
	'''
	if (per<=1 or times<=0) : return array
	array = np.array(array)
	per, atype, shape = int(round(per)), array.dtype, array.shape
	if (axis < 0) : axis = len(shape) + axis
	if (axis >= len(shape)) : Raise(Exception, 'axis='+str(axis)+', array.shape='+str(shape)+', axis out of array.shape')
	# Move axis to axis=0
	array = ArrayAxis(array, axis, 0, 'move')
	shape = array.shape  # new shape
	#--------------------------------------------------
	if (reduceshape) : applr, times = False, 1
	#--------------------------------------------------
	# Append left and right
	if (applr == True) : 
		array = ArrayLeftRightInterp(array, 0, per)
		shape = array.shape
	#--------------------------------------------------
	# sigma, True, False, int/ndarray
	if (sigma is not False) : # True, int/ndarray
		if (sigma is not True) : 
			sigma = npfmt(sigma)
			if (sigma.size == 1) : sigma = array*0.+sigma.take(0)
			elif (applr == True) : 
				sigma = ArrayLeftRightInterp(sigma, 0, per)
		array0 = array*1
	#--------------------------------------------------
	#--------------------------------------------------
	for t in xrange(times) : 
		if (t == times-1) : dosigma = True
		else : dosigma = False
		#--------------------------------------------------
		pool = PoolFor(0, shape[0], Nprocess)
		#--------------------------------------------------
		nsplit = pool.nsplit[:]
		arrayend, arrayfont = [], []
		for i in xrange(len(nsplit)) : 
			arrayfont.append(array[nsplit[i][0]-per:nsplit[i][0]])
			arrayend.append( array[nsplit[i][1]:nsplit[i][1]+per])
		if (sigma is False) : array0font, array0end = None, None
		elif (t == 0) : array0font, array0end = arrayfont[:], arrayend[:]
		#--------------------------------------------------
		if (sigma is not False) : # True, int/ndarray
			if (len(shape) == 1) : array = np.concatenate([array[:,None], array0[:,None]], 1)
			else : array = np.concatenate([array, array0], -1)
		#--------------------------------------------------
		retn = pool.map_async(_DoMultiprocess_Smooth, array, (sigma, reduceshape, per, shape[0], arrayfont, arrayend, array0font, array0end, nsplit, dosigma))
		#--------------------------------------------------
		# Distinguish result and sa
		result, satmp = [], []
		for i in xrange(len(retn)) : 
			result.append(retn[i][0])
			if (retn[i][1] is not None) : satmp.append(retn[i][1])
		array = np.concatenate(result)
		if (len(satmp) > 0) : sa = np.concatenate(satmp)
		del pool, retn, result, satmp
	#--------------------------------------------------
	array0 = 0 #@
	if (applr is True) : 
		array = ArrayLeftRightInterp(array, 0, per, True)
		if (sigma is not False) : 
			sa = ArrayLeftRightInterp(sa, 0, per, True)
	#--------------------------------------------------
	if (reduceshape) : 
		nc = np.linspace(0, shape[0], shape[0]/per+1)
		nc = Edge2Center(nc).astype(int)
		array = array[nc]
		if (sigma is not False) : sa = sa[nc]
	#--------------------------------------------------
	array = ArrayAxis(array, 0, axis, 'move')
	if (sigma is not False) : 
		sa = ArrayAxis(sa, 0, axis, 'move')
		return [array, sa]
	else : return array





def SmoothWeight( per, times, plv=False ) : 
	if (per%2==0 and plv) : 
		print 'Warning: SmoothWeight(), per must be odd, reset per='+str(per)+' to per='+str(per+1)
		per +=1
	a0 = np.ones(per)
	#----- Method 1 -----
#	for n in xrange(1, times+1) : 
#		if (times == 1) : break
#		if (n == 1) : continue
#		a = np.zeros([per, 1+2*n])
#		for j in xrange(len(a)) : 
#			a[j,j:j+len(a0)] = a0
#		a0 = a.sum(0)
#	weight = a0 / (1.*per)**times
	#----- Method 1 END -----
	#----- Method 2 -----
	for n in xrange(1, times+1) : 
		if (times == 1) : break
		if (n == 1) : continue
		a = np.zeros([per, 1+2*n*(per/2)])
		for j in xrange(len(a)) : 
			a[j,j:j+len(a0)] = a0
		a0 = a.sum(0) /per
	weight = a0/per  #@ Remember this !!!
	#----- Method 2 END -----
	return weight


def Smooth( array, axis, per, times=1, sigma=False, reduceshape=False ) : 
	'''
	Smooth/Average/Mean array along one axis.
	We can also use spsn.convolve() to do this, but spsn.convolve() will cost much more memory and time, so, the function written here is the best and fastest.

	Weighted average, sigma will reduce to 1/sqrt{per**times}
	Equivalent to average over per**times

	axis:
		array will be smoothed/averaged along which axis.

	per:
		How many bins/elements to average.

	times:
		How many times to smooth.
		For random noise, times=4 is OK
		Note that this parameter is just for reduceshape=False

	sigma:
		False, True, int, np.array
		If False, don't return the error.
		If True, calculate the error from input array.
		If int or np.array, use this sigma to calculate the error of the result.

	reduceshape:
		False, return.shape = array.shape
		True, return.shape < array.shape

	# Also, we can use Convolve to do it:
	#    w = array*0
	#    w = w[len(array)/2-per/2:len(array)/2+per/2+1] = 1
	#    return Convolve(array, w/w.sum())

	Note that:
		Wide of 2 windows: w1 < w2
		a1  = Convolve(a,  w1)
		a2  = Convolve(a,  w2)
		a12 = Convolve(a1, w2)
		=> a12 = Smooth(a2)
		But the beam sizes of the result maps are similar (roughly the same), Beamsize(a12) >= Beamsize(a2).
	'''
	if (per%2 == 0) : per +=1
	if (per<=1 or times<=0) : return array
	array = np.array(array)
	per, atype, shape = int(round(per)), array.dtype, array.shape
	if (axis < 0) : axis = len(shape) + axis
	if (axis >= len(shape)) : Raise(Exception, 'axis='+str(axis)+', array.shape='+str(shape)+', axis out of array.shape')
	# Move axis to axis=0
	array = ArrayAxis(array, axis, 0, 'move')
	shape = array.shape
	#--------------------------------------------------
	if (not reduceshape) : 
		weight = SmoothWeight( per, times )
		lw = len(weight)
		shapew = npfmt(shape)
		shapew[0] = lw
		shapew[1:] = 1
		weight = weight.reshape(shapew)
		#--------------------------------------------------
		dnwa1 = None
		if (len(array) < lw) : 
			dnwa1 = (lw-len(array))/2
			dnwa2 = lw-len(array) - dnwa1
			if (dnwa1 > 0) : a1 = np.concatenate([array[:1]  for i in xrange(dnwa1)])
			else : a1 = []
			a2 = np.concatenate([array[-1:] for i in xrange(dnwa2)])
			array = np.concatenate([a1, array, a2])
			a1 = a2 = 0 #@
		#--------------------------------------------------
		b = array*0
		for i in xrange(len(array)) : 
			if (i < lw/2) : 
				da = np.concatenate([array[:1] for j in xrange(lw/2-i)])
				ai = np.concatenate([da, array[:lw/2+1+i]])
			elif (i >= len(array)-lw/2) : 
				dn = lw/2 - (len(array)-1 - i)
				da = np.concatenate([array[-1:] for j in xrange(dn)])
				ai = np.concatenate([array[i-lw/2:], da])
			else : ai = array[i-lw/2:i+lw/2+1]
			b[i] = (ai * weight).sum(0)
		if (dnwa1 is not None) : b = b[dnwa1:-dnwa2]
		ai = w = 0 #@
	#--------------------------------------------------
	else : 
		if (times == 1) : Npix = len(array)/per
		else : Npix = times
		n = np.linspace(0, len(array), Npix+1).astype(int)
		b = np.zeros((Npix-1,) + shape[1:])
		for i in xrange(len(n)-1) : 
			b[i] = array[n[i]:n[i+1]].mean(0)
	#--------------------------------------------------
	b = ArrayAxis(b, 0, axis, 'move')
	return b


##################################################
##################################################
##################################################


def PCA( y, nPC=3, malposition1D=20, remove=True, fit=False, save=False ) : 
	'''
	PCA can be also used to smooth the data.

	y: 
		Input y map, can be 1D-array, 2D-array or matrix.
		If y is 1D, it will be converted to 2D.
	y.shape = (n1,n2), row n1 is the number of frequencies, column n2 is the number of pixels at its frequency.
	
	nPC: 
		Number of PC you want to save, default nPC=3, if nPC=0 then save eigen_value.sum()>=99.99% (but not all)
		nPC=0, used for y=1D, means doesn't do the PCA, just return the smooth-remove result.

	malposition1D:
		This parameter is just for 1D y.
      We use this parameter to convert y to 2D.

	remove:
		This parameter is just for 1D y.
		If True, remove points which is too large or too small.

	fit:
		This parameter is just for 1D y.
		When establish 2D matrix, there will be some zero on the top and bottom of the 2D matrix, so fit this blank or not.

	save:
		Save the result to output files. True or False
	
	return: 
		ye = first nPC of y map, ye.shape = y.shape
	'''
	shape = np.array(y.shape)
	if (shape.size > 2) : raise Exception(efn()+'len(y.shape) must <= 2')
	shape1 = shape[shape>1]
	if (shape1.size == 1) : 
		if (malposition1D%2 == 0) : malposition1D = malposition1D + 1
		y = Smooth(y.flatten(), 0, 10)
		dN = malposition1D/2
		if (remove) : 
			ys = abs(y)
			dy = abs(2*ys -np.append(ys[4:],ys[-4*2:-4]) -np.append(ys[4:4*2], ys[:-4]))
			rms = RMS(dy[dy<10*RMS(dy)])
			ys = 0 #@
			n0 = np.arange(y.size)
			dy = np.append(n0[:,None],dy[:,None],1)
			n = dy[dy[:,1]>rms][:,0].astype(int)
			dy = 0 #@
			if (n.size > 0) : 
				if (n[0] == 0) : n = n[1:]
				if (n[-1] == len(y)) : n = n[:-1]
				# block n
				n1, p = [], [n[0]]
				for i in range(1, len(n)) : 
					if (n[i]-n[i-1] < 10) : 
						p = p + [n[i]]
					if (n[i]-n[i-1]>=10 or i==len(n)-1) : 
						n1 = n1 + [[p[0],p[-1]]]
						if (i != len(n)-1) : p = [n[i]]
				n = np.array([])
				for i in range(len(n1)) : 
					n = np.append(n, np.arange(n1[i][0], n1[i][1]+1))
				n = n.astype(int)
				ny = np.append(n0[:,None], y[:,None], 1)
				n0 = abs((n0+1)[:,None] - (n+1)[None,:])
				for  i in range(len(n0)) : 
					if (n0[i].min() == 0) : n0[i,0] = -1
				n0 = n0[:,0]
				ny = ny[n0>0]
				f = interpolate.interp1d(ny[:,0], np.arcsinh(ny[:,1]))
				y[n] = np.sinh( f(n) )
				n1 = n = n0 = ny = 0 #@
		if (nPC == 0) : return y
		if (fit) : 
			y = np.arcsinh(y)
			k1 = y[1] - y[0] # >0
			k2 = y[-2] - y[-1] # >0
			yl = k1*np.arange(-dN,0) + y[0]
			yr = -k2*np.arange(1,dN+1) + y[-1]
			y = np.append(yl, y)
			y = np.append(y, yr)
			y = np.sinh(y)
			yl = yr = 0 #@
		Ny1 = y.size
		Ny2 = Ny1+malposition1D-1
		y2 = np.zeros([Ny2, malposition1D])
		y0l, y0r = y[:dN], y[-dN:]
		n = 0
		for i in range(malposition1D) : 
			y2[n:n+Ny1,i] = y
			n = n + 1
		if (fit) : y = y2[2*dN:Ny1]
		else : y = y2[dN:dN+Ny1]

	# get pixel number
	npix = (y.shape)[1]

	# covariance matrix
	y = np.matrix( y )
	C = y * y.T / npix
	
	# rms
	rms = np.diag(C)**0.5
	C = 0 #@
	if (save) : 
		outname = 'rms_of_each_row_of_y_map.dat'
		np.savetxt( outname, rms )
		print outname+'   --->   OK'
	
	# z map
	z = y / rms[:,None]
	y = 0 #@
	if (save) : 
		outname = 'z_map_normalized'
		np.save( outname, z )
		print outname+'.npy   --->   OK'
	
	# correlation matrix
	R = z * z.T / npix
	
	# also R = C / ( rms[:,None] * rms[None,:] )
	
	# eigenvalue decomposition for the R 
	v, P = np.linalg.eigh( R )
	v = v[::-1]    # we need from large to small
	P = P[:,::-1]  # default from small to large
	
	if (save) : 
		outname = 'eigen_value_of_R.dat'
		np.savetxt( outname, v )
		print outname+'   --->   OK'
		
		outname = 'eigen_vector_matrix_of_R'
		np.save( outname, P )
		print outname+'.npy   --->   OK'
	
	# a map (projection factor)
	a = np.matrix(P).T * z
	if (save) : 
		outname = 'a_map_of_R'
		np.save( outname, a )
		print outname+'.npy   --->   OK'

	if ( nPC == 0 ) : 
		for i in range(len(v)) : 
			v = v / v.sum()
			if ( v[:i].sum() > 0.9999 ) : 
				nPC = i
				print 'nPC =', nPC
				break
	
	# z map estimator with first nPC
	ze = np.array( P[:,:nPC] * a[:nPC] )
	
	# y map estimator with first nPC
	ye = ze * rms[:,None]
	if (save) : 
		outname = 'y_map_estimator_with_'+str(nPC)+'PC'
		np.save( outname, ye )
		print outname+'.npy   --->   OK'

	if (shape1.size == 1) : 
		ye = ye[:,dN].reshape(shape)
		ye[:dN]  = y0l
		ye[-dN:] = y0r
		
	return ye


##################################################
##################################################
##################################################


def Convolve( source, beam, edge='mirror' ) : 
	'''
	This function is used to convolve two nd-array with same shape (source.shape==beam.shape).

	beam:
		You can normalize it to beam.sum()=1 or beam.max()=1 or not normalize depending on the situation.
	
	edge: 
		How to handle with the edge.
		(1) edge='linear', convolve directly.
		(2) edge=value(0, 1, source.min() and so on), outside of the source will be treated as this value.
		(3) edge='mirror', the edge as a mirror.
	'''
	shape = source.shape
	dimension = len(shape)
	if (source.shape != beam.shape) : 
		raise Exception('Two array (source and beam) have different shape. source.shape='+str(source.shape)+', beam.shape='+str(beam.shape))
	if (dimension > 3) : raise Exception('Now can just handle <=3D array, but you can modify this function yourself for >=4D')

	def fft( x ) : 
		if   (dimension == 1) : y = np.fft.fft( x ) 
		elif (dimension == 2) : y = np.fft.fft2( x ) 
		else : y = np.fft.fftn( x ) 
		return y

	def ifft( x ) : 
		if   ( dimension == 1 ) : y = np.fft.ifft( x ) 
		elif ( dimension == 2 ) : y = np.fft.ifft2( x ) 
		else : y = np.fft.ifftn( x ) 
		return y

	# npix should be even when convolve
	if (shape[0]%2 == 1) : 
		source = np.append(source, source[shape[0]-1:shape[0]], 0)
		beam = np.append(beam, beam[shape[0]-1:shape[0]], 0)
	if (dimension >= 2) : 
		if (shape[1]%2 == 1) : 
			source = np.append(source, source[:,shape[1]-1:shape[1]], 1)
			beam = np.append(beam, beam[:shape[1]-1:shape[1]], 1)
	if (dimension >= 3) : 
		if (shape[2]%2 == 1) : 
			source = np.append(source, source[:,:,shape[2]-1:shape[2]], 2)
			beam = np.append(beam, beam[:,:,shape[2]-1:shape[2]], 2)


	if (edge == 'linear') : 
		image = np.fft.fftshift(ifft(fft(source)*fft(beam))).real
	else :
		if (edge == 'mirror') : v, edge = 1, 0
		else : v, edge = 0, float(edge)
		
		if (dimension == 1) : 
			n0 = source.shape[0]
			n1 = int(round(n0/2.))
			source1 = source[::-1] *v+edge
			source = np.append(source1, source)
			source = np.append(source, source1)
			source = source[n0-n1:n1-n0]
			source1 = 0 #@
			# For beam, always add 0
			beam1 = np.zeros([n0+2*n1,], beam.dtype)
			beam1[n0-n1:n1-n0] = beam
			beam = beam1
			image = np.fft.fftshift(ifft(fft(source)*fft(beam))).real
			image = image[n1:-n1]

		elif (dimension == 2) : 
			n0 = npfmt(source.shape)
			n1 = (n0/2.).round().astype(int)
			source1 = source[:,::-1] *v+edge
			source = np.append(source1, source, 1)
			source = np.append(source, source1, 1)
			source1 = source[::-1]
			source = np.append(source1, source, 0)
			source = np.append(source, source1, 0)
			source = source[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1]]
			source1 = 0 #@
			# For beam, always add 0
			beam1 = np.zeros(n0+2*n1, beam.dtype)
			beam1[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1]] = beam
			beam = beam1
			image = np.fft.fftshift(ifft(fft(source)*fft(beam))).real
			image = image[n1[0]:-n1[0],n1[1]:-n1[1]]

		elif (dimension == 3) : 
			n0 = npfmt(source.shape)
			n1 = (n0/2.).round().astype(int)
			# axis 0
			source1 = source[::-1] *v+edge
			source = np.append(source1, source, 0)
			source = np.append(source, source1, 0)
			# axis 1
			source1 = source[:,::-1]
			source = np.append(source1, source, 1)
			source = np.append(source, source1, 1)
			# axis 2
			source1 = source[:,:,::-1]
			source = np.append(source1, source, 2)
			source = np.append(source, source1, 2)
			# result
			source = source[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1],n0[2]-n1[2]:n1[2]-n0[2]]
			source1 = 0 #@
			# For beam, always add 0
			beam1 = np.zeros(n0+2*n1, beam.dtype)
			beam1[n0[0]-n1[0]:n1[0]-n0[0],n0[1]-n1[1]:n1[1]-n0[1],n0[2]-n1[2]:n1[2]-n0[2]] = beam
			beam = beam1
			image = np.fft.fftshift(ifft(fft(source)*fft(beam))).real
			image = image[n1[0]:-n1[0],n1[1]:-n1[1],n1[2]:-n1[2]]

	if   (dimension == 1) : image = image[:shape[0]]
	elif (dimension == 2) : image = image[:shape[0],:shape[1]]
	elif (dimension == 3) : image = image[:shape[0],:shape[1],:shape[2]]
	return image


##################################################
##################################################
##################################################


def arcsincos( sina, cosa, factor='+' ) : 
	'''
	Input Asin(a) and Acos(a), return the angle a.
	
	sina, cosa:
		Can be array, list, number

	factor:
		Factor A is arbitory, but you need to set is sign
		factor = '+' or '-' 

	Because the period question, a cann't be determinded by one sin(a) nor cos(a), it must be determinded with both.
	'''
	sina, cosa = npfmt(sina), npfmt(cosa)
	# shape of sina and cosa
	ssin, scos = sina.shape, cosa.shape
	if (ssin != scos) : raise Exception(enf()+'sina.shape != cosa.shape')
	a = sina *0.
	if (factor != '+') : sina, cosa = -sina, -cosa
#	if (cosa == 0) : 
#		if (sina > 0) : a = np.pi/2
#		else : a = 1.5 * np.pi
#	else : 
#		tana = sina / cosa
#		if (sina == 0) : 
#			if (cosa > 0) : a = 0.
#			else : a = np.pi
#
#		elif (cosa < 0) : a = np.pi + np.arctan(tana)
#
#		elif (cosa>0 and sina>0) : a = np.arctan(tana)
#		elif (cosa>0 and sina<0) : a = 2*np.pi + np.arctan(tana)
#	return a
	# cosa=0, sin=1
	tf90 = ((cosa==0) * (sina>0))
	# cosa=0, sin=-1
	tf270 = ((cosa==0) * (sina<0))
	cosa[cosa==0] = 1e-20
	tana = sina / cosa
	# cosa<0
	tf = (cosa<0)
	a[tf] = np.pi + np.arctan(tana[tf])
	# cosa>0, sina>0
	tf = ((cosa>0) * (sina>0))
	a[tf] = np.arctan(tana[tf])
	# cosa>0, sina<0
	tf = ((cosa>0) * (sina<0))
	a[tf] = 2*np.pi + np.arctan(tana[tf])
	# sina=0, cosa=1
	tf = ((sina==0) * (cosa>0))
	a[tf] = 0.
	# sina=0, cosa=-1
	tf = ((sina==0) * (cosa<0))
	a[tf] = np.pi
	a[tf90] = np.pi
	a[tf270] = 1.5*np.pi
	return a


def HA2azalt( lat, HAoraz, Decoralt, ha2azalt=True ) : 
	'''
	Convert (HA(hour angle),Dec) to (az,alt) or (az,alt) to (HA,Dec).

	lat:
		latitude of the observation place.

	ha2azalt:
		If True, convert (HA,Dec) to (az,alt), at this time, HAoraz=HA, Decoralt=Dec
		If False, convert (az,alt) to (HA,Dec), at this time, HAoraz=az, Decoralt=alt

	Note that az is starting from North, and clockwise.

	All angle in jizhi.py are in rad

	Return:
		[az, alt] or [HA, Dec]
	'''
	a = HAoraz
	b = Decoralt
	siny = np.sin(lat)*np.sin(b) + np.cos(lat)*np.cos(b)*np.cos(a)
	if ( siny == 1 or siny == -1 ) : 
		if ( siny == 1 ) : y = np.pi/2
		if ( siny == -1 ) : y = -np.pi/2
		x = 0
	else : 
		cosycosx = np.cos(lat)*np.sin(b) - np.sin(lat)*np.cos(b)*np.cos(a)
		cosysinx = -np.cos(b)*np.sin(a)
		y = np.arcsin( siny )
		cosy = np.cos( y )
		cosx = cosycosx / cosy
		sinx = cosysinx / cosy
		return [arcsincos( sinx, cosx ), y]


##################################################
##################################################
##################################################


## aipy.phs.AntennaArray.get_baseline can not get the baseline with set pointing direction (HA, Dec)
#
#class AntennaArray( aipy.phs.AntennaArray ) : 
#	def get_baseline_set( self, i, j, HA, Dec ) : 
#		bl = self[j] - self[i]
#		return np.dot(aipy.coord.eq2top_m(HA,Dec), bl)


##################################################
##################################################
##################################################


#class Beam2DGaussian( aipy.amp.Beam2DGaussian ) : 
#	'''
#	For aipy.amp.Beam2DGaussian, xFWHM and yFWHM must be a number (one resolution) not array.
#
#	aipy.amp.Beam2DGaussian return A**0.5, not power pattern
#
#	aipy.amp.Beam2DGaussian.response()'s code resp = n.resize(resp, (self.afreqs.size, resp.size)) has not any valid, so I delete it.
#	
#	From now on, use Beam2DGaussian below instead of aipy.amp.Beam2DGaussian
#	
#	Note that aipy.amp.Beam2DGaussian( freqs, xwidth, ywidth ), where xwidth is actually the xFWHM resolution, not width
#	
#	For this class, 
#	   xwidth, ywidth: width of the beam image (map) in rad
#	   Nx, Ny: shape of the beam image (map)
#	'''
#	def response_power( self, xwidth, ywidth, Nx, Ny ) : 
#		x0 = np.sin( xwidth / 2. )
#		y0 = np.sin( ywidth / 2. )
#		y, x = np.mgrid[ -y0:y0:1j*Ny, -x0:x0:1j*Nx ]
#		nf = self.xwidth.size
#		if ( nf > 1 ) : 
#			x = x.reshape( 1, Ny, Nx )
#			y = y.reshape( 1, Ny, Nx )
#			self.xwidth = self.xwidth.reshape( nf, 1, 1 )
#			self.ywidth = self.ywidth.reshape( nf, 1, 1 )
#		x,y = np.arcsin(x)/self.xwidth, np.arcsin(y)/self.ywidth
#		A = np.exp( -4*np.log(2) * (x**2 + y**2) )
#		return A


##################################################
##################################################
##################################################


def hp_Alm_getidx_lm( lmax, l=-1, m=-1 ) : 
	'''
	Spheric harmonic expansion
	Coefficient Alm
	Theory: l=[-inf, +inf] and m=[-l, +l]
	In healpy, l=[0, lmax], m=[0, +l]

	This function bases on healpy.Alm.getidx().
	Alm is 1D array, for (l=10, m=5), we can get the corresponding coefficient is Alm[n], this function is used to return this index n.
	For different lmax, n is different.
	
	lmax:
		maximun of l in Spheric harmonic expansion in healpy

	l, m:
		(1) give both l and m, return one index n :
			hp_Alm_getidx_lm( lmax=512, l=4, m=2 )
		(2) just give l, return 1D index n=[n1,n2,...] with this l(m=[0,+l]) : 
			hp_Alm_getidx_lm( lmax=512, l=4 )
		(3) just give m, return 1D index n=[n1,n2,...] with this m(l=[m, lmax]) : 
			hp_Alm_getidx_lm( lmax=512, m=2 )
	'''
	def ArithmeticSum( a0, d, n ) : 
		return n*(2*a0+(n-1)*d)/2.
	def getidxl( lmax, l ) : 
		m = np.arange( l+1 )
		ln2 = ArithmeticSum( lmax+1, -1, m+1 )
		d = lmax - l + 1
		ln = ln2 - d
		return ln.astype(int)
	def getidxm( lmax, m ) : 
		mn1 = ArithmeticSum( lmax+1, -1, m )
		mn2 = ArithmeticSum( lmax+1, -1, m+1 )
		mn = np.arange( mn1, mn2 )
		return mn.astype(int)
	if ( l<0 and m<0 ) : 
		raise Exception( 'Wrong values of l<0 and m<0 at the same thim.' )
	if ( m>lmax ) : 
		raise Exception( 'm='+str(m)+ ' > lmax='+str(lmax) )
	if (l>=0 and m<0) : lorm='l'
	elif (l>=0 and m>=0) : lorm='lm'
	elif (l<0 and m>=0) : lorm='m'
	if ( lorm == 'lm' ) : return hp.Alm.getidx( lmax, l, m )
	elif (lorm == 'l' ) : return getidxl( lmax, l )
	elif (lorm == 'm' ) : return getidxm( lmax, m )


##################################################
##################################################
##################################################


def Associated_Legendre( l, m, theta ) : 
	'''
	Mathematical principle of spsp.lpmv( m, l, x ).
	In spsp.lpmv(m, l, x), can be:
		spsp.lpmv([0,1,2,3], 3, 0.567)
		spsp.lpmv(3, [3,4,5,6], 0.567)
		spsp.lpmv([1,2,3,4], [3,4,5,6], 0.567)

	l: 
		order of Legendre polynomials, 0~+N
		l can be an value, list, array

	m: 
		-l <= m <= l, must m.shape == l.shape

	theta: 
		x = cos(theta), 0~pi
		theta can be an value, list, array

	return: 
		shape = ( l.size, theta.size )	
	'''	
	if ( type(l) == list ) : l = np.array(l)
	elif ( type(l) != np.ndarray ) : l = np.array([l])

	if ( type(m) == list ) : m = np.array(m)
	elif ( type(m) != np.ndarray ) : m = np.array([m])

	if ( l.shape != m.shape ) : raise Exception( 'l.shape != m.shape' )

	if ( type(theta) == list ) : theta = np.array(theta)
	elif ( type(theta) != np.ndarray ) : theta = np.array([theta])  # one value
	theta = theta[:,None]

	if ( l.min() < 0 ) : raise Exception( 'l must be >= 0, but now l.min()='+str(l.min()) )
	if ( theta.min()<0 or theta.max()>np.pi ) : 
		raise Exception( 'theta must be [0, pi], but now theta=['+str(theta.min())+','+str(theta.max())+']' )

	x = np.cos( theta )

	Plmxa = np.zeros( [ l.size, theta.size ] )
	for i in range( len(l) ) : 
		li = l[i]
		mi = m[i]
		mi0 = mi
		if ( mi < 0 ) : mi = -mi
		n = np.arange( li/2 + 1 )[None,:]
		a = li - 2*n - mi
		s = a.copy()
		a[a<0] = 0
		s[s>=0] = 1
		s[s<0] = 0
		para1 = (-1)**n * spsp.gamma(2*li-2*n+1)/(2**li * spsp.gamma(n+1) * spsp.gamma(li-n+1))
		para2 = spsp.gamma( a+1 )
		para = para1 / para2
		Plx = para * s * x**a
		Plmx = ((-1)**mi * (1-x**2)**(mi/2.) * Plx).sum(1).flatten()
		if ( mi0 < 0 ) : 
			Plmx = (-1)**mi * spsp.gamma(li-mi+1)/spsp.gamma(li+mi+1) * Plmx
		Plmxa[i] = Plmx
	
	if ( Plmxa.shape[0]==1 or Plmxa.shape[1]==1 ) : Plmxa = Plmxa.flatten()
	if ( Plmxa.size == 1 ) : Plmxa = Plmxa[0]

	return Plmxa


def SphericHarmonic( l, m, theta, phi ) : 
	'''
	Mathematical principle of spsp.sph_harm( m, l, phi, theta ).

	l: 
		order of Legendre polynomials, 0~+N
		l can be an value, list, array
	m: 
		-l <= m <= l, must m.shape == l.shape

	theta: 
		x = cos(theta), 0~pi
		theta can be an value, list, array

	phi: 
		0~2pi, phi can be an value, list, array

	return: 
		shape = ( l.size, theta.size )
		where l.size=m.size, theta.size=phi.size
	'''	
	if ( type(l) == list ) : l = np.array(l)
	elif ( type(l) != np.ndarray ) : l = np.array([l])

	if ( type(m) == list ) : m = np.array(m)
	elif ( type(m) != np.ndarray ) : m = np.array([m])

	if ( l.shape != m.shape ) : raise Exception( 'l.shape != m.shape' )

	if ( type(theta) == list ) : theta = np.array(theta)
	elif ( type(theta) != np.ndarray ) : theta = np.array([theta])  # one value

	if ( l.min() < 0 ) : raise Exception( 'l must be >= 0, but now l.min()='+str(l.min()) )
	if ( theta.min()<0 or theta.max()>np.pi ) : 
		raise Exception( 'theta must be [0, pi], but now theta=['+str(theta.min())+','+str(theta.max())+']' )

	if ( type(phi) == list ) : phi = np.array(phi)
	elif ( type(phi) != np.ndarray ) : phi = np.array([phi])  # one value

	Plmx = Associated_Legendre( l, m, theta )

	theta = theta[:,None]
	phi = phi[None,:]
	l = l[:,None]
	m = m[:,None]

	paralm = ((2*l+1)/4./np.pi * spsp.gamma(l-m+1)/spsp.gamma(l+m+1))**0.5 * np.exp(1j*m*phi)

	Plmx = Plmx.reshape( paralm.shape )
	Ylm = paralm * Plmx

	if ( Ylm.shape[0]==1 or Ylm.shape[1]==1 ) : Ylm = Ylm.flatten()
	if ( Ylm.size == 1 ) : Ylm = Ylm[0]

	return Ylm


##################################################
##################################################
##################################################


def healpix2xy( healpix_map ) : 
	'''
	Order of values of healpix map is range or nest, not square matrix.
	This function convert healpix map to xy square matrix (plane).
	(healpy.projector.GnomonicProj.ang2xy())
	We can devide the sphere along the b-axis (not l-axis), because the value (emission) change with b not l.

	return : [xy, lmatrix, bmatrix]
		xy : shape=(b.size, l.size)
		lmatrix and bmatrix are both in rad !
		lmatrix : from left to right : (180,170,..,0,350,340,..,180)
		bmatrix : from top to bottom : (0,10,..,170,180), start from 0
	'''
	# for nside, healpix map is devided into 4*nside-1 rows and 4*nside columns
	nside = int( (len(healpix_map)/12)**0.5 + 0.1 )
	# but I have tested that in order to reconstruct the healpix map completely, nrow = ncolumn = k*nside, k must >=7, now I set k = 8 (not 4, 4 is not enough)
	nrow    = 8 * nside  # square matrix
	ncolumn = 8 * nside

	l1 = np.linspace( np.pi, 0, ncolumn/2+1 )
	l2 = np.linspace( 2*np.pi, np.pi, ncolumn/2+1 )[1:-1]
	l = np.append( l1, l2 )
	b = np.linspace( 0, np.pi, nrow+1 )[:-1]

	lmatrix = l[None,:] + np.zeros([ nrow, 1 ]) # rad
	bmatrix = b[:,None] + np.zeros([ 1, ncolumn ])

	l = lmatrix.flatten()
	b = bmatrix.flatten()

	n = hp.ang2pix( nside, b, l )

	xy = healpix_map[n].reshape( ncolumn, nrow )

	return [xy, lmatrix, bmatrix]


def xy2healpix( xy_map ) : 
	'''
	This function is the oppositive function of healpix 2xy()
	'''
	if ( xy_map.shape[0] != xy_map.shape[1] ) : 
		raise ValueError( 'xy_map must be a square matrix that nrow = ncolumn = 4*nside, but now xy_map.shape=('+str(xy_map.shape[0])+','+str(xy_map.shape[1])+')' )

	n = np.log(xy_map.shape[0]/8.) / np.log(2)
	if ( (n-abs(n)) != 0. ) : 
		raise ValueError( 'nside = 2**n, but now nside='+str(xy_map.shape[0]/8) )

	nrow = ncolumn = xy_map.shape[0]
	nside = nrow / 8

	l1 = np.linspace( np.pi, 0, ncolumn/2+1 )
	l2 = np.linspace( 2*np.pi, np.pi, ncolumn/2+1 )[1:-1]
	l = np.append( l1, l2 )
	b = np.linspace( 0, np.pi, nrow+1 )[:-1]

	lmatrix = l[None,:] + np.zeros([ nrow, 1 ])
	bmatrix = b[:,None] + np.zeros([ 1, ncolumn ])

	n = hp.ang2pix( nside, bmatrix, lmatrix )
	n = n.flatten()

	healpix_map = np.zeros([ 12*nside**2, ]) + xy_map.mean()
	healpix_map[n] = xy_map.flatten()

	return healpix_map


##################################################
##################################################
##################################################


def HealpixZoom( healpix_map, latitude, longitude, latsize, lonsize ) : 
	'''
	This function is used to select a sky region in healpix map, and zoom it in a flat 2D matrix.
	Healpix map can be in Equatorial coordinate or Galactic coordinate.

	longtitude, latitude are the center coordinate of the sky region (rad).

	lonsize, latsize are the size (rad) along longitude and latitude.
	Note that latitude is uniform and longitude isn't, longtitude must be converted by factor 1/cos(lat)
	'''
	# lat is uniform, we set lat edges
	# the size of small sky region in lat
	latmin = latitude - latsize/2.
	latmax = latitude + latsize/2.
	
	# get nside
	nside = int( (len(healpix_map)/12)**0.5 + 0.1 )
	
	# calculate Nlat
	# we assume the area of each pixel in healpix is same, then the resolution is (41253/(12*nside**2))**0.5
	res = (41253./len(healpix_map))**0.5 * np.pi/180
	# in order to avoid missing data, we set
	res = res / 50 # the smaller the res is, the better the result will be
	Nlat = int(latsize / res)
	if ( Nlat % 2 != 1 ) : Nlat = Nlat + 1
	latlist = np.linspace( latmin, latmax, Nlat )
	
	# the lon line cross the medium of region is the center value longitude
	# so we need to calculate lon_range in each lat !
	# in lat, for lon, the actually large circle angle = lon*cos(lat)
	# so if we set Dlat size in lon direction, then lon range in each lat is
	
	Nlon = int(lonsize / res)
	if ( Nlon % 2 != 1 ) : Nlon = Nlon + 1
	lona = lonsize / np.cos(latlist)
	lonmaxa = longitude + lona/2
	lonmina = longitude - lona/2
#	# we select a square region
#	Nlon = Nlat
	latmatrix = latlist[:,None] + np.zeros([1,Nlon])
	lonmatrix = np.zeros([ Nlat, Nlon ])
	for i in range( Nlat ) : 
		lonmin = lonmina[i]
		lonmax = lonmaxa[i]
		lonlist = np.linspace( lonmin, lonmax, Nlon )
		lonmatrix[i] = lonlist
	
	# flatten
	lonall = lonmatrix.flatten()
	latall = latmatrix.flatten()
	
	# convert lb to healpix number
	number = hp.ang2pix( nside, np.pi/2-latall, lonall )
	sky = healpix_map[number].reshape( Nlat, Nlon )

	xlist = np.linspace(longitude-lonsize/2., longitude+lonsize/2., Nlon) # this is the angles along x axis of the matrix, not angle of longitude! xlist is the value of column
	ylist = latlist # ylist is the value of row

	return [sky, xlist, ylist]


##################################################
##################################################
##################################################


#def k_FFT_smoothing( FWHM ) : 
#	'''
#	To convolve the map, there are two method, hp.smoothing() and FFT.
#	But the result of two method is different, they have a relation:
#		FFT = k * hp.smoothing()
#	where k is a constant.
#	This code is used to calculate k.
#
#	But hp.smoothing().sum()==1, it means the total power doesn't change, so just the maximun of these two method are different.
#	Actually, hp.smoothing() is correct!!!!!!
#	So, this function is useless!!
#	'''
#	nside = 1024
#	
#	longitude = 0
#	latitude  = 0
#	
#	n = hp.ang2pix( nside, np.pi/2-latitude, longitude )
#	
#	maphp = np.zeros([ 12*nside**2, ])
#	maphp[n] = 1000
#	
#	maphp = hp.smoothing( maphp, FWHM )
#	maphpmax = maphp.max()
#
#	res = (41253./len(maphp))**0.5 * np.pi/180
#	res = res / 2
#	maphp = 0 #@
#	
#	Dflat = 6 * FWHM
#
#	N = int(Dflat / res)
#	
#	theta = ThetaPhiMatrix( N, res )[0]
#	
#	beam = GaussianBeam( theta, FWHM )
#	theta = 0 #@
#	
#	mapb = beam * 0
#	mapb[N/2,N/2] = 1000
#	
#	sky = np.fft.fftshift( np.fft.ifft2( np.fft.fft2(beam/beam.sum()) * np.fft.fft2(mapb) ) ).real
#
#	k = sky.max() / maphpmax
#	return k


##################################################
##################################################
##################################################


def RMS( a, axis=None ) : 
	''' Root Mean Square
	a is array, this function return the rms of a 
	'''
	if (axis is None) : 
		rms = ((a**2).sum()/a.size)**0.5
	else : 
		rms = ((a**2).sum(axis)/(a.shape[axis]))**0.5
	return rms


def Sigma( a, axis=None ) : 
	'''
	Standard Deviation 

	If axis == None, return sigma of the whole a.
	If axis == int, calculate sigma along this axis.

	Also, see Smooth()
	'''
	if (axis is None) : 
		stda = (((a - a.mean())**2).sum()/a.size)**0.5
	else : 
		shape = np.array(a.shape)
		shape[axis] = 1
		stda = (((a - a.mean(axis).reshape(shape))**2).sum(axis)/(a.shape[axis]))**0.5
	return stda
	

##################################################
##################################################
##################################################


class IndexConvert( object ) : 


	def N2OneD( self, indexnd, shapend ) : 
		'''
		indexnd:
			list : [(n1,n2,n3), (n4,n5,n6), ...]
			tuple: ((n1,n2,n3), (n4,n5,n6), ...)
			np.ndarray: np.array([n1,n2,n3],
			                     [n4,n5,n6],
			                     [........])
		'''
		indexnd = npfmt(indexnd)
		shapendprod = np.cumprod(shapend[::-1])[:-1][::-1]
		shapendprod = np.concatenate([shapendprod, [1]])[None,:]
		indexnd *= shapendprod
		return indexnd.sum(1)


	def One2ND( self, index1d, shapend, retntype='nparray' ) : 
		'''
		retntype:
			'nparray'/numpy.ndarray
			'list'/list
		NOTE THAT don't return bool because of no order!
		'''
		index1d = npfmt(index1d)
		index1d0 = index1d.copy()
		shapendprod = np.cumprod(shapend[::-1])[:-1][::-1]
		shapendprod = np.concatenate([shapendprod, [1]])
		indexnd = []
		for i in xrange(len(shapendprod)) : 
			n1 = (index1d / shapendprod[i])
			index1d = (index1d % shapendprod[i])
			indexnd.append( n1 )
		retn = npfmt(indexnd).T
		if (retntype in ['list', list]) : 
			retn = list(retn)
			for i in xrange(len(retn)) : retn[i] = tuple(retn[i])
		return retn


	def N2ND( self, indexndin, shapendin, shapendout, retntype='nparray' ) : 
		retn = self.N2OneD(indexndin, shapendin)
		retn = self.One2ND(retn, shapendout, retntype)
		return retn


##################################################
##################################################
##################################################


def Value2Index( array, value, err=0 ) : 
	'''
	Give value of element of array, return the index of this value. 
	For example, array A, and A[2,7]=10, then Index( A, 10 ) return (2,7). The result maybe a list not one index

	array: can be np.array, list

	value: can be np.array, list, number

	err: error of value, 
		make value = value-err ~ value+err

	return:
		list of result
	'''
	array = npfmt(array)
	if (err == 0) : 
		if (array.dtype.name[:3] != 'int') : err = 1e-8
	value, err = npfmt(value), abs(npfmt(err))
	if (err.size == 1) : err = err[0]*np.ones_like(value)
	if (value.size != err.size) : 
		Raise(Exception, 'value.size != err.size')
	shape = array.shape
	array = array.flatten() + 1j*np.arange(array.size)
	result = []
	for i in range(value.size) : 
		v, e = value[i], err[i]
		n = array[((v-e)<=array.real)*(array.real<=(v+e))].imag.astype(int)
		result = result + [IndexConvert(n, shape)]
	return result


##################################################
##################################################
##################################################


def Extrema( maxormin, array, minusRMS=0 ) : 
	'''
	This function is used to get the extrema (maxima or minima) points of array.

	maxormin: 'maxima' or 'minima'

	array : input array, can be 1D, 2D, 3D, doesn't work in and above 4D

	minusRMS : 
		set minusRMS=0, then it will return all extrema points whose much of them are just very small peaks !
		set minusRMS != 0, then do : 
			array = array - minusRMS * RMS(array)
			array[array<0] = 0, for maxima with array>0
		It can remove most of small peaks.
	'''
	if   ( maxormin == 'maxima' ) : maxminma = np.greater
	elif ( maxormin == 'minima' ) : maxminma = np.less
	else : raise Exception( "maxormin isn't 'maxima' or 'minima'" )

	# make array>0
	if ( array.min() < 0 ) : array = array - array.min()

	if ( minusRMS != 0 ) : 
		if ( minusRMS < 0 ) : raise Exception( 'minusRMS must >=0' )
		else : 
			array = array - minusRMS * RMS(array)
			if ( maxormin == 'maxima' ) : array[array<0] = 0
			elif ( maxormin == 'minima' ) : array[array>0] = 0

	result = []
	for i in range( len(array.shape) ) : 
		cr = spsaa( array, maxminma, axis=i )
		# convert cr to N-D array
		ncr1 = len(cr)
		ncr2 = len(cr[0])
		cr = np.array(cr).reshape(ncr2,ncr1)
		# remove same points
		for j in range( ncr2 ) : 
			cr0, cr1 = cr[j]
			if ( cr0 < 0 ) : continue
			xtf = (cr[:,0]==cr0)
			ytf = (cr[:,1]==cr1)
			cr[xtf*ytf] = -1
			xtf = yth = 0 #@
			cr[j] = cr0, cr1
		cr = tuple(cr[cr[:,0]>=0].reshape(ncr1,ncr2))
		result = result + [cr]
	cr = 0 #@

	xshape = tuple([len(result)] + list(array.shape))
	x = np.zeros( xshape, int )

	for i in range( len(x) ) : x[i][result[i]] = 1

	for i in range( 1, len(x) ) : x[0] = x[0] * x[i]
	x = x[0]

	result = Value2Index( x, 1 )

	return result


##################################################
##################################################
##################################################


def VisibilityFunction( lgL, lgLuminosityFunction ) : 
	'''
	Give the luminosity lg[L(W Hz^-1)] and luminosity function lg[rho_m(mag^-1 Mpc^-3)], where lg=log10
	Return the visibility function lg[phi(Jy^3/2)]
	This is a fitting function in one paper
	'''
	lgphi = -28.468 + 3/2.*lgL + lgLuminosityFunction
	return lgphi


##################################################
##################################################
##################################################


def NorDiffCount( S, freq, plindex=0.7 ) : 
	'''
	Return the nornalized differential count S^{5/2}n(S).

	Parameters:
		S: flux density in Jy, can be number, array, list
		freq: frequency of the output, in MHz
		plindex: power-law index of flux density of 1400-freq MHz, default plindex=0.7

	This is the fitting function in my foreground paper
	'''

	def DL( z ) : # OK
		if ( type(z) == np.ndarray ) : 
			za = z.flatten()
			dc = CosmologyDistance(za, H0/100., Om, Ol)[:,-1]
			dc = dc.reshape( z.shape )
		else : dc = CosmologyDistance(z, H0/100., Om, Ol)[-1]
		return dc
	
	def Sni( z, S, p ) : 
		# differential count int
		sni = S**1.5 * c * DL(z)**2/(1+z)**2 * phiLz(z,S,p) / ( H0 * (Ol+Om*(1+z)**3)**0.5 )
		return sni
	
	def phiLz( z, S, p) : 
		''' We just need to modify this '''
		x, y, zc, q = p
		L = Lz( z, S )
		phi = g(z,y,zc,q) * f(z,x) * phi0(L/f(z,x))
		return phi
	
	def Lz( z, S ) : # OK
		L = 4*np.pi * (3.08568*10.**16*10.**6)**2 * 10.**(-26) * (1+z)**(1+a) * DL(z)**2 * S
		return L
	
	def phiSF( L ) : 
		''' Note that L, not lgL '''
		C, L0, a, s = np.array([1.30796192e-03, 1.66802874e+21, 1.03982212, 0.603298054])
		rho = C * (L/L0)**(1-a) * np.exp( -0.5*(np.log10(1.+L/L0)/s)**2. )
		return rho
	
	def phiAGN( L ) : 
		'''
		Note that lgL, not L
		-1.5*lgL, because Condon fit visibility, here fit luminosity
		'''
		lgL = np.log10(L)
		Y, B, X, W = np.array([33.88884469, 2.26538581, 25.96056927, 0.84819407])
		lgrho = -1.5*lgL + Y - ( B**2 + ((lgL-X)/W)**2 )**0.5
		return 10.**lgrho
	
	def phi0( L ) : 
		return phiSF(L)+phiAGN(L)
	
	def f( z, x ) : 
		return (1+z)**x
	
	def g( z, y, zc, q ) : 
		return (1+z)**y * np.exp( -(z/zc)**q )
	
	c = 3e+5
	a = 0.8157
	H0 = 67.11
	Ol = 0.6825
	Om = 1 - Ol

	# input S is at freq, but this code is fitted at 1400MHz
	S1400 = S * (freq/1400.)**plindex
	
	z = np.linspace( 0, 10, 3001 )[1:]
	dz = z[1] - z[0]
	z = z[:,None]
	
	p = np.array([6.676, -8.335, 2.397, 2.526])
	
	Snp1400 = Sni(z, S1400, p).sum(0) * dz

	# there is a law: S1*n(S1) = S2*n(S2)
	Snpfreq = (1400./freq)**(1.5*plindex) * Snp1400
	
	return Snpfreq


##################################################
##################################################
##################################################


def arcsinh2lg( value, y=None ) : 
	'''
	Value of matrix is from - to +, so wen can't use lg(a)
	But we can use arcsinh(a)/k, where arcsinh(|a|)/k~lg(|a|)
	This function calculate factor "k"
	'''
	value = npfmt( value )
	if ( y == None ) : 
		if ( abs(value.mean()) <= 2000 ) : y = 13
		elif ( 2000 < abs(value.mean()) <= 6000 ) : y = 12.5
		else : y = 12
	arclog = np.arcsinh( value ) / np.log10(y)
	return [arclog, y]


##################################################
##################################################
##################################################


def HealpixHeader( nside, ordering, coordsys, freq=None, unit=None, beamsize=None ) : 
	'''
	nside: 2**n
	ordering: 'RING', 'NESTED'
	coordsys: 'EQUATORIAL', 'GALACTIC'
	freq: in MHz
	unit: Unit of the healpix map pixel value
	beamsize: Observation FWHM of the healpix map in arcmin
	'''
	key=['PIXTYPE', 'DATECREA', 'ORDERING', 'NSIDE', 'COORDSYS']
	value = ['HEALPIX', Time()[0], ordering.upper(), nside, coordsys.upper()]
	comment = ['HEALPIX pixelisation', 'Creation date of this FITS', 'Pixel ordering scheme, RING or NESTED', 'Healpix resolution parameter', 'Coordinate system, EQUATORIAL or GALACTIC']
	if (freq is not None) : 
		key, value, comment = key+['FREQ'], value+[freq], comment+['MHz']
	if (unit is not None) : 
		key, value, comment = key+['UNIT'], value+[unit], comment+['Unit of the healpix map pixel value']
	if (freq is not None) : 
		key, value, comment = key+['BEAMSIZE'], value+[beamsize], comment+['arcmin']
	return [key, value, comment]


# multi-extension, Primary+Table, Primary+Image, Image+Table, Primary+Image+Table:
# hdulist = pyfits.HDUList()
# hdulist.append(pyfits.PrimaryHDU())
# hdulist.append(pyfits.ImageHDU())
# hdulist.append(pyfits.TableHDU.from_columns())
# hdulist.writeto(outname)

# keyword must <= 8 characters


class Array2Fits( object ) : 
	dtype = 'class:'+sys._getframe().f_code.co_name


	def __init__( self ) : 
		self.hdulist = pyfits.HDUList()


	def Outname( self, outname=None ) : 
		if (outname is not None) : 
			outname = str(outname)
			if (outname[-5:] not in ['.fits', '.FITS']) : 
				outname += '.fits'
			self.outname = outname
		else : 
			outname = self.outname
			if( os.path.exists(outname) ) : 
				os.system('mv '+outname+' '+outname[:-5]+'_old.fits')
			if( os.path.exists(outname[:-5]+'.FITS') ) : 
				os.system('mv '+outname[:-5]+'.FITS '+outname[:-5]+'_old.fits')


	def Header( self, keys, values, comments ) : 
		''' return pyfits.header.Header '''
		if (values is not None) : 
			if (not len(keys)==len(values)) : Raise(Exception, 'len(keys)='+str(len(keys))+', len(values)='+str(len(values)))
			if (comments is None) : comments = ['' for i in xrange(len(keys))]
			elif (len(comments) < len(keys)) : 
				comments += ['' for i in xrange(len(keys)-len(comments))]
		elif (keys is not None) : 
			keys, values, comments = keys.keys(). keys.values(), list(keys.comments)
		else : keys, values, comments = [], [], []
		return [keys, values, comments]
		

	def Bit( self, bit=None, array=None ) : 
		''' bit: 32 or 64 or None '''
		if (bit is None) : bit = 64
		elif (bit not in [16, 32, 64]) : bit = 64
		self.bit = int(round(bit))
		if (array is None) : return
		#--------------------------------------------------
		adtype = npfmt(array).dtype
		if (adtype.name[:7] == 'complex') : 
			bit = 2*self.bit
			if (bit == 32) : bit = 64
			dtype = np.dtype('complex'+str(bit))
		elif (adtype.name[:3] == 'int') : 
			dtype = np.dtype('int'+str(self.bit))
		elif (adtype.name[:5] == 'float') : 
			bit = self.bit
			if (bit == 16) : bit = 32
			dtype = np.dtype('float'+str(bit))
		else : dtype = adtype
		return dtype


	def ImageHDU( self, array, keys=None, values=None, comments=None ) : 
		'''
		(1) keys==None, values==None, comments==None: 
			don't add header
		(2) keys!=None, values!=None, comments!=None: 
			All keys, values, comments must be list, and have the same size
		(3) keys!=None, values!=None, comments==None: 
			Both keys and values must be list, and have the same size
		(4) keys!=None, values==None, comments==None: 
			type(keys)==pyfits.header.Header, not list
		'''
		dtype = self.Bit(self.bit, array)
		array = npfmt(array).astype(dtype)
		imagehdu = pyfits.ImageHDU( array )
		keys, values, comments = self.Header(keys, values, comments)
		hdr = imagehdu.header
		for i in xrange(len(keys)) : 
			hdr[keys[i]] = (values[i], comments[i])
		self.hdulist.append(imagehdu)


	def TableHDU( self, table, keys=None, values=None, comments=None, colname=None ) : 
		'''
		table:
			(1) list or tuple:
				All table[i] must 1D and have the same size
			(2) 1D np.ndarray:
				table = [table]
			(3) 2D np.ndarray:
				table[:,i] is the column of the table

		colname:
			Name of each column, len(colname)==the number of columns

		keys, values, comments:
			See self.ImageHDU()
		'''
		istype = IsType()
		if (istype.isnparray(table)) : 
			if (len(table.shape) > 2) : Raise(Exception, 'table.shape='+str(table.shape)+' is not 1D or 2D')
			dtype = self.Bit(self.bit, table)
			table = table.astype(dtype)
			if (len(table.shape) == 1) : table = [table]
			else : table = table.T
		elif (istype.islist(table) or istype.istuple(table)) : 
			table = list(table)
			colsize = 0
			for i in xrange(len(table)) : 
				dtype = self.Bit(self.bit, table[i])
				table[i] = npfmt(table).flatten().astype(dtype)
				if (table[i].size > colsize): colsize = table[i].size
			for i in xrange(len(table)) : 
				if (table[i].size < colsize) : 
					table[i] = np.concatenate([table[i], [np.nan for j in xrange(colsize-table[i].size)]])
		else : Raise(Exception, 'type(table)='+str(type(table))+' not in [list, tuple, numpy.ndarray]')
		#--------------------------------------------------
		if (colname is None) : colname = 'col'
		if (istype.isstr(colname)) : 
			colname =[colname+'_'+str(i) for i in xrange(len(table))]
		elif (istype.islist(colname) or istype.istuple(colname) or istype.isnparray(colname)) : 
			colname = list(colname)
			if (len(colname) < len(table)) : 
				colname += ['col_'+str(i) for i in xrange(len(colname), len(table))]
		else : Raise(Exception, 'type(colname)='+str(type(colname))+' not in [list, tuple, numpy.ndarray]')
		#--------------------------------------------------
		cols = []
		for i in xrange(len(table)) : 
			# pyfits.Column
			fmt = table[i].dtype.name
			if   (fmt == 'int16')     : fmt = 'I'
			elif (fmt == 'int32')     : fmt = 'J'
			elif (fmt == 'int64')     : fmt = 'K'
			elif (fmt == 'float32')   : fmt = 'E'
			elif (fmt == 'float64')   : fmt = 'D'
			elif (fmt == 'complex32') : fmt = 'C'
			elif (fmt == 'complex64') : fmt = 'M'
			elif (fmt[:6] =='string') : fmt = 'A'+fmt[6:]
			cols.append( pyfits.Column(name=colname[i], format=fmt, array=table[i]) )
		#--------------------------------------------------
		tablehdu = pyfits.BinTableHDU.from_columns(cols)
		keys, values, comments = self.Header(keys, values, comments)
		hdr = tablehdu.header
		for i in xrange(len(keys)) : 
			hdr[keys[i]] = (values[i], comments[i])
		self.hdulist.append(tablehdu)


	def Save( self ) : 
		self.Outname()
		self.hdulist.writeto(self.outname)
		print self.outname+'  -->  saved'


##################################################
##################################################
##################################################


def SignDec( inname ) : 
	'''
	In astronomy catalog, Dec is from +90 to -90, but for Dec = -00 23 15, if we np.loadtxt() it, it will miss the "-", treat -00 as 00, and make Dec=-0deg23arcmin15arcsec to +0deg23arcmin15arcsec.
	This function is used to get the sign of Dec

	Parameter: 
		inname: the path/name of the input file

	Return:
		1D np.array with the same size of the input file
	'''
	instr = open( inname ).readlines()
	sign = np.zeros([len(instr),])
	for i in range(len(instr[0])) : 
		if (instr[0][i]=="+" or instr[0][i]=="-") : break
	for j in range(len(sign)) : 
		if (instr[j][i] == "-") : sign[j] = -1
		else : sign[j] =  1
	return sign


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


def OutDir( outdir=None ) : 
	'''
	outdir:
		None, str, list of str
	'''
	outdirthis = sys.argv[0][:-3]+'_output/'
	if (outdir in [None,False]) : outdir = outdirthis
	if (Type(outdir) == str) : outdir, islist = [outdir], False
	else : islist = True
	for i in range(len(outdir)) : 
		outdir[i] = DirStr(os.path.expanduser(outdir[i]))
		home = os.path.abspath(outdir[i]).split('/')[1]
		if (home not in ['Users','home']): outdir[i]=outdirthis
		try : mkdir(outdir[i])
		except : 
			outdir[i] = outdirthis
			mkdir(outdir[i])
	if (islist == False) : outdir = outdir[0]
	return outdir

#def OutDir( outdir=None, eachdir=False ) : 
#	'''
#	outdir:
#		str, must be one str
#
#	eachdir:
#		str or list of str
#	'''
#	outdirthis = sys.argv[0][:-3]+'_output/'
#	if (outdir in [None,False]) : outdir = outdirthis
#	else : outdir = DirStr(os.path.expanduser(outdir))
#	home = os.path.abspath(outdir).split('/')[1]
#	if (home not in ['Users', 'home']) : outdir = outdirthis
#	print 'outdir=',outdir
#	try : 
#		print '1.1'
#		print 'outdir=', outdir
#		mkdir(outdir)
#		print '1.2'
#	except : 
#		print '2.1'
#		outdir = outdirthis
#		mkdir(outdir)
#		print '2.2'
#	print 'outdir=',outdir
#	# Check eachdir
#	islist = True if(Type(eachdir)==list)else False
#	eachdir = list(eachdir) if(islist)else [eachdir]
#	for i in range(len(eachdir)) : 
#		if (eachdir[i] == True) : eachdir[i] = Time()[-1]+'/'
#		elif (Type(eachdir[i]) == str) : 
#			if (eachdir[i][-1]!='/') : eachdir[i] +='/'
#		#	if (eachdir[i][-1]=='/') : eachdir[i]=eachdir[i][:-1]
#		#	eachdir[i] = eachdir[i].split('/')[-1].split('.')[0].split('_')[0]+'/'
#		else : eachdir[i] = ''
#		eachdir[i] = outdir + eachdir[i]
#		mkdir(eachdir[i])
#	if (islist == False) : eachdir = eachdir[0]
#	return eachdir


##################################################
##################################################
##################################################


#def LS( path, specialword='', option='' ) : 
#	'''
#	Function likes the shell command ls.
#	shell command ls:
#		ls option path/specialword
#		ls -t study/*.pdf
#	Correspond LS():
#		LS( 'study', '*.pdf', '-t' )
#
#	path:
#		can be str: path='study'
#		or list[str]: path=['study', 'music']
#
#	Return:
#		list[str] of file path (note that is path not just filename)
#	'''
#	if (type(path) == str) : path = [path]
#	elif (type(path) != list) : Error('path must be str or list.')
#	for i in range(len(path)) : 
#		if (path[i] == '') : path[i] = path[i] + './'
#		elif (path[i][-1] != '/') : path[i] = path[i] + '/'
#	a = []
#	for i in range(len(path)) : 
#		f = 'ls'+str(i)+'.txt'
#		s = 'ls '+option+' '+path[i]+specialword+' > '+f
#		os.system(s)
#		b = open(f)
#		b = b.readlines()
#		for j in range(len(b)) : 
#			b[j] = b[j][:-1]
#			if (b[j][:len(path[i])] != path[i]) : 
#				b[j] = path[i] + b[j]
#		a = a + [b]
#		os.system('rm '+f)
#	if (len(a) == 1) : a = a[0]
#	return a


##################################################
##################################################
##################################################


def RemoveBeyond( array, percent=0.95, bins=None, reset=False, randn=False ) : 
	'''
	Remove/Reset the data whose value is obviously not good (too larg or too small)

	array:
		must be real array, not complex

	percent:
		How many data will leave/save.

	bins:
		We use ProbabilityDensity() function to do this.

	reset:
		False: just remove/delete these bad data, array will be flatten().
		True: reset these bad data to a reasonable value (set automatically).
		value: reset these bad data to an value given by reset, for example, reset=0.245

	randn:
		True or False
		reset = reset + np.random.randn()*Sigma(array)  # reset is the mean

	Return:
		(1) reset=True, return [new_array, corresponding_index], where new_array.shape==array.shape
		(2) reset=False, return new_array (1D)
	'''
	array = npfmt(array).copy()
	shapea = array.shape
	if (array.min() == array.max()) : return array
	if (array.dtype.name[:3] == 'com') : raise Exception('array must be real, not complex')
	if (bins is None) : bins = binsarcsinh(array)
	xe, xc, N = ProbabilityDensity( array, bins, density=False )
	xcN = np.append([xc], [N], 0)
	xcN = Sort( xcN, ('row', 1), l2s=True )
	xcN0 = xcN[0,0]
	S, Smax = 0, percent*array.size
	for i in range(len(xcN[1])) : 
		if (S > Smax) : break
		S = S + xcN[1,i]
	xb = xcN[0,i:] # note that xb is xc (center value)
	# from xb to xe
	remove = []
	for i in range(len(xb)) : 
		for j in range(len(xe)-1) : 
			if (xe[j] < xb[i] < xe[j+1]) : remove = remove + [j]
	for i in range(len(remove)) : 
		if (reset is False) : 
			array = array[(array<xe[remove[i]])+(array>xe[remove[i]+1])]
		else : 
			if (reset is True) : reset = xcN0
			tf = (array>=xe[remove[i]])*(array<=xe[remove[i]+1])
			if (randn is False) : 
				array[tf] = reset
			else : 
				sigma = Sigma(array)
				array[tf] = np.random.randn(array[tf].size)*sigma + reset
			array = array.reshape(shapea)
	return array


#def RemoveBeyond(array, n=3, remove=True) : 
#	'''
#	Remove/Reset the data whose value beyonds (larger or smaller than) n*rms
#
#	array:
#		must be real array, not complex
#
#	n:
#		value > n*RMS(array) will be removed
#
#	remove:
#		True: flatten array, and remove that values.
#		False: set that values to be 0.123456789
#
#	Return:
#		(1) remove=True, return [new_array, corresponding_index]
#		(2) remove=True, return new_array
#	'''
#	array = npfmt(array)
#	if (array.dtype.name[:3] == 'com') : raise Exception(efn()+'array must be real, not complex')
#	rms = RMS(array)
#	if (remove == False) : 
#		if (array.dtype.name[:3] == 'int') : 
#			array = np.float32(array)
#		array[abs(array)>n*rms] = 0.123456789
#		return array;
#	else : 
#		array = array.flatten()
#		array = array + 1j*np.arange(array.size)
#		array = array[abs(array.real)<=n*rms]
#		return [array.real, (array.imag).astype(int)]
		
	
##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


def Ank( n, k=None ) : 
	'''
	Calculate the permutation and combination.
	n! = spsp.gamma(n+1)
	'''
	if (k is None) : return np.arange(1., n+1).prod() # n!
	if (k > n) : return 0.
	if (k == 0) : return 1.
	return np.arange(n-k+1., n+1).prod()


#def Cnk( n, k ) : 
#	if (n < k) : raise Exception(efn()+'k>n')
#	a0 = np.arange(1., n+1)
#	m1, m2 = k+1, n-k+1
#	if (n < 2*k) : 
#		m1, m2 = n-k+1, k+1
#	a1 = np.arange(1., m1)
#	a2 = np.arange(1., m2)
#	a12 = np.append(a1, a2)
#	a012 = a0/a12
#	return Int(a012)

def Cnk( n, k, Return=False ) : 
	'''
	Here, C_n^k, when k>n, return 0
	Return:
		Must n.size=1 and k.size=1
		False, don,t return the result array
		True, return array
	'''
	n00, k00 = n, k
	n, k = 1.*npfmt(n), 1.*npfmt(k)
	if (n.min()<0 or k.min()<0) : raise Exception(efn()+'n and k must >0')
	n[n==0] = 1e-6
	k[k==0] = -1
	ns, ks = np.array(n.shape), (k.shape)
	nrow = 1
	if (abs(ns-ks).max() == 0) : shape=(n.size, k.size)
	else : 
		shape = (n*k).shape
		nks = ns - ks
		nks = nks[nks!=0][0]
		if (nks < 0) : nrow = 0
	kmax, nmax = abs(k.max()), n.max()
	nsize, ksize = n.size, k.size
	# 3D
	n = n.reshape(1, n.size, 1)
	k = k.reshape(1, 1, k.size)
	minusone = np.arange(kmax).reshape(kmax,1,1)
	n = n - minusone
	k = k - minusone
	k[k<=0] = -10.**6
	n[n<0] = 0
	nk = n / k
	nktf = (nk[0].reshape(1,nsize,ksize)) * np.ones([kmax,1,1])
	nktf1 = nk.copy()
	nktf1[0] = 2
	nk[nktf1<=0] = 1
	nk[(0<nktf)*(nktf<1)] = 0
	cnk = nk[0]*0 + 1
	for i in range(int(kmax)) : 
		cnk = cnk * nk[i]
	cnk[cnk<0] = 1
	if (nrow == 0) : cnk = cnk.T
	cnk = cnk.reshape(shape)
	if (cnk.size == 1) : 
		cnk = (cnk.flatten())[0]
		if (Return) : 
			n = npfmt(n00).flatten()[0]
			k = npfmt(k00).flatten()[0]
			a = np.zeros([int(cnk+0.1), k], int)
			j = -1 # point column
			i = 0
			while (i < len(a)) : 
				if (i == 0) : 
					a[i] = np.arange(k)
					i = i + 1
				else : 
					if (a[i-1,j] < n+j) : 
						a[i] = a[i-1]
						a[i,j] = a[i-1,j] + 1
						if (a[i,j] != a[i-1,j]) : 
							a[i,j:] = np.arange(a[i,j], a[i,j]-j)
							j = -1
						i = i + 1
					else : j = j - 1
			return [cnk, a]
	return cnk
	

##################################################
##################################################
##################################################


def Stepfunc( a, a0=0, zeroone=True, include=True ) : 
	'''
	Step function.

	a:
		one, list, np.array(same shape)

	a0:
		must be one value

	zeroone:
		=True: a<a0, return 0, a>a0, return 1
		=False: a<a0, return 1, a>a0, return 0

	include:
		=True, when a=a0, return 1
		=False, when a=a0, return 0
	'''
	if (npfmt(a0).size != 1) : raise Exception(efn()+'a0 must be one value')
	a = npfmt(a)
	a = a - a0
	if (include) : a[a==0] = 1
	else : a[a==0] = 0
	if (zeroone) : b = 1
	else : b = -1
	a = b * a
	a[a>0] = 1
	a[a<0] = 0
	if (a.size == 1) : a = (a.flatten())[0]
	return a


def Deltafunc( a, a0, error=1e-6 ) : 
	'''
	For perfect Delta function, when a==a0, get 1, otherwise 0.

	But here, because of the computer, when -error<=(a-a0)/a0<=error, we also assume to get 1. Note that is (a-a0)/a0, not (a-a0)
	'''
	a, a1 = npfmt(a), npfmt(a0)
	if (a1.size != 1) : raise Exception(efn()+'a0 must be one value')
	a1 = a0*1.
	if (a0 == 0) : a1 = 1e-10
	a = abs((a - a0)/a1)
	a = (a<=error) + 0
	if (a.size == 1) : a = (a.flatten())[0]
	return a


##################################################
##################################################
##################################################


def CCompile( codepath, command='', outdir='', sophyapath='' ) : 
	'''
	Compile and link the c/c++ code with SOPHYA.
	The command to compile and link will be different due to different computer system.

	codepath:
		Absolute path of c/c++ code

	outdir:
		Which directory to put the output a.out

	compilecommand:
		str
		"MyMac": use MyMac command
		"bao@bao": use the bao@bao command
		other str: compile c/c++ base on the local machine (always done)

	sophyapath:
		When install the SOPHYA, we will set the export SOPHYABASE and other environments in .bash_profile.
		If sophyapath='', this function will read the .bash_profile or .bashrc to ge the SOPHYABASE.
		Or you can set it by hand here.
	'''
	# compile (CXXCOMPILE)
	# CXXCOMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS) -c 
	# CXX = c++
	# CPPFLAGS = -DDarwin -I$(SOPHYAINCP)  $(PIINC)
	# SOPHYAINCP = $(SOPHYABASE)/include/
	# PIINC = $(PIEXTINC) -I/usr/X11R6/include/
	# PIEXTINC = -I/opt/local/include
	# CXXFLAGS = -fno-common -g -O -fPIC 
	# SOPHYABASE

	# link (CXXLINK + SOPHYAEXTSLBLIST)
	# CXXLINK = $(CXX) $(CXXFLAGS) -bind_at_load
	# SOPHYAEXTSLBLIST = -L$(SOPHYASLBP) -lextsophya -lsophya $(SOPEXTLIBS) $(SOPBASELIBS)
	# SOPHYASLBP = $(SOPHYABASE)/slb/
	# SOPEXTLIBS = $(SOPEXTLIBP) $(SOPEXTLIBLIST) -framework Accelerate
	# SOPEXTLIBP = -L/usr/local/Sophya/ExtLibs/lib
	# SOPEXTLIBLIST = -lcfitsio -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -lxastro
	# SOPBASELIBS = -lpthread -lm -lc -ldl 

	# Check codepath
	if (os.path.exists(codepath) is not True) : 
		print 'Error:  '+codepath+'  is not found'
		exit()

	if (compilecommand is None) : compilecommand = ''
	if (linkcommand is None) : linkcommand = ''
	if (sopyhapath is None) : sophyapath = ''
	if (outdir != '') : 
		if (outdir[-1] != '/') : outdir = outdir + '/'
		mkdir(outdir)


	''' For my mac '''
	mycommand1 = 'c++ -g -DDarwin -I/usr/local/Sophya/include/ -I/opt/local/include -I/usr/X11R6/include/ -fno-common -O -fPIC -c -I./inlib -c '
	mycommand2 = 'c++ -fno-common -g -O -fPIC -bind_at_load -L/usr/local/Sophya/slb/ -lPI -lextsophya -lsophya -L/opt/local/lib -lXm -ledit -lcurses -L/usr/X11R6/lib/ -lXt -lX11 -L/usr/local/Sophya/ExtLibs/lib -lcfitsio -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -lxastro -framework Accelerate -lpthread -lm -lc -ldl '
	
	
	''' For bao@bao3 '''
	CXXCOMPLIE = 'g++ -DLinux -I/ope/local/include -I/Dev/Sophya64/include -I/Dev/ExtLibs/include -I/usr/X11R6/include -Wall -Wpointer-arith -fno-common -O -g -fPIC -c '
	CXXLINK = 'g++ -Wall -Wpointer-arith -O -g -fPIC '
	SOPHYAEXTSLBLIST = '-L/Dev/Sophya64/slb -lextsophya -lsophya -lPI -L/Dev/ExtLibs/lib -lcfitsio -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -llapack -lblas -lxastro -lgfortran -lstdc++ -lpthread -lm -lc -ldl '
	baocommand1 = CXXCOMPLIE
	baocommand2 = CXXLINK + SOPHYAEXTSLBLIST

	if (command.lower() == 'mymac') : 
		compilecommand = mycommand1
		linkcommand = mycommand2
	elif (command.lower() == 'bao@bao') : 
		compilecommand = baocommand1
		linkcommand = baocommand2

	else : 
		if (sophyapath == '') : 
			SOPHYABASE = os.popen('echo ${SOPHYABASE}').read()[:-1]
		else : SOPHYABASE = sophyapath
		if (SOPHYABASE[-1] == '/') : SOPHYABASE[:-1]

		makeinc = open(SOPHYABASE+'/include/sophyamake.inc').readlines()
		compilelink = []
		clname = ['CXXCOMPILE', 'CXXLINK', 'SOPHYAEXTSLBLIST', 'CXX', 'CPPFLAGS', 'SOPHYAINCP', 'PIINC', 'PIEXTINC', 'CXXFLAGS', 'SOPHYASLBP', 'SOPEXTLIBS', 'SOPEXTLIBP', 'SOPEXTLIBLIST', 'SOPBASELIBS'] 
		for i in range(len(clname)) : 
			a = clname[i]+' = '
			for j in range(len(makeinc)) : 
				b, n = makeinc[j][:len(a)], 0
				if (b == a) : 
					compilelink, n = compilelink + [makeinc[j][len(a):-1]], 1
					break
			if (n == 0) : 
				print 'Can not find "'+clname[i]+'" in the '+SOPHYABASE+'/include/sophyamake.inc'
				print 'Fail to compile'
				exit()
	
		# devide
		for i in range(len(compilelink)) : 
			a, b, c, n = compilelink[i], [], [], [0] 
			for j in range(len(a)) : 
				if (a[j] != ' ') : break
			for k in range(len(a)-1, 0, -1) : 
				if (a[k] != ' ') : break
			a = a[j:k+1]
			for j in range(len(a)-1) : 
				if (a[j]==' ' and a[j+1]!=' ') : n = n + [j+1]
			n = n + [len(a)]
			for j in range(len(n)-1) : 
				b = b + [a[n[j]:n[j+1]]]
			for j in range(len(b)) : 
				for m in range(len(b[j])) : 
					if (b[j][m] == '$') : 
						for k in range(m, len(b[j])) : 
							if (b[j][k] == ')') : break
						d = 0
						if (k == len(b[j])-1) : pass
						else : 
							f = b[j][k+1:]
							for p in range(len(f)) : 
									if (f[p] != ' ') : d = 1
						if (m == 0) : 
							if (d == 0) : b[j] = [b[j][2:k]]
							else : b[j] = [b[j][2:k], b[j][k+1:]]
						else : 
							if (d == 0) : b[j] = [b[j][:m], b[j][m+2:k]]
							else : b[j] = [b[j][:m], b[j][m+2:k], b[j][k+1:]]
						break
				if (type(b[j]) != list) : b[j] = [b[j]]
			for j in range(len(b)) : c = c + b[j]
			compilelink[i] = c
				
		# instead
		gowhile = True
		while (gowhile) : 
			gowhile = False
			for i in range(len(compilelink)) : 
				a, b = compilelink[i], []
				for j in range(len(a)) : 
					n = 0
					if (a[j] == 'SOPHYABASE') : a[j] = SOPHYABASE
					for k in range(len(clname)) : 
						if (a[j] == clname[k]) : 
							gowhile, n, a[j] = True, 1, compilelink[k]
							break
					if (n == 0) : a[j] = [a[j]]
				for j in range(len(a)) : b = b + a[j]
				compilelink[i] = b


		compilecommand = ''
		a = compilelink[0]
		for i in range(len(a)) : 
			if (a[i][0] == '/') : b = a[i]
			else : b = ' '+a[i]
			compilecommand = compilecommand + b
	
		linkcommand = ''
		a = compilelink[1] + compilelink[2]
		for i in range(len(a)) : 
			if (a[i][0] == '/') : b = a[i]
			else : b = ' '+a[i]
			linkcommand = linkcommand + b

	
	os.system(compilecommand+'-o a.o '+codepath)
	os.system(linkcommand+'-o '+outdir+codepath[:-3]+'.out a.o')
	os.system('rm a.o')


##################################################
##################################################
##################################################


def SciNot( value ) :
	'''
	scientific notation.

	value can be scale, list, n-D array
	
	Return [a, n], value = a * 10**n
	'''
	value = npfmt(value)
	if (value.dtype.name[:3] == 'int') : value = value*1.
	shape = value.shape
	value = value.flatten()
	a = value*0
	n = value.astype(int)*0
	for j in range(value.size) : 
		v = ('%e' % value[j])
		for i in range(len(v)-1, 0, -1) : 
			if (v[i] == 'e') : break
		a[j] = float(v[:i])
		n[j] = int(v[i+1:])
	a, n = a.reshape(shape), n.reshape(shape)
	return [a, n]


##################################################
##################################################
##################################################


def printa( a, precision=6, suppress=True ) : 
	'''
	Format the printing of np.array
	suppress:
		=False, print 1.23e+4
		=True,  print 12340.
	'''
	np.set_printoptions(precision=precision, suppress=suppress)
	print a


##################################################
##################################################
##################################################


def PAON4LonLat( strlonlat=False ) : 
	'''
	Longitude and latitude of PAON4 (degree)
	strlonlat: return [strlon, strlat] or not?
	'''
	lon = (2+11/60.+58.7/3600.)#*np.pi/180
#	lat = (47+23/60.+2/3600.)#*np.pi/180
	lat = (47+22/60.+55.1/3600.)#*np.pi/180
	strlon = '2d11m43s'
	strlat = '+47d23m2s'
	if (strlonlat == False) : return np.array([lon, lat])
	else : return [strlon, strlat]

#--------------------------------------------------

def PAON4Dec( dayname, Dec=None, lonlat=PAON4LonLat() ) : 
	'''
	Use dayname to get the Dec of the antenna pointting

	dayname:
		str or list of str

	return:
		Dec or [Dec] in degree, the same shape as dayname
	'''
	#--------------------------------------------------
	# PAON4 target source
	Decc = {'CygA':'665S', 'CasA':'1142N'}
	#--------------------------------------------------
	if (Dec is not None) : return Dec
	#--------------------------------------------------
	# PAON4 array location
	lon, lat = lonlat  # degree
	if (Type(dayname) == str) : 
		dayname, islist = [dayname], False
	else : islist = True
	#--------------------------------------------------
	Dec = []
	for i in range(len(dayname)) : 
		# Find the first number
		nang, ang = StrFind(dayname[i], 'NumType')
		nang, ang = nang[0], ang[0]
		# N or S relate to lat
		NoS = dayname[i][nang[1]]
		sourcename = dayname[i][:nang[0]]
		sourcename = CalibrationSource().RegisterSource(sourcename)
		if ((sourcename is not None)and(ang+NoS in Decc.values())):
			RAs, Decs = CalibrationSource().RADec(sourcename)
		else : 
			sign = -1 if(NoS=='S')else 1
			ang = sign * float(ang) /100
			Decs = lat + ang
		Dec.append(round(Decs,4))
	if (islist == False) : Dec = np.array(Dec[0])
	else : Dec = npfmt(Dec)
	return Dec

#--------------------------------------------------

def PAON4Dayname( name ) : 
	if (Type(name) == str) : 
		name, islist = [name], False
	else : islist = True
	dayname = []
	for i in range(len(name)) : 
		dayname.append(name[i].split('/')[-1].split('.')[0].split('_')[0])
	if (islist == False) : dayname = dayname[0]
	return dayname

#--------------------------------------------------

def PAON4DaynameCheck( daynamelist, datadir='', datatail='', n_=0 ) : 
	'''
	Check whether there are the dayname data.

	daynamelist:
		daynamelist = ['CasA1142N10mar15', 'CygA24hmai15']
		if (datadir=='' or datadir is None) : datadir = './'
		if (datatail=='' or datatail is None) : datadir+dayname is the full name of the file

	datadir:
		Which directory to save the data
		datadir = '../../paon4/paon4_data/'

	datatail:
		Nametail except the dayname
		datatail = '_all-1_ac_0-4095-8.fits'

	n_:
		This parameter works when datatail==''/None
		Select which "_" split is dayname

	Return:
		return the valid dayname and absolute path
	'''
	if (datadir is None or datadir=='') : datadir = ''
	elif (datadir[-1] != '/') : datadir = datadir + '/'
	if (datatail == None) : datatail = ''
	if (type(daynamelist) != list) : daynamelist = [daynamelist]
	datapath, dayname = [], []
	for i in range(len(daynamelist)) : 
		dni = daynamelist[i]
		fni = datadir + dni +datatail
		if (fni == dni) : 
			dni = dni.split('/')[-1]
			dni = dni.split('.')[0]
			dni = dni.split('_')[n_]
		fni = os.path.expanduser(fni)
		# check
		if (os.path.exists(fni)) : 
			datapath = datapath + [fni]
			dayname = dayname + [dni]
	return [dayname, datapath]

#--------------------------------------------------

def PAON4Chan2AntHV( which=None ) : 
	'''
	Return the chan2antHV used by PAON4
	'''
	chan2antHV1 = ['1H','2H','3H','4H','1V','2V','3V','4V']
	chan2antHV2 = ['3H','4H','1H','2H','1V','2V','3V','4V']
	chan2antHVJSkyMap = ['1H','4H','3H','2H','1V','4V','3V','2V']
	c2a = [chan2antHV1, chan2antHV2, chan2antHVJSkyMap]
	if (which is None) : return c2a
	elif (which != 'JSkyMap') : return c2a[which]
	else : return chan2antHVJSkyMap

#--------------------------------------------------

class PAON4Pair( object ) : 
	dtype = 'class:'+sys._getframe().f_code.co_name

	def __init__( self, chan2antHV=None ) : 
		if (chan2antHV is None) : 
			chan2antHV = ['1H', '2H', '3H', '4H', '1V', '2V', '3V', '4V']
		self.chan2antHV = chan2antHV
		self._All()


	def ShowFunc( self ) : 
		classdict = self.__class__.__dict__
		classname = classdict.keys()
		n = 0
		for i in range(len(classdict)) : 
			if (classname[i-n][0] == '_') : 
				classname.pop(i-n)
				n +=1
			elif (type(classdict[classname[i-n]]) != FunctionType) : 
				classname.pop(i-n)
				n +=1
			else : classname[i-n] = classname[i-n]+'()'
		print self.dtype+'.ShowFunc():'
		for i in range(len(classname)) : print '   '+classname[i]
		return classname


	def _All( self ) : 
		chan2antHV = self.chan2antHV
		strall, strauto, strcross = [], [], []
		for i in range(len(chan2antHV)) : 
			for j in range(i, len(chan2antHV)) : 
				# auto
				if (chan2antHV[i] == chan2antHV[j]) : 
					strauto.append( chan2antHV[i]+'-'+chan2antHV[j] )
					strall.append( strauto[-1] )
				# cross
				elif (chan2antHV[i][-1] == chan2antHV[j][-1]) : 
					strcross.append( chan2antHV[i]+'-'+chan2antHV[j] )
					strall.append( strcross[-1] )
				else : 
					strall.append( chan2antHV[i]+'-'+chan2antHV[j] )
		self.strall, self.strauto, self.strcross = strall, strauto, strcross
		self.ncross = []
		for i in range(len(strcross)) : 
			key = strcross[i].split('-')
			self.ncross.append((chan2antHV.index(key[0]), chan2antHV.index(key[1])))


	def _check( self, key ) : 
		keys = key.split('-')
		if (len(keys) == 1) : 
			key = keys[0] + '-' + keys[0]
			which = 'auto'
		elif (len(keys) != 2) : which = 'other'
		elif (keys[0] == keys[1]) : which = 'auto'
		else : 
			if (key in self.strcross) : which = 'cross'
			else : which = 'other'
		#------------------------------------
		if (which == 'other') : 
			if (key not in self.strall) : raise Exception('key="'+key+'" not in strall='+str(self.strall))
		if (which == 'auto') : 
			if (key not in self.strauto) : raise Exception('key="'+key+'" not in strauto='+str(self.strauto))
		if (which == 'cross') : 
			if (key not in self.strcross) : raise Exception('key="'+key+'" not in strcross='+str(self.strcross))
		return [key, which]


	def All( self, key=None ) : 
		if (key in [None, '']) : return self.strall
		if (Type(key) == 'NumType') : return self.strall[key]
		key, which = self._check(key)
		return self.strall.index(key)


	def Auto( self, key=None ) : 
		'''
		key:
			==None: return strauto
			=='3H', '3H-3H' and so on: return the index of this auto in chan2antHV/strauto
			==int: return strauto[key]
		'''
		if (key in [None, '']) : return self.strauto
		if (Type(key) == 'NumType') : return self.strauto[key]
		key, which = self._check(key)
		if (which != 'auto') : raise Exception('key="'+key+'" not in strauto='+str(self.strauto))
		return self.strauto.index(key)


	def Cross( self, key=None ) : 
		'''
		key:
			==None: return strcross
			=='3H-2H': return the index of this cross in strcross
			==int: return strcross[key]
		'''
		if (key in [None, '']): return [self.strcross, self.ncross]
		if (Type(key) == 'NumType') : return self.strcross[key]
		key, which = self._check(key)
		if (which != 'cross') : raise Exception('key="'+key+'" not in strcross='+str(self.strcross))
		return self.strcross.index(key)


	def Cross2Auto( self, key ) : 
		'''
		Input key='3H-4H', return (0,1)
		Input key=(0,1)  , return '3H-4H'
		'''
		if (Type(key[0]) == 'NumType') : 
			try : return self.chan2antHV[key[0]]+'-'+self.chan2antHV[key[1]]
			except : raise Exception('key='+str(key)+' out of len(chan2antHV)='+str(len(self.auto)))
		key, which = self._check(key)
		if (which != 'cross') : raise Exception('key="'+key+'" not in strcross='+str(strcross))
		return self.ncross[self.strcross.index(key)]


	def Cross2Triangle( self, key ) : 
		'''
		Triangle relation of channel pair
		For example, 3H4H = 3H1H - 4H1H = 3H2H - 4H2H
		key:
			=='3H4H': return [('+','3H-1H','-','4H-1H'), ('+','3H-2H','-','4H-2H')]
			==(0,1): return [('+',(0,2),'-',(1,2)), ('+',(0,3),'-',(1,3))]
		'''
		typekey = Type(key[0])
		if (typekey == 'NumType') : 
			try : key = self.chan2antHV[key[0]]+'-'+self.chan2antHV[key[1]]
			except : raise Exception('key='+str(key)+' out of len(chan2antHV)='+str(len(self.auto)))
		key, which = self._check(key)
		if (which != 'cross') : raise Exception('key="'+key+'" not in strcross='+str(strcross))
		key0, kl1, kl2 = key, [], []
		key = key.split('-')
		for i in range(len(self.strcross)) : 
			if (key0 == self.strcross[i]) : continue
			sc = self.strcross[i].split('-')
			if (key[0] in sc) : kl1.append(sc)
			if (key[1] in sc) : kl2.append(sc)
		triangle = []
		for i in range(len(kl1)) : 
			for j in range(len(kl2)) : 
				if ((kl1[i][0] in kl2[j]) or (kl1[i][1] in kl2[j])):
					triangle.append((kl1[i], kl2[j]))
		for i in range(len(triangle)) : 
			kl1, kl2 = triangle[i]
			k1, k2 = kl1[0]+'-'+kl1[1], kl2[0]+'-'+kl2[1]
			#--------------------------
			if   (kl1[0] == kl2[0]) : 
				if (kl1[1]+'-'+kl2[1] == key0) : triangle[i] = ('+', k2, '-', k1)
				else : triangle[i] = ('+', k1, '-', 'k2')
			#--------------------------
			elif (kl1[0] == kl2[1]) : 
				if (kl1[1]+'-'+kl2[0] == key0) : triangle[i] = ('-', k1, '-', k2)
				else : triangle[i] = ('+', k1, '+', k2)
			#--------------------------
			elif (kl1[1] == kl2[0]) : 
				if (kl1[0]+'-'+kl2[1] == key0) : triangle[i] = ('+', k1, '+', k2)
				else : triangle[i] = ('-', k1, '-', k2)
			#--------------------------
			elif (kl1[1] == kl2[1]) : 
				if (kl1[0]+'-'+kl2[0] == key0) : triangle[i] = ('+', k1, '-', k2)
				else : triangle[i] = ('+', k2, '-', k1)
		#--------------------------
		if (typekey == 'NumType') : 
			for i in range(len(triangle)) : 
				triangle[i] = (triangle[i][0], self.Cross2Auto(triangle[i][1]), triangle[i][2], self.Cross2Auto(triangle[i][3]))
		return triangle


	def ReorderChan2antHV( self, chan2antHVnew, which=None ) : 
		'''
		This function reorders from self.chan2antHV to chan2antHVnew
		autoarray[nautonew] => order in chan2antHVnew
		crossarray[ncrossnew] => order in chan2antHVnew

		which:
			None, 'auto', 'cross'

		return: 
		if (which is None) : return [nautonew, ncrossnew, sign]
		elif (which.lower() == 'auto') : return nautonew
		elif (which.lower() == 'cross') : return [ncrossnew, sign]
		'''
		newpair = PAON4Pair(chan2antHVnew)
		nautonew = np.zeros([len(self.chan2antHV),], int)
		ncrossnew = np.zeros([len(self.strcross),], int)
		sign = ncrossnew*0
		#--------------------------------------------------
		for i in range(len(newpair.chan2antHV)) : 
			nautonew[i] = self.chan2antHV.index( newpair.chan2antHV[i] )
		#--------------------------------------------------
		for i in range(len(newpair.strcross)) : 
			sc = newpair.strcross[i].split('-')
			sc1, sc2 = sc[0]+'-'+sc[1], sc[1]+'-'+sc[0]
			sign1, sign2 = 1, -1
			if (sc1 in self.strcross) : 
				ncrossnew[i] = self.strcross.index( sc1 )
				sign[i] = sign1
			elif (sc2 in self.strcross) : 
				ncrossnew[i] = self.strcross.index( sc2 )
				sign[i] = sign2
			else : raise Exception(sc1+' or '+sc2+' not in self.strcross='+str(self.strcross))
		if (which is None) : return [nautonew, ncrossnew, sign]
		elif (which.lower() == 'auto') : return nautonew
		elif (which.lower() == 'cross') : return [ncrossnew, sign]
		else : raise Exception('which='+str(which)+" not in [None,'auto','cross']")


	def StrcrossJSkyMap( self ) : 
		strcrossJSkyMap = ['1H-4H','1H-3H','2H-1H','3H-4H','2H-4H','2H-3H','1V-4V','1V-3V','2V-1V','3V-4V','2V-4V','2V-3V']
		return strcrossJSkyMap


	def ReorderJSkyMap( self, which=None, chan2antHVJSkyMap=PAON4Chan2AntHV('JSkyMap') ) : 
		'''
		In Jiao's program, PAON4 configuration is
			vector< pair<double,double> > paon4(double D){
				vector< pair<double, double> > con;
				con.push_back( pair<double,double>(0.,0.) );
				con.push_back( pair<double,double>(4.39,6.) );
				con.push_back( pair<double,double>(4.39,-6.) );
				con.push_back( pair<double,double>(-6.,0.) );
		So, the labels of the antennas are
					         2
					4   1
					         3
		The final baselines are:
			baselines=[abs(1-2), abs(1-2), abs(1-4), abs(2-3), abs(2-4), abs(3-4)]
			(1-2) meas that feedpos[0]-feedpos[1]
			abs(1-2) means that select deltaX>0:
				(1-2)=(0-4.39, 0-6)=(-4.39, -6)
				abs(1-2)=(4.39, 6)
			(3-4)=(4.39-6, -6-0)=(10.39, -6)=abs(3-4), because deltaX>0

		Therefore, strcross of JSkyMap is:
		strcrossJSkyMap = ['1H-4H','1H-3H','2H-1H','3H-4H','2H-4H','2H-3H','1V-4V','1V-3V','2V-1V','3V-4V','2V-4V','2V-3V']

		However, the baselines of the real data are determined by the order of the cables connecting to the correlation machine.
		In Jiao's case, we can assume as:
	chan2antHVJSkyMap = ['1H','4H','3H','2H','1V','4V','3V','2V']

		autoarray[nautonew] => order in JSkyMap
		crossarray[ncrossnew] => order in JSkyMap
		'''
		strcrossJSkyMap = self.StrcrossJSkyMap()
		nautonew, ncrossnew, sign = self.ReorderChan2antHV(chan2antHVJSkyMap)
		strcross = PAON4Pair(chan2antHVJSkyMap).strcross
		for i in range(len(strcross)) : 
			if (strcross[i] != strcrossJSkyMap[i]) : sign[i] *= -1
		if (which is None) : return [nautonew, ncrossnew, sign]
		elif (which.lower() == 'auto') : return nautonew
		elif (which.lower() == 'cross') : return [ncrossnew, sign]
		else : raise Exception('which='+str(which)+" not in [None,'auto','cross']")




#	def AutoJSkyMap( self, strauto=None ) : 
#		'''See CrossJSkyMap() below for detail'''
#	#	chan2antHVJ = ['1H','2H','3H','4H','1V','2V','3V','4V']
#	#	baselinesJ = [(0,0),(4.39,6),(4.39,-6),(-6,0),(0,0),(4.39,6),(4.39,-6),(-6,0)]
#		chan2antHVJ = ['1H','4H','3H','2H','1V','4V','3V','2V']
#		#--------------------------------------------------
#		if (strauto is not None) : 
#			if (Type(strauto) == str) : 
#				strauto, islist = [strauto], False
#			else : islist = True
#			strnum = ['0','1','2','3','4','5','6','7','8','9']
#			for i in range(len(strauto)) : 
#				for j in range(len(strauto[i])) : 
#					if (strauto[i][j] not in strnum) : break
#				sc = strauto[i][:j+1]
#				# Check sc
#				if (sc not in self.chan2antHV) : raise Exception("'"+strauto[j]+"' not in self.chan2antHV="+str(self.chan2antHV))
#				strauto[i] = sc
#		else : strauto, islist = self.chan2antHV, True
#		#--------------------------------------------------
#		restrauto, renauto = [], []
#		# renauto: convert chan2antHV to chan2antHVJ
#		for i in range(len(strauto)) : 
#			n = chan2antHVJ.index(strauto[i])
#			restrauto.append(chan2antHVJ[n])
#			renauto.append(n)
#		renauto = npfmt(renauto) 
#		if (islist == False) : 
#			restrauto, renauto = restrauto[0], renauto[0], renauto4toJ[0]
#		return [restrauto, renauto]
#
#
#	def AutoReorderToJSkyMap( self, strauto=None ) : 
#		'''
#		return:
#			nconvert
#			autoarray[nconvert] => order in JSkyMap
#		'''
#		nauto = self.AutoJSkyMap(strauto)[1]
#		if (Type(nauto) == 'NumType') : return nauto
#		else : 
#			nauto = nauto+1j*np.arange(len(nauto))
#			nauto = Sort(nauto).imag.round().astype(int)
#			return nauto
#
#
#	def CrossJSkyMap( self, strcross=None ) : 
#		'''
#		In Jiao's program, PAON4 configuration is
#			vector< pair<double,double> > paon4(double D){
#				vector< pair<double, double> > con;
#				con.push_back( pair<double,double>(0.,0.) );
#				con.push_back( pair<double,double>(4.39,6.) );
#				con.push_back( pair<double,double>(4.39,-6.) );
#				con.push_back( pair<double,double>(-6.,0.) );
#		So, the labels of the antennas are
#					         2
#					4   1
#					         3
#		The final baselines are:
#			baselines=[abs(1-2), abs(1-2), abs(1-4), abs(2-3), abs(2-4), abs(3-4)]
#			(1-2) meas that feedpos[0]-feedpos[1]
#			abs(1-2) means that select deltaX>0:
#				(1-2)=(0-4.39, 0-6)=(-4.39, -6)
#				abs(1-2)=(4.39, 6)
#			(3-4)=(4.39-6, -6-0)=(10.39, -6)=abs(3-4), because deltaX>0
#
#		However, the baselines of the real data are determined by the order of the cables connecting to the correlation machine.
#		This function is used to convert the real data baselines to Jiao's program case.
#		'''
#		#--------------------------------------------------
#	#	strcrossJ = ['1H-2H','1H-3H','4H-1H','3H-2H','4H-2H','4H-3H','1V-2V','1V-3V','4V-1V','3V-2V','4V-2V','4V-3V']
#		#--------------------------------------------------
#		if (strcross is not None) : 
#			if (Type(strcross) == str) : 
#				strcross, islist = [strcross], False
#			else : islist = True
#			strnum = ['0','1','2','3','4','5','6','7','8','9']
#			for i in range(len(strcross)) : 
#				for j in range(len(strcross[i])) : 
#					if (strcross[i][j] not in strnum) : break
#				sc = strcross[i][:j+1]+'-'+strcross[i][j+1:]
#				# Check sc
#				if (sc not in self.strcross) : raise Exception(strcross[j]+' not in self.strcross='+str(self.strcross))
#				strcross[i] = sc
#		else : strcross, islist = self.strcross, True
#		#--------------------------------------------------
#		restrcrossJ, rencrossJ, resignJ = [], [], []
#		for i in range(len(strcross)) : 
#			sc = strcross[i].split('-')
#			sc = self.AutoJSkyMap(sc)[0]
#			sc1, sc2 = sc[0]+'-'+sc[1], sc[1]+'-'+sc[0]
#			sign1, sign2 = 1, -1
#			if (sc1 in strcrossJ) : 
#				restrcrossJ.append(sc1)
#				rencrossJ.append(strcrossJ.index(sc1))
#				resignJ.append(sign1)
#			elif (sc2 in strcrossJ) : 
#				restrcrossJ.append(sc2)
#				rencrossJ.append(strcrossJ.index(sc2))
#				resignJ.append(sign2)
#			else : raise Exception(sc1+' or '+sc2+' not in strcrossJ='+str(strcrossJ))
#		rencrossJ, resignJ = npfmt(rencrossJ), npfmt(resignJ)
#		if (islist == False) : 
#			restrcrossJ, rencrossJ, resignJ = restrcrossJ[0], rencrossJ[0], resignJ[0]
#		return [restrcrossJ, rencrossJ, resignJ]
#		
#
#	def CrossReorderToJSkyMap( self, strcross=None ) : 
#		'''
#		return:
#			nconvert
#			crossarray[nconvert] => order in JSkyMap
#		'''
#		ncross, sign = self.CrossJSkyMap(strcross)[1:]
#		if (Type(ncross) == 'NumType') : return [ncross, sign]
#		else : 
#			ncross = ncross+1j*np.arange(len(ncross))
#			ncross = Sort(ncross).imag.round().astype(int)
#			sign = sign+1j*np.arange(len(sign))
#			sign = Sort(sign).imag.round().astype(int)
#			return [ncross, sign]





#def PAON4Pair( key=None, chan2antHV=None, cross34=True ) : 
#	'''
#	key:
#		(1) key = 'all'  , Return [nall  , strall]
#		(2) key = 'auto' , Return [nauto , strauto]
#		(3) key = 'cross', Return [ncross, strcross]
#		(4) key = int(visibility index), Return [str], If str is effective cross, return [strcross, (auto-pairs)]
#		(5) key = str, Return [n], If str is effective cross, return [ncross, (auto-pairs)]
#		(6) key = None, Return [[nauto, strauto], [ncross, strcross, (auto-pairs)]]
#		ncross: cross in nall
#		strcross: tuple(,) of auto pair of this cross
#
#	chan2antHV:
#		channel 1, 2, 3, ... correspond to which antenna-polarization?
#		default: chan2antHV = ['1H', '2H', '3H', '4H', '1V', '2V', '3V', '4V'], means:
#		chan1->1H, chan2->2H, chan3->3H, chan4->4H
#		chan5->1V, chan6->2V, chan7->3V, chan8->4V
#
#	cross34:
#		It will valid when key=='cross', cross34 just for old data
#		cross34 == True, return 3H-4H and 3V-4V together
#		cross34 == False, don't return
#	'''
#	if (chan2antHV is None) : 
#		chan2antHV = ['1H', '2H', '3H', '4H', '1V', '2V', '3V', '4V']
#	strall, nall = [], []
#	strauto, strauto2, nauto = [], [], []
#	strcross, strcross2, ncross, nstrcross = [], [], [], []
#	n = -1
#	for i in range(len(chan2antHV)) : 
#		for j in range(i, len(chan2antHV)) : 
#			n = n + 1
#			# all
#			strall = strall + [chan2antHV[i]+'-'+chan2antHV[j]]
#			nall = nall + [n]
#			# auto
#			if (chan2antHV[i] == chan2antHV[j]) : 
#				strauto = strauto + [chan2antHV[i]+'-'+chan2antHV[j]]
#				strauto2 = strauto2 + [chan2antHV[i]]
#				nauto = nauto + [n]
#			# cross
#			elif (chan2antHV[i][-1] == chan2antHV[j][-1]) : 
#				strc = chan2antHV[i]+'-'+chan2antHV[j]
#				if (key=='cross' and cross34==False) : 
#					if (strc=='3H-4H' or strc=='3V-4V' or strc=='4H-3H' or strc=='4V-3V') : continue
#				strcross = strcross + [strc]
#				strcross2 = strcross2 + [[chan2antHV[i],chan2antHV[j]]]
#				ncross = ncross + [n]
#
#	for i in range(len(strcross2)) : 
#		n1, n2 = -1, -1
#		for j in range(len(strauto2)) : 
#			if (strcross2[i][0] == strauto2[j]) : n1 = j
#			if (strcross2[i][1] == strauto2[j]) : n2 = j
#		nstrcross = nstrcross + [(n1,n2)]
#
#	if (key is None) : 
#		return [strauto, nauto, strcross, nstrcross, ncross]
#	elif (key == 'all') : return [strall, nall]
#	elif (key == 'auto') : return [strauto, nauto]
#	elif (key == 'cross') : return [strcross, nstrcross, ncross]
#	elif (npfmt(key).dtype.name[:3] == 'int') : 
#		t = JudgeList(key)
#		key, res = npfmt(key), []
#		for i in range(len(key)) : 
#			if (key[i]<0 or key[i]>=len(nall)) : 
#				raise Exception('key['+str(i)+']='+str(key[i])+' is out of [0:'+str(len(nall))+']')
#			n = 0
#			for j in range(len(nauto)) : 
#				if (key[i] == nauto[j]) : 
#					res = res + [strauto[j]]
#					n = 1
#			if (n == 0) : 
#				for j in range(len(ncross)) : 
#					if (key[i] == ncross[j]) : 
#						res = res + [[strcross[j], nstrcross[j]]]
#						n = 1
#			if (n == 0) : 
#				for j in range(len(nall)) : 
#					if (key[i] == nall[j]) : res = res + [strall[j]]
#		if (t == 1) : return res
#		else : return res[0]
#	elif (npfmt(key).dtype.name[:6] == 'string') : 
#		t = JudgeList(key)
#		if (t == 0) : key = [key]
#		res = []
#		for i in range(len(key)) : 
#			n = 0
#			for j in range(len(nauto)) : 
#				if (key[i] == strauto[j]) : 
#					res = res + [nauto[j]]
#					n = 1
#					break
#			if (n == 0) : 
#				for j in range(len(ncross)) : 
#					keyi = key[i].split('-')
#					key2 = keyi[1] + '-' + keyi[0]
#					if (key[i] == strcross[j]) : 
#						res = res + [[nstrcross[j], ncross[j]]]
#						n = 1
#						break
#					if (key2 == strcross[j]) : 
#						res = res + [[(nstrcross[j][1],nstrcross[j][0]), ncross[j]]]
#						n = 1
#						break
#			if (n == 0) : 
#				for j in range(len(nall)) : 
#					keyi = key[i].split('-')
#					key2 = keyi[1] + '-' + keyi[0]
#					if (key[i] == strall[j]) : 
#						res = res + [nall[j]]
#						n = 1
#						break
#					if (key2 == strall[j]) : 
#						res = res + [nall[j]]
#						n = 1
#						break
#			if (n == 0) : 
#				raise Exception('key['+str(i)+"]='"+key[i]+"' is not in "+str(strall))
#		if (t == 1) : return res
#		else : return res[0]
#	else : raise Exception('key='+str(key)+' is not a correct argument')


#def PAON4Gain( fitsnameORarray, outname='', nPC=1, malposition1D=20, remove=True ) : 
#	'''
#	Calculate and save the Gain of PAON4 data
#
#	fitsnameORarray:
#		(1) can be path(str) of the FITS file which to calculate the gain.
#		(2) can be np.array with shape=(Ntime,36,Nfreq) (must be 3D). In this function, we choose Ntime=0 to calculate the gain.
#
#	outname:
#		If outname='', this function will set a outname automatically.
#	'''
#	if (type(fitsnameORarray) == str) : 
#		dayname = PAON4Dayname( fitsnameORarray )
#		a = pyfits.open( fitsnameORarray )[0].data /1e5
#		if (a.shape[0] != 1) : print 'Note that: '+fitsname+', shape='+str(a.shape)+', may not be the FITS for Gain calculation.'
#	else : a = fitsnameORarray
#
#	nauto, strauto = PAON4Pair()[1]
#	gain = np.zeros([nauto.size, a.shape[2]], np.float32)
#	auto = np.zeros([nauto.size, a.shape[2]], np.float32)
#	
#	for i in range(nauto.size) : 
#		auto[i] = a[0,nauto[i],:]
#		# the first frequency
#		auto[i][0] = 2*auto[i][1] - auto[i][2]
#		gain[i] = PCA( auto[i], nPC, malposition1D, remove )
#		print 'Completed  nvis='+str(nauto[i])+', '+strauto[i]
#	gain = gain *1e5
#	auto = auto *1e5
#	
#	# Save to FITS
#	if (outname == '') : 
#		if (type(fitsnameORarray) == str) : 
#			outname = dayname+'_Gain.fits'
#		else : outname = 'Gain_PAON4_'+ShowTime(1)+'.fits'
#	
#	arraylist = [gain, auto]
#	
#	key1 = ['autocor', 'power', 'scale']
#	keyvalue1 = ['PCA', 'Yes, auto-correlation/gain of power', 'original scale']
#	comment1 = ['', '', '']
#	for i in range(nauto.size) : 
#		key1 = key1 + ['row'+str(i)]
#		comment1 = comment1 + ['=nvis, '+strauto[i]]
#	key = [key1, key1]
#	keyvalue1 = keyvalue1 + list(nauto)
#	keyvalue = [keyvalue1]
#	keyvalue1[0] = 'observable data'
#	keyvalue = keyvalue + [keyvalue1]
#	comment = [comment1, comment1]
#	
#	Array2FitsImage( arraylist, outname, key, keyvalue, comment )
#	return gain


def PAON4Header( paon4hdr ) : 
	'''
	PAON4 fits header to get Nt, Nf, etc

	paon4hdr: 
		paon4hdr = fo[i].header

	return: 
		[ Nt, avet, dt, tbin1, tbin2, t1, t2, 
		  Nf, avef, df, fbin1, fbin2, f1, f2, 
		  deltime, dateobs, Nv ]

	Use:
		Generally: 
			Nt, avet, dt, tbin1, tbin2, t1, t2, Nf, avef, df, fbin1, fbin2, f1, f2, deltime, dateobs, Nv = PAON4Header(hdr)
	'''
	hdr = paon4hdr
	Nt, Nv, Nf = hdr['NAXIS3'], hdr['NAXIS2'], hdr['NAXIS1']
	deltime = hdr['DELTIME']
	avef, avet = hdr['FREQAVE'], hdr['TIMEAVE']
	tbin1, tbin2 = hdr['TIMEBIN1'], hdr['TIMEBIN2']+1
	fbin1, fbin2 = hdr['FREQBIN1'], hdr['FREQBIN2']+1
	dt = 16.384e-6 *hdr['NPAQSUM'] *avet  # second
	df = 250./4096 *avef
	t1, t2 = tbin1*deltime/Nt, tbin2*deltime/Nt
	f1, f2 = 1250+250./4096*fbin1, 1250+250./4096*fbin2
	dateobs = hdr['DATEOBS']
	for j in range(len(dateobs)) : 
		if(dateobs[j]=='-'):dateobs=dateobs[:j]+'/'+dateobs[j+1:]
		if(dateobs[j]=='T'):dateobs=dateobs[:j]+' '+dateobs[j+1:]
	return [Nt, avet, dt, tbin1, tbin2, t1, t2, Nf, avef, df, fbin1, fbin2, f1, f2, deltime, dateobs, Nv]


def PAON4Chan2Antpos( chan2antHV=None ) : 
	'''
	Rectangular coordinate of PAON4 in the xyz zenith pointing system.

	chan2antHV:
		Generally, channel 1 will be linked to 1H, etc:
			chan1 -> 1H, chan2 -> 2H, chan3 -> 3H, chan4 -< 4H
			chan5 -> 1V, chan6 -> 2V, chan7 -> 3V, chan8 -< 4V
		However, the linking can be modified by people, for example: chan2antHV = ['3H', '4H', '1H', '2H', '1V', '2V', '3V', '4V'], means:
			chan1 -> 3H, chan2 -> 4H, chan3 -> 1H, chan4 -< 2H
			chan5 -> 1V, chan6 -> 2V, chan7 -> 3V, chan8 -< 4V
		Then the auto-correlations will be (form 1 to 8):
			3H-3H, 4H-4H, 1H-1H, 2H-2H
		and the cross-correlations:
			3H-4H, 3H-1H, 3H-2H, 4H-1H, 4H-2H, 1H-2H
			1V-2V, 1V-3V, 1V-4V, 2V-3V, 2V-4V, 3V-4V

	return:
		Positions of corresponding antennas of each channels
	'''
	# [ew,ns]
	A = np.array([[0,0], [+0.001,+5.993], [+5.996,-4.380], [-5.995,-4.383]])[:,::-1]
	if (chan2antHV is None) : 
		B = np.append(A, A, 0)
	else : 
		B = np.zeros([2*len(A), len(A[0])])
		n = np.zeros([len(A),], int)
		for i in range(len(chan2antHV)) : 
			n1 = int(chan2antHV[i][0]) - 1
			if (chan2antHV[i][1]=='H' or chan2antHV[i][1]=='h') : 
				pass
			elif (chan2antHV[i][1]=='V' or chan2antHV[i][1]=='v') :
				pass
			else : raise Exception('chan2antHV='+str(chan2antHV)+' is not correct')
			B[i] = A[n1]
			n[n1] = n[n1] + 1
		if (n.min() == 0 or n.max() > 2) : raise Exception('chan2antHV='+str(chan2antHV)+' is not correct')
	return B



##################################################
##################################################
##################################################


def MaxMinma( y, x=0, zeroerr=1e-3, mergen0=3, smoothtimes=4, smooth=True ) : 
	'''
	Calculate the maxima and minima: dy/dx=0
	y and x must be 1D

	x:
		You can set x.
		If x=0, mean x=np.arange(y.size)

	zeroerr:
		maxima and minima locate at dy/dx=0.
		But almost it doesn't equal to 0 exactly.
		dy/dx = [-zeroerr, +zeroerr] will be assumed to be 0

	mergen0:
		Result maybe n0 = [5,6,7,12, 100,101,103, 200,201]
		The actual result maybe n0=[7, 101, 200]
		For example, n0[3]=12, parameter "mergen0" is use to judge n0[3]=12 is in [5,6,7] or an individual point, using (7-12)<=mergen0

	smoothtimes:
		How many times to smooth.

	smooth:
		Generally y contains noise.
		smooth=True, use Mean() function to smooth y, the result will be more correct.

	Return: 
		if (x0 == None) : return [n[nr], y[n[nr]]]
		else : return [n[nr], y[n[nr]], x[n[nr]]]
	'''
	N = y.size
	n = np.arange(N)
	x0 = 1
	if (type(x) == int) : 
		x0 = 0 
		x = n
	if (len((x*y).shape) != 1) : raise Exception(efn()+'y and x must be 1D.')
	if (smooth) : 
		Ns = N/100
		if (Ns < 10) : Ns = 10
		y = Smooth(y, 0, Ns, smoothtimes)
	# If dy.size is small, we linear fit it
	if (N < 2e4) : 
		f = interpolate.interp1d(x, y)
		xnew = np.linspace(x[0], x[-1], 20000)
		n = np.linspace(0, N-1, 20000)#.astype(int)
		y = f(xnew) # xnew-y
	else : xnew = x
	dy = y[1:] - y[:-1]	
	dy = np.append(dy, dy[-1])
	if (x0 == 0) : dx = n[1] - n[0]
	else : 
		dx = xnew[1:] - xnew[:-1]
		dx = np.append(dx, dx[-1])
	dy = dy/dx
	dy = dy /dy.max()
	n0 = Value2Index(dy, 0, zeroerr)
	# merge continuum
	nt, nr = [n0[0]], []
	for i in range(1, n0.size) : 
		if ((n0[i]-nt[-1])<=5) : nt = nt + [n0[i]]
		else : 
			nr = nr + [int(np.array(nt).mean())]
			nt = [n0[i]]
	nr = np.array(nr)
	n = n.astype(int)
	if (x0 == 0) : return [n[nr], y[n[nr]]]
	else : return [n[nr], y[n[nr]], x[n[nr]]]


# Calculate nfr1, nfr2
def nfr( frequency, LAST, Dec, D0, deltime, Nt, ksigma=3.2 ) : 
	'''
	LAST:
		LAST = SiderealTime(lon='', date='', RAsource='')
		
	Dec:
		Dec of the source

	ksigma:
		Width between nfr[0] and nfr[1]
	'''
	fwhm = 1.03 * (300./frequency) / (0.8*D0)
	sigma = fwhm / (8*np.log(2))**0.5
	dra = Sph2Circ(ksigma*sigma, Dec)
	ra1 = LAST[1] - dra
	ra2 = LAST[1] + dra
	nfr1 = int(ra1*180/np.pi/15*3600 / deltime * Nt)
	nfr2 = int(ra2*180/np.pi/15*3600 / deltime * Nt)
	if (nfr1 < 0) : nfr1 = 0
	if (nfr2 > Nt) : nfr2 = Nt
	return np.array([nfr1, nfr2])


##################################################
##################################################
##################################################


def Interp1d( xdata, ydata, xnew, kind='linear' ) : 
	'''
	1D interpolation. Note that all array must be 1D.
	Outside the xdata, will interp linear

	kind:
		'linear' or 'cubic'
	'''
	if (len(xdata.shape)!=1 or len(ydata.shape)!=1 or len(xnew.shape)!=1) : raise Exception('All array (xdata, ydata, xnew) must be 1D.')
	f = interpolate.interp1d( xdata, ydata, kind=kind )

	x = xnew + 1j*np.arange(xnew.size)
	xnew = 0 #@

	xl = x[x.real<xdata[0]]
	xr = x[x.real>xdata[-1]]
	xc = x[(x.real>=xdata[0])*(x.real<=xdata[-1])]
	x = 0 #@

	yc = f(xc.real) + 1j*xc.imag

	kl = (ydata[1]-ydata[0])/(xdata[1]-xdata[0])
	kr = (ydata[-1]-ydata[-2])/(xdata[-1]-xdata[-2])
	yl = kl*(xl.real - xdata[0]) + ydata[0] + 1j*xl.imag
	yr = kr*(xr.real - xdata[-2]) + ydata[-2] + 1j*xr.imag
	xdata = ydata = f = xnew = xl = xc = xr = 0 #@

	y = np.append(yl, yc)
	y = np.append(y, yr)
	yl = yc = yr = 0 #@

	y = np.append(y.real[:,None], y.imag[:,None], 1)
	y = Sort( y, ('col', 1) )[:,0]
	return y


##################################################
##################################################
##################################################


def Spherical2Rectangular( theta, phi ) : 
	'''
	Rectangular coordinate system: (x, y, z)
	Spherical coordinate system: (theta, phi) 
		[+z, -z] => theta=[0, 180]
		[+x, +y, -x, -y] => phi=[0, 90, 180, 270]

	x = sin(theta)*cos(phi)
	y = sin(theta)*sin(phi)
	z = cos(theta)

	All angle in rad.

	Note that this function doesn't broadcast !!!
	'''
	x = np.sin(theta)*np.cos(phi)
	y = np.sin(theta)*np.sin(phi)
	z = np.cos(theta)
	return np.array([x, y, z])
	

def Rectangular2Spherical( x, y, z ) : 
	'''
	Rectangular coordinate system: (x, y, z)
	Spherical coordinate system: (theta, phi) 
		[+z, -z] => theta=[0, 180]
		[+x, +y, -x, -y] => phi=[0, 90, 180, 270]

	x = sin(theta)*cos(phi)
	y = sin(theta)*sin(phi)
	z = cos(theta)

	All angle in rad.

	Note that this function doesn't broadcast !!!
	'''
	x1, y1, z1 = npfmt(x), npfmt(y), npfmt(z)
	if (x1.size!=1 or y1.size!=1 or z1.size!=1) : raise Exception(enf()+'(x,y,z) must be "one" point.')
	# phi
	if (x==0 and y!=0) : 
		if (y > 0) : phi = np.pi/2
		if (y < 0) : phi = np.pi/2*3
	elif (x!=0 and y==0) : 
		if (x > 0) : phi = 0.
		if (x < 0) : phi = np.pi
	elif (x==0 and y==0) : phi = 0.
	elif (x!=0 and y!=0) : 
		tan = y/x
		if (tan > 0) : 
			if (x>0 and y>0) : phi = np.arctan( tan )
			else : phi = np.pi + np.arctan( tan )
		elif (tan < 0) : 
			if (x>0 and y<0) : phi = 2*np.pi + np.arctan( tan )
			else : phi = np.pi + np.arctan( tan )
	# theta
	theta = np.arccos( z )
	return np.array([theta, phi])


def RotateCoord( alpha, beta, xyz, inverse=False ) : 
	'''
	All angle in rad.
	Use two angles alpha and beta, the system can be rotated to any direction!

	x, y, z = xyz
	x, y, z can be np.ndarray/list

	In the interferometry, xyz is local coordinate,
	    x1=w,  y1=u,  z1=v     # Note that !!!

	Rectangular coordinate system: (x, y, z)
		Right hand system, +z is thumb, 4 fingers is from +x to +y
	Spherical coordinate system: (theta, phi) 
		[+z, -z] => theta=[0, 180]
		[+x, +y, -x, -y] => phi=[0, 90, 180, 270]

	Rotated system (x1, y1, z1)
		(1) Also right hand system
		(2) alpha (phi rotation) = [0, 360]: 
				+x rotates alpha angle to +x1, right hand is positive, left hand is negative
		(3) beta (theta rotation) = [0, 180]: 
				select y1 axis as the rotation axis to rotate the system, it means y1 axis is always in xy plane. 
				beta is the angle from z to z1 at -x direction. If rotate z to +x direction, bete is negative
#####				beta is the angle between +z1 and +z (also the angle between +x1 and xy plane), when +x1 is above the xy plane, beta is positive, when +x1 is below the xy plane, beta is negative

	*** Note that this rotation matrix is just for the rotation axis is y1 axis, y1 is always in xy plane!

	xyz rotates alpha+beta to x1y1z1, it means first rotates alpha, second rotates beta. If first rotates beta and then alpha, the result is wrong. 
		alpha, beta = np.pi/3, np.pi/4
		xyz = (5.2, 7.4, 9.6)
		x1y1z1:
			(1) x1y1z1 = RotateCoord(alpha, beta, xyz)
			(2) x1y1z1tmp = RotateCoord(alpha, 0, xyz), then
		       x1y1z1 = RotateCoord(0, beta, x1y1z1tmp)

		From x1y1z1 to xyz, we use method (2) inverse, first rotate beta, then rotate alpha
		Because from z to z1 is at -x, so from z1 to z:
		    (1) at +x: -beta
		    (2) at -x: 2*np.pi-beta  # result is the same as (1)
		From x1 to x: -alpha or 2*np.pi-alpha
		Inverse rotation:
			(1) Rotate beta: xyztmp = RotateCoord(0, -beta, x1y1z1)
		   (2) Then rotate alpha: xyz = RotateCoord(-alpha, 0, xyztmp)

	inverse:
		You can use the method above to do the inverse rotation.
		However, you can also set inverse=True to do that.

	Rotation matrix
x1   | cos(alpha)*cos(beta)  sin(alpha)*cos(beta) sin(beta)| x
y1 = |     -sin(alpha)            cos(alpha)          0    | y
z1   |-cos(alpha)*sin(beta) -sin(alpha)*sin(beta) cos(beta)| z

	Note that this function doesn't broadcast !!!

	xyz:
		(x,y,z) or [x,y,z] or np.array([x,y,z])
	'''
	x, y, z = xyz
	M = np.array([[np.cos(beta)*np.cos(alpha), np.cos(beta)*np.sin(alpha), np.sin(beta)], 
	[-np.sin(alpha), np.cos(alpha), 0],
	[-np.sin(beta)*np.cos(alpha), -np.sin(beta)*np.sin(alpha), np.cos(beta)]])
	if (inverse == True) : M = spla.inv(M)
	x1 = M[0,0]*x + M[0,1]*y + M[0,2]*z
	y1 = M[1,0]*x + M[1,1]*y + M[1,2]*z
	z1 = M[2,0]*x + M[2,1]*y + M[2,2]*z
	return np.array([x1, y1, z1])


##################################################
##################################################
##################################################


def SyntheticBeam( antxyz, antbeam, thetaphi, freq, deviation=[0,0], compensate=False ) : 
	'''
	No matter what direction the antennas point to, the synthetic beams of the array are the same as that of pointing to the zenith, because we should compensate the additional phase when the antennas don't point to the zenith.

	antxyz:
		antxyz.shape=(N, 3), N is the number of antenna, 3 is x,y,z
		Rectangular coordinate of each antenna.
		Rectangular coordinate system we use here is :
			+z: zenith,  +x: South,  +y: East
		Corresponding spheric coordinate system:
			theta: +z to -z: 0 to 90 degree
			phi: +x to +y to -x to -y: 0, 90, 180, 270 degree
								-x
				ground	-y		+y
								+x

	antbeam:
		Beam of each antenna.
		If each beams are not the same, then antbeam is a list of beams.
		If each beams are the same, then antbeam can be 1 same beam (2D beam or 1D beam).

	thetaphi:
		If antbeam is 1D beam (just have theta angle), then thetaphi = theta (1D).
		If antbeam is 2D beam (theta, phi), then theta, phi = thetaphi.
		Must theta.shape == beam.shape
	
	freq:
		In MHz, because there is 2*pi/lambda

	deviation:
		=[alpha, beta], see function RotateCoord().
		Devation angle from the antenna axis(direction) in rad.
		deviation can be a list like [[0.2,0.3], [0.2,0.4], [0.6,2], ...].
		Rectangular coordinate system we use here is :
			+z: zenith,  +x: South,  +y: East
		antenna axis: [0, 0], deviate to south: beta<0, deviate to north: beta>0
	'''
	# Cheak beam
	if (type(antbeam) is np.ndarray) : 
		if (len(antbeam.shape) > 2) : 
			raise Exception('the same beam must be 2D or 1D')
		antbeam = np.array([antbeam for i in range(len(antxyz))])
	elif (type(antbeam) is list) : 
		if (len(antbeam) != len(antxyz)) : 
			raise Exception('len(antbeam) != len(antxyz)')
		for i in range(1, len(antbeam)) : 
			if (npfmt(antbeam[i]).shape != npfmt(antbeam[0]).shape) :
				raise Exception('antbeam['+str(i)+'].shape != antbeam[0].shape')
		antbeam = npfmt(antbeam)
		for i in range(len(antbeam)) : 
			if (len(antbeam[i].shape) > 2) : 
				raise Exception('each beam must be 2D or 1D')
	if (len(antbeam[0].shape) == 1) : ND = 1
	elif (len(antbeam[0].shape) == 2) : ND = 2

	# Cheak thetaphi
	if (ND == 1) : 
		thetaphi = npfmt(thetaphi)
		if (len(thetaphi.shape) != 1) : raise Exception('1D beam, thetaphi must also 1D =theta')
		theta, phi = thetaphi, 0
	elif (ND == 2) : 
		if (len(thetaphi) != 2) : raise Exception('2D beam, but len(thetaphi) != 2')
		theta, phi = thetaphi
		if (theta.shape != phi.shape) : raise Exception('theta.shape != phi.shape')
	if (theta.shape != antbeam[0].shape) : 
		raise Exception('theta.shape != beam.shape')

	# deviation
	for i in range(len(deviation)) : 
		if (npfmt(deviation[i]).size > 2) : 
			raise Exception('len(deviation['+str(i)+']) > 2')
	for i in range(1, len(deviation)) : 
		if (npfmt(deviation[i]).size != npfmt(deviation[0]).size) : 
			raise Exception('len(deviation['+str(i)+']) != len(deviation[0])')
	deviation = npfmt(deviation)
	if (deviation.shape == (2,)) : 
		deviation = np.array([deviation for i in range(len(antxyz))])

	# Use correlation to do this, so that this function can be used to synthetic beam and map making.
	Asyn = 0 + 0j
	k0 = 2*np.pi/(300./freq)
	kx = k0 * np.sin(theta)*np.cos(phi)
	ky = k0 * np.sin(theta)*np.sin(phi)
	kz = k0 * np.cos(theta)
	antxyz = npfmt(antxyz)
	''' Method 1 '''
	for i in range(len(antxyz)) : 
		xi, yi, zi = RotateCoord(deviation[i][0], deviation[i][1], antxyz[i])
		Asyn = Asyn + antbeam[i] * np.exp(1j*(kx*xi + ky*yi + kz*zi))
	Asyn = abs(Asyn)**2
	''' Method 2 '''
#	for i in range(len(antxyz)) : 
#		ri = RotateCoord(deviation[i][0], deviation[i][1], antxyz[i])
#		for j in range(len(antxyz)) : 
#			rj = RotateCoord(deviation[i][0], deviation[i][1], antxyz[j])
#			xij, yij, zij = (rj - ri)
#			Asyn = Asyn + antbeam[i] * (antbeam[j].conj()) * np.exp(1j*(kx*xij + ky*yij + kz*zij))
#	Asyn = Asyn.real
	''' Method 3 '''
#	xi, yi, zi = antxyz[i]
#	Asyn = Asyn + antbeam[i] * np.exp(1j*(kx*xi + ky*yi + kz*zi)) * np.exp(1j*k0* xi*np.sin(deviation[i][1]))
#	Asyn = abs(Asyn)**2
	return Asyn


##################################################
##################################################
##################################################


def Sph2Circ( angle, Dec ) : 
	'''
	Spherical2CircularAngle()
	Convert between spherical and circular angles

	angle:
		scale of an angle between two points

	All angles are in rad
	'''
	dang = np.pi/2-Dec - angle/2.
	if (abs(dang) < 1e-6) : return np.pi
	elif (dang < 0) : 
		raise Exception('Maximum Sph at Dec='+('%.3f' % (Dec*180/np.pi))+'deg is 2*(90-Dec)='+('%.3f' % (2*(90-Dec*180/np.pi)))+'deg, now angle='+('%.3f' % (angle*180/np.pi))+'deg')
	else : 
		return 2*np.arcsin( np.sin(angle/2.) /np.cos(Dec) )

def Circ2Sph( angle, Dec ) : 
	return 2*np.arcsin( np.sin(angle/2.) *np.cos(Dec) )


##################################################
##################################################
##################################################



def Dij2DiDj( Dij, DorSigma=0 ) :
	'''
	Just for PAON4
	Row of Dij and Dj is index ij

	DorSigma:
		=0: mean input Dij
		=1: mean input Sigmaij
	'''
	if (len(Dij) != 10) : raise Exception(efn()+'This function is just for PAON4, len(Dij)=10')
	Dij = Dij**2
	if (DorSigma == 1) : Dij = 1./Dij
	D = Dij[:8]*0
	D[1] = (2*Dij[0]+Dij[3]+Dij[4]-Dij[1]-Dij[2]) /2
	D[0] = 2*Dij[0] - D[1]
	D[2] = 2*Dij[3] - D[1]
	D[3] = 2*Dij[4] - D[1]
	D[5] = (2*Dij[5]+Dij[8]+Dij[9]-Dij[6]-Dij[7]) /2
	D[4] = 2*Dij[5] - D[5]
	D[6] = 2*Dij[8] - D[5]
	D[7] = 2*Dij[9] - D[5]
	D = abs(D)**0.5
	if (DorSigma == 1) : D = 1./D
	return D
	

def tsij2tsitsj( tsij, Di, DorSigma=0 ) : 
	'''
	Please see Dij2DiDj()
	'''
	Di = Di**2
	ts = Di*0
	def d(i,j,n) : 
		if (DorSigma == 1) : 
			if (n == i) : n = j
			elif (n == j) : n = i
		return Di[n]/(Di[i]+Di[j])
	t11 = (d(1,2,2)*(d(0,2,0)*tsij[0] - d(0,1,0)*tsij[1]) + d(0,2,2)*d(0,1,0)*tsij[3]) / (d(1,2,2)*d(0,1,1)*d(0,2,0) + d(1,2,1)*d(0,2,2)*d(0,1,0))
	t12 = (d(1,3,3)*(d(0,3,0)*tsij[0] - d(0,1,0)*tsij[2]) + d(0,3,3)*d(0,1,0)*tsij[4]) / (d(1,3,3)*d(0,1,1)*d(0,3,0) + d(1,3,1)*d(0,3,3)*d(0,1,0))
	ts[1] = (t11+t12)/2
	ts[0] = (tsij[0] - d(0,1,1)*ts[1]) / d(0,1,0)
	ts[2] = (tsij[3] - d(1,2,1)*ts[1]) / d(1,2,2)
	ts[3] = (tsij[4] - d(1,3,1)*ts[1]) / d(1,3,3)
	t11 = t12 = 0 #@
	t51 = (d(5,6,6)*(d(4,6,4)*tsij[5] - d(4,5,4)*tsij[6]) + d(4,6,6)*d(4,5,4)*tsij[8]) / (d(4,6,6)*d(4,5,5)*d(4,6,4) + d(5,6,5)*d(4,6,6)*d(4,5,4))
	t52 = (d(4,7,7)*(d(4,7,4)*tsij[5] - d(4,5,4)*tsij[7]) + d(4,7,7)*d(4,5,4)*tsij[9]) / (d(4,7,7)*d(4,5,5)*d(4,7,4) + d(5,7,5)*d(4,7,7)*d(4,5,4))
	ts[5] = (t51+t52)/2
	ts[4] = (tsij[5] - d(4,5,5)*ts[5]) / d(4,5,4)
	ts[6] = (tsij[8] - d(5,6,5)*ts[5]) / d(5,6,6)
	ts[7] = (tsij[9] - d(5,7,5)*ts[5]) / d(5,7,7)
	t51 = t52 = 0 #@
	return ts


##################################################
##################################################
##################################################


def Phase2Line( array, axis=0, threshold=1.8 ) : 
	'''
	phase = k * freq + a
	However, because of the period property (sin/cos), phase is from 0 to 2pi
	This function is to convert the [0,2pi] phase to linear.

	array:
		Array of phase/angle

	axis:
		If array is N-D, need to tell that which axis is phase

	threshold:
		Difference >= threshold*np.pi will be consider to be the other period
	'''
	# Move axis to 0
	array = ArrayAxis(array, axis, -1, 'move')
	shape = np.array(array.shape)
	L, N = shape[-1], shape[:-1].prod()
	array = array.flatten()

	for i in xrange(N) : 
		a = array[i*L:(i+1)*L]
		da = a[1:] - a[:-1]
		da = np.ma.MaskedArray(da, abs(da)>=threshold*np.pi)
		sign = np.sign(da.mean())
		if (sign > 0) : tf = da<-threshold*np.pi
		else : tf = da>threshold*np.pi
		tf = np.concatenate([[False], tf])
		n = np.arange(L)[tf]
		if (n.size <= 1) : continue

		# Remove not
		dn = n[1:] - n[:-1]
		dnlq = dn[SelectLeastsq(dn)].take(0)
		nbad = np.arange(dn.size)[(dnlq*0.3>dn)+(dn>1.3*dnlq)]

		if (nbad.size > 0) : 
			vbad = dn[nbad]
			dn = list(dn)
			for j in range(len(nbad)) : dn.remove(vbad[j])
			dn = npfmt(dn)
			dnlq = dn[SelectLeastsq(dn)].take(0)
			vgood, tf = [], np.ones(len(n), bool)
			for j in range(len(nbad)) : 
				tf[nbad[j]] = tf[nbad[j]+1] = False
				v1 = npfmt(n[nbad[j]+2] - n[nbad[j]-1])
				v2 = npfmt([n[nbad[j]]-n[nbad[j]-1], n[nbad[j]+2]-n[nbad[j]]])
				v3 = npfmt([n[nbad[j]+1]-n[nbad[j]-1], n[nbad[j]+2]-n[nbad[j]+1]])
				v1 = ((v1-dnlq)**2).mean()
				v2 = ((v2-dnlq)**2).mean()
				v3 = ((v3-dnlq)**2).mean()
				nsel = np.array([v1,v2,v3]) + 1j*np.arange(3)
				nsel = Sort(nsel).imag.astype(int)[0]
				if (nsel == 0) : pass
				elif (nsel == 1) : vgood.append(n[nbad[j]])
				elif (nsel == 2) : vgood.append(n[nbad[j]+1])
			n = Sort(np.append(n[tf], vgood).astype(n.dtype))

		n = np.concatenate([n, [L]])
		for j in range(len(n)-1) : 
			a[n[j]:n[j+1]] += sign*(j+1)*2*np.pi  # the result

		if (nbad.size > 0) : 
			# Interp1d bad
			da = a - Smooth(a, 0, a.size/10)
			n = np.arange(a.size)[abs(da)>2*da.std()]
			tfgood = np.ones(a.size, bool)
			tfgood[n] = False
			ngood = np.arange(a.size)[tfgood]
			if (ngood.size > 0.7*a.size) : 
				a[n] = Interp1d(ngood, a[ngood], n)
		da = tf = n = tfgood = ngood = 0 #@

	array = array.reshape(shape)
	array = ArrayAxis(array, -1, axis, 'move')
	return array



#def Phase2Line( array, axis=0, threshold=1.8, dowhile=0.2 ) : 
#	'''
#	phase = k * freq + a
#	However, because of the period property (sin/cos), phase is from 0 to 2pi
#	This function is to convert the [0,2pi] phase to linear.
#
#	array:
#		Array of phase/angle
#
#	axis:
#		If array is N-D, need to tell that which axis is phase
#
#	threshold:
#		Difference >= threshold*np.pi will be consider to be the other period
#
#	dowhile:
#		If you give a good threshold, the result is good.
#		If you give a bad threshold, the result is bad, in this case, we use dowhile to reset a good threshold(automatically).
#		dowhile = badpoint/totalpoint, if >dowhile, will consider this result is bad, do again.
#	'''
#	# Move axis to 0
#	array = ArrayAxis(array, axis, -1, 'move')
#	shape = np.array(array.shape)
#	L, N = shape[-1], shape[:-1].prod()
#	array = array.flatten()
#	for i in xrange(N) : 
#		a = array[i*L:(i+1)*L]
#		da = a[1:] - a[:-1]
#		dow = True
#		while (dow) : 
#			da = np.ma.MaskedArray(da, abs(da)>=threshold*np.pi)
#			sign = np.sign(da.mean())
#			if (sign > 0) : tf = da<-threshold*np.pi
#			else : tf = da>threshold*np.pi
#			tf = np.concatenate([[False], tf])
#			n = np.arange(L)[tf]
#			print n
#			print n[1:]-n[:-1]
#			plt.plot(np.arange(a.size), a, 'bo')
#			plt.plot(n, a[n], 'ro', markersize=4)
#			plt.show()
#			exit()
#			if (n.size == 0) : dow = False
#			else : 
#				dn = 1.*n[1:] - n[:-1]
#				if ((dn.max()-dn.min())/dn.max()<dowhile) : dow = False
#				else : threshold += 0.05
#		print n
#		exit()
#		if (n.size > 0) : 
#			n = np.concatenate([n, [L]])
#			for j in range(len(n)-1) : 
#				a[n[j]:n[j+1]] += sign*(j+1)*2*np.pi  # the result
#			# Interp1d bad
#			da = a[1:] - a[:-1]
#			tf = abs(da)>2*da.std()
#			n = np.arange(da.size)[tf]
#			n = np.concatenate([n-1, n, n+1])
#			n = np.sort(n[(0<=n)*(n<L)])  # bad
#			tfgood = np.ones(a.size, bool)
#			tfgood[n] = False
#			ngood = np.arange(a.size)[tfgood]
#			if (ngood.size > 0.7*a.size) : 
#				a[n] = Interp1d(ngood, a[ngood], n)
#			da = tf = n = tfgood = ngood = 0 #@
#	array = array.reshape(shape)
#	array = ArrayAxis(array, -1, axis, 'move')
#	return array


##################################################
##################################################
##################################################


def RemoveRFI( *arg, **kwargs ) : 
	'''
	There are 3 initial case: ['cubic','linear'], 'median', 'manual
	arglist = [[('array',np.ndarray), ('axis','NumType'), ('loop','NumType'), ('kind',str), ('thrtimes',1), ('applr',False), ('printnnan',False)],
	           [('array',np.ndarray), ('axis','NumType'), ('npix','NumType'), ('kind','median')],
	           [('array',np.ndarray), ('axis','NumType'), ('pair','AnyType'), ('kind','manual'), ('stdpix',100), ('stdtimes',0.8)]]

	array:
		Any dimension np.ndarray

	axis:
		Along which axis of array to remove RFI?

	kind:
		'cubic', 'linear', 'median', 'manual'
		='cubic': use cubic spline fitting to remove RFI
		='linear': use linear fitting to remove RFI
		='median': use median filter to remove RFI
		='manual': set where is the RFI by hand

	Case 1: kind in ['cubic', 'linear']: 
		loop: int, how many loop to remove RFI?
		thrtimes: float, times of current (array-smooth(array,0,3)).std() to judge which pixel is RFI
		applr:
			True, False, int
			If pixels [7,8,11,12,14,15] are RFI and [9,10,13] may be not. 
			Set applr=False: means that [9,10,13] are not RFI
			Set applr=True: means that [13] is RFI but [9,10] are not, because True means difference is 1
			Set applr=int, for example applr=2, them [9,10,13] are all RFI, because the differences of them are within 2
		printnnan: 
			True/False
			['cubic','linear'] methods will judge some pixels are RFI. If printnnan=True, then will print these RFI pixels on the screen

	Case 2: kind='median':
		npix: window width of the median filter

	Case 3: kind='manual':
	           {'array':np.ndarray, 'axis':'NumType', 'pair':'AnyType', 'kind':'manual', 'stdpix':100, 'stdtimes':0.8}]
		pair: 
			Set which pixels are RFI by hand
			pair can be (n1,n2), [n1,n2], [(n1,n2), (n3,n4)], ...
			If one pair: pair=(n1,n2), means np.arange(n1,n2) are RFI
			If list of pair: pair=[(n1,n2), (n3,n4)], means [np.arange(n1,n2), np.arange(n3,n4)] are RFI
		stdpix:
			int, In order to re-fill the RFI pixels, I use interp1d()
			I use stdpix=100*2 pixels outside the RFI to calculate an average edge, then linear interp
		stdtimes:
			float. After interp1d() with stdpix above, RFI will be re-filled with linear. Then add Gaussian noise with std=thrtimes*sigma to the linear line so that it looks better. sigma is calculated automatically
	'''
	#--------------------------------------------------
	arglist = [[('array',np.ndarray), ('axis','NumType'), ('loop','NumType'), ('kind',str), ('thrtimes',1), ('applr',False), ('printnnan',False)],
	           [('array',np.ndarray), ('axis','NumType'), ('npix','NumType'), ('kind','median')],
	           [('array',np.ndarray), ('axis','NumType'), ('pair','AnyType'), ('kind','manual'), ('stdpix',1000), ('stdtimes',0.6)]]
	arglist, which = ArgInit(arglist, arg, kwargs)
	# Check kind
	if (Type(which) == 'NumType') : 
		arglist, which = [arglist], [which]
	kind = arglist[0][3].lower()
	if (kind in ['cubic', 'linear']) : 
		arglist, which = arglist[0], 0
	elif (kind == 'manual') : 
		arglist, which = arglist[-1], 2
	elif (kind == 'median') : 
		for i in range(len(which)) : 
			if (len(arglist[i]) == 4) : break
		arglist, which = arglist[i], 1
	#--------------------------------------------------
	array, axis = arglist[:2]
	shape, axis = array.shape, int(round(axis))
	# Check axis
	if (axis < 0) : axis = len(shape) + axis
	if (axis >= len(shape)) : raise Exception('axis='+str(axis)+' out of array.shape='+str(shape)+'=>'+str(len(shape))+'D')
	# axis moves to -1
	array = ArrayAxis(array, axis, -1, 'move')
	shape = array.shape
	N = shape[-1]
	#--------------------------------------------------------
	if (which == 0) : 
		loop, kind, thrtimes, applr, printnnan = arglist[2:]
		loop = int(round(loop))
		# Check applr
		if (applr in [None, 0, '', [], ()]) : applr = False
		elif (applr is True) : napplr = 1
		elif (applr is not False) : 
			napplr = applr
			applr = True
		n = np.arange(N)
		#-------------------------
		for j in range(loop) : 
			darray = array - Smooth(array, -1, 3)
			threshold = npfmt(thrtimes * darray.std(-1))
			# flatten()
			array = array.flatten()
			darray = darray.flatten()
			threshold = threshold.flatten()
			for i in range(threshold.size) : 
				nvalid = n[darray[i*N:(i+1)*N] <= threshold[i]]
				nnan = n[darray[i*N:(i+1)*N] > threshold[i]]
				if (printnnan == True) : 
					print 'RemoveRFI(), i='+str(i)+', nnan.size='+str(nnan.size)
				if (applr is True and len(nnan)>0) : 
					for k in range(napplr) : 
						nnan0 = RemoveRepetition(np.append(nnan, nnan-1), 0)[0]
						nnan1 = RemoveRepetition(np.append(nnan, nnan+1), 0)[0]
						nnan = RemoveRepetition(np.append(nnan0, nnan1), 0)[0]
					nvalid = n*1
					for k in range(len(nnan)) : 
						nvalid = nvalid[nvalid!=nnan[k]]
				#-------------------------
				array[i*N:(i+1)*N] = Interp1d(nvalid, array[i*N:(i+1)*N][nvalid], n, kind.lower())
			array = array.reshape(shape)
	#--------------------------------------------------------
	elif (which == 1) : 
		npix, kind = arglist[2:]
		array = array.flatten()
		if (npix < 3) : npix = 3
		if (npix % 2 == 0) : npix +=1
		for i in range(array.size/N) : 
			array[i*N:(i+1)*N] = spsn.medfilt(array[i*N:(i+1)*N], npix)
		array = array.reshape(shape)
	#--------------------------------------------------------
	elif (which == 2) : 
		pair, kind, stdpix, stdtimes = arglist[2:]
		#--------------------
		if (pair in [None, False, '', [], np.array([])]) : 
			array = array.reshape(shape)
			array = ArrayAxis(array, -1, axis, 'move')
			return array
		#--------------------
		# Check pair
		pair = npfmt(pair)
		if (len(pair.shape) == 1) : pair = pair[None,:]
		if (pair.shape[1] != 2) : raise Exception('N='+str(pair)+', N.shape[1]!=2')
		array = array.flatten()
		for j in range(len(pair)) : 
			for i in range(array.size/N) : 
				arr = array[i*N:(i+1)*N]
				# Check pair again
				if (pair[j,0]<10) : pair[j,0] = 10
				if (pair[j,1]>len(arr)-11) : pair[j,1] = len(arr)-11
				na1, na2 = pair[j][0]-stdpix, pair[j][1]+stdpix
				if (na1 < 0) : na1 = 0
				if (na2 >= len(arr)) : na2 = len(arr)
				a1, a2 = arr[na1:pair[j][0]], arr[pair[j][1]+1:na2]
				std = (a1.std()+a2.std())/2.*stdtimes
				a1[-1], a2[0] = a1.mean(), a2.mean()
				nall = np.arange(na2-na1)
				nvalid = np.append(nall[:len(a1)], nall[-len(a2):])
				nnan = np.arange(len(a1), len(nall)-len(a2))
				nnanr = np.arange(pair[j][0], pair[j][1]+1)
				arr[nnanr] = Interp1d(nvalid, np.append(a1,a2), nnan) + np.random.randn(nnan.size)*std
				array[i*N:(i+1)*N] = arr
		array = array.reshape(shape)
	#--------------------------------------------------------
	array = ArrayAxis(array, -1, axis, 'move')
	return array


#def RemoveRFI( array, axis, timesORsize=1, kind='median', thrtimes=1, applr=False, printnnan=False ) : 
#	'''
#	array:
#		Any dimension array
#
#	axis:
#		Along which axis of array to remove RFI?
#
#	kind:
#		'cubic', 'linear', 'median'
#		='cubic': use cubic spline fitting to remove RFI
#		='linear': use linear fitting to remove RFI
#		='median': use median filter to remove RFI
#
#	timesORsize:
#		int
#		kind in ['cubic', 'linuea']: how many loop to remove RFI?
#		kind='median': filter size
#
#	thrtimes:
#		int, times of current (array-smooth(array,0,3)).std()
#		Just for ['cubic', 'linuea'] to judge which pixel is RFI
#
#	applr:
#		True, False, int
#		Just for ['cubic', 'linuea']
#		If pixel of RFI like [10,11,12], large timesORsize may not do well, then use applr to assume [9,13](when applr=1) are also RFI, then timeORsize can be smaller.
#	'''
#	array = npfmt(array)
#	shape = array.shape
#	# Check axis
#	if (axis < 0) : axis = len(shape) + axis
#	if (axis >= len(shape)) : raise Exception('axis='+str(axis)+' out of array.shape='+str(shape)+'=>'+str(len(shape))+'D')
#	# Check kind
#	if (kind.lower() not in ['cubic', 'linear', 'median']) : 
#		raise Exception('kind='+kind+' not in '+str(['cubic', 'linear', 'median']))
#	# axis moves to -1
#	array = ArrayAxis(npfmt(array), axis, -1, 'move')
#	N, shape = array.shape[-1], array.shape
#	#--------------------------------------------------------
#	if (kind.lower() != 'median') : 
#		n = np.arange(N)
#		if (applr in [None, 0, '', [], ()]) : applr = False
#		elif (applr is True) : napplr = 1
#		elif (applr is not False) : 
#			napplr = applr
#			applr = True
#		#------------------------------
#		for j in range(timesORsize) : 
#			darray = array - Smooth(array, -1, 3)
#			threshold = npfmt(thrtimes * darray.std(-1))
#			# flatten()
#			array = array.flatten()
#			darray = darray.flatten()
#			threshold = threshold.flatten()
#			for i in range(threshold.size) : 
#				nvalid = n[darray[i*N:(i+1)*N] <= threshold[i]]
#				#---------------------------------
#				nnan = n[darray[i*N:(i+1)*N] > threshold[i]]
#				if (printnnan == True) : 
#					print 'RemoveRFI(), i='+str(i)+', nnan.size='+str(nnan.size)
#				if (applr is True and len(nnan)>0) : 
#					for k in range(napplr) : 
#						nnan0 = RemoveRepetition(np.append(nnan, nnan-1), 0)[0]
#						nnan1 = RemoveRepetition(np.append(nnan, nnan+1), 0)[0]
#						nnan = RemoveRepetition(np.append(nnan0, nnan1), 0)[0]
#					nvalid = n*1
#					for k in range(len(nnan)) : 
#						nvalid = nvalid[nvalid!=nnan[k]]
#				#---------------------------------
#				array[i*N:(i+1)*N] = Interp1d(nvalid, array[i*N:(i+1)*N][nvalid], n, kind.lower())
#			array = array.reshape(shape)
#	#--------------------------------------------------------
#	else : 
#		array = array.flatten()
#		if (timesORsize < 3) : timesORsize = 3
#		if (timesORsize % 2 == 0) : timesORsize = timesORsize + 1
#		for i in range(array.size/N) : 
#			array[i*N:(i+1)*N] = spsn.medfilt(array[i*N:(i+1)*N], timesORsize)
#		array = array.reshape(shape)
#	array = ArrayAxis(array, -1, axis, 'move')
#	return array


##################################################
##################################################
##################################################


def Compile( code=None, machine='bao' ) : 
	'''
	Compile c/c++ with SOPHYA in my MacBook or in bao service machine.
	'''
	if (code is None) : 
	#	print 'Error: Please set the argument, which c/c++ code do you want to compile?'
		print '----- Help -----'
		print './compile_MyMac_bao.py p1   OR   ./compile_MyMac_bao.py p1 p2'
		print '    p1: c/c++ code will be compiled, such as test.cc'
		print '    p2: ="my" or ="bao", compile in my own Mac or in bao service machine'
		print 'Output .out with the same name as .cc at the same path'
		os._exit(0)
	codedirname = sys.argv[1]
	
	for i in range(len(codedirname)-1, -1, -1) : 
		if (codedirname[i] == '/') : break
	if (i == 0) : 
		codedir = './'
		i = -1
	else : codedir = codedirname[:i+1]
	codename = codedirname[i+1:]
	
	''' For my mac '''
	mycommand1 = 'c++ -g -DDarwin -I/usr/local/Sophya/include/ -I/opt/local/include -I/usr/X11R6/include/ -fno-common -O -fPIC -c -I./inlib -c -o '+codedir+'a.o '+codedir+codename
	mycommand2 = 'c++ -fno-common -g -O -fPIC -bind_at_load -L/usr/local/Sophya/slb/ -lPI -lextsophya -lsophya -L/opt/local/lib -lXm -ledit -lcurses -L/usr/X11R6/lib/ -lXt -lX11 -L/usr/local/Sophya/ExtLibs/lib -lcfitsio -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -lxastro -framework Accelerate -lpthread -lm -lc -ldl -o '+codedir+codename[:-3]+'.out '+codedir+'a.o'
	
	''' For bao@bao3 '''
	CXXCOMPLIE = 'g++ -DLinux -I/ope/local/include -I/Dev/Sophya64/include -I/Dev/ExtLibs/include -I/usr/X11R6/include -Wall -Wpointer-arith -fno-common -O -g -fPIC -c '
	CXXLINK = 'g++ -Wall -Wpointer-arith -O -g -fPIC '
	SOPHYAEXTSLBLIST = '-L/Dev/Sophya64/slb -lextsophya -lsophya -lPI -L/Dev/ExtLibs/lib -lcfitsio -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -llapack -lblas -lxastro -lgfortran -lstdc++ -lpthread -lm -lc -ldl '
	baocommand1 = CXXCOMPLIE+'-o a.o '+codedir+codename
	baocommand2 = CXXLINK+SOPHYAEXTSLBLIST+'-o '+codename[:-3]+'.out a.o '
	
	if (machine == 'my') : 
		command1, command2 = mycommand1, mycommand2
	elif (machine == 'bao') : 
		command1, command2 = baocommand1, baocommand2
	else : 
		print 'Error: the second argement is wrong, must be "my" or "bao"'
		sys.exit();
	
	# Compile
	os.system('rm '+codename[:-3]+'.out')
	os.system(command1)
	os.system(command2)
	os.system('rm '+codedir+'a.o')


##################################################
##################################################
##################################################


def SelectLeastsq(array, axis=0, firstN=None, progress=False): 
	'''
	a is N-D array, for example, a.shape=4x5x6
	I want to get a[:,i,:] which is the resonable one along the second axis

	firstN:
		Return the first N index

	progress:
		Show progress or not

	return:
		which
		array[which] is the least-square
	'''
	# Move axis to 0
	array = ArrayAxis(npfmt(array), axis, 0, 'move')
	shape = npfmt(array.shape)
	# flatten()
	array = array.flatten().reshape(shape[0], shape[1:].prod())
	# Memory of array
	memarray = array.size*1e-8*800 # MB
	totmem = CheckMemory()
	memcan = (totmem - memarray)/2 # MB
	# Size of each block
	BlockSize = int(1e8/800*memcan/2)
	# Handle number of rows once
	Nonce = BlockSize/array.size  # Once row
	N = np.linspace(0, shape[0], shape[0]/Nonce+1).astype(int)
	if (N.size == 1) : N = np.concatenate([N,[shape[0]]])
	# Result
	which = []
	if (progress) : progressbar = ProgressBar('SelectLeastsq():', len(N)-1)
	for i in range(len(N)-1) : 
		if (progress) : progressbar.Progress()
		a = array[N[i]:N[i+1]]*1
		a = (a[:,None] - array[None,:])**2
		shape = npfmt(a.shape)
		a = a.flatten().reshape(shape[0], shape[1:].prod())
		a = a.mean(1)
		which.append(a)
	# Order
	which = np.concatenate(which)
	which = which + 1j*np.arange(which.size)
	which = np.sort(which).imag.astype(int)
	if (firstN is not None) : 
		if (firstN == 0) : firstN = 1
		which = which[:firstN]
	''' OLD, slow
	a = ArrayAxis(npfmt(array), axis, 0, 'move')
	b = np.zeros(len(a))
	for i in range(len(a)) : b[i] = ((a-a[i:i+1])**2).mean()
	which = Sort(b + 1j*np.arange(b.size)).imag.astype(int)
	'''
	return which


##################################################
##################################################
##################################################


def hp_EquatorialGalactic( hpmap, e2g=False, nest=False ) : 
	'''
	Convert healpix map from/to Equatorial to/from Galactic

	e2g:
		Equatorial to Galactic, True or False.

	nest:
		Input hpmap is nested or not, True or False.
	'''
	nside = hp.get_nside(hpmap)
	t, p = hp.pix2ang(2*nside, np.arange(4*12*nside**2), nest=nest)
	t = np.pi/2 - t
	p, t = EquatorialGalactic(p, t, True-e2g)
	n = hp.ang2pix(nside, np.pi/2-t, p, nest=nest)
	hpmap = hp.ud_grade(hpmap[n], nside)
	return hpmap


##################################################
##################################################
##################################################


def I2T( I, freq, Ttype ) : 
	h = Constant('Planck_constant')
	k = Constant('Boltzmann_constant')
	c = Constant('Light_speed')
	if (Ttype == 'Thermodynamic') : 
		T = h*freq/k / np.log(2*h*freq**3./I/c**2.+1)
	if (Ttype == 'Rayleigh-Jeans') : 
		T = I*c**2./2/freq**2./k
	return T


##################################################
##################################################
##################################################


def Header( hdr, scale=False ) : 
	'''
	Return the header of FITS.
	fo = pyfits.open()
	hdr = fo[n].header

	scale:
		Maybe it is SCALED DATA with header 'BSCALE', 'BZERO', 'BLANK':
			physical value = BSCALE * (storage value) + BZERO

	return : 
		[keys, values, comments]
	'''
	def DelKey( key, value, comment=None, delkey=[]) : 
		n, N = 0, len(key)
		for k in range(len(delkey)) : 
			for j in range(N) : 
				i = j - n
				if (key[i] == delkey[k]) : 
					key = key[:i] + key[i+1:]
					value = value[:i] + value[i+1:]
					if (comment is not None) : 
						comment = comment[:i] + comment[i+1:]
					n = n + 1
		if (comment is None) : return [key, value]
		else : return [key, value, comment]
	key = hdr.keys()
	value = hdr.values()
	comment = list(hdr.comments)
	if (scale is False) : 
		key, value, comment = DelKey(key, value, comment, ['BSCALE', 'BZERO', 'BLANK', 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4', 'NAXIS5', 'NAXIS6', 'NAXIS7', 'NAXIS8', 'NAXIS8'])

#		n, N = 0, len(key)
#		for j in range(N) : 
#			i = j - n
#			if (key[i] == 'BSCALE') : 
#				key = key[:i] + key[i+1:]
#				value = value[:i] + value[i+1:]
#				comment = comment[:i] + comment[i+1:]
#				n = n + 1
#			if (key[i] == 'BZERO') : 
#				key = key[:i] + key[i+1:]
#				value = value[:i] + value[i+1:]
#				comment = comment[:i] + comment[i+1:]
#				n = n + 1
#			if (key[i] == 'BLANK') : 
#				key = key[:i] + key[i+1:]
#				value = value[:i] + value[i+1:]
#				comment = comment[:i] + comment[i+1:]
#				n = n + 1
	return [key, value, comment]


##################################################
##################################################
##################################################


def Neighbor( a=None, b=None, c=None, err=0 ) : 
	'''
	Two action:
		(1) a!=None and b!=N and c!=N: Neighbor(array, lse, value, err)
		(2) a!=None and c==None: Neighbor(index, arrayshape=None)

	(1) Return neighbors index where array lse(>=<) value
		array:
			Any n-d array
		lse:
			str, '=='/'=', '!=', '<', '<=', '>', '>='
		value:
			one scale, float/int, not array
		return [index, neighbors, NumNeighbor]

	(2) 
		index:
			index or list of index: (1,2) or [(0,1), (1,2)] 
		arrahshape:
			==None: return all neighbor
			!=None: return effective
	'''
	if (a is not None and b is not None and c is not None) : 
		action = 'neival'
		array, lse, value = a*1, b, c
		err = abs(err)
	elif (a is not None) : 
		action = 'neiind'
		index, arrayshape = a, b
	else : raise Exception('All arguments == None')

	if (action == 'neival') : 
		amin, amax = array.min(), array.max()
		if (amin >= 0) : v = int(amin) - 2
		elif (amax <= 0) : v = int(amax) + 2
		else : 
			if (abs(amax) <= abs(amin)) : v = int(amax) + 2
			else : v = int(amin) - 2
		if (lse=='==' or lse=='=') : 
			tf = (array<(value-err))+(array>(value+err))
			array[tf] = v
		elif (lse == '<') : array[array>=value+err] = v
		elif (lse == '<=') : array[array>value+err] = v
		elif (lse == '>') : array[array<=value-err] = v
		elif (lse == '>=') : array[array<value-err] = v
		elif (lse == '!=') : 
			tf = ((value-err)<=array)*(array<=(value+err))
			array[tf] = v
		else : raise Exception('lse must be "==", "!=", "<", "<=", ">", ">="')
		array1 = array*1
		array1[array1!=v] = v+2
		index = npfmt(Value2Index(array1, v+2))
		if (index.size == 0) : return [[], [[]], []]
		array1 = 0 #@
		si = index.shape
		arrayshape = array.shape

	elif (action == 'neiind') : 
		index = npfmt(index, int)
		if (index.size == 0) : return []
		si = index.shape
		if (index.min()<0 and arrayshape is None) : 
			raise Exception('index.min()<0 but arrayshape is None')
		if (arrayshape is not None) : 
			arrayshape = npfmt(arrayshape, int)
			if (si[1] != len(arrayshape)) : raise Exception("index's shape doesn't fit arrayshape")
			for i in range(si[1]) : 
				index[:,i][index[:,i]<0] = index[:,i][index[:,i]<0] + arrayshape[i]
				if (index[:,i].max() >= arrayshape[i]) : raise Exception('index[:,'+str(i)+'].max()='+str(index[:,i].max())+' out of range arrayshape['+str(i)+']='+str(arrayshape[i]))

	if (len(si) == 1) : index = index[None,:]
	si = index.shape
	if (len(si) != 2) : raise Exception('index is not the effective ndarray indices')

	nef, indextuple, nr = [], [], np.zeros([si[0],],int)
	for i in range(si[0]) : 
		nei = []
		indextuple = indextuple + [tuple(index[i])]
		for j in range(si[1]) : 
			ind = index[i]*1
			if (ind[j] >= 1) : 
				ind[j] = ind[j] - 1
				ind = tuple(ind)
				if (action == 'neival') : 
					if (array[ind] == v) : nei = nei + [ind]
				else : nei = nei + [ind]
			ind = index[i]*1
			if (arrayshape is not None) : 
				if (ind[j] <= arrayshape[j]-2 ) : 
					ind[j] = ind[j] + 1
					ind = tuple(ind)
					if (action == 'neival') : 
						if (array[tuple(ind)] == v) : nei = nei + [ind]
					else : nei = nei + [ind]
			else : 
				ind[j] = ind[j] + 1
				ind = tuple(ind)
				if (action == 'neival') : 
					if (array[tuple(ind)] == v) : nei = nei + [ind]
				else : nei = nei + [ind]
		nr[i] = len(nei)
		nef = nef + [nei]
	if (action == 'neival') : return [indextuple, nef, nr]
	else : return nef


##################################################
##################################################
##################################################


def Desource( mapmtx, lse, value=None, times=3 ) : 
	'''
	mapmtx:
		map(np.ndarray) to desource. Generally 2D

	lse, value:
		mapmtx lse value will be treated as source
		For example: lse='>', value='3', them mapmtx>3 will be treated as source and desource() them

	times:
		if we don't know value, then we use the edge of mapmtx to guess value = times*edge
	'''
	mapmtx = 1*mapmtx
	if (value is None) : 
		value = Sort([mapmtx[:,:2].mean(),mapmtx[:,-2:].mean(),mapmtx[:2].mean(),mapmtx[-2:].mean()])[:2].mean()
		value = times * value
	vnan = int(mapmtx.max()) + 3
	if (lse=='==' or lse=='=') : mapmtx[mapmtx==value] = vnan
	elif (lse == '<')  : mapmtx[mapmtx <value] = vnan
	elif (lse == '<=') : mapmtx[mapmtx<=value] = vnan
	elif (lse == '>')  : mapmtx[mapmtx >value] = vnan
	elif (lse == '>=') : mapmtx[mapmtx>=value] = vnan
	elif (lse == '!=') : mapmtx[mapmtx!=value] = vnan
	else : raise Exception('lse must be "==", "!=", "<", "<=", ">", ">="')
	N = mapmtxp[mapmtx==vnan].size
	print 'Running Desource(), '+str(N)+' pixels ...'
	indeff, neigeff, num = Neighbor(mapmtx, '=', vnan)
	num = npfmt(num)
	neigall = Neighbor(indeff, mapmtx.shape)
	num = np.append(num[:,None], np.arange(num.size)[:,None], 1)
	num = Sort(num, ('col',0), True)
	# Use num, indeff, neigall
	for i in range(len(num)) : 
	#	print str(i+1)+'/'+str(len(num)), '  Desource()'
		ind = indeff[num[i,1]]
		ni = neigall[num[i,1]]
		s, ns = 0., 0
		for j in range(len(ni)) : 
			if (mapmtx[ni[j]] != vnan) : 
				s = s + mapmtx[ni[j]]
				ns = ns + 1
		mapmtx[ind] = s/ns
	return mapmtx


#def Desource( mapmtx, times=2.5, Nalpha=2, instead='interp1d' ) : 
#	'''
#	Remove strong source in mapmtx (flat 2D map)
#
#	times:
#		Remove value > times*mean
#
#	Nalpha:
#		int: Rotate mapmtx to remove again so that the result is better
#		np.ndarray or list: alpha to rotate (rad).
#
#	instead:
#		Instead the pixels of source
#		instead = 'interp1d': Use Interp1d(x,y,xnea) to instead
#		instead = value: Use this value to instead
#	'''
#	Nr, Nc = mapmtx.shape
#	ni, nj = Nr/2, Nc/2
#	mr = (mapmtx[:2].mean()+mapmtx[-2:].mean()+mapmtx[:,:2].mean()+mapmtx[:,-2:].mean())/4
#	m3 = mapmtx*0
#	if (type(Nalpha) is not np.ndarray and type(Nalpha) is not list) : 
#		alpha = np.linspace(0, 2*np.pi, Nalpha+1)[:-1]
#	else : 
#		alpha = npfmt(Nalpha)
#		Nalpha = alpha.size
#	for k in range(Nalpha) : 
#		print str(k+1)+'/'+str(Nalpha), '  ', Time()[1], '   Desource()'
#		m1, m2 = mapmtx*0+mr, mapmtx*1
#		if   (alpha[k]==0 or alpha[k]==np.pi) : m1 = mapmtx
#		elif (alpha[k]==np.pi/2 or alpha[k]==1.5*np.pi) : 
#			m1 = mapmtx.T
#		else : 
#			for i in range(Nr) : 
#				for j in range(Nc) : 
#					x, y, z = RotateCoord(alpha[k],0,[i-ni,j-nj,0])
#					x, y = int(round(x))+ni, int(round(y))+nj
#					if (x<0 or x>=Nr or y<0 or y>=Nc) : pass 
#					else : m1[i,j] = mapmtx[x,y]
#		for i in range(Nr) : 
#			a, n = m1[i], np.arange(Nc)
#			tf = (a<=times*mr)
#			if (type(instead) is str) : 
#				m1[i] = Interp1d(n[tf], a[tf], n)
#			else : m1[i][True-tf] = float(instead)
#		if   (alpha[k]==0 or alpha[k]==np.pi) : m2 = m1
#		elif (alpha[k]==np.pi/2 or alpha[k]==1.5*np.pi) : 
#			m2 = m1.T
#		else : 
#			for i in range(Nr) : 
#				for j in range(Nc) : 
#					x, y, z = RotateCoord(-alpha[k],0,[i-ni,j-nj,0])
#					x, y = int(round(x))+ni, int(round(y))+nj
#					if (x<0 or x>=Nr or y<0 or y>=Nc) : pass 
#					else : m2[i,j] = m1[x,y]
#		m3 = m3 + m2
#	m3 = m3 / Nalpha
#	return m3


##################################################
##################################################
##################################################


def ArrayGroup( array, step=1 ) : 
	'''
	Devide array into several groups basing on step

	array:
		Will first flatten() and Sort() (small to large)

	step:
		array[i]-array[i-1]<=step, we will consider array[i] and array[i-1] are in the same group

	return:
		[arraygroup, arrayrange]
	'''
	array, group = Sort(npfmt(array).flatten()), [0] # group index
	for t in range(1, len(array)) : 
		if (abs(array[t]-array[t-1])<=step) : 
			group.append(group[-1])
		else : group.append(group[-1]+1)
	group = np.array(group)
	arraygroup, arrayrange = [], []
	for t in range(group.max()+1) : 
		arraygroup.append(array[group==t])
		arrayrange.append([arraygroup[-1][0],arraygroup[-1][-1]+1])
	return [arraygroup, npfmt(arrayrange)]


##################################################
##################################################
##################################################


def PAON4Gaint( obsname, eachdir=None, freqselect=1430, Dt=4, smoothtimes=500, h24=False, mask=True, RADec=None, lonlat=PAON4LonLat(1), Dant=5, plot=True, chan2antHV=PAON4Chan2AntHV(1), ylim=None, skip=False ) : 
	'''
	obsname:
		str or list of str
		Must be the absolute path of the FITS file

	eachdir:
		Output will always save to the program file outdir=sys.argv[0][:-3]+'_output/'
		eachdir=None/False, save to outdir
		eachdir=True, save to outdir+Time()[-1]
		eachdir=str, save to outdir+eachdir
		eachdir=[str,str,str], save to outdir+eachdir[i]

	freqselect:
		int/float in MHz, not list
		Select which frequency to get the gain
		This frequency should be less RFI
		Default freqselect=1430 MHz

	Dt:
		int/float, in minute
		Interval to calculate noise's std
		Sidereal time is 23h56m04s, here set the default: 4 minute

	smoothtimes:
		noise's std we obtain here is the result of one measurement, this result may have large uncertainty.
		smoothtimes>=50, smooth the result(std), simulate several times measurements, make the result more reliable

	h24:
		True, False, scale value(float)
		Whether dayname is 24 hours observation (sidereal time).
		h24 = True: two ends should be equal after remove the Gain, valamin(difference between these two ends) will be set automatically.
		h24 = scale value (such 0.5): valamin=h24, h24=True
		An good value is h24=0.5 => valamin=0.5, it means that: 
			abs(font_end - back_end) < valamin after remove Gain
		Note that Deltime in .ppf is UTC, however one Dec circule is 1 sidereal day = 23h56m04s

	mask:
		True, False, [n1,n2] or [[n1,n2],[n3,n4],...](for multiple source), float/int
		The noise_std will become larger when incident emission becomes larger (specially on the bright source)
		=True: Just mask the target source, then you must set lonlat=(lon, lat) of the antenna array, and diameter of the antenna (Dant), then this function will calculate mask automatically.
		=False: set mask=[], means don't mask
		=[n1,n2] or [[n1,n2],[n3,n4],...], use this mask, ignore/don't use lonlat and Dant
		=float/int, set sigma of nfr() for mask=True case
		** Note that, for all dayname(if it's list), we use the same mask!

	RADec, lonlat, Dant:
		RADec:
			None, or [RA,Dec] in degree
			RADec=None: will set RADec automatically basing on the dayname
			RADec=[RA,Dec]: RA,Dec is the center of the mask pixels, use RA to get LAST, use Dec to nfr()
			If mask in [False,None], don't need to set RADec and lonlat
		lonlat:
			Location of the antenna array in degree, used to get LAST and the antenna pointting
		Dant:
			Diameter of single antenna in meter, used to nfr() 

	skip:
		True or False
		=True: if outname exist, skip this and do next
		Default =False

	# Nf=512 => 1420.2, 1420.3, 1420.4, 1420.5, 1420.6 => nfreq=349 are all 21cm

	output:
		FITS file:
			[0]: t in second
			[1]: G(t) with mask
			[2]: G(t) interp mask
			[3]: G(t) smoothing with h24=False
			[4]: G(t) smoothing with h24=True
		#	[4]: cutoffstd: std of this time interval larger than cutoffstd will be considered as RFI
	'''
	print 'PAON4Gaint() ...'
	starttime = Time()
	print 'START:', starttime[1]
	Nedge = 24*60/Dt/20
	valamin = None
	if (type(h24) not in [bool, NoneType, str]) : 
			valamin = npfmt(h24).flatten()[0]
			h24 = True
	elif (h24 is not True) : h24 = False
	#--------------------------------------------------
	# Check/Set mask
	masktf = False
	if (Type(mask) == 'NumType') : nfrsigma, masktf = mask, True
	elif (mask is True) : nfrsigma, masktf = 3, True
	if (masktf is True) : pass
	elif (mask in [False, '', None, []]) : mask = []
	else : 
		if (npfmt(mask).size == 2) : mask = [mask]
		else : 
			mask, ndel = list(mask), 0
			# Check mask and remove some bad mask
			for k in range(len(mask)) : 
				try : mask[k-ndel] = mask[k-ndel][:2]
				except : 
					mask.pop(k)
					ndel +=1
	#--------------------------------------------------
	# Check dayname whether there is that data set
	fitsname = PathExists(obsname)
	dayname = PAON4Dayname(fitsname)
	if (Type(dayname) == str) : 
		islist, dayname, fitsname = False, [dayname], [fitsname]
	else : islist = True
	# Directory to save the output fits and pictures
	if (eachdir is None) : eachdir = ''
	outdirthis = OutDir(None)
	outdir = OutDir(outdirthis+'PAON4Gaint/')
	outdir = OutDir(StrListAdd(outdir,eachdir))
	if (Type(outdir) == str) : 
		outdir = [outdir for i in range(len(dayname))]
	outnamelist = []
	#--------------------------------------------------
	for i in range(len(dayname)) : 
		# outname
		outname = outdir[i]+dayname[i]+'_auto-real-imag_noise-std_'+str(freqselect)+'MHz_'+str(Dt)+'min_st'+str(smoothtimes)+'.fits'
		outnamelist.append(outname)
		if (skip) : 
			if (os.path.exists(outname)) : continue
		# Open FITS and read header
		fo = pyfits.open(fitsname[i])
		hdr = fo[2].header
		Nt, avet, dt, tbin1, tbin2, t1, t2, Nf, avef, df, fbin1, fbin2, f1, f2, deltime, dateobs, Nv = PAON4Header(hdr)
		#---------Set mask-------------------
		if (masktf is True) : 
			if (RADec is None) : 
				RADec = CalibrationSource().RADec(dayname[i][:4])
			if (RADec is None) : mask = []
			else : 
				RAs, Decs = RADec  # degree, antenna pointting
				lon, lat = lonlat  # degree, location of PAON4 
				# Sidereal time
				LAST = SiderealTime(lon, RAs, dateobs)
				nfr1, nfr2 = nfr(freqselect, LAST, Decs*np.pi/180, Dant, deltime, Nt, nfrsigma)	
				mask = [[nfr1, nfr2]]
		#---------Set mask END---------------
		Ndt = int(round(deltime/60./Dt))
		ndt = int(Dt*60./deltime*Nt)
		#--------------------------------------------------
		nt = np.linspace(0, Nt, Ndt+1).round().astype(int)
		tlist = np.linspace(t1, t2, Ndt+1)  # sec for G-t
		tlist = Edge2Center(tlist)
		#--------------------------------------------------
		# For h24=True
		n24b = (nt[0], nt[1])
		try : n24c = int((23*60+56)*60./(deltime/Ndt))
		except : n24c = -2
		n24e = (nt[n24c], nt[n24c+1])
		#--------------------------------------------------
		# Every 10 second, calculate std, decide whether it is value or invalue(string RFI)
		Ndrop = int(deltime/10.)
		ndrop = np.linspace(0, Nt, Ndrop+1).round().astype(int)
		#--------------------------------------------------
		info = fo.info(0)
		Nvtot = 0
		for j in range(len(info)) : 
			Nvtot = Nvtot + info[j][4][1]
		Std = np.zeros([len(nt)-1,Nvtot])-1 # 8auto, 12real, 12imag
		Stdinterp = Std*0
		if (h24 is True) : Std24 = Std*0
		Stds = Std*0
		if (smoothtimes is None) : smoothtimes = 300
		nstd, rv = -1, ['0']
		cutoffstd = np.zeros([Nvtot,], np.float32)
		#--------------------------------------------------
		headback = []
		for j in range(3) :  # auto, real, imag
			Nv = fo[j].header['NAXIS2']
			rv = rv + [str(int(rv[j])+Nv)]
			for v in range(Nv) : 
				nstd = nstd + 1
				print dayname[i]+' => '+str(i+1)+'/'+str(len(dayname))+'    vis => '+str(nstd+1)+'/'+str(Nvtot)+'    '+Time(starttime[0],Time()[0])
				#------------------------------
				# nfclean always be the largest frequency because
				# (1) Least RFI, (2) Smallest amplitude
				# So, 1430MHz is one of the best.
				# Select the most clean frequency
				if (freqselect in ['', None]) : 
					freqselect = 1430
				nfclean = int(round((freqselect-f1)/df))
				#------------------------------
				a = pyfits.getdata(fitsname[i], j)[:,v,nfclean]
				#------------------------------
				# Check a every 60s, drop that 60s if its std is larger(assuming with strong RFI)
				stddrop = np.zeros([Ndrop,])
				for k in range(len(ndrop)-1) : 
					stddrop[k] = a[ndrop[k]:ndrop[k+1]].std()
				meandrop = RemoveBeyond(stddrop, 0.9).mean()
				stdamax = 2*meandrop-stddrop.min()
				cutoffstd[nstd] = stdamax
				#------------------------------
				stddrop[stddrop>stdamax] = 0
				stddrop[stddrop>0] = 1
				stddrop = stddrop.astype(int)
				nsd = np.array(Value2Index(stddrop, 0)).flatten()
				stddrop = 0 #@
				# set anam
				sign = np.sign(a.min())
				if (sign == 1) : anan = -2
				else : anan = int(2*a.min())-2
				#-------------
				for k in range(len(nsd)) : 
					a[ndrop[nsd[k]]:ndrop[nsd[k]+1]] = anan
				nsd = 0 #@
				#------------------------------
				for k in range(len(mask)) : 
					try : a[mask[k][0]:mask[k][1]] = anan-2
					except : pass
				#------------------------------
				# For h24=True
				headback.append([RemoveRFI(a[n24b[0]:n24b[1]],0,5).mean(), RemoveRFI(a[n24e[0]:n24e[1]],0,5).mean()])
				#------------------------------
				nmask = []
				for t in range(len(nt)-1) : 
					n1, n2 = nt[t], nt[t+1]
					b = a[n1:n2]
					if (b[b==anan-2].size > (n2-n1)/3) : 
						nmask.append(t)
						continue
					n11, n22 = n1-60, n2+60
					if (n11 < 0) : n11 = 0
					if (n22 > a.size) : n22 = a.size
					b1, b2 = a[n11:n1], a[n2:n22]
					b, b1, b2 = b[b!=anan-2], b1[b1!=anan-2], b2[b2!=anan-2]
					b, b1, b2 = b[b!=anan], b1[b1!=anan], b2[b2!=anan]
					if (b.size < (n2-n1)/2) : continue
					b = np.append(b1, b)
					b = np.append(b, b2)
				#	b -= Smooth(RemoveRFI(b,0,5),0,40,applr=False)
					b -= Smooth(RemoveRFI(b,0,5), 0, int(Dt*60./10), applr=False)
					Std[t,nstd] = b[len(b1):len(b1)+len(b)].std()
				#	if (b.size < 30) : 
				#		Std[t,nstd] = -1
				#		continue
				#	pdf = ProbabilityDensity(b, ndt/25)[1:] ###
				#	mu, sigma = Leastsq(pdf[0], pdf[1], 'gaussian')
				#	Std[t,nstd] = sigma
				a = b = 0 #@
				#------------------------------
				# Deal with nmask
				nmask = ArrayGroup(nmask)[1]
				# Use mean to predict the edge of mask (interp1d)
				for t in range(len(nmask)) : 
					n1, n2 = nmask[t]
					Std[n1:n2,nstd] = -1
					try : nml = nmask[t-1:t][0,1]
					except : nml = 0
					try : nmr = nmask[t+1:t+2][0,0]
					except : nmr = len(Std)
					n1l, n2r = n1-Nedge, n2+Nedge
					if (n1l < 0) : n1l = 0
					if (n2r >= len(Std)) : n2r = len(Std)
					nml = max(nml, n1l)
					nmr = min(nmr, n2r)
					v1, v2 = Std[nml:n1,nstd], Std[n2:nmr,nstd]
					v1, v2 = RemoveRFI(v1,0,3), RemoveRFI(v2,0,3)
					Std[n1,nstd] = v1[1:-1].mean()
					Std[n2-1,nstd] = v2[1:-1].mean()
				#------------------------------
				# Interp1d RFI
				nna = np.array([], int)
				if (Std[:,nstd].min() < 0) : 
					a = Std[:,nstd]*1
					a[a>=0] = 1 # a=-1 or 1
					nna, npa = Value2Index(a,-1), Value2Index(a,1)
					Std[nna,nstd] = Interp1d(npa, Std[npa,nstd], nna) # interp1d
				#------------------------------
				# Handle nna
				Std[:,nstd] = Smooth(Std[:,nstd], 0, 3)
				Stdtmp = Smooth(Std[:,nstd], 0, 30)
				Std[nna,nstd] = Stdtmp[nna]
				Stdinterp[:,nstd] = Std[:,nstd]*1
				Std24[:,nstd]     = Std[:,nstd]*1
			#	Std[nna,nstd] = -1  #@
				Std[nna,nstd] = np.nan  #@
				#------------------------------
				if (h24 is True) : 
					reachst, nrr = True, 1
					while (reachst is True) : 
						reachst, nrr = False, nrr+2
						a = RemoveRFI(Std24[:,nstd], 0, nrr)
						val0, val1 = headback[-1]
						vala0, vala1 = val0/a[0], val1/a[n24c]
						if (valamin is None) : 
							valamin = min(abs(vala0), abs(vala1))/200.
						dval0 = abs(vala0-vala1)
						dowhile = (dval0<valamin) and False or True
						n24 = 0
						while (dowhile is True) : 
							a = Smooth(a, 0, 3, applr=False) # applr=False in order to affect the ends
							n24 = n24 + 1
							vala0, vala1 = val0/a[0], val1/a[n24c]
							dval = abs(vala0-vala1)
							if (dval > dval0*1.1) : dowhile = False
							if (dval < valamin) : dowhile = False
							if (n24 >= smoothtimes+100) : 
								dowhile = False
								if (nrr < 8) : reachst = True
					a = Smooth(a, 0, 3, smoothtimes-n24)
					Std24[:,nstd] = a  # smoothed
				break
			break

		# Smooth Stdinterp to Stds
		Stds = Smooth(RemoveRFI(Stdinterp, 0, Nedge/3*2), 0, 3, smoothtimes)
		#--------------------------------------------------
		# Save to FITS
		tlist = np.float32(tlist)
		Std = np.float32(Std)             # with =0
		Stdinterp = np.float32(Stdinterp) # interp 0
		Stds = np.float32(Stds)           # h24=False, smoothed
		Std24 = np.float32(Std24)         # h24=True,  smoothed
		#---------------------
		Axis2 = '[0:'+rv[0]+']:Auto,HV; ['+rv[0]+':'+rv[1]+']:Cross,HV,real; ['+rv[1]+':'+rv[2]+']:Cross,HV,imag'
		#---------------------
		key = ['dayname', 'dateobs', 'Axis1', 'Axis2', 'Method', 'What', 'Unit']
		#---------------------
		value = [dayname[i], dateobs, 'Frequency', Axis2, 'Gain(t) is proportional to standard deviation of noise']
		#---------------------
		key04 = ['dayname', 'dateobs', 'What', 'Unit']
		value0 = [dayname[i], dateobs, 'Center time of each interval to calculate noise.std()', 'Second']
		#---------------------
		value1 = value + ['Gain(t) changing with time, original+mask', 'Arbitrary Unit']
		value2 = value + ['Gain(t) changing with time, original+interp1d', 'Arbitrary Unit']
		value3 = value + ['Gain(t) changing with time, smooth '+str(smoothtimes)+', h24=False', 'Arbitrary Unit']
		#---------------------
		value4 = [dayname[i], dateobs, 'stdcutoff, std > stdcutoff will be consider as RFI', 'Arbitrary Unit']
		#---------------------
		value5 = value + ['Gain(t) changing with time, smooth '+str(smoothtimes)+', h24=True', 'Arbitrary Unit']
	#	key = [key04, key, key, key, key04, key]
	#	value = [value0, value1, value2, value3, value4, value5]
	#	imagelist = [tlist, Std, Stdinterp, Stds, cutoffstd, Std24]
		key = [key04, key, key, key, key]
		value = [value0, value1, value2, value3, value5]
		imagelist = [tlist, Std, Stdinterp, Stds, Std24]
		Array2FitsImage(imagelist, outname, key, value)
		print starttime[1], '=>'
		print Time()[1], '   total:', Time(starttime[0], Time()[0])
	#	if (i != len(dayname)-1) : print
	Purge()
	#--------------------------------------------------
	if (plot) : 
		print 'Plotting PAON4Gaint() ...'
		_PAON4GaintPlot(outnamelist, chan2antHV, StrListAdd(outdir,'figure/'), ylim=ylim)
	#--------------------------------------------------
	if (islist == False) : outnamelist = outnamelist[0]
	print 'END:', Time()[1]
	print
	return outnamelist

#--------------------------------------------------

def _PAON4GaintPlot( gainname, chan2antHV=PAON4Chan2AntHV(1), eachdir=None, ylim=None ) : 
	'''
	gainname:
		str or list of str

	chan2antHV:
		=PAON4Chan2AntHV(0,1,...)

	eachdir:
		Whether save the figures of each dayname to each directory
		None: save all figure to ./
		str: save to outdir+eachdir
		list of str: save figures of dayname[i] to outdir+eachdir[i]

	ylim:
		='PAON4', None
		plt.ylim() of each figure.
		Same shape as dayname. For example, dayname='CygA665S1dec15', then ylim=[...]; dayname=['CygA665S1dec15'], then ylim=[[...]]; dayname=['CygA665S1dec15', CygA765S30nov15], then ylim=[[...], [...]]
		Inside ylim: 
			[0]: all autos to 1 figure
			[1:na]: ylim of each auto
			[na:]: ylim of each cross
		Default, None, is for PAON4: 
		# PAON4: 1 auto-normalized, 8 auto, 12 cross(include 34)
		ylim = [[(0.6,1),(100,200),(90,190),(70,170),(300,650),(400,720),(130,230),(400,800),(60,120),None,None,None,(80,170),None,(130,310),None,None,(140,270),None,None,None]]
	'''
	#--------------------------------------------------
	paon4pair = PAON4Pair(chan2antHV)
	strcross = paon4pair.strcross
	colorauto = plt_color(len(chan2antHV))
	#--------------------------------------------------
	# Check dayname whether there is that data set
	gainname = PathExists(gainname)
	if (gainname in ['',[]]) : raise Exception('gainname not exist')
	dayname = PAON4Dayname(gainname)
	if (Type(dayname) == str) : 
		islist, dayname, gainname = False, [dayname], [gainname]
	else : islist = True
	# Directory to save the output fits and pictures
	outdir = OutDir(eachdir)
	if (Type(outdir) == str) : 
		outdir = [outdir for i in range(len(dayname))]
	#--------------------------------------------------
	# Set ylim
	ylimstr = ''
	if (Type(ylim) == str) : 
		ylimstr = ylim.lower()
		if (ylimstr == 'paon4') : 
			ylim = [(0.6,1),(100,200),(90,190),(70,170),(350,700),(400,720),(130,230),(400,800),(60,120),None,None,None,None,None,None,None,None,None,None,None,None]
		#	ylim = [(0.6,1),(100,200),(90,190),(70,170),(350,700),(400,720),(130,230),(400,800),(60,120),None,None,None,(80,120),None,(130,250),(170,290),None,(140,270),None,None,None]
	tf = False
	try : 
		for i in range(len(ylim)) : 
			try : 
				for j in range(len(ylim[i])) : 
					try : 
						if (ylim[i][j] is not None) : tf = True
					except : continue
			except : continue
	except : pass
	if (ylim is None) : pass
	elif (ylimstr == 'paon4') : ylim = [ylim for i in range(len(dayname))]
	elif (tf == True) : pass
	else : ylim = [ylim]
	#--------------------------------------------------
	for i in range(len(dayname)) : 
		ny = 0
		plt.figure(figsize=(10,6))
		hdr = pyfits.getheader(gainname[i], 1)
		date, hour, minute, sec = dateobs2hms(hdr['DATEOBS'])
		tG = pyfits.getdata(gainname[i], 0) # sec
		th = hour+minute/60+sec/3600 + tG/3600 # o'clock
		#--------------------------------------------------
		# 8 auto 1 figure
		Gtmean = np.zeros([len(colorauto),])
		Gtmeanmax = Gtmean*0
		for j in range(len(colorauto)) : 
			Gt = pyfits.getdata(gainname[i], 3)[:,j]
			Gtmean[j] = Gt.mean()
			Gtmeanmax[j] = (Gt/Gt.mean()).max()
		Gtmeanmax = Gtmeanmax.max()*1.02
		Gtmean = Gtmean * Gtmeanmax
		for j in range(len(colorauto)) : 
			Gt = pyfits.getdata(gainname[i], 3)[:,j]
			plt.plot(th, Gt/Gtmean[j], color=colorauto[j], lw=3, label=chan2antHV[j])
		plt.legend(loc=8)
		plt.xlim(th.min(), th.max())
		plt_axes('x', 'both', [1, 0.25])
		plt.xlabel(date+", UTC (o'clock)", size=16)
		try : plt.ylim(ylim[i][ny][0], ylim[i][ny][1])
		except : pass
		plt.ylabel(r'$G(t) \cdot \sigma$ (A.U.)', size=16)
		plt.title(dayname[i]+r', $G(t) \cdot \sigma$, 8 auto', size=16)
		plt.savefig(outdir[i]+dayname[i]+'_Gt.std_auto_8_h24-False.png')
		plt.clf()
		ny +=1
		#--------------------------------------------------
		# Plot a auto, h24=False
		for j in range(len(chan2antHV)) : 
			Gt = pyfits.getdata(gainname[i], 2)[:,j]
			plt.plot(th, Gt, 'co', markersize=5, label='interp')
			Gt = pyfits.getdata(gainname[i], 1)[:,j]
			plt.plot(th, Gt, 'bo', markersize=5, label='mask')
			Gt = pyfits.getdata(gainname[i], 3)[:,j]
			plt.plot(th, Gt, 'r-', lw=3, label=chan2antHV[j])
			plt.legend(loc=8)
			plt.xlim(th.min(), th.max())
			plt_axes('x', 'both', [1, 0.25])
			plt.xlabel(date+", UTC (o'clock)", size=16)
			try : plt.ylim(ylim[i][ny][0], ylim[i][ny][1])
			except : pass
			plt.ylabel(r'$G(t) \cdot \sigma('+chan2antHV[j]+r')$ (A.U.)', size=16)
			plt.title(dayname[i]+r', $G(t) \cdot \sigma('+chan2antHV[j]+r')$', size=16)
			plt.savefig(outdir[i]+dayname[i]+'_Gt.std_auto_'+chan2antHV[j]+'_h24-False.png')
			plt.clf()
			ny += 1
		#--------------------------------------------------
		# Plot 8 auto, h24=True
		ny -= len(colorauto)
		for j in range(8) : 
			Gt = pyfits.getdata(gainname[i], 3)[:,j]
			plt.plot(th, Gt, 'r-', lw=3, label=chan2antHV[j]+', h24=False')
			Gt = pyfits.getdata(gainname[i], -1)[:,j]
			plt.plot(th, Gt, 'b-', lw=2, label=chan2antHV[j]+', h24=True')
			plt.legend(loc=8)
			plt.xlim(th.min(), th.max())
			plt_axes('x', 'both', [1, 0.25])
			plt.xlabel(date+", UTC (o'clock)", size=16)
			try : plt.ylim(ylim[i][ny][0], ylim[i][ny][1])
			except : pass
			plt.ylabel(r'$G(t) \cdot \sigma('+chan2antHV[j]+r')$ (A.U.)', size=16)
			plt.title(dayname[i]+r', $G(t) \cdot \sigma('+chan2antHV[j]+r')$', size=16)
			plt.savefig(outdir[i]+dayname[i]+'_Gt.std_auto_'+chan2antHV[j]+'_h24-True.png')
			plt.clf()
			ny += 1
		#--------------------------------------------------
		# Plot cross, cross-check, Gij*sigmaij = 2**0.5 * Gi*sigmai*Gj*sigmaj
		for j in range(len(strcross)) : 
			na1, na2 = paon4pair.Cross2Auto(strcross[j])
			Gt = pyfits.getdata(gainname[i],3)[:,8+j] # real
			plt.plot(th, Gt, 'b-', lw=3, label=strcross[j]+', real')
			Gt = pyfits.getdata(gainname[i],3)[:,8+j+len(strcross)] # imag 
			plt.plot(th, Gt, 'g-', lw=3, label=strcross[j]+', imag')
			Gt1 = pyfits.getdata(gainname[i], 3)[:,na1] # auto 
			Gt2 = pyfits.getdata(gainname[i], 3)[:,na2] # auto 
			# Cross, two antenna at the same time, observate twice, sigma of noise will decreate to 2**0.5
			Gt12 = (Gt1*Gt2/2)**0.5
			plt.plot(th, Gt12, 'r-', lw=3, label=chan2antHV[na1]+r'$\otimes$'+chan2antHV[na2]+'$/\sqrt{2}$')
			plt.legend(loc=8)
			plt.xlim(th.min(), th.max())
			plt_axes('x', 'both', [1, 0.25])
			plt.xlabel(date+", UTC (o'clock)", size=14)
			try : plt.ylim(ylim[i][ny][0], ylim[i][ny][1])
			except : pass
			plt.ylabel(r'$G(t) \cdot \sigma('+strcross[j]+r')$ (A.U.)', size=16)
			plt.title(dayname[i]+r', $G(t) \cdot \sigma('+strcross[j]+r')$', size=16)
			plt.savefig(outdir[i]+dayname[i]+'_Gt.std_cross_'+strcross[j]+'_h24-False.png')
			plt.clf()
			ny +=1


##################################################
##################################################
##################################################


def PAON4Calibration( obsname, gainname, eachdir=None, chan2antHV=None, lonlat=PAON4LonLat(), Dant=5, nfrs=[], fitbaseline=False, calivis=None, califreq=None, plot=True, plotmask=['3H-4H','3V-4V'], ylim=None, plot2d=True, gnuname=None, skip=False, plev=False ):
	'''
	obsname, gainname:
		str or list of str
		Note that they must match with each other

	eachdir:
		str
		for each dayname[i], mkdir its own dir to save the output files

	chan2antHV:
		See: def PAON4chan2antHV(n=None)
		Generally, channel 1 will be linked to 1H, etc:
			chan1 -> 1H, chan2 -> 2H, chan3 -> 3H, chan4 -< 4H
			chan5 -> 1V, chan6 -> 2V, chan7 -> 3V, chan8 -< 4V
		However, the linking can be modified by people, for example: chan2antHV = ['3H', '4H', '1H', '2H', '1V', '2V', '3V', '4V'], means:
			chan1 -> 3H, chan2 -> 4H, chan3 -> 1H, chan4 -< 2H
			chan5 -> 1V, chan6 -> 2V, chan7 -> 3V, chan8 -< 4V
		Then the auto-correlations will be (form 1 to 8):
			3H-3H, 4H-4H, 1H-1H, 2H-2H
		and the cross-correlations:
			3H-4H, 3H-1H, 3H-2H, 4H-1H, 4H-2H, 1H-2H
			1V-2V, 1V-3V, 1V-4V, 2V-3V, 2V-4V, 3V-4V
	#	meter
	#	rectangular coordinates of each antenna
	#	ewnscoord is 2D array, shape=(Nant,2), each row is (ewcoord, nscoord)

	lonlat:
		rad
		longitude and latitude of the antenna array

	Dant:
		meter
		diameter of one antenna

	nfrs:
		[nfrs1, nfrs2] or () or np.array([]) or int/float
		pixels range of the fringe of the source used to fit
		if ==[], set automatically
		if int/float, xx times of sigma

	fitbaseline:
		Fit baseline or not.
		False: use the baselines measured by other method
		True: fit the baselines using the visibilities

	calivis, califreq:
		Which visibility and freq to be fit
		=None, =[], ='' means fit all
		For example, 
			calivis = [2,4,5]
			califreq = np.arange(300,400) # freq index(int), not MHz
	'''	
	print 'PAON4Calibration() ...'
	# Coordinates of each antennas
	pos = PAON4Chan2Antpos(chan2antHV)
	paon4pair = PAON4Pair(chan2antHV)
	strcross = paon4pair.strcross
	# longitude and latitude of PAON4
	lon, lat = lonlat  # degree
	#--------------------------------------------------
	# Check dayname whether there is that data set
	fitsname = PathExists(obsname)
	gainname = PathExists(gainname)
	dayname = PAON4Dayname(fitsname)
	#----------
	if (Type(dayname) == str) : 
		islist, dayname, fitsname, gainname = False, [dayname], [fitsname], [gainname]
	else : islist = True
	if (len(fitsname) == 0) : raise Exception('obsname not exist')
	if (gainname not in [[''],[]]) : 
		caligain = True
		if (len(fitsname) != len(gainname)) : raise Exception('len(obsname) != len(gainname)')
	else : caligain = False
	print 'dayname:', dayname
	starttime = Time()
	print 'START:', starttime[1]
	#--------------------------------------------------
	# Directory to save the output fits and pictures
	outdirthis = OutDir(None)
	outdir = OutDir(outdirthis+'PAON4Calibration/')
	outdir = OutDir(StrListAdd(outdir,eachdir))
	if (Type(outdir) == str) : 
		outdir = [outdir for i in range(len(dayname))]
	outnamelist = []
	#--------------------------------------------------
	for i in range(len(fitsname)) : 
		# outname
		if (caligain) : ong = 'cali'
		else : ong = 'with'
		if (fitbaseline) : 
			outname = outdir[i]+dayname[i]+'_calibrate-imag-'+ong+'gain_S-ts-sigma-dP-Pns-Lew.fits'
		else : 
			outname = outdir[i]+dayname[i]+'_calibrate-imag-'+ong+'gain_S-ts-sigma-dP-Pns.fits'
		outnamelist.append(outname)
		if (skip) : 
			if (os.path.exists(outname)) : continue
		#--------------------------------------------------
		hdr = pyfits.getheader(fitsname[i], 2)
		Nt, avet, dt, tbin1, tbin2, t1, t2, Nf, avef, df, fbin1, fbin2, f1, f2, deltime, dateobs, Nv = PAON4Header(hdr)
		#--------------------------------------------------
		# CasA or CygA or Crab
		RAs, Decs = CalibrationSource().RADec(dayname[i][:4])
		# Sidereal time
		LAST = SiderealTime(lon, RAs, dateobs)
		freq = np.linspace(f1, f2, Nf+1)  # MHz
		freq = Edge2Center(freq)
		t = np.linspace(t1, t1+deltime, Nt)/60.  # minute
		#--------------------------------------------------
		# Before fit abs(visibility), remove Gain(t)
		if (caligain) : 
			tG = pyfits.getdata(gainname[i], 0)/60.  # minute
			Gt = pyfits.getdata(gainname[i], 3)  # smooth
		#--------------------------------------------------
		''' Calibrate which cross-correlations? => calivis '''
		ncross34 = []
		if (calivis is not None) : 
			calivis = npfmt(calivis)
			if (Type(calivis.take(0))!='NumType'): calivis=None
		if (calivis is not None) : calivis = npfmt(calivis)
		else : calivis = np.arange(Nv)
		#----------
		if (califreq is not None) : 
			califreq = npfmt(califreq)
			if (Type(califreq.take(0))!='NumType'): califreq=None
		if (califreq is not None) : califreq = npfmt(califreq)
		else : califreq = np.arange(Nf)
		#--------------------------------------------------
		Sij     = np.zeros([Nv, len(freq)])
		tsij    = Sij*0
		sigmaij = Sij*0 
		Pij     = Sij*0
		Pns     = Sij*0
		if (fitbaseline) : Lij = Sij*0
		#--------------------------------------------------
		for j in range(Nv) : 
			if (j not in calivis) : continue
			if (plev == False) : 
				print dayname[i]+' => '+str(i+1)+'/'+str(len(dayname))+'    '+strcross[j]+' => '+str(j+1)+'/'+str(Nv)+'    '+Time(starttime[0], Time()[0])
			na = paon4pair.Cross2Auto(strcross[j])
			Lew, Lns = pos[na[0]] - pos[na[1]]
			#--------------------------------------------------
			for k in range(Nf) : 
				if (k not in califreq) : continue
				if (plev != False) : 
					print dayname[i]+' => '+str(i+1)+'/'+str(len(dayname))+'    '+strcross[j]+' => '+str(k+1)+'/'+str(Nf)+'    '+Time(starttime[0], Time()[0])
				#----------------------------------------
				nfr51, nfr52 = nfr(freq[k], LAST, Decs*np.pi/180, Dant, deltime, Nt, 5)	
				imag = pyfits.getdata(fitsname[i],2)[nfr51:nfr52,j,k]*1 
				if (caligain) : 
					Gvi = Interp1d(tG, Gt[:,j+Nv+len(chan2antHV)], t)[nfr51:nfr52]
				else : Gvi = 1.
				# Remove Gt
				imag /= Gvi
				# Correct
				nfr31, nfr32 = nfr(freq[k], LAST, Decs*np.pi/180, Dant, deltime, Nt, 2.4)	
				dn1, dn2 = nfr31-nfr51, nfr52-nfr32
				if (dn1 < 200) : dn1 = 200
				imag = imag - imag[:dn1/2].mean()
				#----------------------------------------
				# Select data to fit, nfrs
				if (nfrs is None) : nfrs = []
				nfrs = npfmt(nfrs).flatten()
				if (nfrs.size == 1) : # times of sigma
					nfrs = nfr(freq[k], LAST, Decs*np.pi/180, Dant, deltime, Nt, nfrs[0])	
				elif (nfrs.size == 0) : 
					dn1 = dn2 = 0 
					nfrs1t, nfrs2t = nfr51+dn1, nfr52-dn2
					nfrs1i, nfrs2i = dn1, len(imag)-dn2
				else : 
					nfrs1t, nfrs2t = nfrs
					dn1, dn2 = nfrs1t-nfr51, nfr52-nfrs2t
					if (dn1 < 0) : dn1, nfrs1t = 0, nfr51
					if (dn2 < 0) : dn2, nfrs2t = 0, nfr52
					nfrs1i, nfrs2i = dn1, len(imag)-dn2
				#----------------------------------------
				# Smooth
				imag = Smooth(imag, 0, 60, applr=False) # wrose but fast, but enough
				#----------------------------------------
				def func( x, p ) : 
					alpha = (x-p[1])/60.*15*np.pi/180  # p[1] is the time in minute
					beta = Circ2Sph(alpha, Decs*np.pi/180)
					if (fitbaseline) : 
						y = p[0] * np.exp(-beta**2 /2 /p[2]**2) * np.sin( (2*np.pi/(300./freq[k])) * ( p[4]*np.sin(beta) - Lns*np.cos(beta)*np.sin((Decs-lat)*np.pi/180) ) + p[3] )
					else : 
						y = p[0] * np.exp(-beta**2 /2 /p[2]**2) * np.sin( (2*np.pi/(300./freq[k])) * ( Lew*np.sin(beta) - Lns*np.cos(beta)*np.sin((Decs-lat)*np.pi/180) ) + p[3] )
					return y
				#----------------------------------------
				p0 = [imag.max(), LAST[1]*180/np.pi/15*60, (300/freq[k])/2/Dant, 1.23]
				if (fitbaseline) : p0.append(Lew)
				p = FuncFit(func, t[nfrs1t:nfrs2t], imag[nfrs1i:nfrs2i], p0)[0]
				#----------------------------------------
				if (p[0] < 0) : 
					p[0] = -p[0]
					p[3] = p[3] + np.pi
				Sij[j,k]     = p[0] 
				tsij[j,k]    = p[1] # RA_source, minute
				sigmaij[j,k] = abs(p[2])
				Pij[j,k]     = p[3]%(2*np.pi)
				if (fitbaseline) : Lij[j,k] = Lew = p[4]
				Pns[j,k] = (2*np.pi/(300./freq[k]))*(-Lns*np.sin(Decs*np.pi/180-lat))
				imag = 0 #@
		#--------------------------------------------------
		Gt = tG = 0 #@
		key0 = ['dayname', 'dateobs', 'LAST', 'FitVis', 'What', 'Unit']
		value0 = [dayname[i], dateobs, LAST[1]*180/np.pi/15*60, 'Fit Imaginary part']
		value1 = ['Sij, source amplitude', 'A.U.']
		value2 = ['tsij, sorce sidereal time', 'minute']
		value3 = ['sigmaij, sigma = lambda / Dant / ksd', '']
		value4 = ['Pij, additional phase without North-South phase', 'rad']
		value5 = ['Pns, 2*pi/wavelength*(-Lns*sin(Decs-lat)), North-South phase. Usage: vis*np.exp(-j*Pns)', 'rad']
		value6 = ['Freq', 'MHz']
		key = [key0, key0, key0, key0, key0, key0]
		value = [value0+value1, value0+value2, value0+value3, value0+value4, value0+value5, value0+value6]
		imagelist = [Sij,tsij,sigmaij,Pij,Pns,freq]
		if (fitbaseline) : 
			key.append(key0)
			value7 = ['Lewij, East-West baseline', 'meter']
			value.append(value0+value7)
			imagelist.append(Lij)
		Array2FitsImage(imagelist, outname, key, value)
		print 'dayname =', dayname
		print starttime[1], '=>'
		print Time()[1], '   total:', Time(starttime[0], Time()[0])
	#--------------------------------------------------
	print 'Plotting PAON4Calibration() ...'
	print 'plot2d='+str(plot2d)
	outnamelistnpk = []
	if (plot) : 
		for i in range(len(outnamelist)) : 
			 outnamelistnpk.append( _PAON4CalibrationPlot(outnamelist[i], chan2antHV, plotmask, StrListAdd(outdir[i],'figure/'), ylim, plot2d, fitsname[i], gnuname[i]), skip )
	#--------------------------------------------------
	if (not skip) : Purge()
	if (islist == False) : 
		outnamelist = outnamelist[0]
		outnamelistnpk = outnamelistnpk[0]
	print 'END:', Time()[1]
	print
#	return [outnamelist, outnamelistnpk]
	return outnamelistnpk


##################################################
##################################################
##################################################


def _PAON4CalibrationPlot( caliphasename, chan2antHV=PAON4Chan2AntHV(1), strcrossmask=['3H-4H','3V-4V'], eachdir=None, ylim=None, plot2d=False, obsname=None, gnuname=None, skip=False ) : 
	'''
	dayname:
		Must be str, one dayname, because just on source dayname can be used to calibrated.
		If calidir=None, then dayname is the absolute/complete path of the FITS

	calidir, calitail:
		FITS result from function PAON4Calibration()
		If calidir=None, then dayname is the absolute/complete path of the FITS

	chan2antHV:
		PAON4Chan2AntHV(1,2,...)

	strcrossmask:
		str or list of str
		strcross that won't be plotted here
		Default ['3H-4H','3V-4V']

	plot2d, obsdir, obstail, gaindir, gaintail:
		plot2d:
			True or False, whether plot 2D map: 
			x-axis=theta, y-axis=freq, pixel-value=phase
		obsdir, obstail:
			Get observation data from obsdir, obstail
			If obstail=None, then obsdir is the complete/absolute path
		gaindir, gaintail:
			Calibrate G(t)
			If gaintail=None, then gaindir is the complete/absolute path

	outdir:
		None or str
		Where to save the .png and .npk
		If outdir=None, then set the outdir automatically

	ylim: 
		ylim = {'Dij':[(3,6),(0.2,0.1,'%.1f')], 'tsij':[(90,100),(2,0.2,'%.1f')], 'Sij':[(0,1),None], 'Pij':[(0,1),(0.1,0.01,'%.1f')]}
	'''
	#--------------------------------------------------
	# Check dayname whether there is that data set
	if (Type(caliphasename) != str) : raise Exception('type(caliphasename) must be str, but now is '+str(Type(caliphasename)))
	caliname = PathExists(caliphasename)
	if (caliname == '') : raise Exception('caliphasename="'+caliphasename+'" not exist')
	dayname = PAON4Dayname(caliname)
	#----------
	if (plot2d) : 
		if (Type(obsname) != str) : raise Exception('type(obsname) must be str, but now is '+str(Type(obsname)))
		obsname = PathExists(obsname)
	#--------------------------------------------------
	# Directory to save the output fits and pictures
	outdir = OutDir(eachdir)
	#--------------------------------------------------
	# Parameters
	paon4pair = PAON4Pair(chan2antHV)
	strcross = paon4pair.strcross
	RADec = CalibrationSource().RADec(dayname[:4])
	RAs, Decs = npfmt(RADec)*np.pi/180  # rad
	outname = outdir[:-7]+dayname+'_caliplot_Dij-D-ts-S-Pab-Pns.npk'
	if (skip) : 
		if (os.path.exists(outname)) : return outname
	npk = NPK(outname)
	#--------------------------------------------------
	# strcrossmask
	if (Type(strcrossmask) == str) : strcrossmask = [strcrossmask]
	strcrossmask, cmtmp = list(strcrossmask), []
	for i in range(len(strcrossmask)) : 
		strc = strcrossmask[i].split('-')
		cmtmp += [strc[0]+'-'+strc[1], strc[1]+'-'+strc[0]]
	strcrossmask = cmtmp
	# ['3H-4H'] => ['3H-4H','4H-3H']
	#--------------------------------------------------
	# Open obs header
	hdr = pyfits.getheader(obsname, 2)
	Nt, avet, dt, tbin1, tbin2, t1, t2, Nf, avef, df, fbin1, fbin2, f1, f2, deltime, dateobs, Nv = PAON4Header(hdr)
	t = np.linspace(t1, t1+deltime, Nt)/60.  # minute
	Nf1, Nf2 = Nf*2/5, Nf*4/5
	#--------------------------------------------------
	# Open calibration
	cali = pyfits.open(caliname)
	Sij, tsij, sigmaij, Pij, Pns, freq = cali[0].data, cali[1].data, cali[2].data, cali[3].data, cali[4].data, cali[5].data
	LAST = cali[0].header['LAST'] # min
	#--------------------------------------------------
	# Check sigmaij, Dij, Di
	# I have checked that kfwhm=1.18 is the best
#	kfwhm = 1.18  # k=1.03=>ksd=2.286, 1.22=>1.93 or 1.18=>2
	kfwhm = 1.13
	ksd = (8*np.log(2))**0.5 / kfwhm
	sigmaijf = RemoveRFI(sigmaij, 1, 5, 'median')
	sigmaijf = Smooth(sigmaijf, 1, 5, 10)
	Dij = 300/(freq*sigmaij)/ksd
	Dijf = 300/(freq*sigmaijf)/ksd
	sigmaij = sigmaijf = 0 #@
	nn34H, nn34V, n34 = [], [], []
	for i in range(Nv) :
		if (strcross[i] in strcrossmask) : 
			Dijf[i], n34 = -1, n34+[i]
			continue
		if   (strcross[i][1] == 'H') : nn34H.append(i)
		elif (strcross[i][1] == 'V') : nn34V.append(i)
		Di = RemoveBeyond(Dijf[i,Nf1:Nf2], 0.7)
		a = Leastsq(np.arange(Di.size),Di,'polynomial',p0index=0)[0]
		Dijf[i] = a
		plt.plot(freq, freq*0+a, 'b-', lw=3, label=('%.2f'%a)+' m')
		Di = a + (Dij[i]-a)/3
		plt.plot(freq, Di, 'ro', markersize=5, label=strcross[i])
		plt.legend()
		plt.xlim(1250, 1500)
		plt_axes('x', 'both', [25,5])
		plt.xlabel(r'$\nu \, (MHz)$', size=16)
		plt.ylabel(r'$D$('+strcross[i]+r') (A.U.)', size=16)
		try : plt.ylim(ylim['Dij'][0][0], ylim['Dij'][0][1])
		except : pass
		try : plt_axes('y', 'both', ylim['Dij'][1][:2], ylim['Dij'][1][2])
		except : pass
		plt.title(r'$D$('+strcross[i]+r')', size=16)
		plt.savefig(outdir+'D_'+strcross[i]+'.png')
	#	plt.clf()
		plt.close()
	Dijf = Dijf[:,0]  # D34=-1, 1D, independent of freq
	Dij = 0 #@
	nn34H, nn34V, n34 = npfmt(nn34H), npfmt(nn34V), npfmt(n34)
	#--------------------------------------------------
	# Solve Di from Dij
	# Which elements in tsijf/Dijf are the same polarization and be used to solve Di
	# Here we don't use 34H and 34V
	#ncross = PAON4Pair('cross')[1][:len(strcross)/2 -1] # -1 for not use 1 cross-correlation. If don't use 2 cross, then -2
	Nant = len(chan2antHV)/2
	A = np.zeros([len(Dijf)/2, Nant])
	ncross = PAON4Pair().ncross
	for i in range(len(ncross)/2) : 
		A[i, ncross[i][0]] = 1
		A[i, ncross[i][1]] = 1
	AH = A[nn34H]
	AV = A[nn34V-len(A)]
	A = 0 #@
	BH = 2*Dijf[nn34H]**2
	BV = 2*Dijf[nn34V]**2
	DH = spla.lstsq(AH, BH)[0]
	DV = spla.lstsq(AV, BV)[0]
	Di = np.append(DH, DV)**0.5  # order as chan2antHV
	nH = nV = A = BH = BV = DH = DV = 0 #@
	for i in range(len(n34)) : 
		n3, n4 = ncross[n34[i]]
		Dijf[n34[i]] = ((Di[n3]**2 + Di[n4]**2)/2)**0.5
	#--------------------------------------------------
	hdrDij = {'DAYNAME':dayname, 'WHAT':('Dij','fitted'), 'UNIT':'meter', 'KFWHM':(kfwhm, '1.03=>2.286, 1.22=>1.930, 1.18=>2'), 'BEAMSTD':'sigma_beam = kfwhm/(8ln2)^0.5 * 300/freq/D'}
	npk.Append(Dijf, 'Dij', hdrDij)
	#--------------------
	hdrDi = {'DAYNAME':dayname, 'WHAT':('Di','extracted from Ax=B'), 'UNIT':'meter', 'KFWHM':(kfwhm, '1.03=>2.286, 1.22=>1.930, 1.18=>2'), 'BEAMSTD':'sigma_beam = kfwhm/(8ln2)^0.5 * 300/freq/D'}
	npk.Append(Di, 'Di', hdrDi)
	#--------------------------------------------------
	# Fit tsij
	tsijf = tsij[:,0]*0
	for i in range(Nv) : 
		ti = RemoveBeyond(tsij[i,Nf1:Nf2], 0.7)
		a = Leastsq(np.arange(ti.size),ti,'polynomial',p0index=0)[0]
		tsijf[i] = a
		dt = a/60/24*4.
		plt.plot(freq, freq*0+a-dt, 'b-', lw=5, label=('%.2f'%(a-dt))+' min')
		plt.plot(freq, tsij[i]-dt, 'ro', markersize=5)
		plt.legend()
		plt.title(strcross[i]+', LAST='+('%.2f'%(LAST)+' min'), size=16)
		try : plt.ylim(ylim['tsij'][0][0], ylim['tsij'][0][1])
		except : pass
		try : plt_axes('y', 'both', ylim['tsij'][1][:2], ylim['tsij'][1][2])
		except : pass
		plt.ylabel(r'$t$ (min)', size=16)
		plt.xlim(1250, 1500)
		plt_axes('x', 'both', [25,5])
		plt.xlabel(r'$\nu \, (MHz)$', size=16)
		plt.savefig(outdir+'ts_'+strcross[i]+'.png')
	#	plt.clf()
		plt.close()
	tsij = 0 #@
	#--------------------------------------------------
	hdrtsij = {'DAYNAME':dayname, 'WHAT':('tsij','fitted'), 'UNIT':'minute'}
	npk.Append(tsijf, 'tsij', hdrtsij)
	#--------------------------------------------------
	# Plot freq-flux of CalibrationSource CygA
	flux = CalibrationSource().FluxDensity(dayname[:4], freq)
	plt.plot(freq, flux, 'b-', lw=3, label='CygA,  J.W.M. Baars, et al., 1964')
	plt.legend()
	plt.xlim(1250, 1500)
	plt_axes('x', 'both', [25,5])
	plt.xlabel(r'$\nu \, (MHz)$', size=16)
	plt.ylabel(r'$S_{CygA} \, (Jy)$', size=16)
	plt.title(r'$S_{CygA}$', size=16)
	plt.savefig(outdir+'S_CygA.png')
#	plt.clf()
	plt.close()
	#--------------------------------------------------
	# Fit Sij
	gnu = np.load(gnuname)
	gnu = Smooth(gnu, 1, 8, reduceshape=True)
	Sijf = RemoveRFI(Sij, 1, 7, 'median')
	Sijf = Smooth(Sijf, 1, 3, 50)
	color = plt_color(len(Sij))
	for i in range(len(Sij)) :
		if (strcross[i] in strcrossmask) : continue
		na = paon4pair.Cross2Auto(strcross[i])
		stra = strcross[i].split('-')
		labelf = r'$S_{CygA}(\nu) \cdot \sqrt{g(\nu,'+stra[0]+r')\cdot g(\nu,'+stra[1]+r')}$'
		SCygAgnu = flux * (gnu[na[0]]*gnu[na[1]])**0.5
		#--------------------------------------------
		plt.plot(freq, SCygAgnu/SCygAgnu.mean()/5, 'b-', lw=3, label=labelf)
		plt.plot(freq, Sij[i]/Sijf[i].mean()/5, color='r', marker='o', markersize=5, ls='', label='Fitting $S('+strcross[i]+')$')
		plt.legend()
		try : plt.ylim(ylim['Sij'][0][0], ylim['Sij'][0][1])
		except : pass
		try : plt_axes('y', 'both', ylim['Sij'][1][:2], ylim['Sij'][1][2])
		except : pass
		plt.xlim(1250, 1500)
		plt_axes('x', 'both', [25,5])
		plt.xlabel(r'$\nu \, (MHz)$', size=16)
		plt.ylabel('$S$('+strcross[i]+') (A.U.)', size=16)
		plt.title('$S$('+strcross[i]+')', size=16)
		plt.savefig(outdir+'S_'+strcross[i]+'.png')
	#	plt.clf()
		plt.close()
	hdrSij = {'DAYNAME':dayname, 'WHAT':('Sij','RemoveRFI(), Smooth()'), 'UNIT':'A.U.'}
	npk.Append(Sijf, 'Sij', hdrSij)
	#--------------------------------------------------
	# Fit Pij-freq
	Pijab = []
	freqf = np.linspace(1250,1500,len(freq)*10)
	freqfit = freq[Nf1:Nf2]
	ncross = paon4pair.ncross
	for i in range(Nv) : 
		if (strcross[i] in strcrossmask) : 
			triangle = paon4pair.Cross2Triangle(strcross[i])
			phasemask, Pijabmask = [], []
			for j in range(len(triangle)) : 
				s1, sc1, s2, sc2 = triangle[j]
				s1 = (s1=='+') and 1 or -1
				s2 = (s2=='+') and 1 or -1
				sc1 = Pij[paon4pair.Cross(sc1)]
				sc2 = Pij[paon4pair.Cross(sc2)]
				phasemask.append( (s1*sc1+s2*sc2) % (2*np.pi) )
				Pijabmask.append( Leastsq(freqfit, Phase2Line(phasemask[-1][Nf1:Nf2]), 'polynomial', p0index=1) )
			Pijabmask = npfmt(Pijabmask).T
			a, b = Pijabmask
			phasefit = (a[:,None]+b[:,None]*freq[None,:])#%(2*np.pi)
			a, b = Leastsq(freq, phasefit.mean(0), 'polynomial', p0index=1)
			phase01 = []
			for j in range(len(phasemask)) : 
				phasemask[j] -= (Pijabmask[0,j]+Pijabmask[1,j]*freq)
				phasemask[j] = phasemask[j] % (2*np.pi)
				p01 = phasemask[j] / (2*np.pi)
				p01[p01>0.5] = p01[p01>0.5] - 1
				phase01.append(p01)
			phase01 = RemoveRFI(npfmt(phase01),1,5,'median')
			std = (phase01[:,Nf1:Nf2]).std(1)
			n = np.arange(std.size)[std==std.min()]
			Pij[i] = (phasemask[n]+(a+b*freq)) %(2*np.pi)
			Pijab.append([a,b])
			p01 = phase01 = phasemask = 0 #@
		else : 
			Pijab.append( Leastsq(freqfit, Phase2Line(Pij[i,Nf1:Nf2]), 'polynomial', p0index=1) )
		# plot fit
		p = (Pijab[-1][0]+Pijab[-1][1]*freqf)%(2*np.pi)/(2*np.pi)
		n = []
		for j in range(len(p)-1) : 
			if (abs(p[j]-p[j+1]) > 0.7) : n.append(j+1)
		n = np.array([0] + n + [len(p)])
		for j in range(len(n)-1) : 
			plt.plot(freqf[n[j]:n[j+1]], p[n[j]:n[j+1]], 'c-', lw=3)
		# plot points
		plt.plot(freq, Pij[i]/(2*np.pi), 'ro', markersize=5)
		try : plt.ylim(ylim['Pij'][0][0], ylim['Pij'][0][1])
		except : pass
		try : plt_axes('y', 'both', ylim['Pij'][1][:2], ylim['Pij'][1][2])
		except : pass
		plt.ylabel(r'$\Delta \Phi ('+strcross[i]+') \, (2\pi)$', size=16)
		plt.xlim(1250, 1500)
		plt_axes('x', 'both', [25,5])
		plt.xlabel(r'$\nu \, (MHz)$', size=16)
		plt.title(r'$\Delta \Phi \,('+strcross[i]+')$', size=16)
		plt.savefig(outdir+'dP_'+strcross[i]+'.png')
	#	plt.clf()
		plt.close()
	freqf = n = p = 0 #@
	hdrPij = {'DAYNAME':dayname, 'WHAT':('Pijab','a,b=Pijab; Pij=a+b*freq'), 'UNIT':'rad'}
	Pijab = npfmt(Pijab).T  # a, b - Pijab
	npk.Append(Pijab, 'Pijab', hdrPij)
	#--------------------
	hdrPij = {'DAYNAME':dayname, 'WHAT':('Pns','usage: Ptot-Pns'), 'UNIT':'rad'}
	npk.Append(Pns, 'Pns', hdrPij)
	npk.Save()
	#--------------------------------------------------
	# x-axis=theta, y-axis=freq, pixel-value=phase, 2D map
	if (plot2d == True) : 
		freqselect = (1250+1500)/2.  # Plot center freq
		nfr1, nfr2 = nfr(freqselect, [0,tsijf.mean()/60.*15*np.pi/180], Decs,Di.mean(),deltime,Nt, 10)
		nfr51, nfr52 = nfr(freqselect, [0,tsijf.mean()/60.*15*np.pi/180], Decs,Di.mean(),deltime,Nt, 5)
		P2d        = np.zeros([Nv, Nf, nfr2-nfr1])
		P2d_dP     = P2d*0
		P2d_dP_Pns = P2d*0
		#-------------------------
		imagallmem = pyfits.getdata(obsname,2)[nfr1:nfr2] 
		imagall = RemoveRFI(imagallmem, 0, 7)
		del imagallmem
		imagall = Smooth(imagall, 0, 120, 5)
		realallmem = pyfits.getdata(obsname,1)[nfr1:nfr2] 
		realall = RemoveRFI(realallmem, 0, 7)
		del realallmem
		realall = Smooth(realall, 0, 120, 5)
		Purge()
		#-------------------------
		for i in range(Nv) : 
			if (strcross[i] in strcrossmask) : continue
			theta = (t[nfr1:nfr2]-tsijf[i])/60.*15*np.cos(Decs)
			for j in range(Nf) : 
				imag = imagall[:,i,j]
				real = realall[:,i,j]
				imag -= imag[nfr51-nfr1-500:nfr51-nfr1].mean()
				real -= real[nfr51-nfr1-500:nfr51-nfr1].mean()
				P2d[i,j] = (arcsincos(imag, real))%(2*np.pi)
				phase = Pij[i,j]                           #@
				phase = Pijab[0][i] + Pijab[1][i]*freq[j]  #@
				P2d_dP[i,j] = (P2d[i,j]-phase) %(2*np.pi)
				P2d_dP_Pns[i,j] = (P2d_dP[i,j]-Pns[i,j]) %(2*np.pi)
			#-------------------------
			plt.figure(figsize=(18,6))
			plt.pcolormesh(theta, freq, P2d[i]/(2*np.pi), vmin=0,vmax=1)
			plt.title(strcross[i]+r', $\Phi(\nu)$, 2D', size=16)
			plt.xlim(theta.min(), theta.max())
			plt_axes('x', 'both', [1,0.1])
			plt.xlabel(r'$\theta$ (deg)', size=16)
			plt.ylim(1250, 1500)
			plt.ylabel(r'$\nu \, (MHz)$', size=16)
			plt_axes('y', 'both', [25,5])
			plt.savefig(outdir+'Ptot_'+strcross[i]+'.png')
		#	plt.clf()
			plt.close()
			#-------------------------
			plt.figure(figsize=(18,6))
			plt.pcolormesh(theta, freq, P2d_dP[i]/(2*np.pi), vmin=0, vmax=1)
			plt.title(strcross[i]+r', $\Phi(\nu)-\Delta\Phi$, 2D', size=16)
			plt.xlim(theta.min(), theta.max())
			plt_axes('x', 'both', [1,0.1])
			plt.xlabel(r'$\theta$ (deg)', size=16)
			plt.ylim(1250, 1500)
			plt.ylabel(r'$\nu \, (MHz)$', size=16)
			plt_axes('y', 'both', [25,5])
			plt.savefig(outdir+'Ptot-dP_'+strcross[i]+'.png')
		#	plt.clf()
			plt.close()
			#-------------------------
			plt.figure(figsize=(18,6))
			plt.pcolormesh(theta, freq, P2d_dP_Pns[i]/(2*np.pi), vmin=0, vmax=1)
			plt.title(strcross[i]+r', $\Phi(\nu)-\Delta\Phi-\Phi_{ns}$, 2D', size=16)
			plt.xlim(theta.min(), theta.max())
			plt_axes('x', 'both', [1,0.1])
			plt.xlabel(r'$\theta$ (deg)', size=16)
			plt.ylim(1250, 1500)
			plt.ylabel(r'$\nu \, (MHz)$', size=16)
			plt_axes('y', 'both', [25,5])
			plt.savefig(outdir+'Ptot-dP-Pns_'+strcross[i]+'.png')
		#	plt.clf()
			plt.close()
		Purge()
	return npk.outname


##################################################
##################################################
##################################################


def freq2velo( dfreq, freq0 ) : 
	'''
	Doppler effect/shift

	dfreq, freq0 in MHz
	return dvelo in km/s
	'''
	return 3e5*dfreq/freq0


def velo2freq( dvelo, freq0 ) : 
	'''
	dvelo in km/s, freq0 in MHz
	return dfreq in MHz
	'''
	return dvelo*freq0/(3e5)


##################################################
##################################################
##################################################


def InvalidMinMax( array ) : 
	'''
	For example, array.min()=-1.23, array.max()=5.67
	Then we can set an value -2 and 6, which value in array ==-2 or ==6 are invalid
	This function is to return [-2, 6]
	'''
	amin, amax = array.min(), array.max()
	if (Invalid(amin).mask == True) :
		array = Invalid(array)
		amin, amax = array.min(), array.max()
	amin = int(amin) -1
	amax = int(amax) +1
	return np.array([amin, amax], int)


##################################################
##################################################
##################################################


def dateobs2hms( dateobs ) : 
	'''
	dateobs in PPF   : '2015-11-11T20:29:4.3'
	dateobs in ephem : '2015/11/11 20:29:4.3'
	Both formats are OK

	return ['2015-11-11', 20.0, 29.0, 4.3] for PPF
	return ['2015/11/11', 20.0, 29.0, 4.3] for ephem
	'''
	n = [len(dateobs)]
	for i in range(len(dateobs)-2, 0, -1) : 
		if (dateobs[i]==':' or dateobs[i]==' ' or dateobs[i]=='T'): 
			n = [i] + n
	hour = float(dateobs[n[0]+1:n[1]])
	minute = float(dateobs[n[1]+1:n[2]])
	sec = float(dateobs[n[2]+1:n[3]])
	date = dateobs[:n[0]]
	return [date, hour, minute, sec]


##################################################
##################################################
##################################################


def DesourceAegean( skyimage, beamobs=None, backtimes=3, beamconvnum=None, aegeandir=None ) : 
	'''
	skyimage:
		str('xxx.fits') or np.ndarray(2D)

	beamobs:
		observation beam of the sky image in degree
		if None, read in FITS header

	beamconvnum:
		Convolve the skyimage or not? Will make the image look better
		beamconvmun is None or int number
		if int number, means convolve a beam with FWHM accounts for int(number) pixels

	aegeandir:
		Where is the aegean.py
	'''
	strimagestr, tf, key, value = '', 0, '', ''
	if (type(skyimage) == str) :
		skyimagestr = skyimage 
		hdr = pyfits.getheader(skyimagestr,0)
		skyimage = pyfits.getdata(skyimagestr,0)
		shape = npfmt(skyimage.shape)
		if (shape.size!=2) : raise Exception('skyimage must be 2D')

		# Cheak keys
		kn = hdr.get('abcdefg')
		k1, k2 = hdr.get('CTYPE1'), hdr.get('CTYPE2')
		if (k1 == kn or k2 == kn) : 
			hdr['CTYPE1'], hdr['CTYPE2'], tf = 'RA', 'DEC', 1
		k1, k2 = hdr.get('CRPIX1'), hdr.get('CRPIX2')
		if (k1 == kn) : hdr['CRPIX1'], tf = 0, 1
		if (k2 == kn) : hdr['CRPIX2'], tf = 0, 1
		k1, k2 = hdr.get('CRVAL1'), hdr.get('CRVAL2')
		if (k1 == kn) : hdr['CRVAL1'], tf = 0, 1
		if (k2 == kn) : hdr['CRVAL2'], tf = 0, 1
		k1, k2 = hdr.get('CDELT1'), hdr.get('CDELT2')
		if (k1 == kn) : hdr['CDELT1'], tf = 1e-3, 1
		if (k2 == kn) : hdr['CDELT2'], tf = 1e-3, 1

		if (beamobs is not None) : 
			hdr['BMAJ'], hdr['BMIN'], hdr['BPA'] = beamobs
		else : 
			k1, k2, k3 = hdr.get('BMAJ'), hdr.get('BMIN'), hdr.get('BPA')
			if (k3 == kn) : hdr['BPA'] = 0
			if (k1 == kn) : 
				if (k2 == kn) : raise Exception('beamobs=None, missing keywords "BMAJ", "BMIN", "BPA"')
				else : hdr['BMAJ'] = hdr['BMIN']
			else :
				if (k2 == kn) : hdr['BMAJ'] = hdr['BMIN']
			beamobs = [hdr['BMAJ'], hdr['BMIN'], hdr['BPA']]

		key = ['CTYPE1', 'CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE2', 'CRPIX2', 'CRVAL2', 'CDELT2']
		value = [hdr['CTYPE1'], hdr['CRPIX1'], hdr['CRVAL1'], hdr['CDELT1'], hdr['CTYPE2'], hdr['CRPIX2'], hdr['CRVAL2'], hdr['CDELT2']]

		if (tf == 1) : 
			skyimagestr = skyimagestr[:-5]+'_aegean_'+Time()[-1]+'.fits'
			Array2FitsImage(skyimage, skyimagestr, hdr.keys(), hdr.values())

	else : 
		tf = 1
		skyimage = npfmt(skyimage)
		shape = npfmt(skyimage.shape)
		if (shape.size!=2) : raise Exception('skyimage must be 2D')
		if (beamobs is None) : raise Exception('skyimage is np.ndarray, you must set beamobs=[BMAJ, BMIN, BPA]')
		if (len(beamobs) != 3) : raise Exception('len(beamobs)=='+str(len(beamobs))+', !=3, must beamobs=[BMAJ, BMIN, BPA]')
		key = ['CTYPE1', 'CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE2', 'CRPIX2', 'CRVAL2', 'CDELT2']
		value = ['RA', 0, 0, 1e-3, 'DEC', 0, 0, 1e-3]
		skyimagestr = 'aegean_'+Time()[-1]+'.fits'
		Array2FitsImage(skyimage, skyimagestr, key, value)

	# Get aegean-background.fits
	if (aegeandir is None) : 
		aegeandir = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/aegean/'
	aegeandir = DirStr(aegeandir)

	beamstr = str(beamobs[0])+' '+str(beamobs[1])+' '+str(beamobs[2])
	cmd = 'python '+aegeandir+'aegean.py --beam '+beamstr+' --save_background '+skyimagestr
	os.system(cmd)
	print key
	print value
	print 'skyimage.shape = '+str(skyimage.shape)
	print 'beamobs =', beamobs
	
	aegeanbackground = pyfits.getdata('aegean-background.fits',0)
	os.system('rm aegean-background.fits aegean-curvature.fits aegean-rms.fits')
	if (tf == 1) : os.system('rm '+skyimagestr)

	if (beamconvnum is not None) : 
		pixnum = shape.max()
		resol = 1./pixnum
		D = 1.22*0.21/(beamconvnum*resol)
		beam = BeamModel('gaussian','dish',D,1420.4,1.,pixnum)[0]
		shape = beam.shape
		if (shape[0] < pixnum) : 
			n1 = (pixnum-shape[0])/2
			n2 = pixnum-shape[0]-n1
			beam = beam[n1:-n2]
		elif (shape[1] < pixnum) : 
			n1 = (pixnum-shape[1])/2
			n2 = pixnum-shape[1]-n1
			beam = beam[:,n1:-n2]
		skyimage = Convolve(skyimage, beam/beam.sum())

	source = skyimage - aegeanbackground*backtimes
	source[source<0] = 0
	background = skyimage - source
#	back = Convolve(back, beam/beam.sum())
	return [background, source]


##################################################
##################################################
##################################################


def RADecMtx( RAc, Decc, RAsize, Decsize, pixsize=None, pixnum=None, nsidelimit=2048 ) : 
	'''
	RAc, Decc:
		degree
		Center of the RADec matrix

	RAsize, Decsize:
		degree
		RAsize: sph angle along RA direction. RArange ~ Sph2Circ(RAsize, Dec[i])
		Decsize: sph angle along Dec direction
	
	pixsize:
		degree
		Set pixel size of 2D matrix
		Along axis=0(row) is Dec, axis=1(column) is RA
		(1) pixsize=scale, pixelsize is scale*scale degree^2
		(2) pixsize=[scale1, scale2], pixelsize is scale1*scale2 degree^2
		pixsize can be None, then pixnum works
		if pixnum and pixsize are both not None, then use the higher resolution

	pixnum:
		int or list[int] or ndarray([int])
		Set the pixel number of 2D matrix
		Along axis=0(row) is Dec, axis=1(column) is RA
		(1) pixnum=scale, matrix.shape=(scale, scale)
		(2) pixnum=[scale1, scale2], matrix.shape=(scale1, scale2)
		pixnum can be None, then pixsize works
		if pixnum and pixsize are both not None, then use the higher resolution

	nsidelimit:
		Because it needs to use healpy, then it must need nside. However, large nside will use more memory. Therefore, set nsidelimit so that the memory is enough

	Return:
		np.array([RA_matrix, Dec_matrix]) in degree
	'''
	RAc, Decc, RAsize, Decsize = 1.*npfmt(RAc)[0], 1.*npfmt(Decc)[0], 1.*npfmt(RAsize)[0], 1.*npfmt(Decsize)[0]
	# Check pixnum and pixsize
	# Convert pixsize to pixnum
	if (pixnum is None and pixsize is None) : raise Exception('pixnum and pixsize can not both be None')
	pixnum2 = None
	if (pixsize is not None) : 
		pixsize = 1.*npfmt(pixsize)
		if (pixsize.size == 1) : pixsize = np.array([pixsize[0], pixsize[0]])
		pixnum2 = np.array([RAsize/pixsize[0], Decsize/pixsize[1]]).round().astype(int)
	if (pixnum is not None and pixnum2 is None) : 
		pixnum = npfmt(pixnum).flatten()
		if (pixnum.size == 1) : pixnum = np.array([pixnum[0], pixnum[0]])
		else : pixnum = pixnum[:2]
	elif (pixnum is not None and pixnum2 is not None) : 
		pixnum = npfmt(pixnum)[:2]
		pixnum[0] = max(pixnum[0], pixnum2[0])
		pixnum[1] = max(pixnum[1], pixnum2[1])
	else : pixnum = pixnum2
	# Now we just have pixnum(2 elements)

	# nside
	pixsizeh = (np.array([RAsize,Decsize])/pixnum).min()
	n = int(np.log(41253/pixsizeh**2/12.) / np.log(4)) + 1
	nside = 2**n
	if (nsidelimit is None) : pass
	elif (nside > nsidelimit) : nside = nsidelimit
	
	# RA, Dec matrix
	Dec, RA = hp.pix2ang(nside, np.arange(12*nside**2))
	Dec, RA = 90-Dec*180/np.pi, RA*180/np.pi
	if ((RA.max()-RA.min()) > 350) : 
		RA[RA>180] = RA[RA>180] - 360

	RA = hp.cartview(RA, rot=(RAc,Decc,0), lonra=[-RAsize/2,RAsize/2], latra=[-Decsize/2,Decsize/2], return_projected_map=True, xsize=pixnum[0])
	plt.close()
	Dec = hp.cartview(Dec, rot=(RAc,Decc,0), lonra=[-RAsize/2,RAsize/2], latra=[-Decsize/2,Decsize/2], return_projected_map=True, xsize=pixnum[1])
	plt.close()
	return np.array([RA, Dec])


##################################################
##################################################
##################################################


def Healpix2Skyregion( hpmap, ordering, coordsysin, coordsysout, RAlc, Decbc, RAlsize=None, Decbsize=None, pixsize=None, pixnum=None, nsidelimit=2048 ) : 
	'''
	Select a sky region from healpix sky map

	hpmap:
		healpix map

	ordering:
		'ring' or 'nest'/'nested'

	coordsysin, coordsysout:
		'galactic' or 'equatorial'

	RAlc, Decbc:
		degree
		(1) (RAc, Decc) or (lc, bc), center of the matrix
		(2) RAlmtx and Decbmtx themselves. If this case, we don't need to set the parameters below

	RAlsize, Decbsize:
		degree
		size(sph angle) along RAl and Decb
		Note that along RAl, RAlsize is sph angle, not circ angle. Finally, the edges of RAl are different

	pixsize:
		degree
		Set pixel size of 2D matrix
		Along axis=0(row) is Decb, axis=1(column) is RAl
		(1) pixsize=scale, pixelsize is scale*scale rad^2
		(2) pixsize=[scale1, scale2], pixelsize is scale1*scale2 rad^2
		pixsize can be None, then pixnum works
		if pixnum and pixsize are both not None, then use the higher resolution

	pixnum:
		Set the pixel number of 2D matrix
		Along axis=0(row) is Decb, axis=1(column) is RAl
		(1) pixnum=scale, matrix.shape=(scale, scale)
		(2) pixnum=[scale1, scale2], matrix.shape=(scale1, scale2)
		pixnum can be None, then pixsize works
		if pixnum and pixsize are both not None, then use the higher resolution

	cutRAl:
		Because ranges of RAl are different along axis=0, this parameter is used to cut the same RAl value, set out of this RAl to be np.nan
	'''
	RAlc, Decbc = 1.*npfmt(RAlc), 1.*npfmt(Decbc)
	if (RAlc.size >= 4) : RAlmtx, Decbmtx = RAlc, Decbc
	else : 
		RAlc, Decbc, RAlsize, Decbsize = RAlc[0], Decbc[0], 1.*npfmt(RAlsize)[0], 1.*npfmt(Decbsize)[0]
		RAlmtx, Decbmtx = RADecMtx( RAlc, Decbc, RAlsize, Decbsize, pixsize, pixnum, nsidelimit)

	# From coordsysout to coordsysin
	RAlmtxout, Decbmtxout = RAlmtx.copy(), Decbmtx.copy()
	RAlmtx, Decbmtx = RAlmtx*np.pi/180, Decbmtx*np.pi/180
	coordsysin, coordsysout = coordsysin.lower(), coordsysout.lower()
	if (coordsysin != coordsysout) : 
		if (coordsysin == 'galactic') : which = 'e2g'
		else : which = 'g2e'
		RAlmtx, Decbmtx = EquatorialGalactic(RAlmtx, Decbmtx, which)

	# Check hpmap
	nside = hp.get_nside(hpmap)
	ordering = ordering.lower()
	if (ordering == 'ring') : nest = False
	else : nest = True

	npix = hp.ang2pix(nside, np.pi/2-Decbmtx, RAlmtx, nest=nest)
	RAlmtx, Decbmtx = RAlmtxout, Decbmtxout

	mtx = hpmap[npix]

#	if (cutRAl) : 
#		amax = int(RAlmtx.max())+2
#		RAlmtx[RAlmtx<RAlmtx[0,0]]  = np.nan
#		RAlmtx[RAlmtx>RAlmtx[0,-1]] = np.nan
#
#		nanmtx = RAlmtx.copy()
#		nanmtx[nanmtx<amax] = 1
#		
#		mtx = np.ma.masked_invalid(mtx * nanmtx)
	return [mtx, RAlmtx, Decbmtx]


##################################################
##################################################
##################################################


def ParserOption() : 
	'''
	Rewrite sys.argv[1:] to options=dict{}
	'''
	sysargv = sys.argv[1:]
	if (len(sysargv) == 0) : return {}
	#--------------------------------------------------
	# Modify '--a' to '-a', and '-freq' to '--freq'
	nopt = []  # which are the options, the rest are the values
	for i in range(len(sysargv)) : 
		if (sysargv[i][0] != '-') : continue
		nopt.append(i)
		k = len(sysargv[i])
		for j in range(len(sysargv[i])) :
			if (sysargv[i][j] != '-') :
				k = j
				break
		a = sysargv[i][k:]
		if (len(a) <= 1) : sysargv[i] = '-' + a
		else : sysargv[i] = '--' + a
	nopt.append(len(sysargv))
	#--------------------------------------------------
	options = {}
	for i in range(len(nopt)-1) : 
		arg = sysargv[nopt[i]+1:nopt[i+1]]
		if (len(arg) == 0) : arg = True
		elif (len(arg) == 1) :
			arg = arg[0]
			if (arg.lower() == 'true') : arg = True
			elif (arg.lower() == 'false') : arg = False
		else : arg = tuple(arg)
		options[sysargv[nopt[i]]] = arg
	#--------------------------------------------------
	# Check ',' and '='
	for k in options.keys() : 
		if (Type(options[k]) != str) : continue
		if ('=' in options[k]) : 
			options[k] = options[k].split('=')
			options[k][0] += '='
			options[k][1] = options[k][1].split(',')
		elif (',' in options[k]) : 
			options[k] = options[k].split(',')
	return options


##################################################
##################################################
##################################################


def StrcrossTriangle( strcross_i, strcross_all=None, chan2antHV=None ) : 
	'''
	Phase(1H-2H) = Phase(1H-3H) - Phase(2H-3H) and so on.
	StrcrossTriangle('1H-2H') return [1, '1H-3H', -1, '2H-3H']
	1 means '+', -1 means '-'

	strcross_i:
		can be str: '1H-2H'
		or list: ['1H-2H'], ['1H-2H', '2H-3H', '3H-4H']

	strcross_all:
		All strcross of this antenna array
		If None, get strcross_all by chan2antHV
		If not None, use it, and ignore chan2antHV

	chan2antHV:
		See PAON2Chan2AntHV
		Used to get strcross_all
		If None, set chan2antHV=PAON4Chan2AntHV(0)
	'''
	if (strcross_all is None) : 
		if (chan2antHV is None) : chan2antHV = PAON4Chan2AntHV(0)
		strcross_all = PAON4Pair('cross', chan2antHV)[0]
	isstr, strcross = False, strcross_i
	if (type(strcross) == str) : 
		strcross = [strcross]
		isstr = True
	strtri = []
	for i in range(len(strcross)) : 
		strsplit = strcross[i].split('-')
		strl, strr = [], []
		for j in range(len(strcross_all)) : 
			# strl
			if (strsplit[0] in strcross_all[j].split('-')) : 
				strl.append(strcross_all[j])
			# strr
			if (strsplit[1] in strcross_all[j].split('-')) : 
				strr.append(strcross_all[j])
		strlr = []
		for j in range(len(strl)) : 	
			strlsplit = strl[j].split('-')
			for k in range(len(strr)) : 	
				if (strl[j] == strr[k]) : continue
				strrsplit = strr[k].split('-')
				strsame = ''
				if (strlsplit[0] in strrsplit) : 
					strsame = strlsplit[0]
				elif (strlsplit[1] in strrsplit) : 
					strsame = strlsplit[1]
				if (strsame=='' or (strsame in strsplit))	: continue
				strlr.append([strl[j], strr[k]])
		strj = []
		for j in range(len(strlr)) : 	
			strl, strr = strlr[j]
			strl = strl.split('-')
			strr = strr.split('-')
			if   (strl[0] == strr[0]) : 
				AB = '+A-B'
				strlrj = strr[1] + '-' + strl[1]
				if (strlrj != strcross[i]) : AB = '+B-A'
			elif (strl[0] == strr[1]) : 
				AB = '+A+B'
				strlrj = strr[0] + '-' + strl[1]
				if (strlrj != strcross[i]) : AB = '-A-B'
			elif (strl[1] == strr[0]) : 
				AB = '+A+B'
				strlrj = strl[0] + '-' + strr[1]
				if (strlrj != strcross[i]) : AB = '-A-B'
			elif (strl[1] == strr[1]) : 
				AB = '+A-B'
				strlrj = strl[0] + '-' + strr[0]
				if (strlrj != strcross[i]) : AB = '+B-A'
			sign0, sign1 = int(AB[0]+'1'), int(AB[2]+'1')
			if (AB[1] == 'A') : 
				strj.append([sign0, strlr[j][0], sign1, strlr[j][1]])
			elif (AB[1] == 'B') : 
				strj.append([sign0, strlr[j][1], sign1, strlr[j][0]])
		strtri.append(strj)
	if (isstr) : strtri = strtri[0]
	return strtri


##################################################
##################################################
##################################################


class NPK( object ) :
	dtype = 'class:'+sys._getframe().f_code.co_name

	def __init__( self, outname='', compress=True ) : 
		''' If __init__ for self.Load(), array means dtype '''
		self.dtype     = self.dtype
		self.outname   = outname
		self.array     = []
		self.label     = []
		self.header    = []
		self.compress  = compress
		self.arraytype = []

#	def __del__( self ) : 
#		self.Save()
#		self.Clear()

	def Show( self ) : 
		cdict = self.__dict__
		keys, n = cdict.keys(), 0
		for i in range(len(keys)) : 
			if (len(keys[i]) > n) : n = len(keys[i])
		n = '%'+str(n+1)+'s'
		print self.dtype+'.show() = {'
		for i in range(len(keys)) : 
			print (n % keys[i]) + ':', cdict.values()[i]
		print '}'


	def ShowFunc( self ) : 
		classdict = self.__class__.__dict__
		classname = classdict.keys()
		n = 0
		for i in range(len(classdict)) : 
			if (classname[i-n][0] == '_') : 
				classname.pop(i-n)
				n +=1
			elif (type(classdict[classname[i-n]])!=FunctionType): 
				classname.pop(i-n)
				n +=1
			else : classname[i-n] = classname[i-n]+'()'
		print self.dtype+'.ShowFunc() :'
		for i in range(len(classname)) : print '   '+classname[i]
		return classname


	def Clear( self ) : 
		''' Will clear all memory used by the object '''
		self.__init__()


	def Append( self, array, label=None, header=None ) : 
		'''
		append array to NPK object.
		If array is ndarray/list/tuple/dict, just append the address, won't increase the memory.
		
		array:
			Must be np.ndarray/class/instance, or list of ndarray [ndarray, ...]
			if type(array)==list : convert each array[i] to ndarray
			else : convert to np.array(array), no matter what array it is
			Maybe we have the case: array=[[a,b],c,[d],[e,f],g,[h]], it will Append [a,b], c, [d], [e,f], g, [h] (same format, include []) as that in array
			It can also Append/save class/instance, it will actually save its __dict__, and save the class name in header. For example, save calss sphmap => a=sphmap(), actually save a.__dict__ to array, Type(a)='class:sphmap' to header. After load() it to d, then use: b=sphmap(), b.__dict__=d to get the saved class

		label:
			str, or list of str [str, str, ...]
			label = list(npfmt(label, str).flatten()[:len(array)])
			if len(label)<len(array), add '' to the end so that make sure len(label)==len(array)

		header:
			dict{}, or list of dict [{}, {}, ...]
			same operation as label
		'''
		# Check array
		# if class:, save self.__dict__
		arraytype = []
		if (type(array) != list) : array = [array]
		for i in range(len(array)) : 
		#	if (type(array[i]) == tuple) : array[i] = list(array[i])
			# Maybe we have the case: array=[[a,b],c,[d],[e,f],g,[h]]
			if (type(array[i]) == list) : 
				arraytypei = []
				for j in range(len(array[i])) : 
					arraytypei.append(Type(array[i][j]))
					# Convert class/instance to its .__dict__
					try : array[i][j] = array[i][j].__dict__
					except : pass
				arraytype.append(arraytypei)
			else : 
				arraytype.append(Type(array[i]))
				try : array[i] = array[i].__dict__
				except : pass
		self.arraytype += arraytype

		# Check label
		if (label is not None) : 
			label = list(npfmt(label, str).flatten()[:len(array)])
			if (len(label) < len(array)) : 
				label += ['' for i in range(len(array)-len(label))]
		else : label = ['' for i in range(len(array))]

		# Check header
		if (header is not None) : 
			if (type(header) not in [dict, list]) : 
				# Invalid header
				header = [{} for i in range(len(array))]
			if (type(header) == dict) : header = [header]
			if (len(label) < len(array)) : 
				header += [{} for i in range(len(array)-len(header))]
		else : header = [{} for i in range(len(array))]

		self.array  += array
		self.label  += label
		self.header += header
		# label='' to 'arr_'
		for i in range(len(self.label)) : 
			if (self.label[i]=='') : self.label[i] = 'arr_'+str(i)

	
	def Save( self, outname=None, array=None, label=None, header=None, compress=True ) :
		'''
		Save array to .npk

		If array is None: save self.array, self.label, ...
		If array is not None: save array, label which given here, and ignore all self.name, self.array, ...

		array, label, header:
			see self.Append()

		compress:
			True : np.savez_compressed()
			False: np.savez()
		'''
		# Check outname
		def _checkname( outname ) : 
			if (outname=='' or outname is None) : 
				outname = 'test_'+Time()[-1]
			if (outname[-4:] in ['.npk', '.npy', '.npz']) : 
				outname = outname[:-4]
			return outname

		if (array is None) : 
			outname = _checkname(self.outname)
			array  = self.array
			label  = self.label
			header = self.header
			arraytype = self.arraytype
		else : 
			outname = _checkname(outname)
			outnpk = NPK(outname+'.npk')  # can create instance
			outnpk.Append(array, label, header)
			array  = outnpk.array
			label  = outnpk.label
			header = outnpk.header
			arraytype = outnpk.arraytype
		print 'Saving --> '+outname+'.npk ...'

		# Move array, label, header to adict
		adict = {}
		for i in range(len(array)) : 
			header[i]['ARRORDER'] = i
			header[i]['DTYPE'] = arraytype[i]
			header[i]['SHAPE'] = npfmt(array[i]).shape
			header[i]['DATECREA'] = (Time()[0], 'Creation date of .NPK')
			header[i]['LABEL'] = label[i]
			adict[label[i]] = array[i]
			adict[label[i]+'_hdr'] = header[i]

		if (compress) : npsavez = np.savez_compressed
		else : npsavez = np.savez
		if (adict == {}) : adict = {'empty':[],'empty_hdr':{}}
		npsavez(outname, **adict)
		os.system('mv '+outname+'.npz '+outname+'.npk')
		outname = outname + '.npk'
		print outname+' --> Saved'
		return outname


	def Load( self, inname=None, orderlabel=None, mmap_mode=None, dtype=None, files=False ) :
		'''
		Load .npk, .npz, .npy

		orderlabel:
			Inside .npk and .npz, there may be several files, use orderlabel to select which files to be loaded
			orderlabel can be int / label(str)
			If want to know there are what labels in .npk, please use parameter => files=True

		files:
			NPK().Load(inname, files=True) will return all orderlabel name

		mmap_mode:
			If inname is .npy digital file, you can set mmap_mode to read the data in memory mapping
			Default mmap_mode=None => read the data directly
			mmap_mode: None, 'r+', 'r', 'w+', 'c'
			'r'   Open existing file for reading only
			'r+'  Open existing file for reading and writing
			'w+'  Create or overwrite existing file for reading and writing
			'c'   Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only

		dtype:
			Try to convert the load data to dtype
			Should be the same shape as orderlabel

		return [array, header]
		'''
		if (inname is None) : 
			inname = self.outname
			if (inname == '') : raise Exception('inname='+inname)
		if (dtype is None) : 
			if (self.array in Type()) : dtype = self.array

		npk = np.load(inname, mmap_mode=mmap_mode)  #@#@#@#@#@
		# Check inname
		if (inname[-4:] != '.npk') : 
			print inname+' is not .npk, return np.load("'+inname+'")'
			if (dtype is not None) : 
				try : npk = np.asarray(npk, dtype)
				except : pass
			return npk

		# Get orderlabel
		orlafs = ['' for i in range(len(npk.files)/2)]
		for nfs in npk.files : 
			if (nfs[-4:] == '_hdr') : continue
			n = npk[nfs+'_hdr'].take(0)['ARRORDER']
			orlafs[n] = (n, nfs, nfs+'_hdr')

		# Return orderlabel using files=True
		if (files) : 
			npk.close()
			return orlafs

		# Check orderlabel
		if (orderlabel is None) : norla = range(len(orlafs))
		else : 
			orderlabel = list(npfmt(orderlabel))
			norla = []
			for i in range(len(orderlabel)) : 
				for j in range(len(orlafs)) : 
					if (orderlabel[i] in orlafs[j]) : norla.append(j)
				if (len(norla) != i+1) : raise Exception('orderlabel='+str(orderlabel)+' not in '+str(orlafs))
		#	norla = list(RemoveRepetition(norla)[0])

		# Check dtype
		if (type(dtype) == tuple) : dtype = list(dtype)
		if (type(dtype) != list) :
			dtype = [dtype for i in range(len(norla))]
		if (len(dtype) < len(norla)) :
			dtype = dtype + [None for i in range(len(norla))-len(dtype)]

		# Read array and header
		array, header, ndo = [], [], []
		for i in range(len(norla)) : 
			n, nfs = orlafs[norla[i]][:2]
			if (n in ndo) : continue
			ndo.append(n)
			header.append(npk[nfs+'_hdr'].take(0)) # hdr is dict{}
			array.append(npk[nfs])
			if (array[-1].dtype == np.object) : # class.__dict__
				try : array[-1] = list(array[-1])
				except : array[-1] = array[-1].take(0)
			else : 
				try : array[-1] = np.array(npk[nfs],dtype[norla[i]])
				except : pass
		return [array, header]


##################################################
##################################################
##################################################


def ArgInit( arglist, arg, kwargs, rewhich=None ) : 
	'''
	If can accept any type, set type as 'AnyType'

	arglist: 
		Must be list of list or tuple, not dict because dict doesn't have order: [[], []] or [(), ()]
		Each [] or () for one __init__ situation
		arglist = [[('VariableName',Type()), ('x',y)], # which==0
		           [('VariableName',InitialValue')]]   # which==1
		First/outest []: for arglist
		Second []: for each __init__ case
		Third []/(): 
			for each argument of this case
			Must (only) contain two elements: (arg1, arg2)
			arg1=key: key of this argument
			arg2=Type() or initial value: initial value is better, because we can use initial value to know its Type
		For example: 
		arglist = [[('lmax'r,512), ('m':CodeType)], 
		           [('jsphskymap',InstanceType), ('share',True)],
		           [('sky','class:Sky'), ('outname','test.fits')]]

	arg, kwargs:
		from def func( *arg, **kwargs)

	rewhich:
		It may return which=[0,2,8], several cases. 
		Set rewhich to return which[rewhich]
		Default: None

	return:
		[arglist, which]
		Useage: x, y, z = arglist
	'''
	#--------------------------------------------------
	# Convert to arglist = [['key',[InitialValue,Type()]]]
	arglist, keylist = list(arglist), []
	for i in range(len(arglist)) : 
		arglist[i], keylisti = list(arglist[i]), []
		for j in range(len(arglist[i])) : 
			arglist[i][j] = list(arglist[i][j])
			keylisti.append(arglist[i][j][0])
			#----------
			a1 = 2  # a1=1 means Type, a1=2 means initial value
			if (arglist[i][j][1] in Type()) : a1 = 1 # norlam Type
			elif (Type(arglist[i][j][1]) == str) : # class
				if (arglist[i][j][1][:6] == 'class:') : a1 = 1
			if (a1 == 1) : 
				arglist[i][j][1] = ['*None*', arglist[i][j][1]]
			else : 
				arglist[i][j][1] = [arglist[i][j][1], Type(arglist[i][j][1])]
		keylist.append(tuple(keylisti))
	#--------------------------------------------------
	# Use kwargs to fill arglist base on keys
	kks, kvs = kwargs.keys(), kwargs.values()
	for i in range(len(kwargs)) : 
		for j in range(len(arglist)) : 
			if (kks[i] not in keylist[j]) : continue
			n = keylist[j].index(kks[i])
			if (arglist[j][n][1][1] in [Type(kvs[i]),'AnyType']) :
				arglist[j][n][1].append(kvs[i])
	#--------------------------------------------------
	# Use arg to fill arglist base on type
	for j in range(len(arglist)) : 
		if (len(arg) > len(arglist[j])) : continue
		for i in range(len(arg)) : 
			if (len(arglist[j][i][1]) == 3) : break
			if (arglist[j][i][1][1] in [Type(arg[i]),'AnyType']) :
				arglist[j][i][1].append(arg[i])
	#--------------------------------------------------
	# Check arglist
	which = []
	for i in range(len(arglist)) : 
		n = 0
		for j in range(len(arglist[i])) : 
			if (len(arglist[i][j][1]) != 3) : 
				if (arglist[i][j][1][0] == '*None*') : 
					n = 1
					break
				arglist[i][j][1].append(arglist[i][j][1][0])
		if (n == 0) : which.append(i)
	#--------------------------------------------------
	if (len(which) == 0) : 
		strarglist = '['
		for i in range(len(arglist)) : 
			strarglist = strarglist + str(arglist[i]) + ',\n'
		strarglist = strarglist[:-2] + ']'
		raise Exception('\n  ArgInit(), arg='+str(arg)+', kwargs='+str(kwargs)+' do not fit any case of:\n    ==> '+strarglist)
	#--------------------------------------------------
	arglistre = []
	for j in range(len(which)) : 
		arglistj = []
		for i in range(len(arglist[which[j]])) : 
			arglistj.append(arglist[which[j]][i][1][2])
		if (len(arglistj) == 1) : arglistj = arglistj[0]
		arglistre.append(arglistj)
	if (len(which) == 1) : 
		arglistre, which = arglistre[0], which[0]
	else : 
		if (rewhich not in [None,False]) : 
			arglistre, which = arglistre[rewhich], which[rewhich]
	#--------------------------------------------------
	return [arglistre, which]


#def ArgInit( arglist, arg, kwargs ) : 
#	'''
#	If can accept any type, set type as 'AnyType'
#
#	arglist: 
#		Must be list of dict: [{}, {}]
#		Each dict{} for one __init__ situation
#		Templet:
#			arglist = [{'VariableName': Type(), 'x':y},  # which==0
#			           {'VariableName': InitialValue'}]  # which==1
#			key:
#				'VariableName': str of the variable name, will be convert to variable/symbol automatically using 
#			value:
#				value of this key can be two cases: Type() or InitialValue. Note that Type(512)==CodeType
#				However, set initial value as much as possible, because when set initial value, function will ust Type() to judge the type automatically.
#
#		For example: 
#		arglist = [{'lmax'r:512), 'm':CodeType}, 
#		           {'jsphskymap':InstanceType, 'share',True},
#		           {'sky':'class:Sky', 'outname':'test.fits'}]
#
#
#
##		All level should be []
##		Lowest level ('lmax',InitialValue) can use tuple(), others level must use [], otherwise may have some problems
##		Templet: 
##		arglist = [[('VariableNane',Type()),()],        # which==0
##		           [('VariableName',InitialValue),()]]  # which==1
##			First element of lowest ()/[]: 
##				'VariableName': str of the variable name, will be convert to variable/symbol automatically using 
##			Second element of lowest ()/[]:
##				Type() or InitialValue: Note that Type(512)==CodeType
##				However, set initial value as much as possible.
##
##		For example: 
##		arglist = [[('lmax',512), ['m',CodeType]], 
##		           [['jsphskymap',InstanceType], ['share',True]]]
#
#	arg, kwargs:
#		from def func( *arg, **kwargs)
#
#	return:
#		[arglist, which]
#		Useage: x, y, z = arglist
#	'''
#	# Convert to arglist = [['key',[InitialValue,Type()]]]
#	narg = []
#	for i in range(len(arglist)) : 
#		narg.append(len(arglist[i]))
#		print arglist[i].keys()
#		exit()
#		arglist[i] = arglist[i].items()
#		print arglist[i]
#		exit()
#		for j in range(len(arglist[i])) : 
#			arglist[i][j] = list(arglist[i][j])
#			a1 = 2  # a1=1 means Type, a1=2 means initial value
#			if (arglist[i][j][1] in Type()) : a1 = 1
#			elif (type(arglist[i][j][1]) == str) : 
#				if (arglist[i][j][1][:6] == 'class:') : a1 = 1
#			if (a1 == 1) : 
#				arglist[i][j][1] = ['*None*', arglist[i][j][1]]
#			else : 
#				arglist[i][j][1] = [arglist[i][j][1], Type(arglist[i][j][1])]
#
#	for i in range(len(arglist)) :
#		print arglist[i]
#		print
#	exit()
#
#	# Use kwargs to fill arglist base on keys
#	kks, kvs = kwargs.keys(), kwargs.values()
#	for i in range(len(kwargs)) : 
#		for j in range(len(arglist)) : 
#			for k in range(len(arglist[i])) : 
#				if (kks[i] == arglist[j][k][0]) : 
#					if (arglist[j][k][1][1] in [Type(kvs[i]), 'AnyType']) : 
#						arglist[j][k][1].append(kvs[i])
#						narg[j] -= 1
#
#	# Use arg to fill arglist base on type
#	for i in range(len(arg)) : 
#		for j in range(len(arglist)) : 
#			if (len(arg) > narg[j]) : continue
#			if (arglist[j][i][1][1] in [Type(arg[i]),'AnyType']) :
#				print arglist[j][i][1]
#				arglist[j][i][1].append(arg[i])
#				print arglist[j][i][1]
#				print
#
#	exit()
#	for i in range(len(arglist)) :
#		print arglist[i]
#		print
#	exit()
#		
#
#	# Check arglist
#	which = []
#	for i in range(len(arglist)) : 
#		n = 0
#		for j in range(len(arglist[i])) : 
#			if (len(arglist[i][j][1]) != 3) : 
#				if (arglist[i][j][1][0] == '*None*') : n += 1
#		if (n == 0) : which.append(i)
#	which = [which, []]
#	for i in which[0] : 
#		for j in range(len(arglist[i])) : 
#			which[1].append(len(arglist[i][j][1]))
#			break
#	which = np.array(which).T
#	which = which[which[:,1]==(which[:,1].max())][:,0]
##	if (len(which) == 1) : which = which[0]
#	if (len(which) == 0) : 
#		strarglist = '['
#		for i in range(len(arglist)) : 
#			strarglist = strarglist + str(arglist[i]) + ',\n'
#		strarglist = strarglist[:-2] + ']'
#		raise Exception('\n  ArgInit(), arg='+str(arg)+', kwargs='+str(kwargs)+' do not fit any case of:\n    ==> '+strarglist)
#
#	arglistre = []
#	for j in range(len(which)) : 
#		arglistw, arglistj, argdictj = arglist[which[j]], [], {}
#		for i in range(len(arglistw)) : 
#			if (len(arglistw[i][1]) == 3) : 
#				arglistj.append(arglistw[i][1][2])
#			else : arglistj.append(arglistw[i][1][0])
#		if (len(arglistj) == 1) : arglistj = arglistj[0]
#		arglistre.append(arglistj)
#	if (len(which) == 1) : 
#		arglistre, which = arglistre[0], which[0]
#	return [arglistre, which]


##################################################
##################################################
##################################################


def RemoveRepetition( array, axis=0, threshold=0 ) : 
	'''
	axis: 
		Along which axis

	threshold:
		Can be one value, or ndarray with shape=array[:,axis,:].shape
	'''
	array = npfmt(array)
	arraydtype = array.dtype
	# Check axis
	if (axis not in range(len(array.shape))) : raise Exception('axis='+str(axis)+' out of '+str(len(array.shape))+'-D array')
	# Move axis to first
	array = ArrayAxis(array*1., axis, 0, 'move')
	threshold = abs(npfmt(threshold))
	if (threshold.size == 1) : 
		threshold = np.zeros(array[0].shape) + threshold[0]
	arr, narr = [], []
	for i in range(len(array)-1) : 
		if (np.isnan(array[i].take(0))) : continue
		# Find same baseline
		arri =  threshold - abs(array[i+1:] - array[i]) # take >=0
		# For nan
		arri[np.isnan(arri)] = -1
		arri[arri>=-threshold.min()/1000.] = 1  # same
		arri[arri< -threshold.min()/1000.] = 0  # different
		arri = arri.astype(int)
		for j in range(len(array.shape)-1) : arri = arri.sum(-1)
		tf = int((threshold*0+1).sum())
		arri[arri!=tf] = 0
		arri[arri==tf] = 1
		arr.append(array[i])
		narr.append(len(arri[arri==1])+1)
		arri = np.array(np.append(np.zeros([i+1,],int), arri), bool)
		array[arri] = np.nan
	if (np.isnan(array[-1].take(0)) == False) : 
		arr.append(array[-1])
		narr.append(1)
	arr, narr = np.array(arr, arraydtype), np.array(narr)
	arr = ArrayAxis(arr, 0, axis, 'move')
	return [arr, narr]


##################################################
##################################################
##################################################


def LAB3d2Equa( lab3dname, outname=None ) : 
	'''
	LAB data downloaded from the website is:
		(1) in Galactic coordinate
		(2) SCALED DATA with header 'BSCALE', 'BZERO', 'BLANK'

	This function will convert SCALED DATA to Normal data, the save that in Galactic[0] and Equatorial[1] coordinates

	return:
		outname
	'''
	#--------------------------------------------------
	hdir = pyfits.getheader(lab3dname, 0)
	# key, value, comment, correct and modify some keys
	key, value, comment = Header(hdr)
	value1 = value[:]
	comment[key.index('CDELT2')] = 'latitude increment'
	comment[key.index('FREQ0')] = 'reference frequency in Hz'
	value1[key.index('CTYPE1')] = 'ELON-CAR'
	value1[key.index('CTYPE2')] = 'ELAT-CAR'
	value1[key.index('SYSTEM')] = 'EQUATORIAL'
	#--------------------------------------------------
	# RA/l, Dec/b, velocity map
	Nl, Nb, Nv = hdr['NAXIS1'], hdr['NAXIS2'], hdr['NAXIS3']
	dl, db, dv = hdr['CDELT1'], hdr['CDELT2'], hdr['CDELT3']
	l0, b0, v0 = hdr['CRVAL1'], hdr['CRVAL2'], hdr['CRVAL3']
	RA  = l = np.linspace(l0, l0+dl*Nl, Nl+1)[:-1]
	Dec = b = np.linspace(b0, b0+db*Nb, Nb+1)[:-1]
	v = np.linspace(v0, v0+dv*Nv, Nv+1)[:-1]
	#--------------------------------------------------
	# data map
	laba = pyfits.getdata(lab3dname, 0)
#	amin = -11
	#amin = InvalidMinMax(laba)[0]
	labb = np.zeros(laba.shape, np.float32)# + amin
	#--------------------------------------------------
	n = -1
	for j in range(len(Dec)) : 
		n +=1
		Progress(n, len(Dec), 10)
		li, bi = npfmt(EquatorialGalactic(RA*np.pi/180, Dec[j]*np.pi/180,'e2g')) *180/np.pi
		li[li>180] = li[li>180] - 360
		nl = ((li-l0)/dl).round().astype(int)
		nb = ((bi-b0)/db).round().astype(int)
		nl[nl<0] = 0
		nl[nl>=Nl] = Nl-1
		nb[nb<0] = 0
		nb[nb>=Nb] = Nb-1
		labb[:,j,:] = laba[:,nb,nl]
	#--------------------------------------------------
#	# Fill all a
#	n = -1
#	for j in range(len(b)) : 
#		n = n + 1
#		Progress(n, len(b)+len(Dec), 10)
#		bi = b[j]*np.pi/180 # -90->0->90
#		for i in range(len(l)) : 
#			li = l[i]*np.pi/180 # 180->0->-180
#			# lb to RADec
#			RAi, Deci = EquatorialGalactic(li,bi,'g2e')[:,0] *180/np.pi
#			if (RAi > 180) : RAi = RAi - 360
#			nRA = int(round((RAi - l0) / dl))
#			nDec = int(round((Deci - b0) / db))
#			if (nRA < 0) : nRA = 0
#			if (nRA >= Nl) : nRA = Nl-1
#			if (nDec < 0) : nDec = 0
#			if (nDec >= Nb) : nDec = Nb-1
#			labb[:,nDec,nRA] = laba[:,j,i]
#	
#	# Fill the blank
#	for j in range(len(Dec)) : 
#		n = n + 1
#		Progress(n, len(b)+len(Dec), 10)
#		Decj = Dec[j]*np.pi/180
#		for i in range(len(RA)) : 
#			if (labb[0,j,i] != amin) : continue
#			RAi = RA[i]*np.pi/180
#			# RADec to lb
#			li, bi = EquatorialGalactic(RAi, Decj).flatten() *180/np.pi
#			if (li > 180) : li = li - 360
#			# le, be to nle, nbe
#			nli = int(round((li - l0) / dl))
#			nbi = int(round((bi - b0) / db))
#			labb[:,j,i] = laba[:,nbi,nli]
	#--------------------------------------------------
	# Save to FITS
	if (outname in [None, '']) : 
		ln1, ln2, ln3 = lab3dname.lower().split('-'), lab3dname.lower().split('_'), lab3dname.lower().split('.')
		if ('3d' in ln1+ln2+ln3) : 
			for i in range(len(lab3dname)-1) : 
				if (lab3dname[i:i+2].lower() == '3d') : break
			lab3dname = lab3dname[:i] + lab3dname[i+2:]
		for i in range(1, len(lab3dname)) : 
			if (lab3dname[-i] == '.') : break
		lab3dname = lab3dname[:-i]
		outname = lab3dname+'-3D_Galactic-Equatorial.fits'
	key = [key, key]
	value = [value, value1]
	comment = [comment, comment]
	Array2FitsImage([laba,labb], outname, key, value, comment)
	Purge()
	return outname


def LAB3d2Healpix( lab3dname, freqc, freqresol, nside=256, coordsys='Equatorial', outname=None, save=False ) : 
	'''
	lab3dname:
		Name of FITS file generated by function LAB3d2Equa()

	freqc, freqresol:
		Scale or ndarray (1D)
		Frequency of output healpix map
		Will average [freqc-freqresol/2:freqc+freqresol/2]

	nside:
		nside of the output healpix

	coordsys:
		'Galactic', 'Equatorial'
		Output coordinate system

	outname:
		outname of the output healpix FITS
		If None: set it automatically
	'''
	# Check nside
	n = np.log(nside)/np.log(2)
	if (n != int(n)) : raise Exception('nside != 2^n')
	
	# Check coordsys
	if (coordsys.lower() == 'galactic') : nhdu = 0
	elif (coordsys.lower() == 'equatorial') : nhdu = 1
	else : raise Exception('coordsys='+coordsys+', must be "Galactic" or "Equatorial"')
	
	# Get header
	fo = pyfits.open(lab3dname)
	hdr = fo[nhdu].header
	keylab, valuelab, commentlab = Header(fo[0].header)
	
	# RA/l, Dec/b, velocity map
	NRAl, NDecb, Nv = hdr['NAXIS1'], hdr['NAXIS2'], hdr['NAXIS3']
	dRAl, dDecb, dv = hdr['CDELT1'], hdr['CDELT2'], hdr['CDELT3']/1e3
	RAl0, Decb0, v0 = hdr['CRVAL1'], hdr['CRVAL2'], hdr['CRVAL3']/1e3
	freq0 = hdr['FREQ0']/1e6 # MHz
	
	# frequency range of LAB
	vlist = np.arange(v0, v0+Nv*dv, dv) 
	dfreq = velo2freq(dv, freq0)
	nfreq0 = np.arange(Nv)[abs(vlist)<dv/10.][0]
	freqmin = freq0 + velo2freq(vlist[0], freq0)
	freqmax = freq0 + velo2freq(vlist[-1], freq0)
	
	# Check freqc and freqresol
	freqc, freqresol = list(npfmt(freqc)), list(npfmt(freqresol))
	# freqresol should be 2D
	if (len(freqc) != len(freqresol)) : raise Exception('len(freqc) != len(freqresol)')
	for i in range(len(freqresol)) : 
		freqresol[i] = npfmt(freqresol[i])[:1]
		if (freqc[i] == None) : freqc = freq0
	freqc, freqresol = npfmt(freqc).flatten(), npfmt(freqresol).flatten()
	freq1 = freqc - freqresol/2.
	freq2 = freqc + freqresol/2.
	if (freq1[freq1<freqmin].size!=0 or freq2[freq2>freqmax].size!=0) : raise Exception('Frequency band of LAB is ['+('%.3f' % freqmin)+', '+('%.3f' % freqmax)+']\n'+'freqc='+str(freqc)+'\n'+'freqresol='+str(freqresol)+'\n'+'out of this band')
	nfreq1 = nfreq0+((freq1-freq0)/dfreq).round().astype(int)
	nfreq2 = nfreq0+((freq2-freq0)/dfreq).round().astype(int)+1  # because we need to get freq2 itself!
	freq1 = freq2 = 0 #@
	
	# theta, phi matrix -> index matrix
	theta, phi = hp.pix2ang(nside, np.arange(12*nside**2))
	theta, phi = 90-theta*180/np.pi, phi*180/np.pi
	phi[phi>180] = phi[phi>180] - 360
	ntheta = ((theta - Decb0) / dDecb).round().astype(np.int32)
	theta = 0 #@
	ntheta[ntheta<0] = 0
	ntheta[ntheta>=NDecb] = NDecb-1
	nphi = ((phi - RAl0) / dRAl).round().astype(np.int32)
	phi = 0 #@
	nphi[nphi<0] = 0
	nphi[nphi>=NRAl] = NRAl-1
	nthetaphi = np.append(ntheta[:,None], nphi[:,None], 1)
	ntheta = nphi = 0 #@
	index = IndexConvert(nthetaphi, (NDecb, NRAl))  #@#@
	nthetaphi = 0 #@
	
	# healpix map
	hpmap = []
	for i in range(len(nfreq1)) : 
		lab = pyfits.getdata(lab3dname, nhdu)[nfreq1[i]:nfreq2[i]].mean(0)
		hpmap.append(np.float32(lab.flatten()[index]))
	
	# Save FITS
	if (save) : 
		# outname
		if (outname is None) : outname = 'labh_healpix_'+coordsys+'.fits'
		elif (outname[-5:].lower() != '.fits') : outname += '.fits'
		# Header
		keyhp = ['EXTNAME', 'CREADATE', 'FREQ', 'UNIT', 'PIXTYPE', 'ORDERING', 'NSIDE', 'COLUMN', 'BLANLINA', 'LABHDR', 'BLANLINB']
		valuehp = [Time()[0], 'K', 'HEALPIX', 'RING', nside, 'Each velocities', '********', '*** Below is the header of LAB data ***', '********']
		commenthp = ['Extension name', 'Creaton date', 'MHz', 'Brightness temperature', 'HEALPIX pixelisation', 'Pixel ordering scheme, RING or NESTED', 'Resolution parameter for HEALPIX', '', '', '', '']
		keyi = keyhp + keylab
		valuei = valuehp + valuelab
		commenti = commenthp + commentlab
		key, value, comment = [], [], []
		for i in range(len(freqc)) : 
			key = key + [keyi]
			comment = comment + [commenti]
			value = value + [[str(freqc[i])+'MHz'] + valuei[0:1] + [freqc[i]] + valuei[1:]]
		# columntag, healpix map saved to table
		columntag = []
		for i in range(len(freqc)) : columntag.append(str(freqc[i])+'MHz')
		# Save
		Array2FitsTable(hpmap, outname, columntag, None, key, value, comment)
		return [hpmap, outname]
	return hpmap


##################################################
##################################################
##################################################


def PAON4CaliAuto( freq, obsname, gainname, Dec=None, lab3dname=None, chan2antHV=PAON4Chan2AntHV(1), rmRFI=None, outdir=None, eachdir=False, Deff=4.6, nside=256, plot=True, ylim=None, skip=False, plev=False ) : 
	'''
	freq:
		MHz, must be one value/scale, not ndarray
		Which frequency of auto to be calibrated

	dayname: 
		Here dayname must be one str/file, not list
		dayname can be just dayname, or the complete/absolute path of the FITS
		If dayname is the complete/absolute path of the FITS, then set obsdir and obstail to be None or ''

	obsdir, obstail:
		Observation data in FITS
		If dayname is the complete/absolute path of the FITS, then set obsdir and obstail to be None or ''

	gaindir, gaintail:
		File from PAON4Gaint(), G(t)
		If gaintail=None or '', means that gaindir is the complete/absolute path of the G(t) FITS file => gainname=gaindir, we don't use dayname at this time

	Dec:
		In degree, declination of the antenna pointting
		If None, use dayname to guess the Dec. If fail to guess the Dec, will raise error

	chan2antHV:
		Channel to antenna HV
		chan2antHV = PAON4Chan2AntHV(1) / PAON4Chan2AntHV(4)

	rmRFI:
		First this function will use two method to find and remove RFI automatically (RemoveRFI())
		However, the result may be not good enough
		Then set rmRFI by hand, means this range of pixels are RFI, and they will be removed
		rmRFI must be 2D:
			[[20,30]], [(20,30)] means pixels 20-30 are RFI. Althought just one range(20,30), you must add [] outside
			[(20,30), [50,60], ...]

	outdir:
		Which directory to save the output .npk and .png
		None means tring to save to sys.argv[0][:-3]+'_output/'
		If it can mkdir the directory above, for example, use this function in ipython, then will set outdir='./'

	lab3dname:
		Complete/Absolute path of lab-3D_Galactic_Equatorial.fits
		If freq=1420.4MHz, must use LAB+GSM to calibrate.
		Beyond 21cm, use GSM is enough
		lab3dname=None: don't use LAB, just use GSM

	Deff:
		meter. Effective diameter of antenna for calculating FWHM

	nside:
		We use LAB and GSM to calibrate the amplitude of auto. They will reduce the resolution to as PAON4, use hp.smoothing(). So we need to set nside of the healpix map

	plot, ylim:
		plot: True, False, plot the figure or not
		ylim: If plot, you can set ylim to control the plt.ylim() of each figure. ylim is a dict{}, ylim.keys()=['autoAU','LAB','GSM','Tsky','autoKo','autoKf']
		ylim for PAON4 21cm observation is
			ylim = {'autoAU':None, 'LAB':[(0,50),(5,1)], 'GSM':[(0,50),(5,1)], 'Tsky':[(0,50),(5,1)], 'autoKo':None, 'autoKf':[(-10,50),(5,1)]}
		If you don't want to use plt.ylim():
			(1) Set the value to be None
			(2) Delete this key
			Effects of two methods above are the same
	'''
	print 'PAON4CaliAuto() ...'
	print 'START:', Time()[1]
	#--------------------------------------------------
	if (ylim is None) : ylim = {}
	#--------------------------------------------------
	# Check dayname whether there is that data set
	obsname = PathExists(obsname)
	gainname = PathExists(gainname)
	dayname = PAON4Dayname(obsname)
	#--------------------------------------------------
	# Directory to save the output fits and pictures
	outdirthis = OutDir(None)
	outdir = OutDir(outdirthis+'PAON4CaliAuto/')
	outdir = OutDir(StrListAdd(outdir,eachdir))
	OutDir(StrListAdd(outdir,'figure/'))
	#--------------------------------------------------
	# Smooth to PAON4 single dish FWHM
	freqselect = freq
	FWHM = 1.22*300/freqselect/Deff  # 1.22 is best, I checked
	outname = outdir+dayname+'_auto_'+str(freqselect)+'MHz_calibrated_K.npk'
	if (skip) : 
		if (os.path.exists(outname)) : 
			print 'END:', Time()[1]
			print
			return outname
	npk = NPK(outname)
	#--------------------------------------------------
	# Check lab3dname
	if (lab3dname is not None) : 
		lab3dname = ShellCmd('ls '+lab3dname)
		if (len(lab3dname) == 0) : exit()
		lab3dname = lab3dname[0]
	#--------------------------------------------------
	Decs = PAON4Dec(dayname, Dec)
	#--------------------------------------------------
	if (chan2antHV is None) : chan2antHV = PAON4Chan2AntHV(1)
	paon4pair = PAON4Pair(chan2antHV)
	strauto = paon4pair.strauto
	color = plt_color(len(chan2antHV))
	#--------------------------------------------------
	if (plev != False) : 
		print 'obsname:     '+obsname
		print 'gainname:    '+gainname
		print 'dayname:     '+dayname
		print 'outdir:      '+outdir
		print 'npk.outname: '+npk.outname
		print 'lab3dname:   '+lab3dname
		print 'Dec:         '+str(round(Decs,4))+'deg'
		print 'chan2antHV:  '+str(chan2antHV)
	#--------------------------------------------------
	# Open obs
	hdr = pyfits.getheader(obsname, 2)
	Nt, avet, dt, tbin1, tbin2, t1, t2, Nf, avef, df, fbin1, fbin2, f1, f2, deltime, dateobs, Nv = PAON4Header(hdr)
	t = np.linspace(t1, t1+deltime, Nt)/60.  # UTC, minute
	# UTC 24hours, however the sidereal time should be 23h56m4s
	#--------------------------------------------------
	# Frequency range of PAON4-21cm with Nfreq=512, n21cm=349 for 512
	# -1 is because of the miss matching the freq bins in LAB
	# PAON4-21cm=>-0, LAB-21cm=>-1
	nfreqselect = int(round(Nf/250.*(freqselect-1250))) 
	#--------------------------------------------------
	hdrnpk = {'FREQ':(freqselect,'MHz'), 'FREQBIN':(round(df,4),'MHz'), 'Dec':(round(Decs,4),'degree'), 'FWHM':(round(FWHM*180/np.pi,4),'degree')}
	#--------------------------------------------------
	# Open gain
	fo = pyfits.open(gainname)
	tG = fo[0].data/60.  # minute, UTC
#	Gt = fo[3].data  # h24 False, bad result
	Gt = fo[-1].data  # h24 True
	#--------------------------------------------------
	# Get 8 auto, cali gain
	automem = pyfits.getdata(obsname, 0)[:,:,nfreqselect]
	auto = []
	# Remove G(t)
	if (plev != False) : print 'Calibrating G(t) ...'
	for i in range(len(chan2antHV)) : 
		Ga = Interp1d(tG, Gt[:,i], t)
		ai = automem[:,i]*1
		auto.append( ai/Ga )
	del automem, Ga, Gt, tG, ai
	auto = npfmt(auto).T
	Purge()
	#--------------------------------------------------
	if (plev != False) : print 'Removing RFI ...'
	# RemoveRFI automatically
	auto = RemoveRFI(auto, 0, 10, 'median')
#	# RemoveRFI by hand, small RFI
	auto = RemoveRFI(auto, 0, rmRFI, 'manual')
	#--------------------------------------------------
	if (plot) : 
		# Plot, Just auto/G(t), with noise
		for i in range(len(chan2antHV)) : 
			plt.plot(t/60, auto[:,i], color=color[i], label=strauto[i])
		plt.legend()
		plt.xlabel(r'UTC (hour)', size=16)
		plt.xlim(0, 24)
		plt_axes('x', 'both', [1, 0.5])
		plt.ylabel('(A.U.)', size=16)
		if ('autoAU' in ylim.keys()) : stry = 'autoAU'
		if ('AUTOAU' in ylim.keys()) : stry = 'AUTOAU'
		if (stry != '') : 
			if (ylim[stry] is not None) : 
				plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				plt_axes('y', 'both', ylim[stry][1])
		plt.title(dayname+', 8 autos calibrated $G(t)$', size=16)
		plt.savefig(outdir+'figure/'+dayname+'_auto8_'+str(freqselect)+'MHz_caligain.png')
		plt.clf()
	#--------------------------------------------------
	# Sidereal time of auto, and RA
	t = t * 24*60/(23*60+56+4./60)  # Sidereal time, minute
	# Where is the Sidereal time=24hour
	dt = abs(t-24*60)
	n24 = int(round(np.arange(t.size)[dt==dt.min()].mean()))
	del dt, t
	# n24+1, the last pixel is 24h=0h, same as the first pixel
	auto, dt = auto[:n24+1], 0
	t = np.linspace(0, 24*60, len(auto)) # Sidereal time, minute, use it to handle 24hour circle miss matching as RFI
	RAlist = np.linspace(0, 360, len(auto)) # degree
	#--------------------------------------------------
	if (lab3dname is not None) : 
		if (plev != False) : print 'Using LAB data ...'
		# Frequency range for taking LAB slice
		# Frequency range of PAON4-21cm with Nfreq=512, n21cm=349 for 512
		# -1 is because of miss matching the freq bins
		freqPAON4 = np.linspace(1250, 1500, 4096)
		#freqPAON4 = Edge2Center(np.linspace(1250, 1500, 4097))
		nsmooth = 4096/Nf
		nf41, nf42, nf43 = (nfreqselect-1)*nsmooth, (nfreqselect+0)*nsmooth, (nfreqselect+1)*nsmooth
		# For PAON4 freq resolution freqbin=8, should be nf41:nf42
		# Test: 
		#   nf4e=nf42-1 => all bad
		#   nf41  nf42  Tsys  left   right
		#    +1    +0    113  high  perfect
		#    +0    +0    108   OK   perfect
		#    -1    +0     99   OK   perfect (best)
		#    +1    +1    126   OK     OK
		#    +0    +1    120   OK     OK
		#    -1    +1    111   OK     OK
		nf4b, nf4e = nf41-1, nf42+0
		freqPAON4 = freqPAON4[nf4b:nf4e]
		freqc = freqPAON4.mean()
		freqbin = freqPAON4[-1] - freqPAON4[0]
		#--------------------------------------------------
		# Get LAB-healpix
		lab = LAB3d2Healpix(lab3dname, freqc, freqbin, nside=nside)[0]
		# Smooth to PAON4 single dish FWHM
		lab = hp.smoothing(lab, FWHM, verbose=False)
		# Sky region of dayname
		n = hp.ang2pix(nside, np.pi/2-Decs*np.pi/180, RAlist*np.pi/180)
		# Take slice
		lab, n = lab[n], 0
		hdrnpktmp = hdrnpk.copy()
		hdrnpktmp.update({'WHAT':'LAB', 'UNIT':'K'})
		npk.Append(lab, 'LAB', hdrnpktmp)
		#--------------------------------------------------
		# Plot lab
		if (plot) : 
			label = 'LAB data\n'+r'$\nu='+str(freqselect)+r'$MHz, $\Delta\nu='+('%.3f' % (250./Nf))+r'$MHz'+'\n'+'Dec$='+('%.2f' % Decs)+'$deg\n'+'FWHM$='+('%.3f' % (FWHM*180/np.pi))+'$deg'
			plt.plot(RAlist, lab, 'b-', lw=2, label=label)
			plt.legend(loc=2)
			plt.xlabel('RA (deg)', size=16)
			plt.xlim(0, 360)
			plt_axes('x', 'both', [30,5])
			plt.ylabel(r'$T_b$ (K)', size=16)
			stry = ''
			if ('lab' in ylim.keys()) : stry = 'lab'
			if ('LAB' in ylim.keys()) : stry = 'LAB'
			if (stry != '') : 
				if (ylim[stry] is not None) : 
					plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
					plt_axes('y', 'both', ylim[stry][1])
			plt.title(r'LAB, $\nu='+str(freqselect)+r'$MHz, Dec$='+('%.2f' % Decs)+'$deg', size=16)
			plt.savefig(outdir+'figure/'+'lab_'+str(freqselect)+'MHz.png')
			plt.clf()
	else : lab = 0
	#--------------------------------------------------
	# Synchrotron @ freqselect GSM)
	if (plev != False) : print 'Using GSM ...'
	gsmname = GSM(freqselect, 'npy')
	gsm = gsmname[1](gsmname[0])  # Galactic coord
	gsm = hp.ud_grade(gsm, nside)
	# Smooth to PAON4 single dish FWHM
	gsm = hp.smoothing(gsm, FWHM, verbose=False)
	l, b = EquatorialGalactic(RAlist*np.pi/180, Decs*np.pi/180)
	n = hp.ang2pix(nside, np.pi/2-b, l)
	gsm = gsm[n] # K, gsd;f slice
	del l, b, n
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'WHAT':'GSM', 'UNIT':'K'})
	npk.Append(gsm, 'GSM', hdrnpktmp)
	#--------------------------------------------------
	# Plot GSM
	if (plot) : 
		# Plot gsm
		label = 'GSM (Synchrotron emission)\n'+r'$\nu='+str(freqselect)+r'$MHz, $\Delta\nu='+('%.3f' % (250./Nf))+r'$MHz'+'\n'+'Dec$='+('%.2f' % Decs)+'$deg\n'+'FWHM$='+('%.3f' % (FWHM*180/np.pi))+'$deg'
		plt.plot(RAlist, gsm, lw=2, label=label)
		plt.legend(loc=2)
		plt.xlabel(r'RA (deg)', size=16)
		plt.xlim(0, 360)
		plt_axes('x', 'both', [30, 5])
		plt.ylabel(r'$T_b$ (K)', size=16)
		stry = ''
		if ('gsm' in ylim.keys()) : stry = 'gsm'
		if ('GSM' in ylim.keys()) : stry = 'GSM'
		if (stry != '') : 
			if (ylim[stry] is not None) : 
				plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				plt_axes('y', 'both', ylim[stry][1])
		plt.title(r'Synchrotron, $\nu='+str(freqselect)+'$MHz, Dec=$'+('%.2f' % Decs)+'$deg', size=16)
		plt.savefig(outdir+'figure/'+'synchrotron_'+str(freqselect)+'MHz.png')
		plt.clf()
	#--------------------------------------------------
	# Total temperature of the sky
	if (plev != False) : print 'Calibrating A.U. to K ...'
	Tsky = gsm + lab
	del gsm, lab
	#--------------------------------------------------
	# Where are the peaks/maxinum value in Tsys
	# Use this to calibrate the amplitude from A.U. to K
	nmaxTsky = int(round((np.arange(len(Tsky))[Tsky==Tsky.max()]).mean()))
	#--------------------------------------------------
	# Smooth auto
	# The most narrow 21cm peak's with is 120 pixels
	# So, auto = Smooth(auto, 0, 120, 20) is OK!
	autos = Smooth(auto, 0, 120, 20)
	#--------------------------------------------------
	# A.U. to K convertion factor also is the sigma of noise, because I devide G(t)*std_noise at the beginning, make sigma_noise to be 1 A.U.
	std_noise = np.zeros([len(chan2antHV),])
	#--------------------------------------------------
	nsplit = [n24+1]
	for i in range(len(chan2antHV)) : 
		# Where is the maxinum/peak
		nmax4 = int(round(np.arange(len(autos))[autos[:,i]==autos[:,i].max()].mean()))
		# Shift auto starting from nmax4
		auto[:,i] = np.append(auto[nmax4:,i], auto[:nmax4,i])
		if (i == 0) : t = np.append(t[nmax4:], t[:nmax4])
		RAlist4 = (RAlist[nmaxTsky] + RAlist) % 360
		# Where is RA=0
		n40 = int(round(np.arange(RAlist4.size)[RAlist4==RAlist4.min()].mean()))
		nsplit.append((nmax4,n40))
		# Reorder auto to 0-360 degree
		auto[:,i] = np.append(auto[n40:,i], auto[:n40,i])
		if (i == 0) : t = np.append(t[n40:], t[:n40])
		# A.U. to K convertion factor also is the sigma of noise, because I devide G(t)*std_noise at the beginning, make sigma_noise to be 1 A.U.
		# auto=Tsky+Tsys+offset, if want to calibrate it, we assume Tsys and offset are constants, then auto.mean()=Tsky.mean()+Tsys+offset, the difference is auto-auto.mean()=Tsky-Tsky.mean(), then we can remove the effect of Tsys and offset, then calibrate the amplitude.
		# We use the peak/maxinum to calibrate: (auto-auto.mean()).max()=auto.max()-auto.mean()
		std_noise[i] = (Tsky.max()-Tsky.mean()) / (autos[:,i].max()-autos[:,i].mean())
	del RAlist4
	#--------------------------------------------------
	# Where is Sidereal time =24h
	nt24 = int(round(np.arange(len(t))[t==t.max()].mean()))
	# 24 hours miss matching, remove it as RFI
	# left: 10 degree, right: 3 degree will be treated as RFI
	nl, nr = int(10/RAlist[1]), int(3/RAlist[1])
	auto = RemoveRFI(auto, 0, [(nt24-nl, nt24+nr)], 'manual')
	del t
	#--------------------------------------------------
	# Calibrate the amplitude
	# A.U to K
	auto *= std_noise # K
	autos = Smooth(auto, 0, 120, 20)
	# Tsys and shift
	Tsys_auto = std_noise * (83886*1e-6*df*1e6)**0.5  # K
	# Baseline/Base-curve of Tsky is about 2K, however, baseline of auto ~ 110K, this 110K=2K+Tsys+offset. shift=Tsys+offset
	shift = auto.mean(0) - Tsky.mean()  # K
	#--------------------------------------------------
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'nsplit, index of UTC to split and re-order to real RAlist', 'USAGE':'auto=auto[:nsplit[0]] is the 24h sidereal time, cut the pixels outside nsplit[0]. For nsplit[1:], each nsplit[i]=(ns1,ns2), do two times re-order: auto=np.append(auto[ns1:],auto[:ns1]), then auto=np.append(auto[ns2:],auto[:ns2]), will re-order auto from UTC to real RAlist=0-360deg'})
	npk.Append(npfmt(nsplit), dayname+'_nsplit', hdrnpktmp)
	#----------
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'Auto-corr, calibrated', 'UNIT':'K'})
	npk.Append(np.float32(auto), dayname+'_auto_K', hdrnpktmp)
	#----------
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'sigma_noise', 'UNIT':'K', 'COMMENT':'sigma of noise used to calculate Tsys'})
	npk.Append(std_noise, dayname+'_sigma_noise', hdrnpktmp)
	#----------
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'Tsys', 'UNIT':'K', 'COMMENT':'Tsys of 8 auto calculated from std of noise'})
	npk.Append(Tsys_auto, dayname+'_Tsys', hdrnpktmp)
	#----------
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'shift', 'UNIT':'K', 'COMMENT':'shift of auto baseline from real Tsky, shift=Tsys+offset'})
	npk.Append(shift, dayname+'_shift', hdrnpktmp)
	npk.Save()
	#--------------------------------------------------
	if (plot) : 
		# Plot Tsky
		labelTsky = '$T_{sky}$\n'+r'$\nu='+str(freqselect)+r'$MHz, $\Delta\nu='+('%.3f' % (250./Nf))+r'$MHz'+'\n'+'Dec$='+('%.2f' % Decs)+'$deg\n'+'FWHM$='+('%.3f' % (FWHM*180/np.pi))+'$deg'
		plt.plot(RAlist, Tsky, lw=2, label=labelTsky)
		plt.legend(loc=2)
		plt.xlabel(r'RA (deg)', size=16)
		plt.xlim(0, 360)
		plt_axes('x', 'both', [30, 5])
		plt.ylabel(r'$T_b$ (K)', size=16)
		stry = ''
		if ('tsky' in ylim.keys()) : stry = 'tsky'
		if ('TSKY' in ylim.keys()) : stry = 'TSKY'
		if ('Tsky' in ylim.keys()) : stry = 'Tsky'
		if (stry != '') : 
			if (ylim[stry] is not None) : 
				plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				plt_axes('y', 'both', ylim[stry][1])
		plt.title(r'$T_{sky}$, $\nu='+str(freqselect)+'$MHz, Dec=$'+('%.2f' % Decs)+'$deg', size=16)
		plt.savefig(outdir+'figure/'+'Tsky_'+str(freqselect)+'MHz.png')
		plt.clf()
		#--------------------------------------------------
		# Plot auto, original
		color = color[1:4] + [plt_color(len(chan2antHV)+1)[3]] + color[4:]
		for i in range(len(chan2antHV)) : 
			plt.plot(RAlist, autos[:,i], color=color[i], lw=2, label=chan2antHV[i])
		plt.plot(RAlist, Tsky, color='k', lw=2, label='LAB+GSM')
	#	plt.legend(loc=9, fontsize=6)
		plt.xlabel(r'RA (deg)', size=16)
		plt.xlim(0, 360)
		plt_axes('x', 'both', [30, 5])
		plt.ylabel(r'$T_b$ (K)', size=16)
		stry = ''
		if ('autoKo' in ylim.keys()) : stry = 'autoKo'
		if ('AUTOKO' in ylim.keys()) : stry = 'AUTOKO'
		if (stry != '') : 
			if (ylim[stry] is not None) : 
				plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				plt_axes('y', 'both', ylim[stry][1])
		plt.title(dayname+', 8 autos, original\n'+r'$\nu='+str(freqselect)+'$MHz, Dec=$'+('%.2f' % Decs)+'$deg', size=16)
		plt.savefig(outdir+'figure/'+dayname+'_auto8_'+str(freqselect)+'MHz_K_original.png')
		plt.clf()
		#--------------------------------------------------
		# Plot the plt.legend() for autoKo above
		for i in range(len(chan2antHV)) : 
			strdn = strdT = strds = ''
			strstd = str(round(std_noise[i],3))
		#	if (round(std_noise[i],1)<1) : strdT = '\,\,\,\,'
			strTsys = str(round(Tsys_auto[i],1))
			if (round(Tsys_auto[i],1)<100) : strdT = '\,\,\,\,'
			strshift = str(round(shift[i],1))
			if (round(shift[i],1)<100) : strds = '\,\,\,\,'
			label = chan2antHV[i]+r',  $\sigma='+strdn+strstd+'\,$K $\Rightarrow$ $T_{sys}='+strdT+strTsys+r'\,$K,  $\Delta T_b='+strds+strshift+r'\,$K'
			plt.plot([1],[1], color=color[i], label=label)
		plt.plot([1],[1], color='k', label='LAB+GSM')
		plt.legend(loc=10)
		plt.savefig(outdir+'figure/'+dayname+'_auto8_'+str(freqselect)+'MHz_K_original_legend.png')
		plt.clf()
		#--------------------------------------------------
		# Plot auto, shift, with Tsys
		plt.plot(RAlist, Tsky, color='k', lw=3, label='$T_{sky}$')
		for i in range(len(chan2antHV)) : 
			plt.plot(RAlist, autos[:,i]-shift[i], color=color[i], label=chan2antHV[i])
		plt.legend(loc=9)
		plt.xlabel(r'RA (deg)', size=16)
		plt.xlim(0, 360)
		plt_axes('x', 'both', [30, 5])
		plt.ylabel(r'$T_b$ (K)', size=16)
		stry = ''
		if ('autoKf' in ylim.keys()) : stry = 'autoKf'
		if ('AUTOKF' in ylim.keys()) : stry = 'AUTOKF'
		if (stry != '') : 
			if (ylim[stry] is not None) : 
				plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				plt_axes('y', 'both', ylim[stry][1])
		plt.title(dayname+', 8 autos, shifted\n'+r'$\nu='+str(freqselect)+'$MHz, Dec=$'+('%.2f' % Decs)+'$deg', size=16)
		plt.savefig(outdir+'figure/'+dayname+'_auto8_'+str(freqselect)+'MHz_K_shifted.png')
		plt.clf()
	print 'END:', Time()[1]
	print
	return npk.outname


##################################################
##################################################
##################################################


def PAON4CaliCross( freq, obsname, gainname, caliphasename, caliautoname, chan2antHV=PAON4Chan2AntHV(1), rmRFI=None, eachdir=None, Dec=None, plot=True, ylim=None, maxbaseline=None, strcrossmask=None, skip=False, plev=False ) : 
	'''
	freq:
		MHz, must be one value/scale, not ndarray
		Which frequency of auto to be calibrated

	dayname: 
		Here dayname must be one str/file, not list
		dayname can be just dayname, or the complete/absolute path of the FITS
		If dayname is the complete/absolute path of the FITS, then set obsdir and obstail to be None or ''

	obsdir, obstail:
		Observation data in FITS
		If dayname is the complete/absolute path of the FITS, then set obsdir and obstail to be None or ''

	gaindir, gaintail:
		File from PAON4Gaint(), G(t)
		If gaintail=None or '', means that gaindir is the complete/absolute path of the G(t) FITS file => gainname=gaindir, we don't use dayname at this time

	chan2antHV:
		Channel to antenna HV
		chan2antHV = PAON4Chan2AntHV(1) / PAON4Chan2AntHV(4)

	rmRFI:
		First this function will use two method to find and remove RFI automatically (RemoveRFI())
		However, the result may be not good enough
		Then set rmRFI by hand, means this range of pixels are RFI, and they will be removed
		rmRFI must be 2D:
			[[20,30]], [(20,30)] means pixels 20-30 are RFI. Althought just one range(20,30), you must add [] outside
			[(20,30), [50,60], ...]

	outdir:
		Which directory to save the output .npk and .png
		None means tring to save to sys.argv[0][:-3]+'_output/'
		If it can mkdir the directory above, for example, use this function in ipython, then will set outdir='./'

	plot, ylim, maxbaseline:
		plot: True, False, plot the figure or not
		ylim: If plot, you can set ylim to control the plt.ylim() of each figure. ylim is a dict{}, ylim.keys()=['autoAU','LAB','GSM','Tsky','autoKo','autoKf']
		ylim for PAON4 21cm observation is
			ylim = {'crossAU':None, 'crossK':None}
		If you don't want to use plt.ylim():
			(1) Set the value to be None
			(2) Delete this key
			Effects of two methods above are the same
		maxbaseline:
			We will plot the smoothing cross, use maxbaseline to the parameter 'per' of function smooth()

		ylim = {'original':[(-10000,10000), (2000,500,'%i')], 
		          'offset':[(-10000,10000), (2000,500,'%i')], 
		             'dPK':[(-8,8), (1,0.1,'%.1f')], 
		            'PnsK':[(-8,8), (1,0.1,'%.1f')]} 
	'''
	print 'PAON4CaliCross() ...'
	print 'START:', Time()[1]
	#--------------------------------------------------
	if (ylim is None) : ylim = {}
	#--------------------------------------------------
	# Check dayname whether there is that data set
	obsname = PathExists(obsname)
	gainname = PathExists(gainname)
	caliphasename = PathExists(caliphasename)
	caliautoname = PathExists(caliautoname)
	dayname = PAON4Dayname(obsname)
	#--------------------------------------------------
	# Directory to save the output fits and pictures
	outdirthis = OutDir(None)
	outdir = OutDir(outdirthis+'PAON4CaliCross/')
	outdir = OutDir(StrListAdd(outdir,eachdir))
	OutDir(StrListAdd(outdir,'figure/'))
	#--------------------------------------------------
	Decs = PAON4Dec(dayname, Dec)
	nang = StrFind(dayname, 'NumType')[0][0]
	RAs = CalibrationSource().RADec(dayname[:nang[0]])[0]
	#--------------------------------------------------
	if (chan2antHV is None) : chan2antHV = PAON4Chan2AntHV(1)
	paon4pair = PAON4Pair(chan2antHV)
	strcross = paon4pair.strcross
	color = plt_color(len(strcross))
	#--------------------------------------------------
	freqselect = freq
	outname = outdir+dayname+'_cross_'+str(freqselect)+'MHz_calibrated_phase-K.npk'
	if (skip) : 
		if (os.path.exists(outname)) : return outname
	npk = NPK(outname)
	#--------------------------------------------------
	if (plev != False) : 
		print 'obsname:       '+obsname
		print 'gainname:      '+gainname
		print 'dayname:       '+dayname
		print 'caliphasename: '+caliphasename
		print 'caliautoname:  '+caliautoname
		print 'outdir:        '+outdir
		print 'npk.outname:   '+npk.outname
		print 'RA, Dec:       '+str(round(RAs,4))+'deg, '+str(round(Decs,4))+'deg'
		print 'chan2antHV:    '+str(chan2antHV)
	#--------------------------------------------------
	# Open obs
	hdr = pyfits.getheader(obsname, 2)
	Nt, avet, dt, tbin1, tbin2, t1, t2, Nf, avef, df, fbin1, fbin2, f1, f2, deltime, dateobs, Nv = PAON4Header(hdr)
	t = np.linspace(t1, t1+deltime, Nt)/60.  # UTC, minute
	# UTC 24hours, however the sidereal time should be 23h56m4s
	#--------------------------------------------------
	# Frequency range of PAON4-21cm with Nfreq=512, n21cm=349 for 512
	# -1 is because of the miss matching the freq bins in LAB
	# PAON4-21cm=>-0, LAB-21cm=>-1
	nfreqselect = int(round(Nf/250.*(freqselect-1250))) 
	hdrnpk = {'FREQ':(freqselect,'MHz'), 'FREQBIN':(round(df,4),'MHz')}
	#--------------------------------------------------
	# Open gain
	fo = pyfits.open(gainname)
	tG = fo[0].data/60.  # minute, UTC
	Gt = fo[3].data  # h24 False, bad result
#	Gt = fo[-1].data  # h24 True
	#--------------------------------------------------
	# Open calibrated phase
	arrlist, hdrlist = NPK().Load(caliphasename, [0,1,2,4,5])
	Dij, Di, tsij, Pijab, Pns = arrlist
	LAST = (0, tsij.mean()/60*15*np.pi/180)
	#--------------------------------------------------
	# Open auto_K.npk
	arrlist, hdrlist = NPK().Load(caliautoname, [2,4])
	nsplit, std_noise = list(arrlist[0]), arrlist[1]
	#--------------------------------------------------
	# Get 6pairs*2polarizations*2realimag cross, cali gain
	if (plev != False) : print 'Removing RFI ...'
	realmem = pyfits.getdata(obsname, 1)[:,:,nfreqselect]
	imagmem = pyfits.getdata(obsname, 2)[:,:,nfreqselect]
	cross = np.append(realmem, imagmem, 1)
	realmem = imagmem = 0 #@
	Purge()
	# I have checked that cross.dtype won't change below, so the dtype of the result is determined here
	cross = cross[:,:len(strcross)] + 1j*cross[:,len(strcross):]
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'cross-corr, original data', 'UNIT':'A.U.'})
	npk.Append(cross, dayname+'_cross_original', hdrnpktmp)
	cross = np.append(cross.real, cross.imag, 1)
	#--------------------------------------------------
	# RemoveRFI automatically
	cross = RemoveRFI(cross, 0, 10, 'median')
	cross = RemoveRFI(cross, 0, rmRFI, 'manual')
	#--------------------
	if (plot) : 
		nfr1, nfr2 = nfr(freqselect, LAST, Decs*np.pi/180, Dij.mean(), deltime, Nt, 6)
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(t[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('t (min)', size=16)
				plt.xlim(t[nfr1:nfr2].min(), t[nfr1:nfr2].max())
				plt_axes('x', 'both', [10, 1])
				plt.ylabel('(A.U.)', size=16)
				stry = None
				if ('original' in ylim.keys()) : stry = 'original'
				if ('ORIGINAL' in ylim.keys()) : stry = 'ORIGINAL'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n original data', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_original.png')
				plt.clf()
	#--------------------------------------------------
	# Remove the correlating noise/distortion
	# Target source range as RFI
	if (plev != False) : print 'Calibrating visibility distortion/offset ...'
	nfr1, nfr2 = nfr(freqselect, LAST, Decs*np.pi/180, Dij.mean(), deltime, Nt, 6)
	nfr2 = nfr2 + (nfr2-nfr1)
	if (nfr2 > len(cross)-1000) : nfr2 = len(cross)-1000
#	crosssmooth = RemoveRFI(cross, 0, [(nfr1,nfr2)], 'manual')
#	crosssmooth = Smooth(crosssmooth, 0, 300, 2, applr=False)
#	crosssmooth = Smooth(crosssmooth, 0, 300, 8)
#	cross -= crosssmooth
	crosssmooth = np.append(cross[:nfr1], cross[nfr2:], 0)
	crossmooth = RemoveRFI(crosssmooth, 0, 100, 'median')
	# Calibrate offset
	for i in range(cross.shape[1]) : 
		cross[:,i] -= Leastsq(np.arange(len(crosssmooth)), crosssmooth[:,i], 'polynomial', p0index=0)[0]
	crosssmooth = 0 #@
	#--------------------
	if (plot) : 
		nfr1, nfr2 = nfr(freqselect, LAST, Decs*np.pi/180, Dij.mean(), deltime, Nt, 6)
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(t[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('t (min)', size=16)
				plt.xlim(t[nfr1:nfr2].min(), t[nfr1:nfr2].max())
				plt_axes('x', 'both', [10, 1])
				plt.ylabel('(A.U.)', size=16)
				stry = None
				if ('offset' in ylim.keys()) : stry = 'offset'
				if ('OFFSET' in ylim.keys()) : stry = 'OFFSET'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n calibrated offset', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_offset.png')
				plt.clf()
	#--------------------------------------------------
	# Calibrate addition phase
	if (plev != False) : print 'Calibrating dP ...'
	cross = cross[:,:len(strcross)] + 1j*cross[:,len(strcross):]
	phase = Pijab[0] + Pijab[1]*freqselect
	cross = np.complex64(cross * np.exp(-1j*phase))
	cross = np.append(cross.real, cross.imag, 1)
	#--------------------
	if (plot) : 
		nfr1, nfr2 = nfr(freqselect, LAST, Decs*np.pi/180, Dij.mean(), deltime, Nt, 6)
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(t[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('t (min)', size=16)
				plt.xlim(t[nfr1:nfr2].min(), t[nfr1:nfr2].max())
				plt_axes('x', 'both', [10, 1])
				plt.ylabel('(A.U.)', size=16)
				stry = None
				if ('dP' in ylim.keys()) : stry = 'dP'
				if ('Pij' in ylim.keys()) : stry = 'Pij'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n calibrated $\Delta \Phi$', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_dP.png')
				plt.clf()
	#--------------------------------------------------
	# Remove G(t)
	print 'Calibrating G(t) ...'
	sqrt2 = np.zeros([len(strcross),])
	for i in range(len(sqrt2)) : 
		Gimag = Gt[:,i+len(chan2antHV)+len(strcross)]*1
		na = paon4pair.Cross2Auto(strcross[i])
		Ga1, Ga2 = Gt[:,na[0]]*1, Gt[:,na[1]]*1
		sqrt2[i] = ((Ga1*Ga2)**0.5).mean() / Gimag.mean()
		Gimag = Interp1d(tG, Gimag, t)
		cross[:,i] /= Gimag
		cross[:,i+len(strcross)] /= Gimag
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'sqrt{2}, ratio ot of sigma of noise between auto and cross'})
	npk.Append(sqrt2, dayname+'_sqrt2', hdrnpktmp)
	np.save('sqrt2', sqrt2)
	exit()
	#--------------------
#	# Method 2
#	sqrt2 = np.zeros([cross.shape[1],])
#	for i in range(len(sqrt2)) : 
#		Gri = Gt[:,i+len(chan2antHV)]*1
#		na = paon4pair.Cross2Auto(strcross[i%len(strcross)])
#		Ga1, Ga2 = Gt[:,na[0]]*1, Gt[:,na[1]]*1
#		sqrt2[i] = ((Ga1*Ga2)**0.5).mean() / Gri.mean()
#		Gri = Interp1d(tG, Gri, t)
#		cross[:,i] /= Gri
	#--------------------
	Gt = tG = Gimag = Greal = Ga1 = Ga2 = Gri = 0 #@
	#--------------------------------------------------
	# Sidereal time 24hour
	cross = cross[:nsplit[0]]
	RAlist = np.linspace(0, 360, len(cross)) # degree
	# Re-order to real RA, calibrate to K
	if (plev != False) : 
		print 'Converting to Equatorial ...'
		print 'Calibrating A.U. to K ...'
	for i in range(len(strcross)) : 
		# Re-order
		na1, na2 = paon4pair.Cross2Auto(strcross[i])
		ni1, ni2 = nsplit[na1+1]
		nj1, nj2 = nsplit[na2+1]
		k1, k2 = Di[na1]**2/(Di[na1]**2+Di[na2]**2), Di[na2]**2/(Di[na1]**2+Di[na2]**2)
		n1, n2 = ni1*k1+nj1*k2, ni2*k1+nj2*k2
		cross[:,i] = np.append(cross[n1:,i], cross[:n1,i])
		cross[:,i] = np.append(cross[n2:,i], cross[:n2,i])
		cross[:,i+len(strcross)] = np.append(cross[n1:,i+len(strcross)], cross[:n1,i+len(strcross)])
		cross[:,i+len(strcross)] = np.append(cross[n2:,i+len(strcross)], cross[:n2,i+len(strcross)])
		# Calibrate to K
		stdc = (std_noise[na1]*std_noise[na2])**0.5 / sqrt2[i]
		cross[:,i] *= stdc
		cross[:,i+len(strcross)] *= stdc
	cross = cross[:,:len(strcross)] + 1j*cross[:,len(strcross):]
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'cross-corr, calibrated to K and additional phase', 'UNIT':'K'})
	npk.Append(cross, dayname+'_cross_K_dP', hdrnpktmp)
	cross = np.append(cross.real, cross.imag, 1)
	#--------------------
	if (plot) : 
		# Plot, cross, 0-360deg, K, cali phase, with noise
		nfr1, nfr2 = nfr(freqselect, (0,RAs*np.pi/180), Decs*np.pi/180, Dij.mean(), 24*3600, len(cross), 6)
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(RAlist[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('RA (deg)', size=16)
				plt.xlim(RAlist[nfr1:nfr2].min(), RAlist[nfr1:nfr2].max())
				plt_axes('x', 'both', [5, 0.5])
				plt.ylabel(r'$T_b$ (K)', size=16)
				stry = None
				if ('dPK' in ylim.keys()) : stry = 'dPK'
				if ('PijK' in ylim.keys()) : stry = 'PijK'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n calibrated $\Delta \Phi$, scale to K', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_dP-K.png')
				plt.clf()
	#--------------------------------------------------
	# Calibrate north-south phase
	if (plev != False) : print 'Calibrating Pns ...'
	cross = cross[:,:len(strcross)] + 1j*cross[:,len(strcross):]
	cross = np.complex64(cross * np.exp(-1j*Pns[:,nfreqselect]))
	cross = np.append(cross.real, cross.imag, 1)
	#--------------------
	if (plot) : 
		# Plot, cross, 0-360deg, K, cali phase, with noise
		nfr1, nfr2 = nfr(freqselect, (0,RAs*np.pi/180), Decs*np.pi/180, Dij.mean(), 24*3600, len(cross), 6)
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(RAlist[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('RA (deg)', size=16)
				plt.xlim(RAlist[nfr1:nfr2].min(), RAlist[nfr1:nfr2].max())
				plt_axes('x', 'both', [5, 0.5])
				plt.ylabel(r'$T_b$ (K)', size=16)
				stry = None
				if ('PnsK' in ylim.keys()) : stry = 'PnsK'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n calibrated $\Delta \Phi$ and $\Phi_{north-south}$, scale to K, before correct 34', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_dP-Pns_before34.png')
				plt.clf()
	#--------------------------------------------------
	# However because we can fit the phases of 3H4H and 3V4V, we just use the phase relationship to predict what the phase of 34 should be, however the result may be not good. Now, we can correct the phase of 34 now after calibrate dP and Pns
	# strcrossmask
	if (strcrossmask is None) : strcrossmask = ['3H-4H','3V-4V']
	if (Type(strcrossmask) == str) : strcrossmask = [strcrossmask]
	strcrossmask, cmtmp, ncrossmask = list(strcrossmask), [], []
	for i in range(len(strcrossmask)) : 
		strc = strcrossmask[i].split('-')
		cmtmp += [strc[0]+'-'+strc[1], strc[1]+'-'+strc[0]]
	strcrossmask = cmtmp
	# ['3H-4H'] => ['3H-4H','4H-3H']
	for i in range(len(strcrossmask)) : 
		try : ncrossmask.append(paon4pair.Cross(strcrossmask[i]))
		except : pass
	#--------------------
	cross = cross[:,:len(strcross)] + 1j*cross[:,len(strcross):]
	cross34 = Smooth(cross, 0, 60, 3, applr=False)
	for i in ncrossmask : 
		cross[:,i] -= cross34[:,i]
		cross[:,i].real += abs(cross34[:,i])
	cross34 = 0 #@
	#--------------------------------------------------
	hdrnpktmp = hdrnpk.copy()
	hdrnpktmp.update({'DAYNAME':dayname, 'WHAT':'cross-corr, calibrated to K and additional phase, north-south phase', 'UNIT':'K'})
	npk.Append(cross, dayname+'_cross_K_dP-Pns', hdrnpktmp)
	npk.Save()
	cross = np.append(cross.real, cross.imag, 1)
	#--------------------------------------------------
	if (plot) : 
		# Plot, cross, 0-360deg, K, cali phase, with noise
		nfr1, nfr2 = nfr(freqselect, (0,RAs*np.pi/180), Decs*np.pi/180, Dij.mean(), 24*3600, len(cross), 6)
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(RAlist[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('RA (deg)', size=16)
				plt.xlim(RAlist[nfr1:nfr2].min(), RAlist[nfr1:nfr2].max())
				plt_axes('x', 'both', [5, 0.5])
				plt.ylabel(r'$T_b$ (K)', size=16)
				stry = None
				if ('PnsK' in ylim.keys()) : stry = 'PnsK'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n calibrated $\Delta \Phi$ and $\Phi_{north-south}$, scale to K', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_dP-Pns.png')
				plt.clf()
	#--------------------------------------------------
	# Plot cross-dP-K and cross-Pns-K without noise
	if (plot) : 
		if (maxbaseline is None) : maxbaseline = 100
		nfr1, nfr2 = nfr(freqselect, (0,RAs*np.pi/180), Decs*np.pi/180, maxbaseline, 24*3600, len(cross), 1)
		smoothper1 = (((nfr2-nfr1)/20)%2==1) and (nfr2-nfr1)/20 or (nfr2-nfr1)/20+1
		smoothper2 = (((nfr2-nfr1)/5)%2==1) and (nfr2-nfr1)/5 or (nfr2-nfr1)/5+1
		nfr1, nfr2 = nfr(freqselect, (0,RAs*np.pi/180), Decs*np.pi/180, Dij.mean(), 24*3600, len(cross), 6)
		#--------------------
		cross = npk.array[1]
		cross = np.append(cross.real, cross.imag, 1)
		crossmax = Smooth(cross[nfr1:nfr2], 0, smoothper1, applr=False)
		crossmax = abs(crossmax).max(0)
		cross[nfr1:nfr2] = Smooth(cross[nfr1:nfr2], 0, smoothper2, applr=False)
		crosssmax = abs(cross[nfr1:nfr2]).max(0)
		cross[nfr1:nfr2] = cross[nfr1:nfr2] /crosssmax*crossmax
		#--------------------
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(RAlist[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('RA (deg)', size=16)
				plt.xlim(RAlist[nfr1:nfr2].min(), RAlist[nfr1:nfr2].max())
				plt_axes('x', 'both', [5, 0.5])
				plt.ylabel(r'$T_b$ (K)', size=16)
				stry = None
				if ('dPK' in ylim.keys()) : stry = 'dPK'
				if ('PijK' in ylim.keys()) : stry = 'PijK'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n calibrated $\Delta \Phi$, scale to K, remove noise', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_dP-K-rmn.png')
				plt.clf()
		#--------------------------------------------------
		cross = npk.array[2]
		cross = np.append(cross.real, cross.imag, 1)
		crossmax = Smooth(cross[nfr1:nfr2], 0, smoothper1, applr=False)
		crossmax = abs(crossmax).max(0)
		cross[nfr1:nfr2] = Smooth(cross[nfr1:nfr2], 0, smoothper2, applr=False)
		crosssmax = abs(cross[nfr1:nfr2]).max(0)
		cross[nfr1:nfr2] = cross[nfr1:nfr2] /crosssmax*crossmax
		#--------------------
		for i in range(cross.shape[1]) : 
			strri = (i<len(strcross)) and 'real' or 'imag'
			plt.plot(RAlist[nfr1:nfr2], cross[nfr1:nfr2,i], color=color[i%len(strcross)], label=strcross[i%len(strcross)]+', '+strri)
			if ((i+1)%len(strcross) == 0) : 
				plt.legend(loc=2, fontsize=8.3)
				plt.xlabel('RA (deg)', size=16)
				plt.xlim(RAlist[nfr1:nfr2].min(), RAlist[nfr1:nfr2].max())
				plt_axes('x', 'both', [5, 0.5])
				plt.ylabel(r'$T_b$ (K)', size=16)
				stry = None
				if ('PnsK' in ylim.keys()) : stry = 'PnsK'
				try : plt.ylim(ylim[stry][0][0], ylim[stry][0][1])
				except : pass
				try : plt_axes('y', 'both', ylim[stry][1][:2], ylim[stry][1][2])
				except : pass
				plt.title(dayname+', cross, '+strri+'\n calibrated $\Delta \Phi$ and $\Phi_{north-south}$, scale to K, remove noise', size=16)
				plt.savefig(outdir+'figure/'+dayname+'_cross_'+strri+'_'+str(freqselect)+'MHz_dP-Pns-rmn.png')
				plt.clf()
	print 'END:', Time()[1]
	print
	return npk.outname


##################################################
##################################################
##################################################


def DiagIndex( shape, which='diag' ) : 
	'''
	Return the bool ndarray

	which:
		='diag': return diag index using bool
		='left': return below diag (lower left) index
		='right': return above diag (upper right) index
      ='leftdiag', 'rightdiag'

	shape:
		shape of the array
	'''
	if (len(shape) > 2) : raise Exception('shape should be 1- or 2-D')
	r = np.arange(shape[0])
	c = np.arange(shape[1])
	tf = np.complex64(r[:,None] + 1j*c[None,:])
	which = which.lower()
	if (which == 'diag') : return (tf.real==tf.imag)
	elif (which == 'left') : return (tf.real>tf.imag)
	elif (which == 'leftdiag') : return (tf.real>=tf.imag)
	elif (which == 'right') : return (tf.real<tf.imag)
	elif (which == 'rightdiag') : return (tf.real<=tf.imag)


##################################################
##################################################
##################################################


def LM( l, m, order=None ) : 
	'''
	Return the lm matrix

	l: 0 to lmax
	m: -l to +l
	l, m can be 1D or 2D

	order:
		None: return 1D(flatten) ndarray with l>=abs(m)
		'2D': return 2D matrix with np.nan
		'l': return 1D array ordered by l from small to large
		'm': return 1D array ordered by m from small to large
		'i': return 1D array ordered by idx from small to large
		* Also return morder and lorder

	All l and m will be flatten()
	'''
	Tl, Tm = Type(l), Type(m)
	if (Tl=='NumType' and Tm=='NumType') : 
		if (abs(m) > l) : 
			m = l if(m>0)else -l
			return [l, m]
	l, m = npfmt(l).flatten(), npfmt(m).flatten()
	lm = l[:,None] + 1j*m[None,:]
	tf = (lm.real>=abs(lm.imag))
	lm[True-tf] = 0
	l = lm.real.round().astype(int)
	m = lm.imag.round().astype(int)
	lm = 0 #@
#	if ('NumType' in [Tl,Tm]) : order = None
	if (order not in ['2D','2d']) : l, m = l[tf], m[tf]
	if (order in ['l','L']) : 
		l = Sort(l + 1j*np.arange(l.size))
		n = l.imag.round().astype(int)
		l = l.real.round().astype(int)
		m = m[n]
	elif (order in ['m','M']) : 
		m = Sort(m + 1j*np.arange(m.size))
		n = m.imag.round().astype(int)
		m = m.real.round().astype(int)
		l = l[n]
	if (order in ['2D','2d']) : 
		if ('NumType' in [Tl,Tm]) : l, m, tf = l.flatten(), m.flatten(), tf.flatten()
		if (Tl == 'NumType') : l = l[tf][0]
		if (Tm == 'NumType') : m = m[tf][0]
	else : 
		if (Tl == 'NumType') : l = l[0]
		if (Tm == 'NumType') : m = m[0]
	if (order not in ['2D','2d']) : return [l, m]
	else : return [l, m, tf]


##################################################
##################################################
##################################################


class CompressHealpixMap( object ) : 
	dtype = 'class:'+sys._getframe().f_code.co_name

	def __init__( self ) : self.compresshpmap = []

	def Append( self, hpmap, value, error=1e-6 ) : 
		'''
		In the healpix map, there may be lots of pixels with the same value (for example, 0), cost lots of memory.
	
		This function is use to return:
			[ValidHpmap, ValidIdx]
			ValidHpmap: pixels which are not that same value
			ValidIdx: healpix index of ValidHpmap
	
		value, error:
			value-error to value+error will be not save
		'''
		#--------------------------------------------------
		nside = []
		# Check whether healpix map
		if (Type(hpmap) in [list, tuple]) : 
			hpmap, islist = list(hpmap), True
		else : hpmap, islist = [hpmap], False
		for i in range(len(hpmap)) : 
			try : nside.append(hp.get_nside(hpmap[i]))
			except : 
				if (islist) : pstr = '['+str(i)+']'
				else : pstr = ''
				print self.dtype+'.Append()/Warning: hpmap'+pstr+' is not a Healpix Map.'
				nside.append(hpmap[i].shape)
		#--------------------------------------------------
		# Check value, error
		Tv, Te = Type(value), Type(error)
		if (Type(value) == 'NumType') : 
			value = [value for i in range(len(hpmap))]
		else : 
			value = list(value)[:len(hpmap)]
			for i in range(len(value), len(hpmap)) : 
				if (nside[i] == 0) : value.append(0)
				else : value.append(hpmap[i].min())
		if (Type(error) == 'NumType') : 
			error = [abs(error) for i in range(len(hpmap))]
		else : 
			error = abs(npfmt(error))[:len(hpmap)]
			error = np.append(error, [error.max() for i in range(len(hpmap)-len(error))])
		value, error = npfmt(value), npfmt(error)
		#--------------------------------------------------
		for i in range(len(hpmap)) : 
			if (nside[i] == 0) : 
				self.compresshpmap.append( [hpmap[i], 0, 0, 0] )
				continue
			tf = ((value[i]-error[i])>hpmap[i])+(hpmap[i]>(value[i]+error[i]))
			n = np.arange(12*nside[i]**2)[tf]
			self.compresshpmap.append( [hpmap[i][tf], n, nside[i], value[i], error[i]] )


##################################################
##################################################
##################################################


def Islist( *arg, **kwargs ) : 
	'''
	Usage: Islist(a, b, c, d, ..., flat=False)
	Such as for wavelength, Deff, pointing, ...

	*arg:
		target to be judged if is list/ndarray/tuple
		Note that input a,b,.... must be NumType/list/ndarray/tuple, can't be other types, otherwise will return a wrong result with warning.

	**kwargs:
		Only accepts one key "flatten"
		If flatten==True, a.flatten()
		If not True, don't flatten
		Default: flatten==False

	return:
		[[islista,shapea,a], [islistb,shapeb,b], ...]
	If just (a), return [islista, shapea, a]

	islista: True or False

	shapea: Original shape of a

	a: a.flatten() if(flatten)else a

	flatten: Must give the key flatten=True/False when you want to use it.
	'''
	result, arg = [], list(arg)
	try : flatten = kwargs['flatten']
	except : flatten = False
	for i in range(len(arg)) : 
		islist = True
		if (Type(arg[i]) in [list,tuple,np.ndarray]) : pass
		elif (Type(arg[i]) == 'NumType') : islist = False
		else : 
			Raise(Warning, "arg["+str(i)+"] not in ['NumType', list, typle, np.ndarray]")
			islist = False
		try : 
			arg[i] = npfmt(arg[i])
			shape = list(arg[i].shape)
			if (flatten) : arg[i] = arg[i].flatten()
		except : shape = []
		result.append([islist, shape, arg[i]])
	if (len(result) == 1) : result = result[0]
	return result


def IslistN( *arg, **kwargs ) : 
	'''
	arg[i] in Islist() are independent each other, 
	while arg[i] here should all have the same shape

	**kwargs:
		Only accepts one key "flatten"
		If flatten==True, a.flatten()
		If not True, don't flatten
		Default: flatten==False

	Usage: CheckShape( a, b, c, ... )
	Such as for (theta,phi), (lon,lat), (x,y,z)

	return:
		[islist, shape, a, b, c, ...]
		a, b, c, ... with the same shape and convert to np.ndarray using npfmt()
	'''
	try : flatten = kwargs['flatten']
	except : flatten = False
	arg = list(arg)
	#--------------------------------------------------
	size, islist = np.array([]), np.array([], int)
	for i in range(len(arg)) : 
		if (Type(arg[i]) == 'NumType') : islist = np.append(islist, [0])
		else : islist = np.append(islist, [1])
		arg[i] = npfmt(arg[i])
		size = np.append(size, [arg[i].size])
	#--------------------------------------------------
	if (islist.sum() == 0) : islist = False
	else : islist = True
	#--------------------------------------------------
	# Can be NumType and list
	if (size[(size>1)*(size<size.max())].size > 0) : Raise(Warning, 'arg[i].shape are not the same. For smaller arg[i], use its first element arg[i].take(0)')
	# shapeX as who has the most number of elements
	n = Value2Index(size, size.max()).take(0)
	shape = arg[n].shape
	#--------------------------------------------------
	# Reshape and flatten
	for i in range(len(arg)) : 
		try : arg[i] = arg[i].reshape(shape)
		except : arg[i] = np.zeros(shape) + arg[i].take(0)
		if (flatten) : arg[i] = arg[i].flatten()
	return [islist, list(shape)] + arg


##################################################
##################################################
##################################################



class MapMaking( object ) : 
	dtype = 'class:'+sys._getframe().f_code.co_name
	''' All angles are in rad '''

	def __init__( self, nside=256, lmax=None, lonlat=None, configuration=None, pointing=None, wavelength=None, Deff=None, beamwhich=None, bit=None, resetlmax=True ) : 
		'''
		lonlat
			=[lon, lat] in rad, longitude and latitude of the Antenna Array, must be NumType

		wavelength:
			meter, relating to the self.skymap, must be NumType

		Deff:
			meter, must be NumType.

		pointing:
			rad, (PointLat - lat), relate to the lat of the Antenna Array, Northener(+), Southener(-). NumType or ndarray

		configuration:
			meter, (x,y,z) of each antenna, ndarray

		lmax:
			When Reconstruct() the skymap, we need to use the pseudo-inverse: V=M*A, M=V*Ainv=M*A*Ainv, if we want A*Ainv=I, then must rowA<=colA. Finally we need: lmax<=2*(len(pointing)*len(baseline))

		bit:
			bit of the data/ndarray usded here to decrease the memory consumption
			Just accepts float>=32, for other case, use np.float64
			bit can be:
				int: 32, 64, 128
				str: '32', '64', '128', 'float32', 'float64', 'float128'
				type: np.float32, np.float64, np.float128
				np.dtype: array.dtype
		'''
		#--------------------------------------------------
		self.nside = self._Nside(nside)
		self.lmax = self._Lmax(lmax)
		self.lon, self.lat = self._LonLat(lonlat)
		self.pointing = self._Other(pointing, 0)
		self.wavelength = self._Other(wavelength, Constant()['21cmwavelength'])
		self.Deff = self._Other(Deff, 4.5)
		self.bit, self.cbit = self._Bit(bit)
		self.beamwhich = self._BeamWhich(beamwhich)
		self.configuration = self._Configuration(configuration)
		self.resetlmax = bool(resetlmax)
	##################################################


	def _Configuration( self, configuration=None ) : 
		if (Type(configuration) in [NoneType,str]) : return self.Configuration(configuration)
		configuration = np.array(configuration)
		if (configuration.size%3 !=0) : Raise(Exception, 'configuration.size %3 !=0')
		if (configuration.size <6) : Raise(Exception, 'configuration.size <6. At least 2 antennas')
		if (len(configuration.shape) !=2) : 
			n = configuration.size/3
			configuration = configuration.reshape(n, 3)
		else : 
			shape = configuration.shape
			if (3 not in shape) : Raise(Exception, '3 not in configuration.shape='+str(shape))
			if (shape[1] == 3) : pass
			else : configuration = configuration.T
		return configuration

	def _Bit( self, bit ) : 
		if (Type(bit) == 'NumType') : bit = str(int(bit))
		if (Type(bit) == str) : 
			bit = bit.lower()
			if (bit[:5] != 'float') : bit += 'float'
		try : bit = np.dtype(bit)
		except : bit = np.dtype(None)  # System
		if (bit not in [np.dtype('float32'), np.dtype('float64'), np.dtype('float128')]) : bit = np.dtype(None)
		if   (bit == np.float32)  : cbit = np.complex64
		elif (bit == np.float128) : cbit = np.complex256
		else : cbit = np.complex128
		return [bit, cbit]

	def _BeamWhich( self, beamwhich ) : 
		try : beamwhich = beamwhich.lower()
		except : beamwhich = 'fwhm'
		if (beamwhich not in ['fwhm','bwfn','resolreza']): beamwhich='fwhm'
		return beamwhich

	def _LonLat( self, lonlat ) : 
		try : lon, lat = npfmt(lonlat).flatten()[:2]
		except : lon, lat = np.array([2.19963889, 47.38197222])*np.pi/180  # rad, PAON4LonLat
		return [lon, lat]

	def _Nside( self, nside ) : 
		try : 
			nside = int(round(float(nside)))
			n = int(round(np.log(nside) / np.log(2)))
			nside = 2**n
		except : nside = 256
		return nside

	def _Lmax( self, lmax ) : 
		try : lmax = int(float(lmax))
		except : lmax = 500
		return lmax

	def _Other( self, arg, argdefault ) : 
		try : 
			arg = np.array(arg, float)
			if (arg.shape == ()) : arg = arg.take(0)
		except : arg = argdefault
		return arg
	##################################################


	def Configuration( self, which='paon4' ) : 
		'''
		configuration:
			Note that: 
				(1) Here just handle 1 polarization
				(2) Must consider the PAON4Chan2AntHV
			x: Ease-West   (East+)
			y: North-South (North+)
			z: zenith
			configuration = [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), .]
		Note that must include z, even if it is 0
			If ==None, use self.configuration

			For POAN4:
				     North                     Ant4
				West       East    Ant2  Ant1
				     South                     Ant3
		'''
		if (Type(which) != str) : which = 'paon4'
		if (which.lower() == 'paon4') : 
			return np.array([(0,0,0), (4.39,6,0), (4.39,-6,0), (-6,0,0)])
	#--------------------------------------------------

	def SetConfiguration( self, configuration=None ) : 
		self.configuration = self._Configuration(configuration)
	##################################################


	def SetSkyMap( self, skymap=None, coordin='', coordout='' ):
		if (Type(skymap) == str) : 
			skymap = os.path.abspath(os.path.expanduser(skymap))
			if (skymap[-5:].lower() == '.fits') : 
				try : self.skymap = hp.ud_grade(hp.read_map(skymap, verbose=False), self.nside)
				except : Raise(Exception, 'Can NOT read the Healpix Map "'+skymap+'" by healpy.read_map()')
			else : 
				try : self.skymap = hp.ud_grade(np.load(skymap), self.nside)
				except : Raise(Warning, 'Can NOT read the Healpix Map "'+skymap+'" by np.load()')
		else : 
			try : 
				self.skymap = hp.ud_grade(self.nside, skymap)
				skymap = 0 #@
			except: Raise(Warning, 'skymap is NOT a Healpix Map.')
		if ((coordin not in [None,'']) and (coordout not in [None,''])) : 
			theta, phi = hp.pix2ang(self.nside, np.arange(self.skymap.size))
			phi, theta = CoordTrans(coordout, coordin, phi, np.pi/2-theta)
			n = hp.ang2pix(self.nside, np.pi/2-theta, phi)
			self.skymap = self.skymap[n]
		return self.skymap
	##################################################


	def Configuration2Baseline( self, plusx=False, zero=1e-7, same=1e-6 ) : 
		'''
		Return "different" baseline of cross-correlations

		NEED:
			self.configuration

		SELFSET:
			self.baseline
			self.baselinecount
			(self.lmax)

		plusx:
			plusx=True: if baseline with x<0, convert to whth x>0
	
		zero, same:
			smaller than zero=1e-7 will be consider as 0
			smaller than same=1e-6 (1 micron) will be consider as the same baseline

		return:
			[baseline, baselonecount]
			baseline.shape = (NB, 3), NB is the number of valid baselines, 3 is xyz
			For PAON4, NB=1+6=7 (1 auto, 6 cross)
			xlist, ylist, zlist = baseline.T

		Note that baseline[0] is the auto-correlation [0,0,0]
		'''
		#--------------------------------------------------
		baseline = np.array([[0,0,0]])
		for i in range(len(self.configuration)-1) : 
			baselinei = self.configuration[i+1:] - self.configuration[i]
			#--------------------------------------------------
			if (plusx) : 
				# x>=0
				tf = baselinei[:,0]<-zero
				baselinei[tf] = -baselinei[tf]
				# x==0, y>=0
				tf=(abs(baselinei[:,0])<=zero)*(baselinei[:,1]<-zero)
				baselinei[tf] = -baselinei[tf]
				# x==0, y==0, z>=0
				tf = (abs(baselinei[:,0])<=zero)*(abs(baselinei[:,1])<=zero)*(baselinei[:,2]<-zero)
				baselinei[tf] = -baselinei[tf]
			#--------------------------------------------------
			# Check zero
			x, y, z = baselinei.T
			x[abs(x)<=zero], y[abs(y)<=zero], z[abs(z)<=zero] = 0,0,0
			baselinei[:,0], baselinei[:,1], baselinei[:,2] = x, y, z
			#--------------------------------------------------
			baseline = np.append(baseline, baselinei, 0)  # All
		#--------------------------------------------------
		baseline,baselinecount = RemoveRepetition(baseline,0,same)
		self.baseline = baseline
		self.baselinecount = baselinecount
		#--------------------------------------------------
		if (self.resetlmax) : 
			NB = npfmt(self.pointing).size * len(baseline)
			if (self.lmax+1 > 2*NB) : 
				print self.dtype+'/Warning: self.resetlmax=True, Reconstruct() needs to use the pseudo-inverse: V=M*A, M=V*Ainv=M*A*Ainv. If want A*Ainv=I, then must rowA<=colA. Finally needs: lmax+1<=2*(len(pointing)*len(baseline)). Now self.lmax+1='+str(self.lmax+1)+'>2*NB='+str(2*NB)+', reset self.lmax+1=2*NB='+str(2*NB)
				self.lmax = 2*NB-1
		#--------------------------------------------------
		return [baseline, baselinecount]
	##################################################
	
	
	def xyz2XYZ( self, x, y, z, theta, phi ) : 
		'''
		Convert local xyz to Equatorial XYZ
	
		x, y, z: 
			Can be NumType or ndarray(any shape)
			But x,y,z must have the same shape
	
		theta, phi:
			Position of the local on the sphere
			For the Antenna Array case, lon, lat are the location of the Array, then: 
				phi=lon, theta=np.pi/2-lat
				xyz2XYZ(xa, ya, 0, np.pi/2-lat, lon, compact)
			convert the antenna position to Equatorial XYZ
			theta, phi can be NumType or ndarray(any shape), and both must have the same shape
	
		compact:
			compact the dimension or not? True or False
	
		return:
			[X, Y, Z], Equatorial XYZ
			X.shape=(theta.shape, x.shape)
			return.shape=(3, theta.shape, x.shape)

			For example: 
				theta.shape=(2,3), x.shape=(5,6)
				Then X.shape=(2,3,5,6)
				Element theta[1,2] and x[3,5] will get the result X[1,2,3,5]
		'''
		#--------------------------------------------------
		islistX, shapeX, x, y, z = IslistN(x, y, z, flatten=True)
		islistT, shapeT, theta, phi = IslistN(theta, phi, flatten=True)
		#--------------------------------------------------
		theta, phi = theta[:,None], phi[:,None]
		x, y, z = x[None,:], y[None,:], z[None,:]
		#--------------------------------------------------
		X = -x*np.sin(phi) - y*np.cos(theta)*np.cos(phi) + z*np.sin(theta)*np.cos(phi)
		Y = x*np.cos(phi) - y*np.cos(theta)*np.sin(phi) + z*np.sin(theta)*np.sin(phi)
		Z = y*np.sin(theta) + z*np.cos(theta)
		#--------------------------------------------------
		shape = np.append(shapeT, shapeX)
		X, Y, Z = X.reshape(shape), Y.reshape(shape), Z.reshape(shape)
		'''X.shape=(theta.shape, x.shape)'''
		return np.array([X, Y, Z])


	def Baseline2XYZ( self ) : 
		'''
		Use xyz2XYZ()

		NEED:
			self.baseline
			self.lon 
			self.lat

		SELFSET:
			self.baselineXYZ
	
		lat, lon:
			latitude and longitude of Antanna Array
			lat: -np.pi/2 to +np.pi/2
			lon: 0 to 2*np.pi
			Can be NumType or ndarray(any shape), but both have the same shape
	
		x, y, z:
			= baseline[:,0], baseline[:,1], baseline[:,2]
	
		return:
			[X, Y, Z], Equatorial XYZ
			X.shape=(lat.shape, len(baseline))
			return.shape=(3, lat.shape, len(baseline))

			For example: 
				lat.shape=(2,3), len(baseline)=6
				Then X.shape=(2,3,6)
				Element lat[1,2], lon[1,2] and baseline[5] will get result X[1,2,3,5]
		'''
		islistL, shapeL, lon, lat = IslistN(self.lon, self.lat, flatten=True)
		x, y, z = self.baseline.T
		# baseline to XYZ
		baselineXYZ = self.xyz2XYZ(x,y,z, np.pi/2-lat, lon)
		'''X.shape=(lat.shape, x.shape)'''
		'''baselineXYZ.shape=(3, lat.shape, x.shape)'''
		shape = shapeL + [len(self.baseline)]
		self.baselineXYZ = baselineXYZ.reshape([3]+shape)
		return self.baselineXYZ
	##################################################
	
	
	def BeamResol( self ) : 
		'''
		sigma = Resol / (8*ln2)
			where Resol in [FWHM, BWFN, ResolReza]
		FWHM      = 1.03   * wavelength / Deff
		BWFN      = 1.22   * wavelength / Deff
		ResolReza = 1.1774 * wavelength / Deff

		NEED:
			self.wavelength
			self.Deff
			self.beamwhich

		SELFSET:
			self.beamresol

		return:
			resol.shape=(Deff.shape, wavelength.shape)
		'''
		#--------------------------------------------------
		islistW, shapeW, wavelength = Islist(self.wavelength, flatten=True)
		islistD, shapeD, Deff = Islist(self.Deff, flatten=True)
		if   (self.beamwhich == 'fwhm') : k = 1.03
		elif (self.beamwhich == 'bwfn') : k = 1.22
		elif (self.beamwhich == 'resolreza') : k = (8*np.log(2))**0.5/2 # 1.1774
		#--------------------------------------------------
		resol = k * wavelength[None,:] / Deff[:,None]
		'''resol.shape=(Deff.shape, wavelength.shape)'''
		#--------------------------------------------------
		shape = shapeD + shapeW
		self.beamresol = resol.reshape(shape)
		return self.beamresol
	
	
	def BeamSigma( self ) : 
		'''
		See BeamResol(), sigma = Resol / (8*ln2)

		NEED:
			self.beamresol (Used first)
		 or
			self.wavelength
			self.Deff
			self.beamwhich

		SELFSET:
			self.beamsigma (self.beamresol)
		'''
		try : self.beamsigma = self.beamresol /(8*np.log(2))**0.5
		except : self.beamsigma = self.BeamResol() /(8*np.log(2))**0.5
		return self.beamsigma
	##################################################
	
	
	def GaussianBeam( self, theta=None ) : 
		'''
		sigma = Resol / (8*ln2)
			where Resol in [FWHM, BWFN, ResolReza]
		FWHM      = 1.03   * wavelength / Deff
		BWFN      = 1.22   * wavelength / Deff
		ResolReza = 1.1774 * wavelength / Deff

		NEED:
			self.beamsigma

		SELFSET:
			self.gaussianbeam
	
		theta: 
			in rad, NumType or ndarray(any shape)
			If ==None, use self.nside: theta = hp.pix2ang(nside)
	
		return:
			GaussianBeam.shape=(Deff.shape, wavelength.shape, theta.shape)
		'''
		#--------------------------------------------------
		# Check theta
		if (theta is None) : 
			nside = self.nside
			theta = hp.pix2ang(nside, np.arange(12*nside**2))[0]
		islistT, shapeT, theta = Islist(theta, flatten=True)
		#--------------------------------------------------
		sigma = self.BeamSigma()
		'''sigma.shape=(Deff.shape, wavelength.shape)'''
		#--------------------------------------------------
		beam = np.exp(-theta[None,None,:]**2/2/sigma[:,:,None]**2)
		sigma = theta = 0 #@
		'''beam.shape=(Deff.shape, wavelength.shape, theta.shape)'''
		#--------------------------------------------------
		shape = list(self.beamsigma.shape) + shapeT
		self.gaussianbeam = beam.reshape(shape)
		return self.gaussianbeam
	##################################################
	
	
	def thetaphi2xyz( self, theta=None, phi=None ) : 
		'''
		Spherical coordinate to rectangular coordinate
			theta: 0 to pi
			phi: 0 to 2*pi
		theta, phi can be NumType and ndarray(any shape), same shape
		if theta=0 is North Pole, means transform to Equatorial XYZ

		SELFSET:
			self.thetaphiXYZ
	
		return:
			[x, y, z]
			x.shape = theta.shape = phi.shape
			return.shape=(3, theta.shape)
		'''
		if ((theta is None) or (phi is None)) : 
			nside = self.nside
			theta, phi = hp.pix2ang(nside, np.arange(12*nside**2))
		# Check theta,phi
		islistT, shapeT, theta, phi = IslistN(theta, phi, flatten=True)
		#--------------------------------------------------
		x = np.sin(theta) * np.cos(phi)
		y = np.sin(theta) * np.sin(phi)
		z = np.cos(theta)
		theta = phi = 0 #@
		#--------------------------------------------------
		x, y, z = x.reshape(shapeT), y.reshape(shapeT), z.reshape(shapeT)
		self.thetaphiXYZ = np.array([x, y, z])
		return self.thetaphiXYZ
	##################################################
	
	
	def BaselineBeam( self, lon=None, lat=None, baseline=None, theta=None, phi=None, wavelength=None, Deff=None, pointing=None, which=None, compact=False, selfset=False ):
		'''
		NEED:
			self.lon
			self.lat
			self.baseline
			self.wavelength
			self.Deff
			self.pointing
			self.beamwhich
			(self.nside)

		SELFSET:
			self.baselinebeam

		Return GaussianBeam * np.exp(1j*k*rij), beam with phase

		theta, phi:
			If ==None, theta, phi = hp.pix2ang(self.4*nside, ...)
			Note that here we use 4*nside
	
		reutrn 
			baselinebeam.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), theta.shape)
		'''
		#--------------------------------------------------
		islistW, shapeW, wavelength = Islist(self.wavelength, flatten=True)
		islistD, shapeD, Deff   = Islist(self.Deff, flatten=True)
		islistP, shapeP, pointing = Islist(self.pointing, flatten=True)
		islistL, shapeL, lon, lat = IslistN(self.lon, self.lat, flatten=True)
		shapeB = [len(self.baseline)]
		#--------------------------------------------------
		'''float64: 84, 70  float32: 84, 70  => self.nside=256'''
		# Check theta, phi
		if ((theta is None) or (phi is None)) : 
			nside = 4*self.nside #@
			theta, phi = np.array(hp.pix2ang(nside, np.arange(12*nside**2)), self.bit)
		else : theta, phi = np.array(theta, self.bit), np.array(phi, self.bit)
		islistT, shapeT, theta, phi = IslistN(theta, phi, flatten=True)
		'''float64: 372, 262     float32: 516, 165'''
		#--------------------------------------------------
		#--------------------------------------------------
		# baseline to XYZ
		XYZb = self.Baseline2XYZ()
		'''Xb.shape=(lat.shape, len(baseline))'''
		# theta,phi to XYZ
		Xt, Yt, Zt = np.array(self.thetaphi2xyz(theta, phi), self.bit) # uvo
		'''dXt=dYt=dZt=0'''
		'''float64: 1110, 550     float32:  900, 309'''
		theta = phi = 0 #@
		'''float64: 1110, 358     float32:  900, 214'''
		'''Xt.shape=theta.shape'''
		Xb, Yb, Zb = XYZb[:,:,:,None]
		# Projected baseline
		rij = np.array(Xb*Xt + Yb*Yt + Zb*Zt, self.bit)
		'''float64: 1110, 454     float32:  900, 262'''
		'''rij.shape=(lat.shape, len(baseline), theta.shape)'''
		#--------------------------------------------------
		phase = np.array(2*np.pi/wavelength[:,None,None,None] * rij, self.bit)
		'''dphase=0'''
		'''float64: 1110, 550     float32:  900, 310'''
		rij = 0 #@
		'''float64: 1110, 454     float32:  900, 262'''
		'''phase.shape=(wavelength.shape, lat.shape, len(baseline), theta.shape)'''
		#--------------------------------------------------
		XYZp = self.thetaphi2xyz(np.pi/2-(lat[None,:]+pointing[:,None]), lon[None,:]+pointing[:,None]*0)
		'''Xp.shape=[pointing.shape, lat.shape]'''
		Xp, Yp, Zp = XYZp[:,:,:,None]
		rb = np.array(Xp*Xt + Yp*Yt + Zp*Zt, self.bit)
		'''float64: 1110, 550     float32:  900, 310'''
		Xp = Yp = Zp = XYZp = Xb = Yb = Zb = XYZB = Xt = Yt = Zt = XYZt = 0 #@
		'''float64: 852, 262     float32: 756, 166'''
		'''rb.shape=(pointing.shape, lat.shape, theta.shape)'''
		gaussianbeam = np.array(self.GaussianBeam(np.arccos(rb)), self.bit)
		'''float64: 852, 358     float32: 756, 214'''
		rb = 0 #@
		'''float64: 852, 262     float32: 756, 166'''
		'''gaussianbeam.shape=(Deff.shape, wavelength.shape, pointing.shape, lat.shape, theta.shape)'''
		#--------------------------------------------------
		# Move axis
		'''phase.shape from (wavelength.shape, lat.shape, len(baseline), theta.shape) to (lat.shape, wavelength.shape, len(baseline), theta.shape)'''
		phase = ArrayAxis(phase, 1, 0, 'move')
		'''gaussianbeam.shape from (Deff.shape, wavelength.shape, pointing.shape, lat.shape, theta.shape) to (lat.shape, Deff.shape, wavelength.shape, pointing.shape, theta.shape)'''
		gaussianbeam = ArrayAxis(gaussianbeam, 3, 0, 'move')
		# Add None axis
		phase = phase[:,None,:,None,:,:]
		gaussianbeam = gaussianbeam[:,:,:,:,None,:]
		baselinebeam = gaussianbeam * np.exp(1j*phase)
		'''float64: 1020, 454     float32:  756, 262'''
		gaussianbeam = phase = 0 #@
		'''float64: 1020, 262     float32:  756, 166'''
		'''baselinebeam.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), theta.shape)'''
		#--------------------------------------------------
		shape = shapeL+shapeD + shapeW + shapeP + shapeB + shapeT
		self.baselinebeam = baselinebeam.reshape(shape)
		#--------------------------------------------------
		'''float64: 1020, 262     float32:  756, 166'''
		'''baselinebeam.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), theta.shape)'''
		return self.baselinebeam
	##################################################
	

	def LM2Index( self, idxfunc, l, m, lmax, order=None ) : 
		'''
		idxfunc:
			str or FunctionType
			If str: 'Almds' or 'Alm', which are used in JSkyMap
			If FunctionType, use this function: idx=idxfunc(l,m,lmax)
	
		l, m, lmax: 
			l, m: NumType or ndarray(any shape)
			lmax: Must be NumType !!!
	
		order:
			'2D': return 2D valid [idx, tfX]
			None: remove all invalid and return 1D vaild [l, m, idx]
			'l': as None, but sort l from small to large
			'm': as None, but sort m from small to large
			'i': as None, but sort idx from small to large
		'''
		#--------------------------------------------------
		def idxAlmds( l, m, lmax=None ) : 
			''' If you want to broadcase, please reshape l, m before pass them to this function '''
			idx = l*(l+1)+m
			return idx
		#--------------------
		def idxAlm( l, m, lmax ) : 
			''' If you want to broadcase, please reshape l, m before pass them to this function '''
			idx = lmax*m - m*(m-1)/2 + l
			return idx
		#--------------------------------------------------
		if (Type(idxfunc) == str) : 
			if (idxfunc.lower() == 'almds') : idxfunc = idxAlmds
			elif (idxfunc.lower() == 'alm') : idxfunc = idxAlm
			else : idxfunc = idxAlm
		#--------------------------------------------------
		if (order is None) : order = ''
		else : order = order.lower()
		islistL, shapeL, l = Islist(l, flatten=True)
		islistM, shapeM, m = Islist(m, flatten=True)
		#--------------------------------------------------
		# 0<=l<=lmax
		tfL = (0<=l)*(l<=lmax)  # True is valid, False is invalid
		l[True-tfL] = 0  # Set invalid to be 0 so that we won't encounter ERROR when using it
		# -lmax<=m<=lmax
		tfM = (abs(m)<=lmax)  # True is valid, False is invalid
		m[True-tfM] = 0
		#--------------------------------------------------
		# healpix to (l,m)
		idx = idxfunc(l[:,None], m[None,:], lmax)
		tfX = np.array(tfL[:,None]*tfM[None,:], bool)
		tfL = tfM = 0 #@
		'''idx.shape=(l.shape, m.shape)'''
		#--------------------------------------------------
		# Remove abs(m)>l in idx
		lX = l[:,None] + m[None,:]*0  # l in idx (2D)
		mX = l[:,None]*0 + m[None,:]  # m in idx (2D)
		tfX[(abs(mX)-lX)>0] = False
		idx[True-tfX] = 0
		#--------------------------------------------------
		if (islistL==islistM==False) : 
			l, m, idx, tfX = l.take(0), m.take(0), idx.take(0), tfX.take(0)
			if (tfX == False) : l, m, idx = lmax, lmax, lmax*(lmax+2) # Take lmax so that the value ~0
			if (order == '2d') : return [idx, tfX]
			else : return np.array([l, m, idx])
		#----------
		if (order == '2d') : 
		#	if (compact) : 
		#		if (not islistL) : shapeL = []
		#		if (not islistM) : shapeM = []
			shape = shapeL + shapeM
			idx, tfX = idx.reshape(shape), tfX.reshape(shape)
			return [idx, tfX] 
			'''2D idx.shape=(l.shape, m.shape)'''
		#----------
		lmi = np.array([lX[tfX], mX[tfX], idx[tfX]])  # 3x 1D
		if (order == '') : return lmi
		if (order == 'l') : return Sort(lmi, ('row',0))
		if (order == 'm') : return Sort(lmi, ('row',1))
		if (order == 'i') : return Sort(lmi, ('row',2))
	##################################################
	
	
	def BeamLM( self, release=True ):
		'''
		NEED:
			self.lmax
			self.baselinebeam

		SELFSET:
			self.beamlm

		baselinebeam = GaussianBeam*np.exp(1j*k*rij), is a complex beam
		beamlm is the Spherical Harmonic transforma of baselinebeam

		However, hp.map2alm() just handles real map. For complex map baselinebeam, we use a special method as shown here.

		release:
			Release the memory.
			Just delete "baselinebeam", because it may contain several Healpix Maps. For other terms, they just use a little memory, leave them.

		return:
			beamlm.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), (lmax+1)**2)
		'''
		#--------------------------------------------------
		shapeB = list(self.baselinebeam.shape)
		NB = npfmt(shapeB)[:-1].prod()
		self.baselinebeam = self.baselinebeam.reshape(NB, shapeB[-1])
		'''float64: 405, 276     float32: 309, 180'''
		#--------------------------------------------------
		lmax = self.lmax
		l = np.arange(lmax+1)
		mn = np.arange(-lmax, 0)
		mp = np.arange(0, lmax+1)
		morder, idxdsn = self.LM2Index('Almds', l, mn, lmax)[1:]
		idxdsp = self.LM2Index('Almds', l, mp, lmax)[2]
		idxmn  = self.LM2Index('Alm', l, -mn, lmax)[2]
		idxmp  = self.LM2Index('Alm', l,  mp, lmax)[2]
		'''float64:405,276  float32:261,179 => self.nside=256'''
		#--------------------------------------------------
		almR, almI = [], []
		for i in range(NB) : 
			almR.append(np.array(hp.map2alm(self.baselinebeam[i].real, lmax=lmax), self.cbit))
			almI.append(np.array(hp.map2alm(self.baselinebeam[i].imag, lmax=lmax), self.cbit))
		almR, almI = npfmt(almR), npfmt(almI)
		'''float64: 498, 281     float32: 400, 183'''
		#--------------------------------------------------
		almds = np.zeros([NB, (lmax+1)**2], self.cbit) # Value of alm of baselinebeam
		almds[:,idxdsn] = (-1)**morder[None,:] * (np.conj(almR[:,idxmn]) + 1j*np.conj(almI[:,idxmn]))
		almds[:,idxdsp] = almR[:,idxmp] + 1j*almI[:,idxmp]
		'''float64: 502, 285     float32: 403, 185'''
		#--------------------------------------------------
		if (release) : self.__dict__.pop('baselinebeam')
		else : self.baselinebeam = self.baselinebeam.reshape(shapeB)
		'''float64: 310, 93     float32: 403, 90'''
		# Reshape
		shapeA = shapeB[:-1] + [almds.shape[-1]]
		self.beamlm = almds.reshape(shapeA)
		#--------------------------------------------------
		'''float64: 502, 285     float32: 403, 185'''
		'''beamlm.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), (lmax+1)**2)'''
		return self.beamlm
	##################################################


	def VisibilityM( self ) : 
		'''
		NEED:
			self.lmax
			self.beamlm
			self.skymap

		SELFSET:
			self.visibm

		return:
			visibm=Vij(m) is the FFT of observation data visibt=Vij(t)
			visibm.shape=(2, lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), lmax+1)
			visibm[0] with m>=0
			visibm[1] with m<=0
		'''
		#--------------------------------------------------
		shapeB = list(self.beamlm.shape)
		'''beamlm.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), (lmax+1)**2)'''
		# Reshape to beamlm.shape=(pointing.size*baseline.size, (lmax+1)**2)
		NB = npfmt(shapeB[:-1]).prod()
		self.beamlm = self.beamlm.reshape(NB, shapeB[-1])
		#--------------------------------------------------
		lmax = self.lmax
		l = m = np.arange(lmax+1)
		# Index of maplm
		idxmaplm, idxtf = self.LM2Index('Alm', l, m, lmax, '2D')
		# Index of Vij(m)
		idxbeamlmp = self.LM2Index('Almds', l, -m, lmax, '2D')[0]
		# Index of Vij(-m)
		idxbeamlmn = self.LM2Index('Almds', l, m, lmax, '2D')[0]
		#--------------------------------------------------
		# alm of skymap
		try : maplm = np.array(hp.map2alm(self.skymap, lmax=lmax), self.cbit)
		except : Raise(Exception, 'self.skymap is not a Healpix Map')
		#--------------------------------------------------
		# visibm matrix
		visibm = []
		# Vij(m), 0,1,2,...,lmax
		visibm.append(((-1)**m * maplm[idxmaplm] * self.beamlm[:,idxbeamlmp] * idxtf).sum(-2).astype(complex))
		# Vij(-m), 0,-1,-2,...,-lmax
		visibm.append((maplm[idxmaplm] * np.conj(self.beamlm[:,idxbeamlmn]) * idxtf).sum(-2).astype(complex))
		#--------------------------------------------------
		self.beamlm = self.beamlm.reshape(shapeB)
		'''beamlm.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), (lmax+1)**2)'''
		#--------------------------------------------------
		shapeV = [2] + shapeB[:-1] + [lmax+1]
		self.visibm = np.array(visibm).reshape(shapeV)
		'''visibm.shape=(2, lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), lmax+1)'''
		'''visibm[0]: 0,1,2,...,lmax, visibm[1]: 0,-1,-2,...,-lmax'''
		return self.visibm
	##################################################


	def Reconstruct( self, cond=0.01, rcond=0.02 ) : 
		'''
		visibm:
			FFT of observation visibt=Vij(t)

		selfset:
			self.lmax
			self.visibm
			self.beamlm
			self.reskymap
		'''
		#--------------------------------------------------
		shapeB = list(self.beamlm.shape)
		'''beamlm.shape=(lat.shape, Deff.shape, wavelength.shape, pointing.shape, len(baseline), (lmax+1)**2)'''
		## Reshape to beamlm.shape=(pointing.size*baseline.size, (lmax+1)**2)
		NB = npfmt(shapeB[:-1]).prod()
		self.beamlm = self.beamlm.reshape(NB, shapeB[-1])
		#--------------------------------------------------
		if (self.resetlmax) : 
			if (self.lmax+1 > 2*NB) : 
				Raise(Exception, self.dtype+'.Reconstruct()/Error: Reconstruct() needs to use the pseudo-inverse: V=M*A, M=V*Ainv=M*A*Ainv. If want A*Ainv=I, then must rowA<=colA. Finally needs: lmax+1<=2*(len(pointing)*len(baseline)). Now self.lmax+1='+str(self.lmax+1)+'>2*NB='+str(2*NB))
		#--------------------------------------------------
		shapeV = list(self.visibm.shape)
		self.visibm = self.visibm.reshape(2*NB, shapeV[-1])
		'''visib.shape=(2*NB, lmax+1)'''
		#--------------------------------------------------
		lmax = self.lmax
		remaplm = np.zeros([(lmax+1)*(lmax+2)/2,], complex)
		A = []
		for m in range(lmax+1) : 
			Ap, An, l = [], [], np.arange(m, lmax+1)
			idxbeamlmp = self.LM2Index('Almds', l, -m, lmax)[2]
			idxbeamlmn = self.LM2Index('Almds', l,  m, lmax)[2]
			#----------
			# (l,49)
			Ap = ((-1)**m * self.beamlm[:,idxbeamlmp]).T
			An = (np.conj(self.beamlm[:,idxbeamlmn])).T
			# (l,49+49), first 49: +m, second 49: -m
			A.append(np.matrix(np.append(Ap, An, 1)))
			'''A.shape=(x, 2*NB)'''
		#	Ainv = np.matrix(spla.pinv2(A[-1], cond, rcond))
			np.save('A', A[-1])
			Raise()
			Ainv = np.matrix(spla.pinv2(A[-1]))

			# Reconstruct maplm
			idxmaplm   = self.LM2Index('Alm',   l,  m, lmax)[2]
			# (1,x) = (1,2*NB)x(2*NB,x)
			remaplm[idxmaplm] = np.array(np.matrix(self.visibm[:,m].flatten()) * Ainv).flatten()
		#--------------------------------------------------
		self.reskymap = hp.alm2map(remaplm, self.nside, lmax, verbose=False)
		#--------------------------------------------------
		self.visibm = self.visibm.reshape(shapeV)
		self.beamlm = self.beamlm.reshape(shapeB)
		return self.reskymap


##################################################
##################################################
##################################################


class CArray( object ) : 
	'''
	In some array, (such as healpix map), there may be lots of pixels with the same value (for example, 0), cost lots of memory.
	This class is use to save the "valid" data so that we can decrease the menory consumption.

	Usage:
		b = CArray(a, para1, para2)
		a = b.Take()
	'''
	dtype = 'class:'+sys._getframe().f_code.co_name

	def __init__( self, *arg, **kwargs ) : 
		'''
		value, err: range [value-err, value+err] will be consider as "invalid", don't save then.

		perc, bins:
			Save the percentage of the data
			perc, bins are used in ProbabilityDensity()
			perc: str, 'xx%', for example, '20%'
			bins: int
		'''
		arglist = [[('array','AnyType'),('value','NumType'),('err','NumType')],
		           [('array','AnyType'),('perc',str),('bins','NumType')]]
		arglist, which = ArgInit(arglist, arg, kwargs)
		self.value, self.err, self.perc, self.bins = None, None, None, None
		if (which == 0) : self.value, self.err = arglist[1], abs(arglist[2])
		if (which == 1) : 
			if (arglist[1][-1] == '%') : 
				self.perc = float(arglist[1][:-1])/100
				self.bins = int(round(arglist[2]))
			else : Raise(Exception, 'perc="'+arglist[1]+'" not as "xx%"')
		self.array = npfmt(arglist[0])
		self.shape = self.array.shape
		self.Compress()
		##################################################


	def Compress( self ) : 
		self.array = self.array.flatten()
		if (self.value is not None) : 
			vmax, vmin = self.value+self.err, self.value-self.err
			tf = (self.array>vmax)+(self.array<vmin)
		elif (self.perc is not None) : 
			x,xc,y = ProbabilityDensity(self.array, self.bins, density=False)
			x, y = x[::-1], y[::-1]
			counto, countn, N = 0, 0, 1.*self.array.size
			for i in range(len(y)) : 
				if (abs(countn-self.perc)<1e-3) : break
				counto = countn
				countn += y[i]/N
			dco,dcn = abs(counto-self.perc), abs(countn-self.perc)/2
			if (dcn >= dco) : i -=1
			tf = False
			for j in range(i) : 
				dx = abs(x[j] - x[j+1])
				xmax, xmin = x[j]+dx/100., x[j+1]-dx/100.
				tf += (xmin<=self.array)*(self.array<=xmax)
		#--------------------------------------------------
		self.idx = np.arange(self.array.size)[tf]
		self.data = self.array[self.idx]
		self.__dict__.pop('array')
		##################################################


	def Take( self ) : 
		array = np.zeros(self.shape, self.data.dtype).flatten()
		array[self.idx] = self.data
		array = array.reshape(self.shape)
		return array


##################################################
##################################################
##################################################


def MergeGnu( gnudir, dayname, plot=True, chan2antHV=PAON4Chan2AntHV(1), skip=False ) : 
	'''
	gnudir:
		Absolute path of the folder saving the gnu*.fits
		Can be str or list of str

	dayname:
		Name of the data set.
		Must has the same shape as gnudir
		For example, dayname='CygA665S1dec15'

	plot:
		Plot some figure
		If plot=False, don't need to set chan2antHV
	'''
	#--------------------------------------------------
	if (Type(gnudir) == str) : gnudir = [gnudir]
	if (Type(dayname) == str) : dayname = [dayname]
	outdirthis = OutDir(None)
	outdir = OutDir(outdirthis+'MergeGnu/')
	outdir = OutDir(StrListAdd(outdir,dayname))
	#--------------------------------------------------
	rename = []
	for j in range(len(gnudir)) : 
		outname = outdir[j]+'gnu_'+dayname[j]+'.npy'
		rename.append(outname)
		if (skip) : continue
		namedir = DirStr(os.path.expanduser(gnudir[j]))
		fitsname = ShellCmd('ls '+namedir+dayname[j]+'_gnu_rdvisip4_*.fits')
		#--------------------------------------------------
		if (len(fitsname) <= 20) : ra = range(len(fitsname))
		else: ra = np.linspace(0,len(fitsname),20).astype(int)[:-1]
		gnu = []
		for i in ra : 
			a = pyfits.getdata(fitsname[i], 0)
			gnu.append(a)
			a = 0 #@
		gnu = npfmt(gnu)
		#--------------------------------------------------
		gnutmp = []
		for i in range(gnu.shape[1]) : 
			gnutmp.append(SelectLeastsq(gnu[:,i], 0)[0].flatten())
		gnu = Smooth(npfmt(gnutmp), 1, 30, 20)
		gnu[gnu<=0] = gnu[gnu>0].min()
		#--------------------------------------------------
		np.save(outname, gnu)
		print outname+'  -->  saved'
		#--------------------------------------------------
		if (plot) : 
			vmax = gnu.max()*0.9
			freq = np.linspace(1250, 1500, gnu.shape[1])
			color = plt_color(len(gnu))
			for i in range(len(gnu)) : 
				plt.plot(freq, gnu[i]/vmax, color=color[i], lw=2, label=chan2antHV[i])
			plt.legend()
			plt.xlabel(r'$\nu \, (MHz)$', size=20)
			plt.xlim(1250, 1500)
			plt_axes('x', 'both', [25,5], fontsize=14)
			plt.ylabel(r'$g(\nu)$ (A.U.)', size=20)
			plt_axes('y', 'major', 0.1, '%.1f', fontsize=14)
			plt.title(dayname[j]+r', $g(\nu)$', size=20)
			plt.savefig(outdir[j]+dayname[j]+'_gnu_8auto.png')
			plt.close()
	return rename


##################################################
##################################################
##################################################


class Phase( object ) : 
	dtype = 'class:'+sys._getframe().f_code.co_name


	def __init__( self, freq=None, phase=None, axis=None ) : 
		'''
		phase:
			 Can be N-D array.

		axis:
			Along which axis is the phase (generally is the frequency axis).
			axis starts from 0.
		'''
		if (freq  is not None) : self.freq = npfmt(freq)
		if (axis  is not None) : self.axis = axis
		if (phase is not None) : 
			self._phase = npfmt(phase)
			self.shape = self._phase.shape
			# Move from axis to -1, and flatten
			self._phase = ArrayAxis(self._phase, self.axis, -1, 'move')
			self._shape = npfmt(self._phase.shape)
			self._phase = self._phase.flatten()
		self._smooth = False
		self._initialguess = False


	def Smooth( self, per=None, times=1, medfilt=False ) : 
		'''
		Smooth the phase.
		Note that phase/angle is periodic with period 2pi, so we use sin(phase) (a continous function) to Smooth().
		'''
		Nf = self.shape[self.axis]
		#----------
		if (per is None) : per = Nf/50
		per = per if(per%2==1)else per+1
		if (per <= 1) : per = 3
		#----------
		vis = np.exp(1j*self._phase)
		cosp, sinp = vis.real*1, vis.imag*1
		vis = 0 #@
		#----------
		# medfilt
		if (medfilt) : 
			for i in range(int(self._shape[:-1].prod())) : 
				cosp[i*Nf:(i+1)*Nf] = spsn.medfilt(cosp[i*Nf:(i+1)*Nf], per)
				sinp[i*Nf:(i+1)*Nf] = spsn.medfilt(sinp[i*Nf:(i+1)*Nf], per)
		#----------
		# Smooth
		cosp = cosp.reshape(self._shape)
		sinp = sinp.reshape(self._shape)
		sinp = Smooth(sinp, -1, per, times)
		cosp = Smooth(cosp, -1, per, times)
		#----------
		self._phase=(np.angle(cosp+1j*sinp)%(2*np.pi)).flatten()
		self._smooth = True
	

	def Mean( self, per=None ) : 
		'''
		Return phase.mean()
		per: Used in spsn.medfilt(phase, per)
		Won't affect the self. properties of the instance of class:Phase
		'''
		Nf = self.shape[self.axis]
		vis = np.exp(1j*self._phase)
		#----------
		if (per is not None) : 
			per = per if(per%2==1)else per+1
			if (per <= 1) : per = 3
			for i in range(int(self._shape[:-1].prod())) : 
				vis.real[i*Nf:(i+1)*Nf] = spsn.medfilt(vis.real[i*Nf:(i+1)*Nf], per)
				vis.imag[i*Nf:(i+1)*Nf] = spsn.medfilt(vis.imag[i*Nf:(i+1)*Nf], per)
		#----------
		shapevis = self._shape*1
		shapevis[-1] = 1
		vis = vis.reshape(self._shape)
		vis = vis.mean(-1).reshape(shapevis)
		vis = np.angle(vis) %(2*np.pi)
		vis = ArrayAxis(vis, -1, self.axis, 'move')
		return vis

	
	def InitialGuess( self, progress=False ) : 
		'''
		phase = x0*freq + x1
		Return the initial guess of x0 and x1.
		'''
		self._x0guess = np.zeros(int(self._shape[:-1].prod()))
		self._x1guess = self._x0guess*0
		#----------
		Nf = self.freq.size
	#	if (not self._smooth) : self.Smooth()
		#----------
		progressbar = ProgressBar('Phase.InitialGuess():', int(self._shape[:-1].prod())) #@
		for i in xrange(int(self._shape[:-1].prod())) : 
			if (progress) : progressbar.Progress() #@
			phase = self._phase[i*Nf:(i+1)*Nf]
			x0 = (phase[5:]-phase[:-5]) / (self.freq[5:]-self.freq[:-5])
			# Don't use the edge
			x0 = x0[Nf/10:Nf*9/10]
			#----------
			sign = np.sign(np.sign(x0).sum())
			x0 = x0[np.sign(x0)==sign]
			per = x0.size/20
			per = per if(per%2==1)else per+1
			x0 = spsn.medfilt(x0, per)
			if (x0.size < 3*per) : x0 = x0.mean()
			else : x0 = x0[per:x0.size-per].mean()
			def sinphasefunc(f, x1) : 
				return np.sin((x0*f + x1) %(2*np.pi))
		#	x1 = FuncFit(sinphasefunc, self.freq[Nf/10:Nf*9/10], np.sin(phase[Nf/10:Nf*9/10]), np.pi)[0]
			x1 = Leastsq(sinphasefunc, self.freq[Nf/10:Nf*9/10], np.sin(phase[Nf/10:Nf*9/10]), np.pi)
			self._x0guess[i] = x0
			self._x1guess[i] = x1 %(2*np.pi)
		#----------
		self._initialguess = True


	def ReturnInitialGuess( self ) : 
		x0shape = self._shape*1
		x0shape[-1] = 1
		x0guess = self._x0guess.reshape(x0shape)
		x1guess = self._x1guess.reshape(x0shape)
		x0guess = ArrayAxis(x0guess, -1, self.axis, 'move')
		x1guess = ArrayAxis(x1guess, -1, self.axis, 'move')
		return [x0guess, x1guess]
	
	
	def Fit( self, progress=False ) : 
		'''
		phase = x0*freq + x1
		Fit x0 and x1.
		'''
		self._x0fit = np.zeros(int(self._shape[:-1].prod()))
		self._x1fit = self._x0fit*0
		self._x0fiterr = self._x0fit*0
		self._x1fiterr = self._x0fit*0
		Nf = self.freq.size
		#----------
		if (not self._initialguess) : self.InitialGuess()
		def sinphasefunc(f, x) : 
			return np.sin((x[0]*f + x[1]) %(2*np.pi))
		#----------
		progressbar = ProgressBar('Phase.Fit():', int(self._shape[:-1].prod())) #@
		for i in xrange(int(self._shape[:-1].prod())) : 
			if (progress) : progressbar.Progress() #@
			phase = self._phase[i*Nf:(i+1)*Nf]
			(x0,x1), (x0err,x1err) = FuncFit(sinphasefunc, self.freq[Nf/10:Nf*9/10], np.sin(phase[Nf/10:Nf*9/10]), [self._x0guess[i], self._x1guess[i]])
			self._x0fit[i] = x0
			self._x1fit[i] = x1 %(2*np.pi)
			self._x0fiterr[i] = x0err
			self._x1fiterr[i] = x1err %(2*np.pi)


	def ReturnFit( self ) : 
		x0shape = self._shape*1
		x0shape[-1] = 1
		x0fit = self._x0fit.reshape(x0shape)
		x1fit = self._x1fit.reshape(x0shape)
		x0fiterr = self._x0fiterr.reshape(x0shape)
		x1fiterr = self._x1fiterr.reshape(x0shape)
		x0fit = ArrayAxis(x0fit, -1, self.axis, 'move')
		x1fit = ArrayAxis(x1fit, -1, self.axis, 'move')
		x0fiterr = ArrayAxis(x0fiterr, -1, self.axis, 'move')
		x1fiterr = ArrayAxis(x1fiterr, -1, self.axis, 'move')
		return [x0fit, x1fit, x0fiterr, x1fiterr]


	def ReturnPhase( self ) : 
		phase = self._phase.reshape(self._shape)
		phase = ArrayAxis(phase, -1, self.axis, 'move')
		return phase


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################


class QueueFor( object ) : 
	dtype = 'class:'+sys._getframe().f_code.co_name
	"""
	PoolFor() is much simpler than QueueFor() !!!
	And the time consumptions are the same.

	Nstart   = 0
	Nend     = 30000
	Nprocess = 4  # Sometimes you increase Nprocess, the total speed does not increase

	def DoMultiprocess( n, queuefor ) : 
		n1, n2 = queuefor.nsplit[n], queuefor.nsplit[n+1]
		progressbar = ProgressBar('pid='+str(os.getpid())+':', n2-n1)
		pfit = []
		for i in xrange(n1, n2) : 
			progressbar.Progress()
			pfit.append( FuncFit(func, x, V[i], p0)[0] )
		queuefor.put( npfmt(pfit) )

	queuefor = QueueFor(Nstart, Nend, Nprocess)
	getlist = []
	
	processlist = []
	for i in xrange(queuefor.Nprocess) : 
		processlist.append( multiprocessing.Process(target=DoMultiprocess, args=(i, queuefor)) )

	for process in processlist : process.start()

	while (len(getlist) != 2*queuefor.Nprocess) : 
		# You must .get() after .start() and before .join(), otherwise maybe jam in .join() forever.
		getlist.append(queuefor.get())

	for process in processlist : process.join()
	
	data = queuefor.data(getlist)
	data = np.concatenate(data)
	del queuefor, getlist, processlist, process

	Nend-Nstart = 30000
	Nprocess        :  1    2    3    4    5    6    9    10
	Time consumption: 2:13 1:20 1:07 1:03 1:04 1:05 1:01 1:02
	"""

	def __init__( self, Nstart, Nend, Nprocess ) : 
		if (Nprocess > Nend-Nstart) : Nprocess = Nend-Nstart
		self.Nstart, self.Nend, self.Nprocess = Nstart, Nend, Nprocess
		self.nsplit = np.linspace(Nstart, Nend, Nprocess+1).astype(int)
		self.queue = multiprocessing.Queue()

	def put( self, obj ) : 
		self.queue.put(obj)
		self.queue.put('pid='+str(os.getpid()))

	def get( self ) : 
		return self.queue.get()

	def empty( self ) : 
		return self.queue.empyt()

	def data( self, getlist ) : 
		data, pid = [], []
		for i in xrange(len(getlist)) : 
			if (type(getlist[i]) == str) : 
				if (getlist[i][:4] == 'pid=') : 
					pid.append(int(getlist[i][4:]))
				else : data.append(getlist[i])
			else : data.append(getlist[i])
		pid = np.array(pid) + 1j*np.arange(len(pid))
		pid = np.sort(pid).imag.astype(int)
		datar = []
		for i in xrange(len(pid)) : 
			datar.append(data[pid[i]])
		return datar


##################################################
##################################################
##################################################


def RandomSeed() : 
	seed = int(('%.11f' % time.time()).split('.')[1][2:] + str(int(np.random.random()*99)))
	np.random.seed(seed)
	return seed


##################################################
##################################################
##################################################


##################################################
##################################################
##################################################



##################################################
##################################################
##################################################


#def Fits2Hdf5( fitspath, *arg ) : 
#	'''
#	Convert .fits to .hdf5
#
#	*arg:
#		dicts that will update to each level .attrs
#		If you want to update level 2 and leave level 0 and 1, you can: 
#			FitsHdf5( fitspath, {}, {}, {'a':1,'b':2} )
#		OR FitsHdf5( fitspath, [{}, {}, {'a':1,'b':2}] )
#	'''
#	hdf5name0 = fitspath[:-5] + '.hdf5'
#	fitspath = os.path.expanduser(fitspath)
#	hdf5name = os.path.expanduser(hdf5name0)
#	fo = pyfits.open(fitspath)
#	if (len(arg)==1 and type(arg)!=dict) : arg = arg[0]
#	#--------------------------------------------------
#	if (os.path.exists(hdf5name)) : 
#		os.system('mv '+hdf5name+' '+hdf5name[:-5]+'_old.hdf5')
#	try : ho = h5py.File(hdf5name, 'w')
#	except : 
#		hdf5name = hdf5name.split('/')[-1]
#		ho = h5py.File(hdf5name, 'w')
#	#--------------------------------------------------
#	maskkeys = ['XTENSION','BITPIX','NAXIS','NAXIS1','NAXIS2','NAXIS3','PCOUNT','GCOUNT','SOPHYAFV']
#	if (len(arg) > 0) : ho.attrs.update(arg[0])
#	for i in xrange(len(fo.info(0))) : 
#		name = fo[i].name
#		if (name == '') : name = 'DATA_'+str(i)
#		ho[name] = fo[i].data
#		hdr = fo[i].header
#		key, value, comment =hdr.keys(),hdr.values(),hdr.comments
#		ncom = 0
#		for j in xrange(len(hdr)) : 
#			if (key[j] in maskkeys) : continue
#			if (key[j] == 'COMMENT') : 
#				key[j] = key[j] + '_'+str(ncom)
#				ncom += 1
#			ho[name].attrs[key[j]] = value[j]
#			if (comment[j] != '') : 
#				ho[name].attrs[key[j]+'_COM'] = comment[j]
#		if (i < len(arg)-1) : ho[name].attrs.update(arg[i+1])
#	print hdf5name0+'  -->  saved'
#	ho.close()


##################################################
##################################################
##################################################


#class Fits2Hdf5( object ) : 
#
#
#	def __init__( self, fitspath=None ) : 
#		if (fitspath is None) : return
#		self.fitspath = fitspath
#		self.hdf5path = fitspath[:-5]+'.hdf5'
#		self.fits = pyfits.open(os.path.expanduser(fitspath))
#		self._h5pyFile()
#		self.IgnoreKeys()
#
#
#	def _h5pyFile( self ) : 
#		hdf5path = os.path.expanduser(self.hdf5path)
#		if (os.path.exists(hdf5path)) : 
#			os.system('mv '+hdf5path+' '+hdf5path[:-5]+'_old.hdf5')
#		try : self.hdf5 = h5py.File(hdf5path, 'w')
#		except : 
#			self.hdf5path = hdf5path.split('/')[-1]
#			self.hdf5 = h5py.File(self.hdf5path, 'w')
#
#
#	def UpdateHDUAttrs( self, hduattrs ) : 
#		'''
#		For example, Fits contains 3 HDUs: Auto, CrossReal, CrossImag, then its Hdf5 has 4 attrs: f.attrs, f['Auto'].attrs, f['CrossReal'].attrs, f['CrossImag'].attrs
#		self.UpdateHDUAttrs() is used to update these 4 attrs
#		len(attrslist) <= len(HDUs), it will update one by one until finish
#
#		hduattrs:
#			list of dict: [{}, {}, {}, {}]
#			hduattrs[0] is for f.attrs
#			hduattrs[1] is for f['Auto'].attrs
#			......
#		'''
#		self.hduattrs = attrslist
#
#
#	def UpdateHDUName( self, hduname ) : 
#		''' Use this hduname to instead of that in FITS '''
#		self.hduname = hduname
#
#
#	def Append( self, array, name, attrs=None ) : 
#		'''
#		Append new array to .hdf5
#		attrs: dict{}
#		'''
#		self.hdf5[name] = array
#		self.hdf5[name].attrs.update(attrs)
#
#
#	def IgnoreKeys( self, ignorekeys=None ) : 
#		'''
#		ignorekeys:
#			keys in Fits that won't save to Hdf5
#		'''
#		self.ignorekeys = ['XTENSION','BITPIX','NAXIS','NAXIS1','NAXIS2','NAXIS3','PCOUNT','GCOUNT','SOPHYAFV']
#		if (ignorekeys is None) : return
#		istype = IsType()
#		if (istype.isstr(ignorekeys)) : ignorekeys = [ignorekeys]
#		elif (type(ignorekeys) in [list, tuple, np.ndarray]) : 
#			ignorekeys = list(ignorekeys)
#		else : return
#		self.ignorekeys += ignorekeys
#
#
#	def Convert( self ) : 
#		try : self.hdf5.attrs.update(self.hduattrs[0])
#		except : pass
#		for i in xrange(len(self.fits.info(0))) : 
#			try : name = self.hduname[i]
#			except :
#				name = fo[i].name
#				if (name == '') : name = 'HDU_'+str(i)
#			self.hdf5[name] = self.fits[i].data
#			hdr = self.fits[i].header
#			keys, values, comments = hdr.keys(), hdr.values(), list(hdr.comments)
#			ncom = 1
#			for j in xrange(len(keys)) : 
#				if (keys[j] in self.ignorekeys) : continue
#				if (keys[j] == 'COMMENT') : 
#					keys[j] = keys[j] + '_'+str(ncom)
#					ncom += 1
#				self.hdf5[name].attrs[keys[j]] = values[j]
#				if (comments[j] != '') : 
#					self.hdf5[name].attrs[keys[j]+'_COM']=comments[j]
#			try : self.hdf5[name].attrs.update(self.hduattrs[i+1])
#			except : pass
#}
#
#		def Save( self ) : 
#			print self.hdf5path+'  -->  saved'
#			self.hdf5.close()
#			self.__dict__.clear()


##################################################
##################################################
##################################################


def SymbolAdd( string ) : 
	'''
	For example:
		SymbolAdd(5a-7+3b+4c+7a-13b+2)
	return:
		'12a-10b+4c-5'
	'''
	# Remove space
	stringsplit = string.split(' ')
	string = ''
	for i in xrange(len(stringsplit)) : string += stringsplit[i]
	# Find operator: '+' and '-'
	operator = []
	for i in xrange(len(string)) : 
		if (string[i] in ['+', '-']) : operator.append(i)
	operator = [0] + operator + [len(string)]
	# Extract number and symbol
	number, symbol = [], []
	for i in xrange(len(operator)-1) : 
		s = string[operator[i]:operator[i+1]]
		try : 
			v = float(s)
			if (v == int(v)) : v = int(v)
			number.append(v)
			symbol.append('')
			continue
		except : pass
		for j in xrange(len(s)-1, -1, -1) : 
			if (j == 0) : 
				number.append(1)
				symbol.append(s)
			try : 
				v = float(s[:j])
				if (v == int(v)) : v = int(v)
				number.append(v)
				symbol.append(s[j:])
				break
			except : pass
	# Calculate
	renumber, resymbol = [number[0]], [symbol[0]]
	for i in xrange(1, len(symbol)) : 
		try : 
			n = resymbol.index(symbol[i])
			renumber[n] += number[i]
		except : 
			resymbol.append(symbol[i])
			renumber.append(number[i])
	# Merge
	restring, numstring = '', ''
	for i in xrange(len(resymbol)) : 
		strnum = str(renumber[i])
		if (strnum[0] != '-') : strnum = '+' + strnum
		if (resymbol[i] == '') : numstring = strnum
		else : restring += strnum + resymbol[i]
	restring += numstring
	if (restring[0] == '+') : restring = restring[1:]
	return restring


##################################################
##################################################
##################################################


class FAST21cm( object ) : 


	def __init__( self, fast21cmdir=None ) : 
		if (fast21cmdir is None) : self.fast21cmdir = '/usr/bin/21cmFAST/'
		else : self.fast21cmdir = DirStr(fast21cmdir, True)
		self.message = False


	def Cosmology( self, h=None, OMm=None, OMb=None, OMn=None, OMk=None, OMr=None, SIGMA8=None, Y_He=None, POWER_INDEX=None, wl=None ) : 
		''' Cosmology parameters, default are the results from Plank 2015 '''
		if (h   is None) : h   = 0.6781
		if (OMm is None) : OMm = 0.308
		if (OMb is None) : OMb = 0.02226
		if (OMn is None) : OMn = 0.0
		if (OMk is None) : OMk = 0.0
		if (OMr is None) : OMr = 8.6e-5
		if (wl  is None) : wl = -1.0
		if (Y_He is None) : Y_He = 0.245
		if (SIGMA8 is None) : SIGMA8 = 0.8149
		if (POWER_INDEX is None) : POWER_INDEX = 0.9677  # ns
		#--------------------------------------------------
		self.cosmology = {'hlittle':h, 'OMm':OMm, 'OMb':OMb, 'OMn':OMn, 'OMk':OMk, 'OMr':OMr, 'SIGMA8':SIGMA8, 'Y_He':Y_He, 'POWER_INDEX':POWER_INDEX, 'wl':wl}
		key, value = self.cosmology.keys(), self.cosmology.values()
		#--------------------------------------------------
		filename = self.fast21cmdir+'Parameter_files/COSMOLOGY.H'
		filebak = self.fast21cmdir+'Parameter_files/COSMOLOGY_original.H'
		if (not os.path.exists(filebak)) : os.system('cp '+filename+' '+filebak)
		fo = open(filename).readlines()
		#--------------------------------------------------
		for i in xrange(len(key)) : 
			for j in xrange(len(fo)) : 
				if (fo[j][:8+len(key[i])] != '#define '+key[i]) : continue
				if (key[i] == 'OMb') : 
					fo[j] = '#define OMb  (float) (('+str(value[i])+'/hlittle)/hlittle) // at z=0\n'
					break
				n1 = fo[j].find('(')
				n2 = fo[j].find(')')
				fo[j] = fo[j][:n1+1] + str(value[i]) + fo[j][n2:]
				break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.cosmology


	def Redshift( self, ZSTART=None, ZEND=None, ZSTEP=None ) : 
		''' Set the redshift z you want '''
		if (ZSTART is None) : ZSTART = 12
		if (ZEND is None) : ZEND = 6
		if (ZSTEP is None) : ZSTEP = -0.2
		ZEND, ZSTART = np.sort(np.array([ZSTART, ZEND]))
		if (ZSTEP == 0) : ZSTEP = -0.1
		else : ZSTEP = -abs(ZSTEP)
		self.redshift ={'ZSTART':ZSTART, 'ZEND':ZEND, 'ZSTEP':ZSTEP}
		key, value = self.redshift.keys(), self.redshift.values()
		#--------------------------------------------------
		filename = self.fast21cmdir+'Programs/drive_zscroll_noTs.c'
		filebak = self.fast21cmdir+'Programs/drive_zscroll_noTs_original.c'
		if (not os.path.exists(filebak)) : os.system('cp '+filename+' '+filebak)
		fo = open(filename).readlines()
		#--------------------------------------------------
		for i in xrange(len(key)) : 
			for j in xrange(len(fo)) : 
				if (fo[j][:8+len(key[i])] != '#define '+key[i]) : continue
				n1 = fo[j].find('(')
				n2 = fo[j].find(')')
				fo[j] = fo[j][:n1+1] + str(value[i]) + fo[j][n2:]
				break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.redshift


	def Box( self, BOX_LEN=None, DIM=None, RANDOM_SEED=None ) : 
		'''
		BOX_LEN : 
			in Mpc
		DIM : 
			int, box.shape=(DIM,DIM,DIM)
		RANDOM_SEED : 
			int or 'time' => use localtime as seed
		'''
		if (BOX_LEN is None) : BOX_LEN = 300
		if (DIM is None) : DIM = 768
		if (RANDOM_SEED is None) : RANDOM_SEED = 1
		DIM = int(DIM)
		if (RANDOM_SEED == 'time') : RANDOM_SEED = int(('%.11f' % time.time()).split('.')[1][2:] + str(int(np.random.random()*99)))
		self.box ={'BOX_LEN':BOX_LEN, 'DIM':DIM, 'RANDOM_SEED':RANDOM_SEED}
		key, value = self.box.keys(), self.box.values()
		#--------------------------------------------------
		filename = self.fast21cmdir+'Parameter_files/INIT_PARAMS.H'
		filebak = self.fast21cmdir+'Parameter_files/INIT_PARAMS_original.H'
		if (not os.path.exists(filebak)) : os.system('cp '+filename+' '+filebak)
		fo = open(filename).readlines()
		#--------------------------------------------------
		for i in xrange(len(key)) : 
			for j in xrange(len(fo)) : 
				if (fo[j][:8+len(key[i])] != '#define '+key[i]) : continue
				if (key[i] == 'RANDOM_SEED') : 
					fo[j] = '#define RANDOM_SEED (long) ('+str(value[i])+') // seed for the random number generator\n'
				elif (key[i] == 'BOX_LEN') : 
					fo[j] = '#define BOX_LEN (float) '+str(value[i])+' // in Mpc\n'
				elif (key[i] == 'DIM') : 
					fo[j] = '#define DIM (int) '+str(value[i])+' // number of cells for the high-res box (sampling ICs) along a principal axis\n'
				break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.box


	def Linear( self, linear=False ) : 
		'''
		linear=True, evolve the density field with linear theory. Note that make sure that your cell size is in the linear regime at the redshift of interest.
		linear=False, evolve the density field with 1st order perturbation theory. Note that make sure that you resolve small enough scales, roughly we find BOX_LEN/DIM should be < 1Mpc.
		'''
		if (linear is True) : linear = '1'
		elif (linear is False) : linear = '0'
		else : 
			Raise(Warning, 'linear='+str(linear)+' is not True or False, reset linear=False')
			linear = '0'
		self.linear = bool(int(linear))
		#--------------------------------------------------
		filename = self.fast21cmdir+'Parameter_files/ANAL_PARAMS.H'
		filebak = self.fast21cmdir+'Parameter_files/ANAL_PARAMS_original.H'
		if (not os.path.exists(filebak)) : os.system('cp '+filename+' '+filebak)
		fo = open(filename).readlines()
		#--------------------------------------------------
		string = '#define EVOLVE_DENSITY_LINEARLY'
		for j in xrange(len(fo)) : 
			if (fo[j][:len(string)] != string) : continue
			fo[j] = string + ' (int) (' + linear + ')\n'
			break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.linear


	def Message( self, message=False ) : 
		try : self.message = bool(message)
		except : pass


	def Make( self ) : 
		pwd = ShellCmd('pwd')[0]
		ShellCmd('cd '+self.fast21cmdir+'Programs/')
		if (self.message) : os.system('make')
		else : ShellCmd('make')
		ShellCmd('cd '+pwd)


	def Run( self, which='drive_zscroll_noTs' ) : 
		pwd = ShellCmd('pwd')[0]
		ShellCmd('cd '+self.fast21cmdir+'Programs/')
		if (self.message) : ShellCmd('./'+which)
		else : os.system('./'+which)
		ShellCmd('cd '+pwd)


##################################################
##################################################
##################################################


