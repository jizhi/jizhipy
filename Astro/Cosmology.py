
class Cosmology( object ) : 


	def Distance( self, z, h=None, Omega_m=0.308, Omega_l=0.692, Omega_k=0, Omega_r=0 ) : 
		'''
		Calculate the cosmology distance with redshift.

		Result is the same as:
		http://www.astro.ucla.edu/~wright/CosmoCalc.html

		Newest cosmological parameter from Plank 2015:
		http://adsabs.harvard.edu/abs/2016A%26A...594A..13P
			H0 = 67.81 +- 0.92  km s^-1 Mpc^-1
			Omega_Lambda = 0.692 +- 0.012
			Omega_m = 0.308 +- 0.012
				( Omega_m + Omega_Lambda = 1 )
			Omega_b h^2 = 0.02226 += 0.00023
		
		Note that, the formulas here are for the flat Lambda_CDM model
		Unit of the distance: 
			h == False: unit is h^{-1}Mpc; 
			else : unit is Mpc. Here h=H0/100
				h=None=0.6781
		
		z:
			redshift, can be a number, a list/ndarray (1D)
	
		Return [dc, dA, dL]
		'''
		import numpy as np
		from scipy.integrate import quad
		from jizhipy.Array import Asarray
		from jizhipy.Basic import IsType
		if (h is None) : h = 0.6781
		c = 3e+5  # km/s
		A = c / 100.0     # c/H0, H0=h*100km/s/Mpc, note that c in km/s
		# judge type of redshift
		islist = False if(IsType.isnum(z))else True
		zlist, dlist = npfmt(z, float), []
		# defind function using lambda
		func = lambda x : 1/( Omega_l + Omega_m*(1+x)**3 )**0.5
		for z in zlist : 
			# note that quad return two element, first is the result, second is the error
			dc = quad( func, 0, z )[0]
			dc = A * dc
			if (h is not False) : dc = dc / h
			dL = dc * (1+z)
			dA = dc / (1+z)  # comoving
			dlist.append( np.array([dc, dA, dL]) )
		dlist = np.array( dlist )
		if (not islist) : dlist = dlist[0]
		return dlist
	
	
	
	
	
	def DistanceApprox( z, h=0, Omega_m=0.308, Omega_l=0.692, Omega_k=0, Omega_r=0 ) : 
		'''
		This function is just for z<<1, save to z^5
		Actually, we don't use this function, use CosmologyDistance() above.
		Note, 
			z=1.0, err=1.8% (error of the result compared with the theory)
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
	
	
	
	
	
	def Redshift( distance, zdpath='/usr/bin/redshift-distance.dat', typed=0, h=0, Omega_m=0.208, Omega_l=0.692, Omega_k=0, Omega_r=0 ) : 
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
					# !!! x must be increasing in interp1d
					f=interpolate.interp1d(x2[::-1], y2[::-1])
					z2 = f( d )
					z2 = float( '%.4f' % z2 )
				if ( z1 >= 0 and z2 < 0 ) : z = [ z1 ]
				elif ( z1 < 0 and z2 >= 0 ) : z = [ z2 ]
				elif ( z1 >= 0 and z2 >= 0 ) : z = [ z1, z2 ]
				else : print('angular diameter distance out of np.arange( dA.min(), dA.max() )')
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





	def CAMB( z, cambpath='/usr/bin/camb' ) : 
		'''
		Produce dark matter 1D power spectrum with CAMB at redshift z.
		z: Redshift can be a number, a list/ndarray (1D)
		
		cambpath: where is the camb program folder, default is '/usr/bin/camb'
		
		First step, create a directory named 'output' (not include '') at current directory.
		Second step, create a directory named 'camb_output' (not include '') in 'output' directory
		
		Note that os.system() can not use cd : os.system( 'cd file1' ), so for the path not in the current folder, use the absolute path
		'''
		redshift = z
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
				outfile.write(parafile[i])
			#	print >> outfile, parafile[i],
			outfile.close()
	
			# calculate with camb with different params.ini
			os.system( './cambpy/camb cambpy/params.ini' )
			# modify and calculate several times OK !
	
		paramsfile.close()
		# delete the temp file
		os.system( 'rm -r cambpy' )





Cosmology = Cosmology()
