
class FAST21cm( object ) : 
	'''
	Use 21cmFAST program
	'''


	def __init__( self, fast21cmdir=None, verbose=False ) : 
		'''
		fast21cmdir:
			Where is the 21cmFAST program

		Make sure the current User of computer system has the permission to write something to this directory
		'''
		import os
		if (fast21cmdir is None) : self.fast21cmdir = os.path.expanduser('~/.mysoftware/python-packages/jizhipy/21cmFAST_kit/21cmFAST/')
		else : self.fast21cmdir = jp.Outdir(fast21cmdir)
		self.verbose = bool(verbose)





	def FAST21cmDir( self, fast21cmdir=None, verbose=None ) : 
		if (verbose is None) : verbose = self.verbose
		self.__init__(fast21cmdir, verbose)





	def Cosmology( self, h=None, OMm=None, OMb=None, OMn=None, OMk=None, OMr=None, SIGMA8=None, Y_He=None, POWER_INDEX=None, wl=None ) : 
		''' Cosmology parameters, default are the results from Plank 2015 '''
		import os
		if (h   is None) : h   = 0.6781
		if (OMm is None) : OMm = 0.308
		if (OMb is None) : OMb = 0.0484
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
		filename=self.fast21cmdir+'Parameter_files/COSMOLOGY.H'
		filebak =self.fast21cmdir+'Parameter_files/COSMOLOGY_original.H'
		if (not os.path.exists(filebak)) : os.system('cp '+filename+' '+filebak)
		fo = open(filename).readlines()
		#--------------------------------------------------
		for i in range(len(key)) : 
			for j in range(len(fo)) : 
				if (fo[j][:8+len(key[i])] != '#define '+key[i]) : continue
				if (key[i] == 'OMb') : 
					fo[j] = '#define OMb  (float) (('+str(value[i])+'/hlittle)/hlittle) // at z=0\n'
					break
				n1 = fo[j].find('(')
				n2 = fo[j].find(')')
				fo[j] =fo[j][:n1+1] +str(value[i]) +fo[j][n2:]
				break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.cosmology





	def Redshift( self, ZSTART=None, ZEND=None, ZSTEP=None ) : 
		''' Set the redshift z you want '''
		import numpy as np
		import os
		if (ZSTART is None) : ZSTART = 12
		if (ZEND is None) : ZEND = 6
		if (ZSTEP is None) : ZSTEP = -0.2
		ZEND, ZSTART = np.sort(np.array([ZSTART, ZEND]))
		if (ZSTEP == 0) : ZSTEP = -0.1
		else : ZSTEP = -abs(ZSTEP)
		self.redshift ={'ZSTART':ZSTART, 'ZEND':ZEND, 'ZSTEP':ZSTEP}
		key, value=self.redshift.keys(), self.redshift.values()
		#--------------------------------------------------
		filename = self.fast21cmdir+'Programs/drive_zscroll_noTs.c'
		filebak = self.fast21cmdir+'Programs/drive_zscroll_noTs_original.c'
		if (not os.path.exists(filebak)) : os.system('cp '+filename+' '+filebak)
		fo = open(filename).readlines()
		#--------------------------------------------------
		for i in range(len(key)) : 
			for j in range(len(fo)) : 
				if (fo[j][:8+len(key[i])] != '#define '+key[i]) : continue
				n1 = fo[j].find('(')
				n2 = fo[j].find(')')
				fo[j] =fo[j][:n1+1] +str(value[i]) +fo[j][n2:]
				break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.redshift





	def Box( self, BOX_LEN=None, DIM=None, HII_DIM=None, RANDOM_SEED=None ) : 
		'''
		BOX_LEN : 
			in Mpc
		DIM : 
			int, box.shape=(DIM,DIM,DIM)
		RANDOM_SEED : 
			int or 'time' => use localtime as seed
		'''
		import os
		if (BOX_LEN is None) : BOX_LEN = 300
		if (DIM is None) : DIM = 768
		if (HII_DIM is None) : HII_DIM = 256
		if (RANDOM_SEED is None) : RANDOM_SEED = 1
		DIM, HII_DIM = int(DIM), int(HII_DIM)
		if (RANDOM_SEED == 'time') : RANDOM_SEED = int(('%.11f' % time.time()).split('.')[1][2:] + str(int(np.random.random()*99)))
		self.box ={'BOX_LEN':BOX_LEN, 'DIM':DIM, 'HII_DIM':HII_DIM, 'RANDOM_SEED':RANDOM_SEED}
		key, value = self.box.keys(), self.box.values()
		#--------------------------------------------------
		filename = self.fast21cmdir+'Parameter_files/INIT_PARAMS.H'
		filebak = self.fast21cmdir+'Parameter_files/INIT_PARAMS_original.H'
		if (not os.path.exists(filebak)) : os.system('cp '+filename+' '+filebak)
		fo = open(filename).readlines()
		#--------------------------------------------------
		for i in range(len(key)) : 
			for j in range(len(fo)) : 
				if (fo[j][:8+len(key[i])] != '#define '+key[i]) : continue
				if (key[i] == 'RANDOM_SEED') : 
					fo[j] = '#define RANDOM_SEED (long) ('+str(value[i])+') // seed for the random number generator\n'
				elif (key[i] == 'BOX_LEN') : 
					fo[j] = '#define BOX_LEN (float) '+str(value[i])+' // in Mpc\n'
				elif (key[i] == 'DIM') : 
					fo[j] = '#define DIM (int) '+str(value[i])+' // number of cells for the high-res box (sampling ICs) along a principal axis\n'
				elif (key[i] == 'HII_DIM') : 
					fo[j] = '#define HII_DIM (int) '+str(value[i])+' // number of cells for the low-res box\n'
				break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.box





	def Linear( self, linear=True ) : 
		'''
		linear=True, evolve the density field with linear theory. Note that make sure that your cell size is in the linear regime at the redshift of interest.
		linear=False, evolve the density field with 1st order perturbation theory. Note that make sure that you resolve small enough scales, roughly we find BOX_LEN/DIM should be < 1Mpc.
		'''
		import os
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
		for j in range(len(fo)) : 
			if (fo[j][:len(string)] != string) : continue
			fo[j] = string + ' (int) (' + linear + ')\n'
			break
		#--------------------------------------------------
		open(filename, 'w').writelines(fo)
		return self.linear





	def Make( self ) : 
		from ShellCmd import ShellCmd, Cd
		import os
		Cd.Go(self.fast21cmdir+'Programs/')
		if (self.verbose) : os.system('make')
		else : ShellCmd('make')
		Cd.Back()





	def Run( self, which='drive_zscroll_noTs' ) : 
		from ShellCmd import ShellCmd, Cd
		import os
		Cd.Go(self.fast21cmdir+'Programs/')
		if (self.verbose) : os.system('./'+which)
		else : ShellCmd('./'+which)
		Cd.Back()





	def DeltaX2Healpix( self, deltaxpath=None, dtype=None ) : 
		'''
		'''
		import numpy as np
		if (dtype is None) : dtype = np.float32
		a = np.fromfile(deltaxpath, dtype)
		n = int(round(a.size**(1./3)))
		a = a.reshape(n, n, n)
		# Flatten
		n1 = int(n**0.5)
		b, n = [], 0
		a = a.T
		for i in range(n1) : 
			c = []
			for j in range(n1) : 
				c.append(a[:,:,n])
				n += 1
			b.append( np.concatenate(c, 1) )
		b = np.concatenate(b, 0)
		return b





def T21cm( z, Omega_HI=0.55e-3, Omega_m=0.0484+0.2596, Omega_l=0.692 ) : 
	'''
	Calculate the average brightness temperature of the 21cm signal as a function of redshift.

	T21_ave = 0.39 * Omega_HI / 1e-3 * ( ( Omega_m + (1+z)**(-3)*Omega_l ) / 0.29 )**(-0.5) * ( (1+z) / 2.5 )**0.5
	
	bias that HI traces the dark matter delta_HI = bHI * delta_x (Masui et al. 2013, ApJL, 763,20)
		b_{opt} = 1.22
		b_{HI}  = 0.70
		\Omega_{HI} = 6.6e-4
	
	Mass fraction of HI with critical density unit (Masui et al. 2013, ApJL, 763,20)
		Omega_HI = 0.5e-3
	
	Hubble constant h = H0 / (100 km/s/Mpc)
	h = 0.6932
	
	Other cosmological parameters
		Omega_b  = 0.04628
		Omega_dm = 0.2402
		Omega_m = Omega_dm + Omega_b
		Omega_l  = 0.7135

	Plank 2015
		h    = 0.6781
		OMm  = 0.308
		OMdm = 0.2596
		OMb  = 0.0484
		OMr  = 8.6e-5
		OMl  = 0.692
	
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





if __name__ == '__main__' : 

	a = FAST21cm('~/Downloads/21cmFAST', True)
	a.Linear(False)
	a.Cosmology()
	a.Redshift(0.8939, 0.8939, 0)
	a.Box(2, 7, 4)
	a.Make()
	a.Run()
