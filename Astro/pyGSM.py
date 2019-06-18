"""
gsm.py
======

Python interface for the Global Sky Model (GSM) or Oliveira-Costa et. al.
This is a python-based equivalent to the Fortran `gsm.f` that comes with the original data. Instead of the original ASCII DAT files that contain the PCA data, data are stored in HDF5, which is more efficient.

References
----------
A. de Oliveira-Costa, M. Tegmark, B.M. Gaensler, J. Jonas, T.L. Landecker and P. Reich
A model of diffuse Galactic radio emission from 10 MHz to 100 GHz
Mon. Not. R. Astron. Soc. 388, 247-260 (2008)
doi:10.111/j.1365-2966.2008.13376.x

PCA data from: space.mit.edu/home/angelica/gsm
"""



#from datetime import datetime


########################################
########################################
########################################





#class GlobalSkyModel(object):
class pyGSM2008( object ) :  # old GSM
	""" Global sky model (GSM) class for generating sky models.
	"""

	def __init__( self, freq_unit='MHz', basemap='haslam', interpolation='pchip', pygsmpath='gsm_components.hdf5' ) : 
		"""
		Global sky model (GSM) class for generating sky models.

		Upon initialization, the map PCA data are loaded into memory and interpolation
		functions are pre-computed.

		Parameters
		----------
		freq_unit: 'Hz', 'MHz', or 'GHz'
			Unit of frequency. Defaults to 'MHz'.

		gsm_version: 'haslam', 'wmap' or '5deg'
			GSM version to generate. The 5deg map has 5.1 degree resolution.
			This is a synthesized map made of all the maps considered in
			the de Oliveira-Costa et. al. paper
			At frequencies below 1GHz, haslam is preferred; you get higher
			resolution (1 degree) by locking to the Haslam 408 MHz map.
			At CMB frequencies, it is best to lock to the WMAP 23 GHz
			map, which is presented denoised with 2 degree resolution.

		interpolation: 'cubic' or 'pchip'
			Choose whether to use cubic spline interpolation or
			piecewise cubic hermitian interpolating polynomial (PCHIP).
			PCHIP is designed to never locally overshoot data, whereas
			splines are designed to have smooth first and second derivatives.

		Notes
		-----
		The scipy `interp1d` function does not allow one to explicitly
		set second derivatives to zero at the endpoints, as is done in
		the original GSM. As such, results will differ. Further, we default
		to use PCHIP interpolation.
		"""
		import h5py
		from jizhipy.Basic import Path
		if (pygsmpath is None): pygsmpath='gsm_components.hdf5'
		if (not Path.ExistsPath(pygsmpath)) : 
			pygsmpath = Path.jizhipyPath('jizhipy_tool/pyGSM_component/gsm_components.hdf5')
			if (not Path.ExistsPath(pygsmpath)) : Raise(Exception, 'pygsmpath="'+pygsmpath+'" NOT exists')
		self.GSM_FILEPATH = pygsmpath
		#----------------------------------------
		try:
			assert basemap in {'5deg', 'wmap', 'haslam'}
		except AssertionError:
			raise RuntimeError("GSM basemap unknown: %s. Choose '5deg', 'haslam' or 'wmap'" % basemap)

		try:
			assert interpolation in {'cubic', 'pchip'}
		except AssertionError:
			raise RuntimeError("Interpolation must be set to either 'cubic' or 'pchip'")

		self.h5 = h5py.File(self.GSM_FILEPATH, "r")
		self.basemap = basemap
		self.interpolation_method = interpolation
		self.freq_unit = freq_unit

		self.pca_map_data = None
		self.interp_comps = None
		self.update_interpolants()

		self.generated_map_data = None
		self.generated_map_freqs = None



	def update_interpolants(self):
		import numpy as np
	   # Choose the PCA map to load from the HDF5 file
		pca_map_dict = {"5deg": "component_maps_5deg",
						"haslam": "component_maps_408locked",
						"wmap": "component_maps_23klocked"}
		pca_map_key = pca_map_dict[self.basemap]
		self.pca_map_data = self.h5[pca_map_key][:]

		# Now, load the PCA eigenvalues
		pca_table = self.h5["components"][:]
		pca_freqs_mhz = pca_table[:, 0]
		pca_scaling   = pca_table[:, 1]
		pca_comps	 = pca_table[:, 2:].T

		# Interpolate to the desired frequency values
		ln_pca_freqs = np.log(pca_freqs_mhz)

		if self.interpolation_method == 'cubic':
			from scipy.interpolate import interp1d
			spl_scaling = interp1d(ln_pca_freqs, np.log(pca_scaling), kind='cubic')
			spl1 = interp1d(ln_pca_freqs,   pca_comps[0],   kind='cubic')
			spl2 = interp1d(ln_pca_freqs,   pca_comps[1],   kind='cubic')
			spl3 = interp1d(ln_pca_freqs,   pca_comps[2],   kind='cubic')

		else:
			from scipy.interpolate import pchip
			spl_scaling = pchip(ln_pca_freqs, np.log(pca_scaling))
			spl1 = pchip(ln_pca_freqs,   pca_comps[0])
			spl2 = pchip(ln_pca_freqs,   pca_comps[1])
			spl3 = pchip(ln_pca_freqs,   pca_comps[2])
		self.interp_comps = (spl_scaling, spl1, spl2, spl3)



	def generate(self, freqs):
		""" Generate a global sky model at a given frequency or frequencies

		Parameters
		----------
		freqs: float or np.array
			Frequency for which to return GSM model

		Returns
		-------
		gsm: np.array
			Global sky model in healpix format, with NSIDE=512. Output map
			is in galactic coordinates, and in antenna temperature units (K).

		"""
		import numpy as np
		from astropy import units
		# convert frequency values into Hz
		freqs = np.array(freqs) * units.Unit(self.freq_unit)
		freqs_mhz = freqs.to('MHz').value

		if isinstance(freqs_mhz, float):
			freqs_mhz = np.array([freqs_mhz])

		try:
			assert np.min(freqs_mhz) >= 10
			assert np.max(freqs_mhz) <= 94000
		except AssertionError:
			raise RuntimeError("Frequency values lie outside 10 MHz < f < 94 GHz")

		# Load interpolators and do interpolation
		ln_freqs	 = np.log(freqs_mhz)
		spl_scaling, spl1, spl2, spl3 = self.interp_comps
		comps = np.row_stack((spl1(ln_freqs), spl2(ln_freqs), spl3(ln_freqs)))
		scaling = np.exp(spl_scaling(ln_freqs))

		# Finally, compute the dot product via einsum (awesome function)
		# c=comp, f=freq, p=pixel. We want to dot product over c for each freq
		#print(comps.shape, self.pca_map_data.shape, scaling.shape)
		map_out = np.einsum('cf,pc,f->fp', comps, self.pca_map_data, scaling)

		if map_out.shape[0] == 1:
			map_out = map_out[0]
		self.generated_map_data = map_out
		self.generated_map_freqs = freqs
		return map_out



	def view(self, idx=0, logged=False):
		""" View generated map using healpy's mollweide projection.

		Parameters
		----------
		idx: int
			index of map to view. Only required if you generated maps at
			multiple frequencies.
		logged: bool
			Take the log of the data before plotting. Defaults to False.

		"""
		import healpy as hp
		import pylab as plt
		if self.generated_map_data is None:
			raise RuntimeError("No GSM map has been generated yet. Run generate() first.")

		if self.generated_map_data.ndim == 2:
			gmap = self.generated_map_data[idx]
			freq = self.generated_map_freqs[idx]
		else:
			gmap = self.generated_map_data
			freq = self.generated_map_freqs

		if logged:
			import numpy as np
			gmap = np.log2(gmap)

		hp.mollview(gmap, coord='G', title='Global Sky Model %s, %s' % (str(freq), self.basemap))
		plt.show()



	def write_fits(self, filename):
		""" Write out map data as FITS file.

		Parameters
		----------
		filename: str
			file name for output FITS file
		"""
		import healpy as hp
		hp.write_map(filename, self.generated_map_data, column_units='K')



	def set_basemap(self, new_basemap):
		self.basemap = new_basemap
		self.update_interpolants()
		if self.generated_map_freqs is not None:
			self.generate(self.generated_map_freqs)



	def set_freq_unit(self, new_unit):
		self.freq_unit = new_unit
		self.update_interpolants()
		if self.generated_map_freqs is not None:
			self.generate(self.generated_map_freqs)



	def set_interpolation_method(self, new_method):
		self.interpolation_method = new_method
		self.update_interpolants()
		if self.generated_map_freqs is not None:
			self.generate(self.generated_map_freqs)





########################################
########################################
########################################




try: 
	import ephem
	#class GSMObserver(ephem.Observer):
	class pyGSM2008Obs(ephem.Observer):
		""" Observer of the Global Sky Model.
	
		Generates the Observed sky, for a given point on Earth.
		Applies the necessary rotations and coordinate transformations
		so that the observed 'sky' can be returned, instead of the
		full galaxy-centered GSM.
	
		This class is based on pyephem's Observer(). The GSM bit can be thought of
		as an 'add on' to ephem.Observer, adding the methods generate() and  view(),
		which allows all-sky images for a given point on earth to be produced.
		"""
	
		def __init__(self):
			""" Initialize the Observer object.
	
			Calls ephem.Observer.__init__ function and adds on gsm
			"""
			import numpy as np
			import healpy as hp
			super(GSMObserver, self).__init__()
			self.gsm = GlobalSkyModel()
			self.observed_sky = None
	
			# Generate mapping from pix <-> angles
			self.gsm.generate(100)
			self._n_pix  = hp.get_map_size(self.gsm.generated_map_data)
			self._n_side = hp.npix2nside(self._n_pix)
			self._theta, self._phi = hp.pix2ang(self._n_side, np.arange(self._n_pix))
	
		def generate(self, freq):
			""" Generate the observed sky for the observer, based on the GSM.
	
			Parameters
			----------
			freq: float
				Frequency of map to generate, in units of MHz (default).
	
			Returns
			-------
			observed_sky: np.array
				Numpy array representing the healpix image, centered on zenith,
				with below the horizon masked.
			"""
			import numpy as np
			import healpy as hp
			self.gsm.generate(freq)
			sky = self.gsm.generated_map_data
	
			# Get RA and DEC of zenith
			ra_rad, dec_rad = self.radec_of(0, np.pi/2)
			ra_deg  = ra_rad / np.pi * 180
			dec_deg = dec_rad / np.pi * 180
	
	
			# Apply rotation
			hrot = hp.Rotator(rot=[ra_deg, dec_deg], coord=['G', 'C'], inv=True)
			g0, g1 = hrot(self._theta, self._phi)
			pix0 = hp.ang2pix(512, g0, g1)
			sky_rotated = sky[pix0]
	
			# Generate a mask for below horizon
			mask1 = self._phi + np.pi / 2 > 2 * np.pi
			mask2 = self._phi < np.pi / 2
			mask = np.invert(np.logical_or(mask1, mask2))
	
			self.observed_sky = hp.ma(sky_rotated)
			self.observed_sky.mask = mask
	
			return self.observed_sky
	
	
		def view(self, logged=False, show=False, **kwargs):
			""" View the local sky, in orthographic projection.
	
			Parameters
			----------
			logged: bool
				Default False, return the log2 image
			"""
			import healpy as hp
			sky = self.observed_sky
			if logged:
				import numpy as np
				sky = np.log2(sky)
	
			hp.orthview(sky, half_sky=True, **kwargs)
	
			if show:
				import pylab as plt
				plt.show()
	
			return sky
	
		def view_observed_gsm(self, logged=False, show=False, **kwargs):
			""" View the GSM (Mollweide), with below-horizon area masked. """
			import numpy as np
			import healpy as hp
			sky = self.observed_sky
			if logged:
				sky = np.log2(sky)
	
			# Get RA and DEC of zenith
			ra_rad, dec_rad = self.radec_of(0, np.pi / 2)
			ra_deg  = ra_rad / np.pi * 180
			dec_deg = dec_rad / np.pi * 180
	
			# Apply rotation
			derotate = hp.Rotator(rot=[ra_deg, dec_deg])
			g0, g1 = derotate(self._theta, self._phi)
			pix0 = hp.ang2pix(self._n_side, g0, g1)
			sky = sky[pix0]
	
			coordrotate = hp.Rotator(coord=['C', 'G'], inv=True)
			g0, g1 = coordrotate(self._theta, self._phi)
			pix0 = hp.ang2pix(self._n_side, g0, g1)
			sky = sky[pix0]
	
			hp.mollview(sky, coord='G', **kwargs)
	
			if show:
				import pylab as plt
				plt.show()
	
			return sky
except: pass





########################################
########################################
########################################







#from scipy.interpolate import interp1d, pchip

def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
	import numpy as np
	kB = 1.38065e-23
	C = 2.99792e8
	h = 6.62607e-34
	T = 2.725
	hoverk = h / kB
	B_nu = 2 * (h * nu)* (nu / C)**2 / (np.exp(hoverk * nu / T) - 1)
	conversion_factor = (B_nu * C / nu / T)**2 / 2 * np.exp(hoverk * nu / T) / kB
	return  K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy



def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
	kB = 1.38065e-23
	C = 2.99792e8
	h = 6.62607e-34
	T = 2.725
	hoverk = h / kB
	conversion_factor = 2 * (nu / C)**2 * kB
	return  K_RJ * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy





#class GlobalSkyModel2016(object):
class pyGSM2016( object ) :  # new GSM
	""" Global sky model (GSM) class for generating sky models.
	"""

	def __init__( self, freq_unit='MHz', unit='TCMB', resolution='hi', pygsmpath='gsm_components.hdf5' ) : 
		""" Global sky model (GSM) class for generating sky models.

		Upon initialization, the map PCA data are loaded into memory and interpolation
		functions are pre-computed.

		Parameters
		----------
		freq_unit: 'Hz', 'MHz', or 'GHz'
			Unit of frequency. Defaults to 'MHz'.

		unit: 'MJysr', 'TCMB', 'TRJ'
			Unit of output data. MJy/Steradian, T_CMB in Kelvin, or T_RJ.

		resolution: 'hi' or 'low'
			Resolution of output map. Either 300 arcmin (low) or 24 arcmin (hi).
			For frequencies under 10 GHz, output is 48 arcmin.

		Notes
		-----
		"""
		import numpy as np
		import h5py
		if (pygsmpath is None): pygsmpath='gsm2016_components.hdf5'
		if (not Path.ExistsPath(pygsmpath)) : 
			pygsmpath = Path.jizhipyPath('jizhipy_tool/pyGSM_component/gsm2016_components.hdf5')
			if (not Path.ExistsPath(pygsmpath)) : Raise(Exception, 'pygsmpath="'+pygsmpath+'" NOT exists')
		self.GSM2016_FILEPATH = pygsmpath
		#----------------------------------------
		if unit not in ['MJysr', 'TCMB', 'TRJ']:
			raise RuntimeError("UNIT ERROR: %s not supported. Only MJysr, TCMB, TRJ are allowed." % unit)

		if resolution.lower() in ('hi', 'high', 'h'):
			resolution = 'hi'
		elif resolution.lower() in ('low', 'lo', 'l'):
			resolution = 'low'
		else:
			raise RuntimeError("RESOLUTION ERROR: Must be either hi or low, not %s" % resolution)

		self.h5 = h5py.File(self.GSM2016_FILEPATH, "r")
		self.freq_unit = freq_unit
		self.unit = unit
		self.resolution = resolution

		# Map data to load
		labels = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']
		self.n_comp = len(labels)
		self.map_ni_hr = np.array([self.h5['highres_%s_map'%lb][:] for lb in labels])
		self.map_ni_lr = self.h5['lowres_maps']
		self.spec_nf = self.h5['spectra'][:]

		self.generated_map_data = None
		self.generated_map_freqs = None



	def generate(self, freqs):
		""" Generate a global sky model at a given frequency or frequencies

		Parameters
		----------
		freqs: float or np.array
			Frequency for which to return GSM model

		Returns
		-------
		gsm: np.array
			Global sky model in healpix format, with NSIDE=1024. Output map
			is in galactic coordinates, ring format.

		"""
		
		import numpy as np
		from astropy import units
		import healpy as hp
		# convert frequency values into Hz
		freqs = np.array(freqs) * units.Unit(self.freq_unit)
		freqs_ghz = freqs.to('GHz').value

		if isinstance(freqs_ghz, float):
			freqs_ghz = np.array([freqs_ghz])

		try:
			assert np.min(freqs_ghz) >= 0.01
			assert np.max(freqs_ghz) <= 5000
		except AssertionError:
			raise RuntimeError("Frequency values lie outside 10 MHz < f < 5 THz: %s")

		if self.resolution == 'hi':
			map_ni = self.map_ni_hr
		else:
			map_ni = self.map_ni_lr

		spec_nf = self.spec_nf
		nfreq = spec_nf.shape[1]

		output = np.zeros((len(freqs_ghz), map_ni.shape[1]), dtype='float32')
		for ifreq, freq in enumerate(freqs_ghz):

			left_index = -1
			for i in range(nfreq - 1):
				if spec_nf[0, i] <= freq <= spec_nf[0, i + 1]:
					left_index = i
					break

			# Do the interpolation
			interp_spec_nf = np.copy(spec_nf)
			interp_spec_nf[0:2] = np.log10(interp_spec_nf[0:2])
			x1 = interp_spec_nf[0, left_index]
			x2 = interp_spec_nf[0, left_index + 1]
			y1 = interp_spec_nf[1:, left_index]
			y2 = interp_spec_nf[1:, left_index + 1]
			x = np.log10(freq)
			interpolated_vals = (x * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1)
			output[ifreq] = np.sum(10.**interpolated_vals[0] * (interpolated_vals[1:, None] * map_ni), axis=0)

			output[ifreq] = hp.pixelfunc.reorder(output[ifreq], n2r=True)

			if self.unit == 'TCMB':
				conversion = 1. / K_CMB2MJysr(1., 1e9 * freq)
			elif self.unit == 'TRJ':
				conversion = 1. / K_RJ2MJysr(1., 1e9 * freq)
			else:
				conversion = 1.
			output[ifreq] *= conversion

#			output.append(result)

		if len(output) == 1:
			output = output[0]
		#else:
		#	map_data = np.row_stack(output)

		self.generated_map_freqs = freqs
		self.generated_map_data = output

		return output



	def view(self, idx=0, logged=False):
		""" View generated map using healpy's mollweide projection.

		Parameters
		----------
		idx: int
			index of map to view. Only required if you generated maps at
			multiple frequencies.
		logged: bool
			Take the log of the data before plotting. Defaults to False.

		"""

		import healpy as hp
		import pylab as plt
		if self.generated_map_data is None:
			raise RuntimeError("No GSM map has been generated yet. Run generate() first.")

		if self.generated_map_data.ndim == 2:
			gmap = self.generated_map_data[idx]
			freq = self.generated_map_freqs[idx]
		else:
			gmap = self.generated_map_data
			freq = self.generated_map_freqs

		if logged:
			import numpy as np
			gmap = np.log2(gmap)

		hp.mollview(gmap, coord='G',
					title='Global Sky Model %s %s' % (str(freq), self.unit))
		plt.show()



	def write_fits(self, filename):
		""" Write out map data as FITS file.

		Parameters
		----------
		filename: str
			file name for output FITS file
		"""
		import healpy as hp
		hp.write_map(filename, self.generated_map_data, column_units=self.unit)





########################################
########################################
########################################





try: 
	import ephem
	#class GSMObserver2016(ephem.Observer):
	class pyGSM2016Obs(ephem.Observer):
		""" Observer of the Global Sky Model.
	
		Generates the Observed sky, for a given point on Earth.
		Applies the necessary rotations and coordinate transformations
		so that the observed 'sky' can be returned, instead of the
		full galaxy-centered GSM.
	
		This class is based on pyephem's Observer(). The GSM bit can be thought of
		as an 'add on' to ephem.Observer, adding the methods generate() and  view(),
		which allows all-sky images for a given point on earth to be produced.
		"""
	
		def __init__(self):
			""" Initialize the Observer object.
	
			Calls ephem.Observer.__init__ function and adds on gsm
			"""
			import numpy as np
			import healpy as hp
			super(GSMObserver2016, self).__init__()
			self.gsm = GlobalSkyModel2016(freq_unit='MHz')
			self.observed_sky = None
	
			# Generate mapping from pix <-> angles
			self.gsm.generate(1000)
			self._n_pix  = hp.get_map_size(self.gsm.generated_map_data)
			self._n_side = hp.npix2nside(self._n_pix)
			self._theta, self._phi = hp.pix2ang(self._n_side, np.arange(self._n_pix))
	
		def generate(self, freq):
			""" Generate the observed sky for the observer, based on the GSM.
	
			Parameters
			----------
			freq: float
				Frequency of map to generate, in units of MHz (default).
	
			Returns
			-------
			observed_sky: np.array
				Numpy array representing the healpix image, centered on zenith,
				with below the horizon masked.
			"""
			import numpy as np
			import healpy as hp
			self.gsm.generate(freq)
			sky = self.gsm.generated_map_data
	
			# Get RA and DEC of zenith
			ra_rad, dec_rad = self.radec_of(0, np.pi/2)
			ra_deg  = ra_rad / np.pi * 180
			dec_deg = dec_rad / np.pi * 180
	
			# Apply rotation
			hrot = hp.Rotator(rot=[ra_deg, dec_deg], coord=['G', 'C'], inv=True)
			g0, g1 = hrot(self._theta, self._phi)
			n_side = hp.npix2nside(self.gsm.generated_map_data.shape[0])
			pix0 = hp.ang2pix(n_side, g0, g1)
			sky_rotated = sky[pix0]
	
			# Generate a mask for below horizon
			mask1 = self._phi + np.pi / 2 > 2 * np.pi
			mask2 = self._phi < np.pi / 2
			mask = np.invert(np.logical_or(mask1, mask2))
	
			self.observed_sky = hp.ma(sky_rotated)
			self.observed_sky.mask = mask
	
			return self.observed_sky
	
	
		def view(self, logged=False, show=False, **kwargs):
			""" View the local sky, in orthographic projection.
	
			Parameters
			----------
			logged: bool
				Default False, return the log2 image
			"""
			import healpy as hp
			import pylab as plt
			sky = self.observed_sky
			if logged:
				import numpy as np
				sky = np.log2(sky)
	
			hp.orthview(sky, half_sky=True, **kwargs)
	
			if show:
				plt.show()
	
			return sky
	
		def view_observed_gsm(self, logged=False, show=False, **kwargs):
			""" View the GSM (Mollweide), with below-horizon area masked. """
			import numpy as np
			import healpy as hp
			import pylab as plt
			sky = self.observed_sky
			if logged:
				sky = np.log2(sky)
	
			# Get RA and DEC of zenith
			ra_rad, dec_rad = self.radec_of(0, np.pi / 2)
			ra_deg  = ra_rad / np.pi * 180
			dec_deg = dec_rad / np.pi * 180
	
			# Apply rotation
			derotate = hp.Rotator(rot=[ra_deg, dec_deg])
			g0, g1 = derotate(self._theta, self._phi)
			pix0 = hp.ang2pix(self._n_side, g0, g1)
			sky = sky[pix0]
	
			coordrotate = hp.Rotator(coord=['C', 'G'], inv=True)
			g0, g1 = coordrotate(self._theta, self._phi)
			pix0 = hp.ang2pix(self._n_side, g0, g1)
			sky = sky[pix0]
	
			hp.mollview(sky, coord='G', **kwargs)
	
			if show:
				plt.show()
	
			return sky
except: pass


