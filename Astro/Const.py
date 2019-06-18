'''
1 Jy = 10^{-26} W m^{-2} Hz^{-1}
'''


class Const( object ) : 

	def __init__( self ) : 
		# light speed
		self.c = 299792458.  # m/s

		# HI freq, wavelength
		self.freq21cm = 1420.40575177  # MHz 
		self.wavelength21cm = 0.2110611405413  # m

		# Plank constant
		self.Plank = 6.6260695729e-34

		# Boltzmann constant
		self.Boltzmann = 1.380648813e-23

		# Gravitational constant
		self.G = 6.67408e-11  # N*m^2*kg^-2

		# Sidereal day [hour]
		self.siderealday = 23+56/60.+4.0916/3600




Const = Const()
