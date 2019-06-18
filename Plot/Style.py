
def Style( style=None, cmap=1, facecolor=1, grid=1, cap=1, math=1, font=1, zorder=1 ) : 
	'''
	style:
		=1: use 1.x style
		=2: use 2.x style
		=None: use style set here(this function)

	zorder:
		Ticks and grids are now plotted above solid elements such as filled contours, but below lines. To return to the previous behavior of plotting ticks and grids above lines, set mpl.rcParams['axes.axisbelow'] = False

	http://matplotlib.org/users/dflt_style_changes.html
	'''
	import matplotlib as mpl
	if (style==1) : 
		mpl.style.use('classic')
		return
	# Below is for style=None
	if (cmap ==1) : mpl.rcParams['image.cmap'] = 'jet'
	if (facecolor==1): mpl.rcParams['figure.facecolor']='0.75'
	if (grid==1) : 
		mpl.rcParams['grid.color'] = 'k'
		mpl.rcParams['grid.linestyle'] = ':'
		mpl.rcParams['grid.linewidth'] = 0.5
	if (cap==1) : mpl.rcParams['errorbar.capsize'] = 3
	if (math==1) : 
		# cm: Computer Modern
		mpl.rcParams['mathtext.fontset'] = 'cm'
		mpl.rcParams['mathtext.rm'] = 'serif'
	if (zorder==1) : mpl.rcParams['axes.axisbelow']=False

