import sklearn

class class_x(object):
#	import sklearn
#	print('import test_import.C.class_x.sklearn')

	def a(self):
		print(sklearn.__path__)
		print('import test_import.C.class_x.a.sklearn')

	def x(self):
		import sklearn
		print('import test_import.C.class_x.x.sklearn')

class_x = class_x()
