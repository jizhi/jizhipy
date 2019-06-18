#import numpy
#print('import test_import.B.a.numpy')

def func_a():
	import sklearn
	print('import test_import.B.a.func_a.sklearn')


class class_a(object):

	def __init__(self):
		import sklearn
		print('import test_import.B.a.class_a.sklearn')

	def a(self):
		import sklearn
		print('import test_import.B.a.class_a.a.sklearn')

	def x(self):
		import sklearn
		print('import test_import.B.a.class_a.x.sklearn')


class class_b(object):

	def __init__(self):
		import sklearn
		print('import test_import.B.a.class_b.sklearn')

	def a(self):
		import sklearn
		print('import test_import.B.a.class_b.a.sklearn')

	def x(self):
		import sklearn
		print('import test_import.B.a.class_b.x.sklearn')

class_b = class_b()
