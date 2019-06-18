# -*- coding:utf-8 -*-
import sys
sys.path.append('../')
#import test_import_1
#import test_import_2
import test_import_3

'''
In [3]: import test_import
import test_import.B.a.numpy
import test_import.B.a.class_a.sklearn
import test_import.B.a.class_b.sklearn
import test_import.B.class_x.sklearn

可见，放在class名下作为类属性的模块，会先自动加载。
放在普通def函数、类下的函数里的模块，均不会自动加载。

在任何地方，如类中或函数中加载过的模块，再次import，是不会重新加载的，会很快。

类名下加载的模块，在类函数里是调用不到的。。。
在类外加载，类函数里才能调用



In [3]: from test_import import C
import test_import.B.a.numpy
import test_import.B.a.class_a.sklearn
import test_import.B.a.class_b.sklearn
import test_import.B.class_x.sklearn
import test_import.B.a.numpy
import test_import.B.a.class_a.sklearn
import test_import.B.a.class_b.sklearn

可以看到，虽然明面上是import C，实际上是将所有模块都加载了，
跟import test_import一模一样



In [4]: import test_import_2
In [5]: test_import_2.C.class_x.a()
['/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sklearn']
import test_import.C.class_x.a.sklearn
In [6]: import sklearn  # 很快，表明在test_import的类的函数里import过sklearn后，在其他任意地方再import sklearn，是不重新import的



关于__init__(self)，加载类时不执行，实例化时才执行



********** 结论 **********
如果想import jizhipy时很快，则用当前写法：将所有import，放到函数里（普通函数或类函数），不放在.py顶格位置和类名下！！！
'''
