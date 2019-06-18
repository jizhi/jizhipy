# See jizhipy_extended/jizhipy_Learned/

from jizhipy.Learned.LabelMap import LabelMap

#def _ModelPath():
#	import jizhipy as jp
#	path = jp.__path__[0]+'_extend/jizhipy_Learned/'
#	return path
#_path = _ModelPath()
_path = __path__[0][:-8]+'_extend/jizhipy_Learned/'

# def ObjectDetector( image, framework, modelPath, txtPath=None, labelMap=None, minConfidence=None, scaleFactor=None, size=None, mean=None, swapRB=False, crop=False, **kwargs ):

# (framework, modelPath, txtPath, labelMap, scaleFactor, size, mean)

# face detection
_labelMapFace = LabelMap(['background', 'face'])
haarcascade_frontalface_alt_xml = {'framework':'Haar', 
	'modelPath':_path+'haarcascade_frontalface_alt.xml/haarcascade_frontalface_alt.xml', 
	'txtPath':None, 
	'labelMap':_labelMapFace, 
	'minConfidence':3, 
	'scaleFactor':1.1,
	'size':0, 
	'mean':None}


# face detection
haarcascade_frontalface_alt2_xml = {'framework':'Haar', 
	'modelPath':_path+'haarcascade_frontalface_alt2.xml/haarcascade_frontalface_alt2.xml', 
	'txtPath':None, 
	'labelMap':_labelMapFace, 
	'minConfidence':3, 
	'scaleFactor':1.1,
	'size':0, 
	'mean':None}


# face detection
haarcascade_frontalface_alt_tree_xml = {'framework':'Haar', 
	'modelPath':_path+'haarcascade_frontalface_alt_tree.xml/haarcascade_frontalface_alt_tree.xml', 
	'txtPath':None, 
	'labelMap':_labelMapFace, 
	'minConfidence':3, 
	'scaleFactor':1.1,
	'size':0, 
	'mean':None}


# face detection
haarcascade_frontalface_default_xml = {'framework':'Haar', 
	'modelPath':_path+'haarcascade_frontalface_default.xml/haarcascade_frontalface_default.xml', 
	'txtPath':None, 
	'labelMap':_labelMapFace, 
	'minConfidence':3, 
	'scaleFactor':1.1,
	'size':0, 
	'mean':None}



# face detection
res10_300x300_ssd_iter_140000_caffemodel ={'framework':'Caffe', 
	'modelPath':_path+'res10_300x300_ssd_iter_140000.caffemodel/res10_300x300_ssd_iter_140000.caffemodel', 
	'txtPath':_path+'res10_300x300_ssd_iter_140000.caffemodel/deploy.prototxt', 
	'labelMap':_labelMapFace, 
	'minConfidence':0.6, 
	'scaleFactor':1.0, 
	'size':(300,300), 
	'mean':(104,117,123)}





# object detection
_labelMap = ['background', 'plane', 'bicycle', 'bird', 
            'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 
            'cow', 'table', 'dog', 'horse', 'motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 
            'train', 'tv']
mobilenet_iter_73000_caffemodel = {'framework':'Caffe', 
	'modelPath':_path+'MobileNet-SSD/mobilenet_iter_73000.caffemodel', 
	'txtPath':_path+'MobileNet-SSD/deploy.prototxt', 
	'labelMap':LabelMap(_labelMap), 
	'minConfidence':0.6, 
	'scaleFactor':1/127.5, 
	'size':(300,300), 
	'mean':(127.5,127.5,127.5)}


# object detection
_labelMapCOCO = LabelMap(_path+'ssd_inception_v2_coco_2018_01_28.pb/mscoco_complete_label_map.pbtxt')
ssd_inception_v2_coco_2018_01_28_pb ={'framework':'Tensorflow',
	'modelPath':_path+'ssd_inception_v2_coco_2018_01_28.pb/frozen_inference_graph_ssd_inception_v2_coco_2018_01_28.pb', 
	'txtPath':_path+'ssd_inception_v2_coco_2018_01_28.pb/graph_ssd_inception_v2_coco_2018_01_28.pbtxt', 
	'labelMap':_labelMapCOCO, 
	'minConfidence':0.6, 
	'scaleFactor':1/127.5, 
	'size':(300,300), 
	'mean':(127.5,127.5,127.5)}


# object detection
ssd_mobilenet_v1_coco_2018_01_28_pb ={'framework':'Tensorflow',
	'modelPath':_path+'ssd_mobilenet_v1_coco_2018_01_28.pb/frozen_inference_graph_ssd_mobilenet_v1_coco_2018_01_28.pb', 
	'txtPath':_path+'ssd_mobilenet_v1_coco_2018_01_28.pb/graph_ssd_mobilenet_v1_coco_2018_01_28.pbtxt', 
	'labelMap':_labelMapCOCO, 
	'minConfidence':0.6, 
	'scaleFactor':1/127.5, 
	'size':(300,300), 
	'mean':(127.5,127.5,127.5)}





availableModel = ['haarcascade_frontalface_alt_xml', 
	'haarcascade_frontalface_alt2_xml',
	'haarcascade_frontalface_alt_tree_xml',
	'haarcascade_frontalface_default_xml',
	'res10_300x300_ssd_iter_140000_caffemodel',
	'mobilenet_iter_73000_caffemodel',
	'ssd_inception_v2_coco_2017_11_17_pb',
	'ssd_mobilenet_v1_coco_11_06_2017_pb'
]
