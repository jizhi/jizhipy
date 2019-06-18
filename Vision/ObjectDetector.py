
def ObjectDetector( image, framework, modelPath, txtPath=None, labelMap=None, minConfidence=None, scaleFactor=None, size=None, mean=None, crop=False, **kwargs ):
	'''
	Parameters
	----------
	image:
		Must be 3D/2D array/matrix, NOT path

	framework:
		'Tensorflow' | 'Caffe' | 'Torch' | 'Darknet' | 'ONNX' | 'ModelOptimizer' | 'Haar' (OpenCV .xml)

	modelPath, txtPath:
		(1) for Tensorflow: 
			* modelPath: path of .pb (binary frozen graph) file with binary protobuf description of the network architecture
			* txtPath: path of .pbtxt file that contains text graph definition in protobuf format
		(2) for Caffe:
			* modelPath: path of .caffemodel file with learned network
			* txtPath: path of .prototxt file with text description of the network architecture
		(3) for OpenCV.xml:
			* modelPath: path of .xml file
			* txtPath: set txtPath=None

	minConfidence:
		(1) for NOT Haar: 
			* Detect objects with confidence >= minConfidence
		(2) for Haar: 
			* minConfidence=minNeighbors, if a candidate target is consist of >= minNeighbors windows, it may be a real target, otherwise, drop it

	scaleFactor:
		(1) for NOT Haar:
			* Use what image to train this model(network)
			* scaleFactor=1: use original image range 0~255
			* scaleFactor=1/255: use normalized image range 0~1
			* For Caffe, use scaleFactor=1
			* For Tensorflow, use scaleFactor=1/127.5
			* (effect: image *= scaleFactor)
		(2) for Haar:
			* scaleFactor = NextWindowSize / CurrentWindowSize
			(default scaleFactor=1.1)

	size:
		(1) for NOT Haar:
			* size=(height, width) for output image
		(2) for Haar:
			* size=flags, usually use 3 cases:
				flags=0 (default)
				flags=cv2.CV_HAAR_DO_CANNY_PRUNING
				flags=cv2.CASCADE_SCALE_IMAGE

	mean:
		(only for NOT Haar)
    	scalar with mean values which are subtracted from channels. Values are intended to be in (mean-R, mean-G, mean-B) order if @p image is BGR (NOT RGB) and @p swapRB=True

	swapRB:
		(only for NOT Haar)
		True/False, swap R(red) and B(blue) or not

	crop:
		(only for NOT Haar)
		True/False, crop the image or not after resize
		@p size is used to resize @p image:
			(1) crop=False: resize directly and don't preserve aspect ratio
			(2) crop=True: first, resize @p image preserving aspect ratio, so that @p image.shape >= size; then, crop the longer axis from the image center, make @p image.shape==size
	'''
	from jizhipy.Basic import Path, Raise
	import numpy as np
	import cv2
	# Check input parameters
	framework_valid = ['tensorflow', 'caffe', 'torch', 'darknet', 'onnx', 'modeloptimizer', 'haar']
	framework, fw0 = str(framework).lower(), framework
	if (framework not in framework_valid): Raise(Exception, 'framework="'+framework+'" is NOT valid, must be one of '+str(framework_valid))
	#----------
	if (minConfidence is None):
		if (framework != 'haar'): minConfidence = 0.7
		else: minConfidence = 3
	#----------
	if (scaleFactor is not None):
		scaleFactor = float(scaleFactor)
	else:
		if (framework == 'tensorflow'): scaleFactor = 1/127.5
		elif (framework == 'haar'): scaleFactor = 1.1
		else: scaleFactor = 1.0
	#----------
	if (size is not None): 
		try: size = tuple(size)[:2]
		except: pass
	else:
		if (framework == 'haar'): size = 0
	#----------
	swapRB = False
	swapRB, crop = bool(swapRB), bool(crop)
	#----------
	modelPath = Path.AbsPath(modelPath)
	Path.ExistsPath(modelPath, stop=True)
	if (txtPath is not None):
		txtPath = Path.AbsPath(txtPath)
		Path.ExistsPath(txtPath, stop=True)
	height, width = image.shape[:2]
	#----------------------------------------
	#----------------------------------------
	if (framework != 'haar'): 
	    # Creates 4-dimensional blob from image
		blob = cv2.dnn.blobFromImage(image, scaleFactor, size, mean, swapRB, crop)
	#	blob = cv2.dnn.blobFromImage(image)
		# Read the model(weight)
		if (framework == 'tensorflow'):
			net = cv2.dnn.readNetFromTensorflow(modelPath, txtPath)
		elif (framework == 'caffe'):
			net = cv2.dnn.readNetFromCaffe(txtPath, modelPath)
		elif (framework == 'torch'):
			isBinary = txtPath
			net = cv2.dnn.readNetFromTorch(modelPath, isBinary)
		elif (framework == 'darknet'):
			net =cv2.dnn.readNetFromDarknet(txtPath, modelPath)
		elif (framework == 'onnx'):
			net = cv2.dnn.readNetFromONNX(modelPath)
		elif (framework == 'ModelOptimizer'):
			net = cv2.dnn.readNetFromModelOptimizer(txtPath, modelPath)
		else: Raise(Exception, 'Not support framework: '+fw0)
		# Pass the blob through the network
		net.setInput(blob)
		# detection[0,0].shape(num_object, 7)
		# 7: 0:?  1:ClassIndex  2:Confidence  3,4,5,6:Box
		detection = net.forward()[0,0]
		# Select >= minConfidence
		n = np.arange(len(detection))[detection[:,2]>=minConfidence]
		detection = detection[n]
		# Detection confidence and rects
		confidence = detection[:,2]
		rects = (detection[:,3:] * np.array([[width, height, width, height]])).astype(int)  # xmin, ymin, xmax, ymax
		# Confidence to text to draw
		if (labelMap is None): label = len(rects)*['']
		else: label = labelMap[detection[:,1].astype(int)]
		text = []
		for i in range(len(rects)):
			c = ('%.1f' % (100*confidence[i]))+'%'
			if (label[i] != ''): c = str(label[i])+': '+c
			text.append(c)
	#----------------------------------------
	#----------------------------------------
	else:
		cascade = cv2.CascadeClassifier(modelPath)
		rects = cascade.detectMultiScale(image, scaleFactor, minConfidence, minSize=(5,5), flags=size)
		if (len(rects)>0): rects[:,2:] += rects[:,:2]
		confidence = np.zeros(len(rects), int)
		text = len(rects)*['']
	#----------------------------------------
	#----------------------------------------
#	from imutils.object_detection import non_max_suppression
#	rects = non_max_suppression(rects, probs=None, overlapThresh=0.7)
	return (rects, confidence, text)










if __name__=='__main__':
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import jizhipy as jp
	import cv2
	modellist = [jp.Learned.haarcascade_frontalface_alt_xml, 
		jp.Learned.haarcascade_frontalface_alt2_xml,
		jp.Learned.haarcascade_frontalface_alt_tree_xml,
		jp.Learned.haarcascade_frontalface_default_xml,
		jp.Learned.res10_300x300_ssd_iter_140000_caffemodel,
		jp.Learned.mobilenet_iter_73000_caffemodel,
		jp.Learned.ssd_inception_v2_coco_2017_11_17_pb,
		jp.Learned.ssd_mobilenet_v1_coco_11_06_2017_pb]

#	for i in range(1, 21):
#		stri = '0'+str(i) if(i < 10) else str(i)
#		imagePath = 'image/image_'+stri+'.jpg'
	for i in range(10):
		imagePath = '../ImageData/image_'+str(i)+'.jpg'
		jp.Basic.Path.ExistsPath(imagePath, stop=True)
		image0 = cv2.imread(imagePath)
	#	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if (len(image0.shape)==3): color = (0,255,0)
		else: color = (255,255,255)
	
		for j in range(len(modellist)):
			image = image0.copy()
			modelArgs = modellist[j]
		#	image = cv2.resize(image, modelArgs['size'])
			modelArgs['minConfidence'] = 0.5
		#	try: rects, confidence, text = ObjectDetector(image, framework, modelPath, txtPath)
		#	except: continue
			rects, confidence, text = ObjectDetector(image, **modelArgs)

			modelName = modelArgs['modelPath'].split('/')[-1]
			print(modelName+' found '+str(len(rects)))
			image = jp.Vision.cv2rectangle(image, rects, text, 'box', color, 2, fontScale=1, loc='bottom')
			if(len(image.shape)==3): plt.imshow(image[:,:,::-1])
			else: plt.imshow(image, cmap=plt.cm.gray)
			plt.title(modelName+'\nfound '+str(len(rects)))
			plt.savefig('compare/image_'+str(i)+'_'+modelName+'_'+str(len(rects))+'.png')
			plt.close()
		#	plt.show()
		#	exit()
		
