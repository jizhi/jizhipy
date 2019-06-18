
def cv2plot( image, bbox, text=None, color=(0,255,0), lw=None, style='box', boxScale=1.2, fontSize=None, loc=None, **kwargs ):
	'''
	Draw the box/cross on the image

	image:
		np.ndarray or matrix, NOT path
		Can be gray or RGB image

	bbox:
		xmin, ymin, xmax, ymax = bbox.T

	boxScale:
		Scale the area of the box
		boxScale=1: face
		boxScale=1.2: the whole head

	text:
		Put text on the image
		str | list of str with len(text)==len(bbox)

	color:
		color=(red, green, blue), 0~255

	style:
		'cross' | 'box' | 'circle' | 'point'

	loc:
		Location of text
			None | [int] 0 ~ 11 (see below)
			None: find the best loc automatically, try the order 0, 3, 1, 2, 4, 5, 6, 8, 7
		-----------------------
		|          6          |
		|                     |
		|          0          |
		|     -----------     |
		|     |    1    |     |
		|7    | 4       | 5   |
		|     |    2    |     |
		|     -----------     |
		|          3          |
		|                     |
		|          8          |
		-----------------------

	fontSize:
		None | [int]
		width=1000 for image, fontSize=40
		if (fontSize is None): 
			fontSize = image.shape[1]/1000*40

	fontFace:
		cv2.FONT_HERSHEY_SIMPLEX  # normal size sans-serif
		cv2.FONT_HERSHEY_PLAIN    # small size sans-serif
		cv2.FONT_HERSHEY_DUPLEX   # normal size sans-serif (more complex than FONT_HERSHEY_SIMPLEX)
		cv2.FONT_HERSHEY_COMPLEX  # normal size serif
		cv2.FONT_HERSHEY_TRIPLEX  # normal size serif (more complex than FONT_HERSHEY_COMPLEX)
		cv2.FONT_HERSHEY_COMPLEX_SMALL   # smaller version of FONT_HERSHEY_COMPLEX
		cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # Hand-writing style
		cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # More complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
		cv2.FONT_ITALIC                  # Flag for italic
	'''
	if (bbox is None or len(bbox)==0): return image
	import cv2
	from jizhipy.Basic import IsType, Path
	from PIL import Image, ImageDraw, ImageFont
	import numpy as np
	image, loc0 = np.array(image), loc
	here = Path.DirPath(__file__, 1)
	width, height = image.shape[:2][::-1]
	if (fontSize is None): fontSize = int(40 * width / 1000)
	if (lw is None): lw = 2. * width / 1000
	if (lw < 1): lw = 1
	if (boxScale != 1):
		dx = ((bbox[:,2]-bbox[:,0])*(boxScale-1)/2).astype(int)
		dy = ((bbox[:,3]-bbox[:,1])*(boxScale-1)/2).astype(int)
		bbox[:,0] -= dx
		bbox[:,2] += dx
		bbox[:,1] -= dy
		bbox[:,3] += dy
	xmin, ymin, xmax, ymax = bbox.T
	xmin[xmin<0] = 0
	xmax[xmax>=width] = width-1
	ymin[ymin<0] = 0
	ymax[ymax>=height] = height-1
	bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3] = xmin, ymin, xmax, ymax
	if (text is None): pass
	elif (IsType.isstr(text)): text = len(bbox)*[text]
	else: 
		text = list(text)
		if (len(text) < len(bbox)): 
			text = list(text)+(len(bbox)-len(text))*['']
	if (loc is None): loc = 'up'
	lw = int(round(lw))
	textlw = 1 if(lw==1)else lw-1
	#----------------------------------------
	def _loc(xmin, xmax, ymin, ymax, loc):
		if   (loc==1): x, y = xmin+lw, ymin+lw
		elif (loc==2): x, y = xmin+lw, ymax-fontSize-lw
		elif (loc==3): x, y = xmin+lw, ymax+lw
		elif (loc==4): x, y = xmin+lw, int((ymax+ymin-fontSize)/2)
		elif (loc==5): x, y = xmax+lw, int((ymax+ymin-fontSize)/2)
		elif (loc==6): x, y = int(width/2-1.5*fontSize), lw
		elif (loc==7): x, y = lw, int((height-fontSize)/2)
		elif (loc==8): x, y = int(width/2-1.5*fontSize), height-fontSize-lw
		else:          x, y = xmin, ymin-fontSize-lw  # loc=0
		return (x, y)
	#----------------------------------------
	def _putText(image, xmin, xmax, ymin, ymax, text):
		try: text = text.decode('utf8')  # python2
		except: pass  # python3
		if (loc0 is not None):
			x, y = _loc(xmin, xmax, ymin, ymax, int(loc0))
		else: 
			x, y = _loc(xmin, xmax, ymin, ymax, 0)
			locOrder, repeat =[3, 1, 2, 4, 5, 6, 8, 7, 1],True
			for loc in locOrder:
				repeat = (len(text)*fontSize > width-x) or (y < 0) or (fontSize > height-y)
				if (not repeat): break
				x, y = _loc(xmin, xmax, ymin, ymax, loc)
		try: # Use PIL, can put chinese
			font = ImageFont.truetype(here+'/simsun.ttc', fontSize, encoding='utf-8')
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = Image.fromarray(image)
			draw = ImageDraw.Draw(image)
			draw.text((x, y), text, color, font) 
			image = np.array(image)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		except: # Use cv2.putText(), but can NOT put chinese
			image = cv2.putText(image, text, (x, y+fontSize-lw), cv2.FONT_HERSHEY_SIMPLEX, 1, color, lw)
		return image
	#----------------------------------------
	if (str(style).lower() == 'box'):
		for i in range(len(bbox)): 
			xmin, ymin, xmax, ymax = bbox[i]
			image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, lw)
			if (text is not None):
				image = _putText(image, xmin, xmax, ymin, ymax, text[i])
	#----------------------------------------
	elif (str(style).lower() == 'cross'):
		for i in range(len(bbox)): 
			xmin, ymin, xmax, ymax = bbox[i]
			w, h = xmax-xmin, ymax-ymin
			x1, w1 = xmin + (w-lw)/2, lw
			y1, h1 = ymin + (h-lw)/2, lw
			image = cv2.rectangle(image, (x1,ymin), (x1+w1,ymax), color,lw)
			image = cv2.rectangle(image, (xmin,y1), (xmax,y1+h1), color,lw)
			if (text is not None):
				image = _putText(image, xmin, xmax, ymin, ymax, text[i])
	#----------------------------------------
	elif (str(style).lower() == 'point'):
		x, y = bbox.T.reshape(2,-1)
		for i in range(len(x)):
			image = cv2.circle(image, (x[i],y[i]), lw,color,lw)
	#----------------------------------------
	try: image = image.get()
	except: pass
	return image
