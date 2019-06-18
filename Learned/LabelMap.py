
def LabelMap( filename ):
	'''
	(1) Tensorflow: https://github.com/tensorflow/models/tree/master/research/object_detection/data

	Convert @p filename to a normal text file
	'''
	import numpy as np
	if (type(filename) !=str): return np.array(filename)
#	if (type(filename) !=str): return filename
	txt = open(filename).read()
	if ('item {' in txt and 'id:' in txt and 'display_name:' in txt):
		txt = txt.split('item {')[1:]
		name, maxidx = 2*len(txt)*['?'], -1
		for s in txt: 
			idx = int(s.split('id:')[1].split('\n')[0])
			display_name = s.split('display_name: "')[1].split('"\n}')[0]
			try: 
				int(display_name)
				display_name += '?'
			except: pass
			name[idx] = display_name
			if (idx > maxidx): maxidx = idx
		name = name[:idx+1]
	#	fo = open(filename+'.txt', 'w')
	#	for s in name: fo.write(s+'\n')
	#	fo.close()
		return np.array(name)
	#	return name


