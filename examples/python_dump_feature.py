import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
											 caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
net.set_phase_test()
#net.set_mode_cpu()
net.set_mode_gpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# image list, output file
basefolder = '/home/nash/Documents/TmallImgCtr/data/ImageMF/'
img_list = np.loadtxt(basefolder+'mapping/auction_id.txt',delimiter=',')
fo = file(basefolder+'alexnet.txt','w')
# for each of the images
for i in range(img_list.shape[0]):
	if i%100 == 0:
		print i

	try:
		scores = net.predict([caffe.io.load_image(basefolder+'original/images/'+'%d.jpg'%img_list[i,0])])
		feat = net.blobs['fc7'].data[4]
		feat = feat.flatten()
	except Exception, x:
		print 'error while loading %d'%img_list[i,0]
		print x
		feat = np.zeros(feat.shape)
	
	fo.write('%d,%d,'%(img_list[i,1],img_list[i,0])) # using the mapped id as feature id.
	featstr = ','.join(['%f'%feat[j] for j in range(len(feat))])
	fo.write(featstr+'\n')

fo.close()
print 'All finished.'