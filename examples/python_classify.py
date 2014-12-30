import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
IMAGE_FILE = sys.argv[1]

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#IMAGE_FILE = 'images/cat.jpg'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
net.set_phase_test()
net.set_mode_cpu()
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show()

prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
plt.show()

prediction = net.predict([input_image], oversample=False)
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
plt.show()

#%timeit net.predict([input_image])
net.predict([input_image])

# Resize the image to the standard (256, 256) and oversample net input sized crops.
input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
# 'data' is the input blob name in the model definition, so we preprocess for that input.
caffe_input = np.asarray([net.preprocess('data', in_) for in_ in input_oversampled])
# forward() takes keyword args for the input blobs with preprocessed input arrays.
#%timeit net.forward(data=caffe_input)
net.forward(data=caffe_input)

net.set_mode_gpu()

prediction = net.predict([input_image])
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
plt.show()

# Full pipeline timing.
#%timeit net.predict([input_image])
net.predict([input_image])

# Forward pass timing.
#%timeit net.forward(data=caffe_input)
net.forward(data=caffe_input)
