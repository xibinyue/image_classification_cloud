'''
E6895 final project
Image Classification As A Cloud Service

This code is based on https://github.com/BVLC/caffe/tree/master/examples
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
caffe_root = '/Users/seanlu/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import urllib


import os

def initialize_caffe():
    if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print 'CaffeNet found.'
    else:
        print 'Downloading pre-trained CaffeNet model...'
        #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

    caffe.set_mode_cpu()

    model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    # load ImageNet labels
    labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    if not os.path.exists(labels_file):
        os.system('../data/ilsvrc12/get_ilsvrc_aux.sh')
        
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    return net, labels, transformer

if __name__ == "__main__":

    net,labels,transformer = initialize_caffe()
    # download an image
    my_image_url = "https://upload.wikimedia.org/wikipedia/commons/4/4c/Push_van_cat.jpg"  
    # for example:
    #my_image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG"
    #os.system('wget -O'+ ' image.jpg'+ my_image_url)
    urllib.urlretrieve(my_image_url, filename="image.png")

    # transform it and copy it into the net
    image = caffe.io.load_image('image.png')
    net.blobs['data'].data[...] = transformer.preprocess('data', image)

    # perform classification
    net.forward()

    # obtain the output probabilities
    output_prob = net.blobs['prob'].data[0]

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]


    print 'probabilities and labels:'
    print (zip(output_prob[top_inds], labels[top_inds]))

