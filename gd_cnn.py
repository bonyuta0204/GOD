'''
Convolutional Neural Network class for Generic Decoding
'''


import caffe
import numpy as np
import os
import PIL.Image
from scipy.misc import imresize


class CnnModel(object):
    '''
    CNN model class
    '''

    def __init__(self, model_def, model_param, mean_image, batch_size=8, rand_seed=None, layers=None):
        '''
        Load a pre-trained Caffe CNN model

        Original version was developed by Guohua Shen
        '''

        # Init random seed
        if not rand_seed is None:
            np.random.seed(rand_seed)


        if layers is None:
            self.layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8')
        else:
            self.layers = layers

        # Enable GPU mode
        caffe.set_mode_gpu()
            
        # Prepare a mean image
        img_mean = np.load(mean_image)
        img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

        # Init the model
        channel_swap = (2, 1, 0)
        self.net = caffe.Classifier(model_def, model_param, mean=img_mean, channel_swap=channel_swap)

        h, w = self.net.blobs['data'].data.shape[-2:]
        self.image_size = (h, w)
        self.batch_size = batch_size

        self.net.blobs['data'].reshape(self.batch_size, 3, h, w)

        ## Init select feature index

        self.feat_index = self.init_feature_index()



    def init_feature_index(self, n=4096):
        feat_index = {}

        for layer in self.layers:
            if layer in ("conv1", "conv2", "conv3", "conv4", "conv5"):
                f_index = []
                blob_shape = self.net.blobs[layer].data[0].shape
                resolution = blob_shape[1] * blob_shape[2]
                n_channel = self.net.blobs[layer].data.shape[1]
                n_per_channel = 4096 / n_channel
                for ch in range(n_channel):

                    # choonse numbers between ch * resolution to (ch + 1) * resolution - 1
                    rand_ind = np.random.randint(ch * resolution, (ch + 1) * resolution - 1,size=n_per_channel,     )
                    f_index += list(rand_ind)

                feat_index[layer] = f_index
            elif layer in ("fc6", "fc7"):
                f_index = np.arange(n)
                feat_index[layer] = f_index
            else:
                #FIX ME
                f_index = np.arange(1000)
                feat_index[layer] = f_index
        return feat_index

    def get_feature(self, images, layers):
        '''
        Returns CNN features

        Original version was developed by Guohua Shen (compute_cnn_feat_mL)
        '''

        # Convert 'images' to a list
        if not isinstance(images, list):
            images = [images]

        num_images = len(images)
        num_loop = int(np.ceil(num_images / float(self.batch_size)))

        image_index = [[ind for ind in xrange(i * self.batch_size, (i + 1) * self.batch_size) if ind < num_images]
                       for i in xrange(num_loop)]

        (h, w) = self.image_size
        mean_image = self.net.transformer.mean['data']

        feature_all = []

        for i, imgind in enumerate(image_index):
            for j, k in enumerate(imgind):
                img = imresize(PIL.Image.open(images[k]).convert('RGB'), (h, w), interp='bicubic')
                img = np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(mean_image, (3, 1, 1))
                self.net.blobs['data'].data[j] = img

            self.net.forward(end=layers[-1])

            for j, k in enumerate(imgind):

                feat_list = [self.net.blobs[lay].data[j].flatten()[self.feat_index[lay]] for lay in layers]
                feature_all.append(feat_list)

        feature_all = np.array(feature_all)
        return feature_all

    def get_channel(self, layer, unit_num):
        """
        unit num starts from 1
        :param layer:
        :param unit_num:
        :return:
            int: channel number
        """
        if layer in ("conv1", "conv2", "conv3", "conv4", "conv5"):
            flatten_unit = self.feat_index[layer][unit_num - 1]
            blob_shape = self.net.blobs[layer].data[0].shape
            resolution = blob_shape[1] * blob_shape[2]
            channel = flatten_unit // resolution

            return channel
        else:
            return unit_num - 1

if __name__ == "__main__":
    # Feture selection settings
    num_features = 4096

    # CNN model settings
    model_def = './data/cnn/caffe_reference_imagenet/caffe_reference_imagenet.prototxt'
    model_param = './data/cnn/caffe_reference_imagenet/caffe_reference_imagenet.caffemodel'
    cnn_layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8')

    mean_image_file = './data/images/ilsvrc_2012_mean.npy'  # ImageNet Large Scale Visual Recognition Challenge 2012

    # Stimulus image settings
    exp_stimuli_dir = ('./data/images/image_training', './data/images/image_test')
    catave_image_dir = ('./data/images/category_test', './data/images/category_candidate')

    # Results file
    data_dir = './data_original_alex'
    featuredir = os.path.join(data_dir, 'ImageFeatures_caffe_test/')
    outputfile = os.path.join(data_dir, 'ImageFeatures_caffe_test.pkl')

    # Misc settings
    rand_seed = 2501

    model = CnnModel(model_def, model_param, mean_image_file, batch_size=128, rand_seed=rand_seed)
    print(model.feat_index)