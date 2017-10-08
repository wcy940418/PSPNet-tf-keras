import os
import argparse
import numpy as np

from keras import backend as K
import tensorflow as tf

import layers_builder as layers
import utils
import drawImage

from time import time

from PIL import Image

WEIGHTS = 'pspnet50_ade20k.npy'
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]]) # RGB

class PSPNet:

    def __init__(self):
        self.model = layers.build_pspnet()
        # set_npy_weights(self.model, WEIGHTS)

    def predict(self, img):
        # Preprocess
        img = img.resize((473, 473))
        input_ = np.array(img, dtype=np.float32)
        input_ = input_ - DATA_MEAN
        input_ = input_[:,:,::-1]

        probs = self.feed_forward(input_)
        return probs

    def predict_sliding_window(self, img):
        pass

    def feed_forward(self, data):
        assert data.shape == (473,473,3)
        data = data[np.newaxis,:,:,:]

        # utils.debug(self.model, data)
        pred = self.model.predict(data)
        return pred[0]

def set_npy_weights(model, npy_weights):
    weights = np.load(npy_weights).item()

    for layer in model.layers:
        print layer.name
        if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
            mean = weights[layer.name]['mean'].reshape(-1)
            variance = weights[layer.name]['variance'].reshape(-1)
            scale = weights[layer.name]['scale'].reshape(-1)
            offset = weights[layer.name]['offset'].reshape(-1)
            
            model.get_layer(layer.name).set_weights([mean, variance, scale, offset])

        elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
            try:
                weight = weights[layer.name]['weights']
                model.get_layer(layer.name).set_weights([weight])
            except Exception as err:
                biases = weights[layer.name]['biases']
                model.get_layer(layer.name).set_weights([weight, biases])
    print 'Finished.'
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', required=True, help='Path the input image')
    parser.add_argument('--output_path', type=str, default='', required=True, help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('--save_weight', type=str, default='0')
    parser.add_argument('--load_weight', type=str, default='0')
    parser.add_argument('--blended', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = Image.open(args.input_path)
        
        pspnet = PSPNet()

        if args.load_weight != '0':
            T = time()
            pspnet.model.load_weights(args.load_weight)
            print "Setting time with hdf5 uses: %f s    " % (time() - T)
        else:
            T = time()
            set_npy_weights(pspnet.model, WEIGHTS)
            print "Setting time with npy use: %f s" % (time() - T)
        T = time()
        probs = pspnet.predict(img)
        print "Forwarding uses: %f s" % (time() - T)
        colors = 'utils/colorization/color150.mat'
        objects = 'utils/colorization/objectName150.mat'
        anns = 'utils/color150'
        predicted_classes = np.argmax(probs, axis=2)
        im_Width = predicted_classes.shape[1]
        im_Height = predicted_classes.shape[0]
        draw = drawImage.BaseDraw(colors, objects, anns,
                            img, (im_Width, im_Height),
                            predicted_classes)
        if args.blended != '0':
            simpleSegmentImage, ann = draw.drawSimpleSegment(True);
        else:
            simpleSegmentImage, ann = draw.drawSimpleSegment();
        simpleSegmentImage.save(args.output_path,"JPEG")
        ann.save(os.path.splitext(args.output_path)[0] + '_anns' + os.path.splitext(args.output_path)[1], "JPEG")
        if args.save_weight != '0':
            pspnet.model.save_weights(args.save_weight)
            print "model saved at: %s" % args.save_weight
