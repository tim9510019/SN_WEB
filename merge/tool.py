#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:41:34 2018

@author: timgenius
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io

def tf_resize_images(imagefile_path,imagsize):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (imagsize[0], imagsize[1]), tf.image.ResizeMethod.AREA)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = Image.open(imagefile_path)
        imgdata = np.asarray(img, dtype='float32')
        X_data = sess.run(tf_img, feed_dict = {X: imgdata})
        X_data = np.array(X_data, dtype = np.float32)
    return X_data

def tf_web_resize_images(imgbinary,imagsize):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (imagsize[0], imagsize[1]), tf.image.ResizeMethod.AREA)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #img = Image.open(imagefile_path)
        imgdata = np.asarray(Image.open(io.BytesIO(imgbinary)), dtype='float32')
        X_data = sess.run(tf_img, feed_dict = {X: imgdata})
        X_data = np.array(X_data, dtype = np.float32)
    return X_data

def preprocess_image(image_data):
    image_data = (np.array(image_data,dtype=np.float32)/255.0 - 0.5) * 2.0
    return image_data

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def listCompare(a,b):
    d = [c for c in b if c not in a]
    if not d:
        return 1
    else:
        return 0
