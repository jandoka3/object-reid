#!/usr/bin/env python3

import os
from importlib import import_module
from itertools import count
import numpy as np
import tensorflow as tf
import common


def flip_augment(image, fid, pid):
    """ Returns both the original and the horizontal flip of an image. """
    images = tf.stack([image, tf.reverse(image, [1])])
    # I changed dimension with tf
    # return images, [fid]*2, [pid]*2
    return images, tf.stack([fid]*2), tf.stack([pid]*2)


def five_crops(image, crop_size):
    """ Returns the central and four corner crops of `crop_size` from `image`. """
    image_size = tf.shape(image)[:2]
    crop_margin = tf.subtract(image_size, crop_size)
    assert_size = tf.assert_non_negative(
        crop_margin, message='Crop size must be smaller or equal to the image size.')
    with tf.control_dependencies([assert_size]):
        top_left = tf.floor_div(crop_margin, 2)
        bottom_right = tf.add(top_left, crop_size)
    center = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    top_left = image[:-crop_margin[0], :-crop_margin[1]]
    top_right = image[:-crop_margin[0], crop_margin[1]:]
    bottom_left = image[crop_margin[0]:, :-crop_margin[1]]
    bottom_right = image[crop_margin[0]:, crop_margin[1]:]
    return center, top_left, top_right, bottom_left, bottom_right

def calculate_emb_for_fids(args, data_fids):
    '''
    Calculate embeddings

    :param args: input arguments
    :param data_fids: relative paths to the imagies
    :return: matrix with shape len(data_fids) x embedding_dim (embedding vector for each image - one row)
    '''
    ###################################################################################################################
    # LOAD DATA
    ###################################################################################################################
    # Load the args from the original experiment.
    net_input_height=256
    net_input_width=128
    pre_crop_height=288
    pre_crop_width=144
    net_input_size = (net_input_height, net_input_width)
    pre_crop_size = (pre_crop_height, pre_crop_width)


    ###################################################################################################################
    # PREPARE DATA
    ###################################################################################################################
    # Setup a tf Dataset containing all images.
    dataset = tf.data.Dataset.from_tensor_slices(data_fids)

    # Convert filenames to actual image tensors.
    # dataset tensor: [image_resized, fid, pid]
    dataset = dataset.map(
        lambda fid: common.fid_to_image(
            fid, tf.constant("dummy", dtype=tf.string), image_root=args.image_root,
            image_size=pre_crop_size),
        num_parallel_calls=8)

    # Augment the data if specified by the arguments.
    # `modifiers` is a list of strings that keeps track of which augmentations
    # have been applied, so that a human can understand it later on.
    modifiers = ['original']
    dataset = dataset.map(flip_augment)
    dataset = dataset.apply(tf.contrib.data.unbatch())
    modifiers = [o + m for m in ['', '_flip'] for o in modifiers]

    dataset = dataset.map(lambda im, fid, pid:(tf.stack(five_crops(im, net_input_size)), tf.stack([fid] * 5), tf.stack([pid] * 5)))
    dataset = dataset.apply(tf.contrib.data.unbatch())
    modifiers = [o + m for o in modifiers for m in ['_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]

    # Group it back into PK batches.
    dataset = dataset.batch(256)

    # Overlap producing and consuming.
    dataset = dataset.prefetch(1)

    images, _, _ = dataset.make_one_shot_iterator().get_next()

    ###################################################################################################################
    # CREATE MODEL
    ###################################################################################################################

    # Get the weights
    model = import_module('nets.resnet_v1_50')
    embedding_dim = 128
    block4_units = 1
    endpoints = model.endpoints(images, block4_units = block4_units, is_training=False, embedding_dim=embedding_dim)

    with tf.Session() as sess:
        # Initialize the network/load the checkpoint.
        print('Restoring from checkpoint: {}'.format(args.checkpoint))
        tf.train.Saver().restore(sess, args.checkpoint)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros(
            (len(data_fids) * len(modifiers), embedding_dim), np.float32)
        for start_idx in count(step=256):
            try:
                emb = sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(
                        start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.

        print()
        print("Done with embedding, aggregating augmentations...", flush=True)

        # Pull out the augmentations into a separate first dimension.
        emb_storage = emb_storage.reshape(len(data_fids), len(modifiers), -1)
        emb_storage = emb_storage.transpose((1,0,2))  # (Aug,FID,128D)

        # Aggregate according to the specified parameter.
        emb_storage = np.mean(emb_storage, axis=0)

    tf.reset_default_graph()
    return emb_storage

def run_embedding(args, dataset):
    # Load the data from the CSV file.
    # pids - person id (array corresponding to the images)
    # fids - array of the paths to the images ({str_})
    dataset=os.path.join(os.getcwd(), dataset)
    img_root=os.path.join(os.getcwd(),args.image_root)
    data_pids, data_fids = common.load_dataset(dataset, img_root, False)

    return calculate_emb_for_fids(args, data_fids)