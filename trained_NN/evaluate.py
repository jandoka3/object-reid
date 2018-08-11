#!/usr/bin/env python3
import json, csv
import os
from importlib import import_module
from itertools import count
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
import common
import visualiseRes
from re_ranking import re_ranking_feature

def  cdist(a, b):
    """
    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    """
    with tf.name_scope("cdist"):
        diffs = tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)
        return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)

def saveResults(args, results_images, array_of_queries, query_fids, num_of_col):
    # make csv file with all queries and for each query save ten best images from gallery
    row = -1
    array_output = np.empty(shape=(len(array_of_queries), num_of_col+1), dtype=np.dtype('U256'))
    for i in array_of_queries:
        # write into the first column the query image
        row += 1
        array_output[row,0] = query_fids[i]
        for j in range(0, num_of_col):
            # next column would be the 10 most similar images from gallery
            array_output[row, j + 1] = results_images[i][j]


    rerank_arg = '_with_rerank' if args.re_rank else ''
    image_to_track = '_image_to_track_with_' + args.selector if args.image_to_track else ''
    filename = 'datasets/'+args.dataset+'/results/'+args.dataset+rerank_arg+image_to_track+"_ranked_images.csv"
    os.makedirs(os.path.dirname(filename), exist_ok = True)

    with open(os.path.join(os.getcwd(), filename), 'wt') as csv_query:
        writer_q = csv.writer(csv_query, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer_q.writerows(array_output)

    return array_output

def evaluate_embs(args, query_pids, query_fids, query_embs, gallery_pids, gallery_fids, gallery_embs):
    # Just a quick sanity check that both have the same embedding dimension!
    query_dim = query_embs.shape[1]
    gallery_dim = gallery_embs.shape[1]
    batch_size = 256

    if query_dim != gallery_dim:
        raise ValueError('Shape mismatch between query ({}) and gallery ({}) '
                         'dimension'.format(query_dim, gallery_dim))

    # Setup the dataset specific matching function
    excluder = import_module('excluders.' + args.excluder).Excluder(gallery_fids)

    # We go through the queries in batches, but we always need the whole gallery
    batch_pids, batch_fids, batch_embs = tf.data.Dataset.from_tensor_slices(
        (query_pids, query_fids, query_embs)
    ).batch(batch_size).make_one_shot_iterator().get_next()

    # Perform reranking if requested
    if args.re_rank:
        rerank_k1 = 20
        rerank_k2 = 6
        rerank_lambda = 0.3
        batch_distances = tf.py_func(re_ranking_feature.re_ranking, [batch_embs, gallery_embs,
                                                        rerank_k1, rerank_k2, rerank_lambda,
                                                        True, batch_size], tf.float16)
    else:
        batch_distances = cdist(batch_embs, gallery_embs)

    # Loop over the query embeddings and compute their APs and the CMC curve.
    aps = []
    correct_rank = []
    results_images = []
    scores_of_queries = []
    pid_matches_all = np.zeros(shape=(1, len(gallery_fids)))
    num_of_NaN = 0
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)
    num_of_paired_img = len(query_pids)

    with tf.Session() as sess:
        for start_idx in count(step=batch_size):
            try:
                # Compute distance to all gallery embeddings for the batch of queries
                distances, pids, fids = sess.run([batch_distances, batch_pids, batch_fids])
                print('\rEvaluating batch {}-{}/{}'.format(
                   start_idx, start_idx + len(fids), len(query_fids)),
                   flush=True, end='')
            except tf.errors.OutOfRangeError:
                print()  # Done!
                break

            # Convert the array of objects back to array of strings
            pids, fids = np.array(pids, '|U'), np.array(fids, '|U')

            # Compute the pid matches
            pid_matches = gallery_pids[None] == pids[:, None]

            # Get a mask indicating True for those gallery entries that should
            # be ignored for whatever reason (same camera, junk, ...) and
            # exclude those in a way that doesn't affect CMC and mAP.
            mask = excluder(fids)
            distances[mask] = np.inf
            pid_matches[mask] = False
            pid_matches_all = np.concatenate((pid_matches_all, pid_matches), axis=0)

            # Keep track of statistics. Invert distances to scores using any
            # arbitrary inversion, as long as it's monotonic and well-behaved,
            # it won't change anything.
            scores = 1 / (1 + distances)
            num_of_col = 10
            for i in range(len(distances)):
                ap = average_precision_score(pid_matches[i], scores[i])
                sorted_distances_inds = np.argsort(distances[i])

                if np.isnan(ap):
                    print()
                    print(str(num_of_NaN) + ". WARNING: encountered an AP of NaN!")
                    print("This usually means a person only appears once.")
                    print("In this case, it's because of {}.".format(fids[i]))
                    print("I'm excluding this person from eval and carrying on.")
                    print()
                    correct_rank.append(-1)
                    results_images.append(gallery_fids[sorted_distances_inds[0:num_of_col]])
                    num_of_NaN += 1
                    num_of_paired_img -= 1
                    scores_of_queries.append(-1)
                    continue

                aps.append(ap)
                scores_of_queries.append(ap)
                # Find the first true match and increment the cmc data from there on.
                rank_k = np.where(pid_matches[i, sorted_distances_inds])[0][0]
                cmc[rank_k:] += 1
                # Save five more similar images to each of image and correct rank of each image
                if (len(gallery_fids) < num_of_col):
                    num_of_col = len(gallery_fids)
                correct_rank.append(rank_k)
                results_images.append(gallery_fids[sorted_distances_inds[0:num_of_col]])

    # Compute the actual cmc and mAP values
    cmc = cmc / num_of_paired_img
    mean_ap = np.mean(aps)

    # Save important data
    rerank_arg = '_with_rerank' if args.re_rank else ''
    filename = 'datasets/'+args.dataset+'/results/'+args.dataset+rerank_arg+"_evaluation.json"
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    out_file = open(os.path.join(os.getcwd(), filename), "w")
    json.dump({'mAP': mean_ap, 'CMC': list(cmc), 'aps': list(aps)}, out_file)
    ranked_images = saveResults(args, results_images, np.argsort(scores_of_queries)[::-1], query_fids, 10)
    visualiseRes.visualiseRankResults(args, ranked_images)

    # Print out a short summary
    if len(cmc) > 9:
        print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(mean_ap, cmc[0], cmc[1],cmc[4], cmc[9]))
    elif len(cmc) > 5:
        print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%}'.format(mean_ap, cmc[0], cmc[1], cmc[4]))
    elif len(cmc) > 2:
        print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%}'.format(mean_ap, cmc[0], cmc[1]))
    else:
        print('mAP: {:.2%} | top-1: {:.2%}'.format(mean_ap, cmc[0]))

    return [mean_ap, cmc[0]]

def run_evaluation(args, query_embs, gallery_embs):

    # Load the query and gallery data from the CSV files.
    query_pids, query_fids = common.load_dataset(args.query_dataset, args.image_root, False)
    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, args.image_root, False)

    [mAP, rank1] = evaluate_embs(args, query_pids, query_fids, query_embs, gallery_pids, gallery_fids, gallery_embs)
    return [mAP, rank1]
