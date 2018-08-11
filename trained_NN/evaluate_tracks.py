#!/usr/bin/env python3
import json
import os
import time
from datetime import timedelta
import numpy as np
from sklearn.metrics import average_precision_score
import common
import evaluate
import visualiseRes
from excluders import excluder_parser as data_parser

def evaluate_run(args, query_pids, query_fids, query_embs, gallery_pids,
             gallery_fids, gallery_embs):
    '''
    Initiates an image-to-track evaluation

    :param args:
    :param query_pids: The person ids of the query images
    :param query_fids: The figure ids of the query images
    :param query_embs: The embeddings of the query images
    :param gallery_pids: The person ids of the gallery images
    :param gallery_fids: The figure ids of the gallery images
    :param gallery_embs: The embeddings of the gallery images
    :return:
    '''

    # Extract person ids and figure ids from the tracks csv file
    [track_pids, tracks_fids] = common.load_tracks(args.tracks_dataset)

    # Number of images to visualise from best-middle-worst results
    #todo now it is number of the best images which will be saved in the text file for each query image (onlythe paths to the images are saved)
    images_to_viz = 10

    track_pids = np.array(track_pids)

    # Extract camera ids from tracks and queries
    track_cam_ids = np.array([[ data_parser.get_camera_ids(args.excluder, x) for x in tracks_fids[i]][0] for i in range(len(tracks_fids))])
    query_cam_ids = np.array(data_parser.get_camera_ids(args.excluder, query_fids))

    # Associate figures to embeddings of gallery and query set
    gallery_fids_to_embs = dict(zip(gallery_fids,gallery_embs))
    query_figs_to_embs = dict(zip(query_fids,query_embs))

    # Matrix to hold the embeddings of the images in the tracks set
    emb_matrix = [[gallery_fids_to_embs[x] for x in tracks_fids[i]] for i in range(len(tracks_fids))]

    # Metrics initialisation
    aps = []
    correct_rank = []
    results_images = []
    scores_of_queries = []
    num_of_NaN = 0
    num_of_paired_img = len(query_pids)
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)

    # Calculation of scores for every query image
    for i, q in enumerate(query_fids):
        start_time = time.time()

        # Mask to exclude the query image from the tracks set
        mask = ((query_pids[i]!=track_pids) | (query_cam_ids[i]!=track_cam_ids)).astype(bool)
        mask = np.array(mask)
#.replace('query','test')
        # Calculating the distances between the query embedding and the embeddings of the tracks set
        distances = [[np.linalg.norm(query_figs_to_embs[q]- x) for x in emb_matrix[i]] for i in range(len(emb_matrix))]
        distances = np.array(distances)

        # Filter out the query image from the distances matrix
        distances = distances[mask]

        # Get a representative distance between the query image and the track
        if args.selector == 'mean':
            rep_distance = [np.mean(np.array(x), axis=0) for x in distances]
        elif args.selector == 'max':
            rep_distance = [np.max(np.array(x), axis=0) for x in distances]

        # High scores are inverted distances
        scores = 1 / (1 + np.array(rep_distance))

        # Mask for filtering out the query image from the ap scores
        gt_mask = track_pids[mask] == query_pids[i]
        sorted_distances_inds = np.argsort(scores)[::-1]
        ap = average_precision_score(gt_mask, scores)

        # Error handling
        if np.isnan(ap):
            print()
            print("This usually means a person only appears once.")
            print("In this case, it's because of {}.".format(query_fids[i]))
            print("I'm excluding this person from eval and carrying on.")
            print()

            correct_rank.append(-1)
            results_images.append(gallery_fids[sorted_distances_inds[0:images_to_viz]])
            num_of_NaN += 1
            num_of_paired_img -= 1
            scores_of_queries.append(-1)
            continue

        aps.append(ap)
        scores_of_queries.append(ap)

        # Find the first true match and increment the cmc data from there on.
        rank_k = np.where(gt_mask[sorted_distances_inds])[0][0]
        cmc[rank_k:] += 1

        # Save few more similar images to each of image and correct rank of each image for visualization the results
        if (len(gallery_fids) < images_to_viz):
            images_to_viz = len(gallery_fids)
        correct_rank.append(rank_k)

        first_img_paths_of_tracks = np.array([tracks_fids[i][0] for i in range(len(tracks_fids))])
        results_images.append(first_img_paths_of_tracks[sorted_distances_inds[0:images_to_viz]])

        # measure time and print
        elapsed_time = time.time() - start_time
        seconds_todo = (len(query_fids) - i) * elapsed_time
        print('\rEvaluate query {}/{}, ETA: {} ({:.2f}s/it)'.format(i, len(query_fids),
                                                                    timedelta(seconds=int(seconds_todo)), elapsed_time),
              flush=True, end='')
        print()

    cmc = cmc / num_of_paired_img
    mean_ap = np.mean(aps)

    # Save important data
    ranked_images = evaluate.saveResults(args, results_images, np.argsort(scores_of_queries)[::-1], query_fids, 10)
    filename = 'datasets/' + args.dataset + '/results/'+args.dataset+'_image_to_track_with_' + args.selector + "_evaluation.json"
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    out_file = open(os.path.join(os.getcwd(), filename), "w")
    json.dump({'mAP': mean_ap, 'CMC': list(cmc), 'aps': list(aps)}, out_file)
    out_file.close()
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

'''
Function to initiate the evaluation from test.py
'''
def run_evaluation(args, query_embs, gallery_embs):

    # Load the query and gallery data from the CSV files.
    query_pids, query_fids = common.load_dataset(args.query_dataset, args.image_root, False)
    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, args.image_root, False)

    [mAP, rank1] = evaluate_run(args, query_pids, query_fids, query_embs, gallery_pids, gallery_fids, gallery_embs)

    return [mAP, rank1]