import os

import cv2
import numpy as np

# Initialisation
def visualiseRankResults(args, data):

    # constants
    NUM_OF_RANKING = 10
    START_QUERY = 0
    END_QUERY = 10

    # load data
    query_fids = data[:, 0]
    results_images = data[:, 1:]

    height_new_im = ((NUM_OF_RANKING + 1) * 128)
    new_im = np.zeros((1, height_new_im, 3))

    drawAndSaveResults(args, new_im, results_images, query_fids, NUM_OF_RANKING, START_QUERY, END_QUERY)

# Checking if the figure is of the same person
def isTruePositive(query_fid, gallery_fid, dataset):

    # check if the images are from the same person
    if (dataset=='VeRi') | (dataset=='Market1501') | (dataset=='VehicleReId'):
        query_pid = query_fid.split("/")[1].split("_")[0]
        gal_pid = gallery_fid.split("/")[1].split("_")[0]
    elif (dataset=='Cuhk03_detected') | (dataset=='Cuhk03_labeled'):
        query_pid = query_fid.split("_")[0]
        gal_pid = gallery_fid.split("_")[0]
    else:
        raise ValueError("Dataset " + dataset + " is inexistent.")

    if query_pid == gal_pid:
        return True
    else:
        return False

def getSortedPids(fids, dataset):
    # check if the images are from the same person
    if (dataset == 'VeRi') | (dataset == 'Market1501') | (dataset == 'VehicleReId'):
        pids = np.array([x.split("/")[1].split("_")[0] for x in fids])
    elif (dataset == 'Cuhk03_detected') | (dataset == 'Cuhk03_labeled'):
        pids = np.array([x.split("_")[0] for x in fids])
    else:
        raise ValueError("Dataset " + dataset + " is inexistent.")
    return pids

def drawAndSaveResults(args, new_im, results_images, query_fids, num_of_col, start_q, end_q):
    '''
    Function to save results on file and provide visualisation
    :param args: arguments
    :param new_im: initialization of the final image
    :param results_images: array that holds the ranked images to be visualised
    :param query_fids: array of the query paths
    :param num_of_col: number of the n-top ranked images
    :return: -
    '''
    # we want visualise the n-top query images with unique ids
    # (it is obvious that if one car from cam02 will be easy for re-identification, the same car from cam03 will be probably in n-top too)
    drawen_pids = set()
    # extract query pids
    query_pids = getSortedPids(query_fids, args.dataset)
    # draw result images into image
    i = start_q-1
    count = 0
    while count < (end_q-start_q):
        i += 1
        # check if the car id is already drawen
        if query_pids[i] in drawen_pids:
            continue
        # the car with the id has not been visualised yet
        drawen_pids.add(query_pids[i])
        q_im = cv2.resize(cv2.imread(os.path.join(args.image_root, query_fids[i])), dsize=(128, 256), interpolation=cv2.INTER_CUBIC)
        image_size = q_im.shape
        images = []
        images.append(q_im)
        rect_topleft_list = []
        rect_bottomright_list = []
        colors = []
        rect_topleft = (0, 0)
        rect_bottomright = (0, 0)
        color = (0, 0, 0)
        for j in range(0, num_of_col):
            images.append(cv2.resize(cv2.imread(os.path.join(args.image_root, results_images[i][j])),
                                     dsize=(128, 256), interpolation=cv2.INTER_CUBIC))
            if isTruePositive(query_fids[i], results_images[i][j], args.dataset):
                # true positive image
                rect_topleft = ((j + 1) * image_size[1] + 2, count * image_size[0] + 2)
                rect_topleft_list.append(rect_topleft)
                rect_bottomright = (rect_topleft[0] + image_size[1] - 4, rect_topleft[1] + image_size[0] - 4)
                rect_bottomright_list.append(rect_bottomright)
                color = (0, 255, 0)
                colors.append(color)
            else:
                # false positive image
                rect_topleft = ((j + 1) * image_size[1] + 2, count * image_size[0] + 2)
                rect_topleft_list.append(rect_topleft)
                rect_bottomright = (rect_topleft[0] + image_size[1] - 4, rect_topleft[1] + image_size[0] - 4)
                rect_bottomright_list.append(rect_bottomright)
                color = (0, 0, 255)
                colors.append(color)

        new_sub_im = np.concatenate(images, 1)
        new_im = np.concatenate((new_im, new_sub_im), 0)
        for k in range(0, len(rect_topleft_list)):
            cv2.rectangle(new_im, rect_topleft_list[k], rect_bottomright_list[k], colors[k], 5)
        cv2.rectangle(new_im, rect_topleft, rect_bottomright, color, 5)
        count += 1

    new_im = np.concatenate((new_im, np.zeros((15, ((num_of_col+1)*image_size[1]), image_size[2]))))

    rerank_arg = '_with_rerank' if args.re_rank else ''
    image_to_track = '_image_to_track_with_' + args.selector if args.image_to_track else ''
    filename = 'datasets/' + args.dataset + '/results/' + args.dataset + rerank_arg + image_to_track + "_qualitative.jpg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(os.path.join(os.getcwd(), filename), new_im)
