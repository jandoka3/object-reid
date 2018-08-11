#!/usr/bin/env python3
from argparse import ArgumentParser

import embed, evaluate, evaluate_tracks
from store_args import Arguments

parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--dataset', required=True,
    choices=('Cuhk03_labeled', 'Cuhk03_detected','Market1501', 'VeRi', 'VehicleReId'),
    help='Name of the dataset')

parser.add_argument(
    '--gpu', default=0, type=int,
    help='ID of GPU which should be use for running.')

parser.add_argument(
    '--re_rank', action='store_true', default=False,
    help='When this flag is provided, re-ranking is performed.')

parser.add_argument(
    '--image_to_track', action='store_true', default=False,
    help='When this flag is provided, image-to-track evaluation is performed.')

parser.add_argument(
    '--selector', default='mean', choices=('mean', 'max'),
    help='')

def run_test(args):

    print("Starting evaluation with the following parameters:")
    args.toString()
    query_embs = embed.run_embedding(args, args.query_dataset)
    gallery_embs = embed.run_embedding(args, args.gallery_dataset)

    if args.image_to_track:
        [mAP, rank1] = evaluate_tracks.run_evaluation(args, query_embs, gallery_embs)
    else: [mAP, rank1] = evaluate.run_evaluation(args, query_embs, gallery_embs)

    return [mAP, rank1]

if __name__ == '__main__':
    import os
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    stored_args = Arguments()
    stored_args.save_args(args)
    run_test(stored_args)