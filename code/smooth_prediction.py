# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" This script loads a base classifier and then runs PREDICT on many examples from a dataset."""
import os
import argparse
import datetime
from time import time

import setGPU
import torch
import numpy as np

from third_party.core import Smooth
from architectures import get_architecture
from datasets import get_dataset, DATASETS, get_num_classes


parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    n_classes = get_num_classes(args.dataset)
    smoothed_classifier = Smooth(base_classifier, n_classes, args.sigma)

    # prepare output file
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    f = open(args.outfile, 'w')
    print("idx\tlabel\tp1\tp2\tc1\tc2\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    confidences = np.ones((len(dataset), n_classes)) / n_classes
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        before_time = time()

        # make the prediction
        top2, ps, confidence = smoothed_classifier.confidence_top2(x, args.N, args.batch)

        confidences[i, :] = confidence

        after_time = time()
        correct = int(top2[0] == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{}\t{}".format(i, label, top2[0], top2[1], ps[0], ps[1],
                                                      correct, time_elapsed), file=f, flush=True)

    f.close()
    np.save(outdir + f'/model_conf_{args.sigma}_{args.split}.npy', confidences)
