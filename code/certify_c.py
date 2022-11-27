# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" Evaluate a smoothed classifier on a dataset. """
import argparse
import os
import datetime
from time import time

import torch

from third_party.core import Smooth
from architectures import get_architecture
#from datasets import get_dataset, DATASETS, get_num_classes
from torchvision import datasets, transforms
import numpy as np

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--base_c_path", type=str, help="path of corrupted dataset")

args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    corruptions = ["identity","brightness", "dotted_line", "glass_blur", "impulse_noise",
            "rotate", "shear", "spatter", "translate", "canny_edges", "fog", "motion_blur", "scale", "shot_noise", "stripe", "zigzag"]

    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], 'mnist')
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, 10, 0.25)

    # prepare output file
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for (idx, corruption) in enumerate(corruptions):
        f = open(args.outfile + corruption.split(".")[0] + ".tsv", 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

        base_c_path = args.base_c_path
        data = torch.Tensor(np.load(base_c_path +corruption + '/test_images.npy')).squeeze(-1)
        targets = torch.LongTensor(np.load(base_c_path + corruption + '/test_labels.npy'))
        
        for i in range(len(data)):

            # only certify every args.skip examples, and stop after args.max examples
            if i % args.skip != 0:
                continue
            if i == args.max:
                break
            (x, label) = (data[i])/255.0, targets[i]
            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
            after_time = time()
            correct = int(prediction == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

        f.close()
