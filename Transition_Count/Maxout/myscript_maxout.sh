#!/bin/bash

python train_feedforward_maxout_grad.py --width=512 --depth=2 --scale=10 --acc=100000 --times=5 --eps=1e-6 --batchsize=16