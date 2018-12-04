#!/bin/bash

# branch number
branch=2
# the checkpoint and logs.
train_fold="log"
eval_fold="logeval"
best_fold="logbest"

# the dataset.
DATASET_DIR=~/tensorflowproject/data/cifar100v

if [ ! -d ${DATASET_DIR} ]; then
    echo "the path ${DATASET_DIR} is not exits!!"
    exit
fi
cwd=$(cd `dirname $0`; pwd)
echo "${cwd}"

BRANCH_DIR="${cwd}/${branch}"
TRAIN_DIR="${cwd}/${branch}/${train_fold}"
EVAL_DIR="${cwd}/${branch}/${eval_fold}"
BEST_DIR="${cwd}/${branch}/${best_fold}"
echo "${BRANCH_DIR}"
echo "${TRAIN_DIR}"
# Run training.
echo "------------------------begin the branch:${branch}"
python train_main.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cifar100 \
  --data_dir=${DATASET_DIR} \
  --model_name=dendritenet \
  --train_epochs=100 \
  --batch_size=100 \
  --learning_rate=0.01 \
  --normalization=No \
  --factor=0.1 \
  --l2_scale=0 \
  --keep_probablity=1.0 \
  --branches=${branch} \
  --layer_number=3 \
  --net_length=512 \
  --branch_dir=${BRANCH_DIR} \
  --best_dir=${BEST_DIR}
echo "------------------------finish the branch:${branch}"


