#!/bin/bash

# cd ../..

# custom config
DATA=/srv/home/zxu444/datasets
TRAINER=MTCoCoOp

DATASET=$1
SEED=$2

# CFG=vit_b32_c4_ep10_batch5_RanClsSmlr_ctxv1
CFG=vit_b32_c4_ep100_batch5_RanClsSmlr_ctxv1
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}/
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
# bash scripts/cocoop/mt_train.sh tiered_imagenet 1