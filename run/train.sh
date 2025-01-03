#!/bin/bash
NUM_PROC=2
GPUS=0,1

cd src
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./main.py prob \
 --exp_id optmig \
 --data_dir ../data/optmig \
 --num_rounds 10 \
 --gpus ${GPUS} --batch_size 16 \
 --aggr_function aggnconv \
 --wx_update 
