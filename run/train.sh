#!/bin/bash
NUM_PROC=4

cd src
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./main.py prob \
 --exp_id rawmig \
 --data_dir ../data/rawmig_10k \
 --num_rounds 10 \
 --gpus 0,1,2,3 --batch_size 16 \
 --aggr_function aggnconv \
 --wx_update
