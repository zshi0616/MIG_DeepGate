cd src
python3 main.py prob --exp_id recgnn_deepgate --data_dir ../data/benchmarks/merged/ --num_rounds 10 --dataset benchmarks --gpus 0 --gate_types INPUT,AND,NOT --dim_node_feature 3 --no_node_cop --aggr_function aggnconv --wx_update --reconv_skip_connection --use_logic_diff