cd src
python main.py prob --exp_id conv_aggnconv --data_dir ../data/benchmarks/merged/ --num_rounds 10 --dataset benchmarks --gpus 0 --gate_types INPUT,AND,NOT --dim_node_feature 3 --no_node_cop --aggr_function aggnconv --wx_update --arch convgnn --num_aggr 3 --un_directed
