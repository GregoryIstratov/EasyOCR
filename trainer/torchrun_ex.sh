torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=172.25.0.81 --master_port=12355 \
distributed/train.py --config=config_files/resnet_lstm_attn.yaml

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=172.25.0.81 --master_port=12355 distributed/train.py --config=config_files/resnet_lstm_attn.yaml

docker run --gpus all --env CUDA_VISIBLE_DEVICES=0,1,2,3 --shm-size 50G --rm -it -v /home/greg/EasyOCR:/EasyOCR trainer /bin/bash \
-c 'cd /EasyOCR/trainer; torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=172.25.0.81 --master_port=12355 distributed/train.py --config=config_files/resnet_lstm_attn.yaml'
