export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
python launch.py --name leo_tuning \
                 --qos lv0b \
                 --mode accelerate \
                 --mem_per_gpu 100 \
                 --time 48 \
                 --config configs/default.yaml \
                 --port 2050 \
                 --gpu_per_node 6 \
                 --num_nodes 1 \
                 --partition HGX \
                 task=tuning_noact \
                 note=tuning_noact \
                 dataloader.train.batchsize=2 \
                 llm=opt1.3b \
                #  pretrained_ckpt_path=ckpts/align_frozen/pytorch_model.bin \
