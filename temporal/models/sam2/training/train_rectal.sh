export CUDA_VISIBLE_DEVICES=1,2 && nohup python -m training.train  -c configs/sam2.1_training/sam2.1_hiera_b+_RectalCancer_finetune.yaml  --use-cluster 0  --num-gpus 2  > sam2.1_hiera_b+_RectalCancer_finetune.log  &


# export CUDA_VISIBLE_DEVICES=1,2 && python -m training.train  -c configs/sam2.1_training/sam2.1_hiera_b+_RectalCancer_finetune.yaml  --use-cluster 0  --num-gpus 2 

