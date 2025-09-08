export CUDA_VISIBLE_DEVICES=2,3 && nohup python -m training.train \
    -c configs/sam2.1_training/sam2.1_hiera_b+_Hip_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 2 \
    > logs/sam2.1_hiera_b+_Hip_finetune.log 2>&1 &


export CUDA_VISIBLE_DEVICES=2,3 && nohup python -m training.train \
    -c configs/sam2.1_training/sam2.1_hiera_t_hip_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 2 \
    > sam2.1_hiera_b+_Hip_finetune.log &



export CUDA_VISIBLE_DEVICES=2,3 && python -m training.train \
    -c configs/sam2.1_training/sam2.1_hiera_t_hip_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 2






# OAI
export CUDA_VISIBLE_DEVICES=1,2,3,5 && nohup python -m training.train \
    -c configs/sam2.1_training/sam2.1_hiera_b+_OAI_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 2 \
    > logs/sam2.1_hiera_b+_OAI_finetune.log > &



