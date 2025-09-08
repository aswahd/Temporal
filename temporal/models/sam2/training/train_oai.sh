export CUDA_VISIBLE_DEVICES=3 && nohup python -m training.train  -c configs/sam2.1_training/OAI_joint_training.yaml  --use-cluster 0  --num-gpus 1 > oai_joint_training.log &  

