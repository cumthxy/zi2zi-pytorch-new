
export CUDA_VISIBLE_DEVICES=0

python train.py \
--experiment_dir experiment \
--gpu_ids cuda:0 \
--epoch 50 \
--batch_size 16 \
--sample_steps 100 \
--checkpoint_steps 500 \
--input_nc 1 \
