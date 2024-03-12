export DATA_PATH=/root/data/root/data/8M/c4_en_t5-base_512/
export OMP_NUM_THREADS=32

torchrun --nproc-per-node 8 torchrun_main.py \
    --model_config configs/llama_20m.json \
    --dataset_path $DATA_PATH \
    --batch_size 64 \
    --total_batch_size 512 \
    --lr 2e-3 \
    --max_length 512 \
    --save_every 1000 \
    --eval_every 1000 \
    --num_training_steps 30900 \
    --tags regular-llama-20m

