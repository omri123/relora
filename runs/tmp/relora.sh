export DATA_PATH=/root/data/root/data/8M/c4_en_t5-base_512/
export OMP_NUM_THREADS=32

# on 4090, bs 24 is the best. for 130m.
torchrun --nproc-per-node 8 torchrun_main.py \
    --model_config configs/llama_20m.json \
    --dataset_path $DATA_PATH \
    --batch_size 32 \
    --total_batch_size 256 \
    --lr 2e-3 \
    --max_length 512 \
    --save_every 1000 \
    --eval_every 1000 \
    --num_training_steps 25900 \
    --tags regular-llama-20m \
        --warmed_up_model checkpoints/colorful-spaceship-28/model_1000/ \
        --warmup_steps 50 \
        --use_peft true \
        --lora_r 64 \
        --tags relora \
        --relora 100 \
        --cycle_length 100 \
        --restart_warmup_steps 50 \
        --scheduler cosine_restarts \
        --reset_optimizer_on_relora True

