TODO: pay attention to add safe_tensors=False when warming up

# ReLoRA -- PEFT Pretraining
> Official code for Stack More Layers Differently: High-Rank Training Through Low-Rank Updates https://arxiv.org/abs/2307.05695
<img width="813" alt="ReLoRA" src="https://github.com/Guitaricet/peft_pretraining/assets/2821124/41415bd0-b39f-4f2c-9bbd-5fd6555e87a7">

## Setup

Requires Python 3.10+ (due to param annotaitons style) and PyTorch 2.0+ (for flash attention).
All requirements are listed in `requirements.txt` and kept up-to-date.

```bash
pip install -e .
pip install flash-attn
```

> We do not have flash attention in our requirements, because for some reason flash attention installation script requires torch and some other requirements to already be installed

## 1B training script

The rule of thumb of selecting the learning rate I use for now is 2X regular training learning rate.
It might require tuning on larger models.
Microbatch size depends on the GPU memory and needs to be tuned to maximize the throughput.
Note that relora allows to use larger microbatch sizes than regular training.

Number of steps is 143K (Pythia) minus 10K, because we start from the checkpoint at 10K steps.
Relora reset frequency is 5320 so that the number of steps is would be divisible by it.

```bash
torchrun --nproc-per-node 8 --nnodes 1 torchrun_main.py --training_config training_configs/1B_v1.0.yaml
```

## Usage

Pre-process data (might take some time)

```bash
python pretokenize.py \
    --save_dir preprocessed_data \
    --tokenizer t5-base \
    --dataset c4 \
    --dataset_config en \
    --text_field text \
    --sequence_length 512
```

The script will log where the pre-processed data is saved. It should be something like `preprocessed_data/<dataset>_<tokenizer>_<sequence_length>`.

To train a model using ReLoRA, first, perform a warmup through regular training.

```bash
export DATA_PATH=<path to preprocessed data>

torchrun --nproc-per-node <N_GPUS> torchrun_main.py \
    --model_config configs/llama_250m.json \
    --dataset_path $DATA_PATH \
    --batch_size 24 \
    --total_batch_size 1152 \
    --lr 5e-4 \
    --max_length 512 \
    --save_every 1000 \
    --eval_every 1000 \
    --num_training_steps 20000
    --tags warm_start_250M
```

> **Reproducibility note:** The way we ran the experiments in the paper was by specifying full num_training_steps, including both the warmup and the ReLoRA training, and stopping it after the desired number of steps was completed. Providing only the number of training steps should work too. The only difference will be the LR schedule during the warmup period.

When you have a warmed-up network checkpoint, run the script with ReLoRA enabled. Note that we use a larger LR during the ReLoRA stage.

Train with PEFT
```bash
torchrun --nproc-per-node <N_GPUS> torchrun_main.py \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 1152 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft \
    --relora 5000 \
    --cycle_length 5000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_relora True \
    --num_training_steps 20000 \
    --save_every 5000 \
    --eval_every 5000 \
    --warmed_up_model checkpoints/llama_250m-2023-06-09-11-29-56/model_5000 \
    --tags relora_250M
```



## Note on batch sizes

To minimize the pain with multi-GPU setups, we recommend avoiding using `--gradient_accumulation` option directly. Instead, specify `--total_batch_size` and allow the script to figure out the gradient accumulation option based on `--batch_size` and the number of GPUs used.

## Relora

Relora integrates existing LoRA parameters into the main network and resets them.
In principle, such an approach can be more flexible than LoRA, but you need to be careful with

1. Optimizer states
2. Learning rate schedule during and right after the reset
3. How frequently you reset

Reset frequency is determined by `--relora` parameter (in the number of update steps, not global steps).
Optimizer reset options are:
```
"--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true"
"--optimizer_random_pruning", default=False, type=float
"--optimizer_magnitude_pruning", default=False, type=float
```

We found that using `--optimizer_magnitude_pruning 0.9` or plain `--reset_optimizer_on_relora` usually performs well.
Note that `--reset_optimizer_on_relora is True by default` and you need to provide `--reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.9` if you want to do magnitude pruning.

ReLoRA currently only supports cosine decay learning rate scheduler.
Specifically `cosine_restarts` that works in cyclical mode that repeats the warmup every `--cycle_length` update steps.

## Warm starts

You can start LoRa from a partially trained checkpoint. To do that, provide `--warmed_up_model` option. For example:

```
torchrun torchrun_main.py ... <other options> .. --warmed_up_model checkpoints/llama_1b-2023-05-05-20-12-43/model_1000
```

## Distributed training

We support single-node distributed training using vanilla PyTorch DDP.
| `main.py` script does not have all features required for relora and will be deleted soon. We recommend to use `torchrun --nproc-per-node 1` for a single-GPU training.

An example of using torchrun
```bash
torchrun --nproc-per-node 8 torchrun_main.py \
    --model_config configs/llama_35m.json \
    --use_peft \
    --lora_r 128 \
    --relora 500 \
    --cycle_length 500 \
    --warmup_steps 250 \
    --reset_optimizer_on_relora False \
    --lr 0.001 \
    --batch_size 60 \
    --total_batch_size 480 \
    --num_training_steps 5000 \
    --save_every 5000 \
    --dtype bfloat16 \
    --tags relora_debug,example
```

Where `--nproc-per-node` is the nubmer of GPUs you are using.

## Citation

```
@misc{lialin2023stack,
    title={Stack More Layers Differently: High-Rank Training Through Low-Rank Updates},
    author={Vladislav Lialin and Namrata Shivagunde and Sherin Muckatira and Anna Rumshisky},
    year={2023},
    eprint={2307.05695},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
