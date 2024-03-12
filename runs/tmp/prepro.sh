python pretokenize.py \
    --save_dir /root/data/20M-256/ \
    --tokenizer t5-base \
    --dataset c4 \
    --dataset_config en \
    --text_field text \
    --sequence_length 256 \
    --take 20000000