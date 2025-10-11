# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR=./ckpt/$DATASET/
USER_NUM=5000

for DATASET in Instruments Games Arts
do
    for LR in 5e-4 1e-3
    do
        OUTPUT_DIR=./ckpt/${DATASET}_${LR}
        torchrun --nproc_per_node=1 --master_port=2314 ./finetune.py \
            --output_dir $OUTPUT_DIR \
            --dataset $DATASET \
            --per_device_batch_size 1024 \
            --learning_rate $LR \
            --user_num $USER_NUM \
            --epochs 200 \
            --index_file .index.json \
            --temperature 1.0
    done
done