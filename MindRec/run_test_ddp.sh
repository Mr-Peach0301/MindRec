# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Arts # Games Instruments

declare -A category_length=(
    ["Instruments"]=9
    ["Arts"]=9
    ["Games"]=8
)

BASE_MODEL=/model/LLaDA-8B-Instruct
DATA_PATH=../data
EPOCHS=3
TEST_BATCH_SIZE=8
USER_NUM=5000
TASK=mindrec
NUM_BEAMS=10

for lr in 5e-5 7e-5 1e-4
do
    for eps in 0.4 0.5 0.6 0.7
    do
        GEN_LENGTH=${category_length[$DATASET]}
        OUTPUT_DIR=./ckpt/${DATASET}_${lr}_${eps}_${EPOCHS}_${TASK}_${USER_NUM}/
        RESULTS_FILE=./results/$DATASET/${lr}_${eps}_${TASK}.json
        nohup torchrun --nproc_per_node=1 --master_port=4327 test_ddp.py > ./out/test.out \
            --ckpt_path $OUTPUT_DIR \
            --base_model $BASE_MODEL\
            --dataset $DATASET \
            --user_num $USER_NUM \
            --data_path $DATA_PATH \
            --results_file $RESULTS_FILE \
            --test_batch_size $TEST_BATCH_SIZE \
            --gen_length $GEN_LENGTH \
            --num_beams $NUM_BEAMS \
            --test_task $TASK \
            --test_prompt_ids 0 \
            --index_file .index.json
    done
done
