# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Arts

declare -A category_length=(
    ["Instruments"]=9
    ["Arts"]=9
    ["Games"]=8
)
gen_length_value=${category_length[$DATASET]}
declare -A gen_length=(
    ["seqrec_c"]=50
    ["seqrec_c_simple"]=$gen_length_value
)

BASE_MODEL=/model/LLaDA-8B-Instruct
DATA_PATH=../data
EPOCHS=3
SIDE=left
TEST_BATCH_SIZE=8
USER_NUM=5000
TRAIN_TASK=seqrec_c_simple
TASK=seqrec_c_simple
NUM_BEAMS=10

for lr in 1e-4
do
    for eps in 0.6
    do
        GEN_LENGTH=${gen_length[$TASK]}
        OUTPUT_DIR=./ckpt/${DATASET}_${lr}_${eps}_${EPOCHS}_${SIDE}_${TRAIN_TASK}_8_${USER_NUM}/
        RESULTS_FILE=./results/$DATASET/${lr}_${eps}_${TASK}.json
        nohup torchrun --nproc_per_node=2 --master_port=4327 test_ddp.py > log1.out \
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
