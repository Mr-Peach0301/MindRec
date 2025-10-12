# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Games
BASE_MODEL=/model/LLaDA-8B-Instruct
DATA_PATH=/MindRec/data
EPOCHS=3
TEST_BATCH_SIZE=8
USER_NUM=5000

for lr in 5e-5 7e-5 1e-4
do
    for eps in 0.4 0.5 0.6 0.7
    do
        OUTPUT_DIR=./ckpt/${DATASET}_${lr}_${eps}_${EPOCHS}_seqrec_${USER_NUM}/
        RESULTS_FILE=./results/$DATASET/${lr}_${eps}.json
        nohup torchrun --nproc_per_node=2 --master_port=4323 test_ddp.py > ./out/test_${DATASET}_${lr}_${eps}.out \
            --ckpt_path $OUTPUT_DIR \
            --base_model $BASE_MODEL\
            --dataset $DATASET \
            --user_num $USER_NUM \
            --data_path $DATA_PATH \
            --results_file $RESULTS_FILE \
            --test_batch_size ${TEST_BATCH_SIZE} \
            --num_beams 10 \
            --test_prompt_ids 0 \
            --index_file .index.json
    done
done
