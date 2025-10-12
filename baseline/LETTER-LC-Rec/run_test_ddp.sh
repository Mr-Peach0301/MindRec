# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

declare -A path=(
    ["llama3"]=/model/Meta-Llama-3-8B-Instruct/
    ["qwen2.5"]=/model/Qwen2.5-7B-Instruct/
)

DATASET=Games
DATA_PATH=/MindRec/data
USER_NUM=5000

for LR in 5e-5 1e-4 2e-4
do
    for MODEL_TYPE in llama3 qwen2.5
    do
        BASE_MODEL=${path[$MODEL_TYPE]}
        OUTPUT_DIR=./ckpt/${MODEL_TYPE}/${DATASET}_${LR}_seqrec_${USER_NUM}/
        RESULTS_FILE=./results/$DATASET/${MODEL_TYPE}/${LR}.json
        nohup torchrun --nproc_per_node=2 --master_port=4325 ${MODEL_TYPE}_test_ddp.py > ./out/test_${DATASET}_${MODEL_TYPE}_${LR}.out \
            --ckpt_path $OUTPUT_DIR \
            --base_model $BASE_MODEL\
            --dataset $DATASET \
            --user_num $USER_NUM \
            --data_path $DATA_PATH \
            --model_type $MODEL_TYPE \
            --results_file $RESULTS_FILE \
            --test_batch_size 8 \
            --num_beams 10 \
            --test_prompt_ids 0 \
            --index_file .index.json
    done
done
