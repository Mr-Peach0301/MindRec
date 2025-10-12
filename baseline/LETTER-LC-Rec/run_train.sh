declare -A path=(
    ["llama3"]=/model/Meta-Llama-3-8B-Instruct/
    ["qwen2.5"]=/model/Qwen2.5-7B-Instruct/
)
DATASET=Games
DATA_PATH=/MindRec/data
EPOCHS=3
USER_NUM=5000
TASK=seqrec

for LR in 5e-5 1e-4 2e-4
do
    for MODEL_TYPE in llama3 qwen2.5
    do
        BASE_MODEL=${path[$MODEL_TYPE]}
        OUTPUT_DIR=./ckpt/${MODEL_TYPE}/${DATASET}_${LR}_${TASK}_${USER_NUM}/
        echo ${DATASET} ${LR} ${eps} ${MODEL_TYPE}
        CUDA_VISIBLE_DEVICES=0 nohup python3 ${MODEL_TYPE}_finetune.py > ./out/${MODEL_TYPE}_${DATASET}_${LR}_sft_${TASK}_${USER_NUM}.out \
            --base_model $BASE_MODEL\
            --output_dir $OUTPUT_DIR \
            --dataset $DATASET \
            --user_num $USER_NUM \
            --data_path $DATA_PATH \
            --model_type $MODEL_TYPE \
            --per_device_batch_size 8 \
            --learning_rate $LR \
            --epochs $EPOCHS \
            --tasks ${TASK} \
            --train_prompt_sample_num 1 \
            --train_data_sample_num 0 \
            --index_file .index.json\
            --temperature 1.0
    done
done
