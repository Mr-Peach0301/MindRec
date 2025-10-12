DATASET=Arts # Games Instruments
BASE_MODEL=/model/LLaDA-8B-Instruct
DATA_PATH=../data
EPOCHS=3
USER_NUM=5000
TASK=mindrec

for LR in 5e-5 7e-5 1e-4
do
    for eps in 0.4 0.5 0.6 0.7
    do
        OUTPUT_DIR=./ckpt/${DATASET}_${LR}_${eps}_${EPOCHS}_${TASK}_${USER_NUM}
        echo ${DATASET} ${LR} ${eps} ${TASK}
        CUDA_VISIBLE_DEVICES=0 nohup python lora_finetune.py > ./out/${DATASET}_${LR}_${eps}_${TASK}.out \
            --base_model $BASE_MODEL\
            --output_dir $OUTPUT_DIR \
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --eps $eps \
            --user_num $USER_NUM \
            --per_device_batch_size 8 \
            --learning_rate $LR \
            --epochs $EPOCHS \
            --tasks $TASK \
            --train_prompt_sample_num 1 \
            --train_data_sample_num 0 \
            --index_file .index.json\
            --temperature 1.0
    done
done
