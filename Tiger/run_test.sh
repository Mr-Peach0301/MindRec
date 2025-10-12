DATA_PATH=../data
USER_NUM=5000

for DATASET in Instruments Games Arts
do
    for LR in 5e-4 1e-3
    do
        RESULTS_FILE=./results/$DATASET/${LR}.json
        CKPT_PATH=./ckpt/${DATASET}_${LR}
        python test.py \
            --gpu_id 0 \
            --ckpt_path $CKPT_PATH \
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --user_num $USER_NUM \
            --results_file $RESULTS_FILE \
            --test_batch_size 64 \
            --num_beams 10 \
            --test_prompt_ids 0 \
            --index_file .index.json
    done
done
