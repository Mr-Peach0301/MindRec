DATASET=Arts # Yelp Instruments
BASE_MODEL=/model/LLaDA-8B-Instruct
DATA_PATH=../data
EPOCHS=3
TASK=mindrec

for lr in 7e-5 1e-4 2e-4
do
    for eps in 0.4 0.5 0.6 0.7
    do
        GEN_LENGTH=9
        OUTPUT_DIR=./ckpt/${DATASET}_${lr}_${eps}_${EPOCHS}_${TASK}/
        RESULTS_FILE=./results/$DATASET/${lr}_${eps}_${TASK}.json
        CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 --master_port=4327 test_ddp.py > ./out/test_${DATASET}_${lr}_${eps}.out \
            --ckpt_path $OUTPUT_DIR \
            --base_model $BASE_MODEL\
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --results_file $RESULTS_FILE \
            --test_batch_size 4 \
            --gen_length $GEN_LENGTH \
            --test_task $TASK \
            --test_prompt_ids 0 \
            --index_file .index.json
    done
done
