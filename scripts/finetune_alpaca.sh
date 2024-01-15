 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4  ./src/pipeline/train.py  \
    --deepspeed ./scripts/zero2.json  \
    --model_name_or_path google/flan-t5-base  \
    --data_path ./src/eval/data/mmlu  \
    --data_name mmlu \
    --bf16 True   \
    --output_dir ./ckpts/  \
    --use_peft mixlora \
    --num_train_epochs 3  \
    --per_device_train_batch_size 8  \
    --gradient_accumulation_steps 2  \
    --evaluation_strategy "no"   \
    --save_strategy "steps" \
    --save_steps 500
    --learning_rate 5e-4  \
    --weight_decay 0.  \
    --warmup_ratio 0.03  \
    --lr_scheduler_type "cosine"  \
    --logging_steps 1   \
    --dataloader_num_workers 4  \
    --tf32 True  \
    --report_to wandb