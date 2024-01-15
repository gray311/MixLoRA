mixlora_config = dict(
    target_modules=r'.*DenseReluDense.*\.(wi_0|wi_1|wo)',
    r=8 ,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    num_experts=8,
    num_experts_per_tok=2,
    expert_capacity=64,
    num_tasks=1,
)

lora_config = dict(
    target_modules=r'.*DenseReluDense.*\.(wi_0|wi_1|wo)',
    r=8 ,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
