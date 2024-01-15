from transformers import T5Config

class T5MoEConfig(T5Config):
    """
    Config of T5MoE.
    """

    def __init__(
            self,
            vocab_size=32128,
            d_model=512,
            d_kv=64,
            d_ff=2048,
            num_layers=6,
            num_decoder_layers=None,
            num_heads=8,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            is_moe=False,
            num_experts=8,
            num_experts_per_tok=2,
            expert_capacity=64,
            router_bias=False,
            router_jitter_noise=0.01,
            router_dtype="float32",
            router_ignore_padding_tokens=False,
            router_z_loss_coef=0.001,
            router_aux_loss_coef=0.001,
            initializer_factor=1.0,
            add_router_probs=False,
            lora_config={
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "merge_weights": False,
            },
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
            classifier_dropout=0.0,
            **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            is_encoder_decoder=is_encoder_decoder,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            classifier_dropout=classifier_dropout,
            **kwargs
        )

        # [FIXME: INTRODUCE MORE CONFIGURATION HERE!]
        
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_capacity = expert_capacity
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype

        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.add_router_probs = add_router_probs

        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef

        self.lora_config = lora_config