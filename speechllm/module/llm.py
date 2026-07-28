from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
import torch


def get_llm(name, use_lora=False, lora_r=8, lora_alpha=16, finetune=False, gradient_checkpointing=False, special_tokens=None, enable_dropout=True, dropout_rate=0.05):
    print(f"Loading LLM: {name}, LoRA: {use_lora}, GC: {gradient_checkpointing}")

    llm_tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    llm_config = AutoConfig.from_pretrained(name, trust_remote_code=True, local_files_only=True)
    if enable_dropout:
        print(f"🔧 Injecting attention_dropout={dropout_rate}")
        llm_config.attention_dropout = dropout_rate
    else:
        print("🔧 Dropout disabled, setting attention_dropout=0.0")
        llm_config.attention_dropout = 0.0
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        name, 
        config=llm_config, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # 关键：在包装 LoRA 之前添加特殊 token 并 Resize
    if special_tokens:
        print(f"Adding special tokens before PEFT: {special_tokens}")
        num_added = llm_tokenizer.add_tokens(special_tokens)
        if num_added > 0:
            llm_model.resize_token_embeddings(len(llm_tokenizer))

    if gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        llm_model.gradient_checkpointing_enable()
        if hasattr(llm_model, "enable_input_require_grads"):
            llm_model.enable_input_require_grads()

    if finetune:
        if use_lora:
            # 使用 LoRA 微调，冻结主干参数
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
                
            peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules = [
                        "q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"
                    ],
                    bias="none",
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )

            llm_model = get_peft_model(llm_model, peft_config)
            llm_model.print_trainable_parameters()
        else:
            # 全量微调，不冻结任何参数
            pass
    else:
        # 不微调，冻结所有参数
        for name, param in llm_model.named_parameters():
            param.requires_grad = False

    return llm_tokenizer, llm_model