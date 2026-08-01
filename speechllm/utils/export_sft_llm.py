"""
从 SFT checkpoint 中提取 LLM 权重，导出为 HuggingFace 格式，供 SGLang 加载。
用法：
    python export_sft_llm.py \
        --ckpt /path/to/step-step=92000.pt \
        --base_model /path/to/mt_0.6b_3_10 \
        --output /path/to/sglang_init_model
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="SFT checkpoint .pt 路径")
parser.add_argument("--base_model", required=True, help="原始 HuggingFace 模型路径")
parser.add_argument("--output", required=True, help="导出目录")
args = parser.parse_args()

print(f"Loading base model from {args.base_model} ...")
model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)

# 添加 special tokens（与训练代码保持一致）
special_tokens = ['<A>', '</A>', '<W>']
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))
print(f"Tokenizer vocab size after adding special tokens: {len(tokenizer)}")

print(f"Loading SFT checkpoint from {args.ckpt} ...")
ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
# DeepSpeed exports a flat state_dict directly; Lightning .ckpt wraps it under "state_dict"
flat = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# 提取 llm_model.* 权重
llm_dict = {k[len("llm_model."):]: v for k, v in flat.items() if k.startswith("llm_model.")}
print(f"Found {len(llm_dict)} llm_model parameters in checkpoint")

result = model.load_state_dict(llm_dict, strict=False)
if result.missing_keys:
    print(f"WARNING missing_keys: {result.missing_keys[:10]}")
if result.unexpected_keys:
    print(f"WARNING unexpected_keys: {result.unexpected_keys[:10]}")

# 覆盖 special token embedding（input patch）
if "special_token_input_patch" in flat:
    patch = flat["special_token_input_patch"].to(torch.bfloat16)  # (3, D)
    # special token ids
    ids = tokenizer.convert_tokens_to_ids(special_tokens)
    print(f"Special token ids: {dict(zip(special_tokens, ids))}")
    with torch.no_grad():
        for i, tok_id in enumerate(ids):
            model.model.embed_tokens.weight[tok_id] = patch[i]
    print("special_token_input_patch applied to embed_tokens")

if "special_token_output_patch" in flat:
    patch = flat["special_token_output_patch"].to(torch.bfloat16)  # (3, D)
    ids = tokenizer.convert_tokens_to_ids(special_tokens)
    with torch.no_grad():
        for i, tok_id in enumerate(ids):
            model.lm_head.weight[tok_id] = patch[i]
    print("special_token_output_patch applied to lm_head")

print(f"Saving to {args.output} ...")
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
print("Done.")
