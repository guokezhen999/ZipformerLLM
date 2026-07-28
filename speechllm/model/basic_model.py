import torch
import torch.nn as nn
import os
import math
import json
import logging
from typing import List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

from speechllm.module.encoder import get_encoder
from speechllm.module.connector import get_connector
from speechllm.module.llm import get_llm
from addict import Dict

class SpeechLLM(nn.Module):
    def __init__(self, config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # 1. Init models
        logging.info("Initializing models...")
        self.audio_encoder = get_encoder(config.model.zipformer)
        self.connector = get_connector(config.model.adapter.name, config.model.adapter)
        
        # 安全地获取 pooling_factor
        ds_factor = config.model.adapter.get('downsampling_factor', None)
        if ds_factor is not None and isinstance(ds_factor, int):
            self.connector.pooling_factor = ds_factor
            logging.info(f"Set connector pooling factor to: {ds_factor}")
        elif not hasattr(self.connector, 'pooling_factor'):
            self.connector.pooling_factor = 1
            logging.info("Pooling factor not found in config, defaulting to 1.")
        
        special_tokens = ["<A>", "</A>", "<W>"]
        self.llm_tokenizer, self.llm_model = get_llm(
            config.model.llm.model_name,
            use_lora=config.model.llm.enable_lora,
            lora_r=config.model.llm.get('lora_r', 8),
            lora_alpha=config.model.llm.get('lora_alpha', 16),
            finetune=True,      # 必须设为 True 以初始化 LoRA 结构进行加载
            special_tokens=special_tokens
        )

        # 获取并保存特殊 Token ID
        self.token_A_id = self.llm_tokenizer.convert_tokens_to_ids("<A>")
        self.token_A_end_id = self.llm_tokenizer.convert_tokens_to_ids("</A>")
        self.token_W_id = self.llm_tokenizer.convert_tokens_to_ids("<W>")
        
        if isinstance(self.token_A_id, list): self.token_A_id = self.token_A_id[0]
        if isinstance(self.token_A_end_id, list): self.token_A_end_id = self.token_A_end_id[0]
        if isinstance(self.token_W_id, list): self.token_W_id = self.token_W_id[0]

        self.use_lora = config.model.llm.enable_lora
        
        # =====================================================================
        # [修改点 1] 无论是否 LoRA，统一初始化 Patch 参数，并设置动态开关
        # =====================================================================
        hidden_size = self.llm_model.config.hidden_size
        self.special_token_input_patch = nn.Parameter(torch.zeros(3, hidden_size))
        self.special_token_output_patch = nn.Parameter(torch.zeros(3, hidden_size))
        
        # 动态开关：只有在加载 Checkpoint 时发现确实有 Patch 权重，才激活 Hook
        self.use_patch_for_generation = False 

        # 注册 Output Patch Hook
        def output_patch_hook(module, input, output):
            # 如果开关未激活（例如：权重里没找到 Patch，或者是原生全量微调），直接放行
            if not getattr(self, 'use_patch_for_generation', False):
                return output
                
            hidden_states = input[0]
            patch_logits = torch.matmul(hidden_states, self.special_token_output_patch.to(hidden_states.device, dtype=hidden_states.dtype).T)
            output[..., self.token_A_id] = patch_logits[..., 0]
            output[..., self.token_A_end_id] = patch_logits[..., 1]
            output[..., self.token_W_id] = patch_logits[..., 2]
            return output

        if hasattr(self.llm_model, 'get_output_embeddings') and self.llm_model.get_output_embeddings() is not None:
            self.llm_model.get_output_embeddings().register_forward_hook(output_patch_hook)
        
        logging.info(f"Special Token IDs: <A>={self.token_A_id}, </A>={self.token_A_end_id}, <W>={self.token_W_id}")

        self.to(device)
        
        # 确保 Connector 与 LLM 的 dtype 一致
        if device.type != "cpu":
            target_dtype = self.llm_model.dtype if hasattr(self.llm_model, "dtype") else torch.bfloat16
            self.connector.to(dtype=target_dtype)
        else:
            self.llm_model.float() 
            self.connector.float()

        self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id if self.llm_tokenizer.pad_token_id is not None else self.llm_tokenizer.eos_token_id
        
        encoder_dtype = next(self.audio_encoder.parameters()).dtype
        connector_dtype = next(self.connector.parameters()).dtype
        llm_dtype = next(self.llm_model.parameters()).dtype
        logging.info(f"Model Precision: Audio Encoder={encoder_dtype}, Connector={connector_dtype}, LLM={llm_dtype}")

    def load_checkpoint(self, checkpoint_path):
        """从检查点加载权重，支持 DeepSpeed 目录和单文件"""
        print_param_status(self, "BEFORE LOADING")
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        
        if os.path.isdir(checkpoint_path):
            logging.info("Detected DeepSpeed checkpoint directory. Consolidating stages...")
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            try:
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
            except Exception as e:
                logging.error(f"❌ Error consolidating DeepSpeed checkpoint: {e}")
                return
        else:
            if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([Dict])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
        
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        def extract_sub_dict(full_dict, prefix):
            return {k[len(prefix):]: v for k, v in full_dict.items() if k.startswith(prefix)}

        # 1. Zipformer Encoder
        audio_prefixes = ["audio_encoder.", "model.audio_encoder."]
        audio_found = False
        for p in audio_prefixes:
            sub = extract_sub_dict(state_dict, p)
            if sub:
                model_keys = set(self.audio_encoder.state_dict().keys())
                sub = {k: v for k, v in sub.items() if k in model_keys}
                self.audio_encoder.load_state_dict(sub, strict=False)
                logging.info(f"✅ Loaded audio_encoder from prefix '{p}' ({len(sub)} keys)")
                audio_found = True
                break
        if not audio_found and "audio_encoder" in state_dict:
            self.audio_encoder.load_state_dict(state_dict["audio_encoder"])
            audio_found = True

        # 2. Connector
        conn_prefixes = ["connector.", "model.connector."]
        conn_found = False
        for p in conn_prefixes:
            sub = extract_sub_dict(state_dict, p)
            if sub:
                self.connector.load_state_dict(sub, strict=False)
                logging.info(f"✅ Loaded connector from prefix '{p}' ({len(sub)} keys)")
                conn_found = True
                break
        if not conn_found and "connector" in state_dict:
            self.connector.load_state_dict(state_dict["connector"])
            conn_found = True

        # 3. LLM (including LoRA)
        llm_prefixes = ["llm_model.", "model.llm_model.", "llm."]
        llm_found = False
        for p in llm_prefixes:
            sub = extract_sub_dict(state_dict, p)
            if sub:
                self.llm_model.load_state_dict(sub, strict=False)
                logging.info(f"✅ Loaded LLM/LoRA from prefix '{p}' ({len(sub)} keys)")
                llm_found = True
                break
        
        if not llm_found and "llm_lora" in state_dict:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.llm_model, state_dict["llm_lora"])
            llm_found = True

        # =====================================================================
        # [修改点 2] Special Token Patching (自动适配全量微调与 Patch 模式)
        # =====================================================================
        patch_keys = ["special_token_input_patch", "special_token_output_patch"]
        patch_loaded_count = 0
        
        for pk in patch_keys:
            target_key = pk if pk in state_dict else f"model.{pk}"
            if target_key in state_dict:
                # 记录加载前的值以计算变化量
                old_val = getattr(self, pk).data.clone()
                getattr(self, pk).data.copy_(state_dict[target_key])
                new_val = getattr(self, pk).data
                diff = torch.abs(old_val - new_val).sum().item()
                
                logging.info(f"✅ 成功加载 {pk} (from {target_key})")
                logging.info(f"      -> 变化量(L1 Diff): {diff:.6f}")
                logging.info(f"      -> 当前均值(Mean): {new_val.mean().item():.8f}, 范数(Norm): {new_val.norm().item():.4f}")
                patch_loaded_count += 1
            else:
                logging.info(f"⏩ Checkpoint 中未找到 {pk} 独立权重。")

        # 核心逻辑：判断是否激活 Patch 机制
        # 核心逻辑：判断是否激活 Patch 机制
        if patch_loaded_count > 0:
            self.use_patch_for_generation = True
            logging.info("🔧 检测到外部 Patch 权重，已激活 Special Token Output Hook。")
            
            # ==============================================================
            # [终极补丁] 将真实的 Input Patch 写入底层 Embedding，防止自回归时查错表
            # ==============================================================
            with torch.no_grad():
                embed_weight = self.llm_model.get_input_embeddings().weight
                
                # patch.data 的 shape 是 [3, hidden_size]，索引对应 <A>, </A>, <W>
                embed_weight[self.token_A_id].copy_(self.special_token_input_patch.data[0])
                embed_weight[self.token_A_end_id].copy_(self.special_token_input_patch.data[1])
                embed_weight[self.token_W_id].copy_(self.special_token_input_patch.data[2])
                
                logging.info("🔧 已将真实的 Input Patch 权重硬编码覆盖到底层 Embedding 矩阵中！")
        else:
            self.use_patch_for_generation = False
            logging.info("🔧 未检测到 Patch 权重 (推断为纯原生全量微调)，关闭 Hook，使用 LLM 原生 Embedding。")
                             
        print_param_status(self, "AFTER LOADING")      
                
    def forward_audio(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_dtype = next(self.audio_encoder.parameters()).dtype
        x = x.to(dtype=encoder_dtype)
        
        encoder_out, encoder_lens = self.audio_encoder(x, x_lens)
        encoder_lens = torch.clamp(encoder_lens - 2, min=0)
        
        connector_dtype = next(self.connector.parameters()).dtype
        encoder_out = encoder_out.to(dtype=connector_dtype)
        
        projector_out = self.connector(encoder_out)
        
        if hasattr(self.connector, 'pooling_factor'):
            output_lens = (encoder_lens + self.connector.pooling_factor - 1) // self.connector.pooling_factor
        return projector_out, output_lens

def print_param_status(model, stage_name):
    """专门针对全量微调：检查词表中特定 ID 对应的权重行"""
    logging.info(f"--- Full SFT Special Tokens Check: {stage_name} ---")
    
    target_tokens = {
        "<A>": model.token_A_id,
        "</A>": model.token_A_end_id,
        "<W>": model.token_W_id
    }

    try:
        embed_table = model.llm_model.model.embed_tokens.weight.data
        logging.info(f"  [Input Embedding] Table Shape: {embed_table.shape}")
        for name, tid in target_tokens.items():
            if tid < embed_table.shape[0]:
                row = embed_table[tid]
                logging.info(f"    Token {name} (ID:{tid}): Mean={row.mean().item():.8f}, L2={row.norm().item():.4f}")
            else:
                logging.error(f"    ❌ Token {name} (ID:{tid}) is OUT OF BOUNDS!")
    except Exception as e:
        pass

    try:
        head_table = None
        if hasattr(model.llm_model, 'lm_head'):
            head_table = model.llm_model.lm_head.weight.data
        elif hasattr(model.llm_model, 'get_output_embeddings'):
            head_table = model.llm_model.get_output_embeddings().weight.data
            
        if head_table is not None:
            logging.info(f"  [Output Head] Table Shape: {head_table.shape}")
            for name, tid in target_tokens.items():
                if tid < head_table.shape[0]:
                    row = head_table[tid]
                    logging.info(f"    Token {name} (ID:{tid}): Mean={row.mean().item():.8f}, L2={row.norm().item():.4f}")
    except Exception as e:
        pass