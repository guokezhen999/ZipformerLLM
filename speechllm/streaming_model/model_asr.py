import torch
import torch.nn as nn
import os
import math
import json
from typing import List, Optional, Tuple

from speechllm.module.encoder import get_encoder
from speechllm.module.connector import get_connector
from speechllm.module.llm import get_llm
from speechllm.zipformer.model import StreamingEncoderAdapter

class StreamingSpeechLLM(nn.Module):
    def __init__(self, config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # 1. Init models
        print("Initializing models...")
        self.base_encoder = get_encoder(config.model.zipformer)
        self.connector = get_connector(config.model.adapter.name, config.model.adapter)

        # Align with ZipformerLLM SpeechLLM: optional downsampling_factor override
        ds_factor = config.model.adapter.get("downsampling_factor", None)
        if ds_factor is not None and isinstance(ds_factor, int):
            self.connector.pooling_factor = ds_factor
            print(f"Set connector pooling factor to: {ds_factor}")
        elif not hasattr(self.connector, "pooling_factor"):
            self.connector.pooling_factor = 1
            print("Pooling factor not found in config, defaulting to 1.")

        special_tokens = ["<A>", "</A>", "<W>"]
        self.llm_tokenizer, self.llm_model = get_llm(
            config.model.llm.model_name,
            use_lora=config.model.llm.enable_lora,
            lora_r=config.model.llm.get('lora_r', 8),
            lora_alpha=config.model.llm.get('lora_alpha', 16),
            finetune=True,      # 必须设为 True 以初始化 LoRA 结构进行加载
            special_tokens=special_tokens
        )

        # 实例化 kv cache 计算器
        self.kv_calculator = IncrementalKvCalculator(self.llm_model)

        # 获取并保存特殊 Token ID
        self.token_A_id = self.llm_tokenizer.convert_tokens_to_ids("<A>")
        self.token_A_end_id = self.llm_tokenizer.convert_tokens_to_ids("</A>")
        self.token_W_id = self.llm_tokenizer.convert_tokens_to_ids("<W>")
        
        # 处理可能返回列表的情况
        if isinstance(self.token_A_id, list): self.token_A_id = self.token_A_id[0]
        if isinstance(self.token_A_end_id, list): self.token_A_end_id = self.token_A_end_id[0]
        if isinstance(self.token_W_id, list): self.token_W_id = self.token_W_id[0]

        # 2. 初始化特殊 Token 的训练策略
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
        
        print(f"Special Token IDs: <A>={self.token_A_id}, </A>={self.token_A_end_id}, <W>={self.token_W_id}")

        self.streaming_adapter = StreamingEncoderAdapter(self.base_encoder, self.connector)
        self.to(device)
        
        if device.type != "cpu":
            target_dtype = self.llm_model.dtype if hasattr(self.llm_model, "dtype") else torch.bfloat16
            # 注意：这里我们转换 connector，streaming_adapter 会引用这个 connector
            self.connector.to(dtype=target_dtype)
            print(f"Converting Connector to {target_dtype} to match LLM.")
        else:
            self.llm_model.float()

        self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id if self.llm_tokenizer.pad_token_id is not None else self.llm_tokenizer.eos_token_id

        # 打印各部分计算精度
        encoder_dtype = next(self.base_encoder.parameters()).dtype
        connector_dtype = next(self.connector.parameters()).dtype
        llm_dtype = next(self.llm_model.parameters()).dtype
        print(f"Model Precision: Audio Encoder={encoder_dtype}, Connector={connector_dtype}, LLM={llm_dtype}")

    def load_checkpoint(self, checkpoint_path):
        """从检查点加载权重，支持 DeepSpeed 目录和单文件"""
        print_param_status(self, "BEFORE LOADING")
        print(f"Loading checkpoint from {checkpoint_path}")
        
        if os.path.isdir(checkpoint_path):
            print("Detected DeepSpeed checkpoint directory. Consolidating stages...")
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            try:
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
            except Exception as e:
                print(f"❌ Error consolidating DeepSpeed checkpoint: {e}")
                return
        else:
            from addict import Dict
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
                self.streaming_adapter.streaming_encoder.load_state_dict(sub, strict=False)
                print(f"✅ Loaded audio_encoder from prefix '{p}' ({len(sub)} keys)")
                audio_found = True
                break
        if not audio_found and "audio_encoder" in state_dict:
            self.streaming_adapter.streaming_encoder.load_state_dict(state_dict["audio_encoder"])
            print(f"✅ Loaded audio_encoder (nested) ({len(state_dict['audio_encoder'])} keys)")
            audio_found = True
        if not audio_found:
            print("❌ Warning: audio_encoder weights NOT found!")

        # 2. Connector
        conn_prefixes = ["connector.", "model.connector."]
        conn_found = False
        for p in conn_prefixes:
            sub = extract_sub_dict(state_dict, p)
            if sub:
                self.streaming_adapter.connector.load_state_dict(sub, strict=False)
                print(f"✅ Loaded connector from prefix '{p}' ({len(sub)} keys)")
                conn_found = True
                break
        if not conn_found and "connector" in state_dict:
            self.streaming_adapter.connector.load_state_dict(state_dict["connector"])
            print(f"✅ Loaded connector (nested) ({len(state_dict['connector'])} keys)")
            conn_found = True
        if not conn_found:
            print("❌ Warning: connector weights NOT found!")

        # 3. LLM (including LoRA)
        llm_prefixes = ["llm_model.", "model.llm_model.", "llm."]
        llm_found = False
        for p in llm_prefixes:
            sub = extract_sub_dict(state_dict, p)
            if sub:
                # 使用 load_state_dict 直接加载到 PeftModel，这会同时处理 LoRA 和基础模型权重
                missing, unexpected = self.llm_model.load_state_dict(sub, strict=False)
                print(f"✅ Loaded LLM/LoRA from prefix '{p}' ({len(sub)} keys)")
                llm_found = True
                break
        
        if not llm_found and "llm_lora" in state_dict:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.llm_model, state_dict["llm_lora"])
            print(f"✅ Loaded llm_lora (nested PEFT format)")
            llm_found = True
            
        if not llm_found:
            print("❌ Warning: LLM/LoRA weights NOT found in checkpoint!")

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
                
                print(f"✅ 成功加载 {pk} (from {target_key})")
                print(f"      -> 变化量(L1 Diff): {diff:.6f}")
                print(f"      -> 当前均值(Mean): {new_val.mean().item():.8f}, 范数(Norm): {new_val.norm().item():.4f}")
                patch_loaded_count += 1
            else:
                print(f"⏩ Checkpoint 中未找到 {pk} 独立权重。")

        # 核心逻辑：判断是否激活 Patch 机制
        if patch_loaded_count > 0:
            self.use_patch_for_generation = True
            print("🔧 检测到外部 Patch 权重，已激活 Special Token Output Hook。")
            
            # ==============================================================
            # [终极补丁] 将真实的 Input Patch 写入底层 Embedding，防止自回归时查错表
            # ==============================================================
            with torch.no_grad():
                embed_weight = self.llm_model.get_input_embeddings().weight
                
                # patch.data 的 shape 是 [3, hidden_size]，索引对应 <A>, </A>, <W>
                embed_weight[self.token_A_id].copy_(self.special_token_input_patch.data[0])
                embed_weight[self.token_A_end_id].copy_(self.special_token_input_patch.data[1])
                embed_weight[self.token_W_id].copy_(self.special_token_input_patch.data[2])
                
                print("🔧 已将真实的 Input Patch 权重硬编码覆盖到底层 Embedding 矩阵中！")
        else:
            self.use_patch_for_generation = False
            print("🔧 未检测到 Patch 权重 (推断为纯原生全量微调)，关闭 Hook，使用 LLM 原生 Embedding。")
                             
        print_param_status(self, "AFTER LOADING")

    def get_init_states(self, batch_size: int = 1):
        """生成初始状态列表"""
        return self.streaming_adapter.get_init_states(batch_size, device=self.device)

    def forward_audio(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        处理音频。如果是流式，则按 chunk 处理。
        x: (batch, seq_len, feat_dim)
        states: 状态列表
        """
        T = x.size(1)
        chunk_size = self.config.model.zipformer.chunk_size
        stride = chunk_size * 2
        tail_length = chunk_size * 2 + 7 + 2 * 3
        
        # x_stream_padded = torch.nn.functional.pad(x, (0, 0, 0, tail_length), "constant", 0)
        if states is None:
            states = self.get_init_states(x.size(0))
            
        stream_outputs = []
        num_steps = math.ceil(T / stride)
        for i in range(num_steps):
            start = i * stride
            end = start + tail_length
            chunk_in = x[:, start:end, :]
            if chunk_in.size(1) < tail_length:
                 chunk_in = torch.nn.functional.pad(chunk_in, (0, 0, 0, tail_length - chunk_in.size(1)), "constant", 0)
            with torch.no_grad():
                chunk_out, states = self.streaming_adapter(chunk_in, states)
            # print(start, end, chunk_out.shape)
            stream_outputs.append(chunk_out)

        audio_embeds = torch.cat(stream_outputs, dim=1)
        # print(audio_embeds.shape)
        return audio_embeds, states

    def generate(self, audio_embeds: torch.Tensor, prompt_text: Optional[str] = None, gen_kwargs: Optional[dict] = None) -> str:
        """ 根据音频 Embeds 生成文本。 """
        # 1. 初始化默认参数
        if gen_kwargs is None:
            gen_kwargs = {
                "max_new_tokens": 200,
                "do_sample": False,
                "num_beams": 1,
                "pad_token_id": self.llm_model.config.pad_token_id,
                "eos_token_id": self.token_W_id,
                "top_p": None,
                "top_k": None,
                "temperature": None,
            }

        if prompt_text is None:
            try:
                lang_name = self.config.data.get("lang_name", "Spanish")
            except:
                lang_name = "Spanish"
            
            prompt_text = f"Transcribe the audio in {lang_name}: "
        
        # 2. Tokenize Prompt
        prompt_ids = self.llm_tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
        
        # 获取 Embedding 层
        if hasattr(self.llm_model, "get_input_embeddings"):
            embedder = self.llm_model.get_input_embeddings()
        elif hasattr(self.llm_model.model, "embed_tokens"):
            embedder = self.llm_model.model.embed_tokens
        else:
            embedder = self.llm_model.model.model.embed_tokens

        prompt_embeds = embedder(prompt_ids)
        
        # 3. 准备特殊 Token 并拼接 
        target_dtype = prompt_embeds.dtype 

        emb_A = self._get_special_embed(self.token_A_id, 0, target_dtype)
        emb_A_end = self._get_special_embed(self.token_A_end_id, 1, target_dtype)

        # 拼接 Input Embeddings (确保 audio_embeds 也是 target_dtype)
        inputs_embeds = torch.cat([
            prompt_embeds,
            emb_A,
            audio_embeds, 
            emb_A_end
        ], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)

        # 4. 执行生成
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs
        )

        decoded_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded_text

    def stream_prefill_text(self, prompt_text: str = None):
        """
        初始化 LLM Session，计算 Prompt + <A> 的 KV Cache 
        返回: past_key_values, past_seq_len
        """
        if prompt_text is None:
            prompt_text = "Transcribe the audio: "
            
        # 1. 准备 Embeddings
        ids = self.llm_tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
        embedder = self.llm_model.get_input_embeddings()
        prompt_embeds = embedder(ids)
        
        # 获取 <A> Embedding
        target_dtype = prompt_embeds.dtype
        if getattr(self, 'use_patch_for_generation', False):
            emb_A = self.special_token_input_patch[0].unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=target_dtype)
        else:
            emb_A = embedder(torch.tensor([self.token_A_id], device=self.device).unsqueeze(0))
        
        # 拼接: [Prompt, <A>]
        inputs_embeds = torch.cat([prompt_embeds, emb_A], dim=1)
        
        # 2. Forward pass 
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True
            )
        
        return outputs.past_key_values, inputs_embeds.shape[1]

    def stream_prefill_audio_chunk(self, audio_chunk_embeds, past_key_values, past_seq_len):
        """
        增量处理音频 Chunk，更新 KV Cache
        audio_chunk_embeds: (Batch, Chunk_Len, Dim)
        """
        return self.kv_calculator.step(audio_chunk_embeds, past_key_values, past_seq_len)

    def stream_generate(self, past_key_values, past_seq_len, gen_kwargs: Optional[dict] = None):
        if gen_kwargs is None: 
            gen_kwargs = {}
        max_new_tokens = gen_kwargs.get("max_new_tokens", 200)
        eos_token_id = self.token_W_id
        
        # 1. 准备 </A> Embedding
        target_dtype = past_key_values[0][0].dtype
        emb_A_end = self._get_special_embed(self.token_A_end_id, 1, target_dtype)

        # 2. 构造 Attention Mask (历史 + </A>)
        curr_seq_len = past_seq_len + 1
        attention_mask = torch.ones((1, curr_seq_len), device=self.device, dtype=torch.long)
        # --- 手动执行一次 Forward (预测第一个字) ---
        with torch.no_grad():
            outputs = self.llm_model(
                inputs_embeds=emb_A_end,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True
            )
            
        # 更新状态
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1) # [1, 1]
        
        # 收集结果
        generated_ids = []
        if next_token_id.item() != eos_token_id:
            generated_ids.append(next_token_id.item())
            
        for _ in range(max_new_tokens - 1):
            # 如果生成了 EOS，立即停止
            if next_token_id.item() == eos_token_id:
                break
            
            # 更新 Mask (每次增加 1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=1)
            
            with torch.no_grad():
                outputs = self.llm_model(
                    input_ids=next_token_id, 
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            
            # 更新 KV 和 Token
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # 记录结果
            if next_token_id.item() != eos_token_id:
                generated_ids.append(next_token_id.item())
            else:
                break # 遇到 EOS

        # 4. 解码文本
        decoded_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded_text

    def _get_special_embed(self, token_id, patch_idx, target_dtype):
        """获取特殊 Token 的 Embedding，兼容 LoRA Patch"""
        if getattr(self, 'use_patch_for_generation', False):
            # patch_idx: 0 for <A>, 1 for </A>, 2 for <W>
            return self.special_token_input_patch[patch_idx].unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=target_dtype)
        else:
            embedder = self.llm_model.get_input_embeddings()
            return embedder(torch.tensor([token_id], device=self.device).unsqueeze(0))

    def _get_text_embeds(self, text):
        """获取文本的 Embeddings"""
        ids = self.llm_tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
        embedder = self.llm_model.get_input_embeddings()
        return embedder(ids)


class IncrementalKvCalculator:
    """
    增量 KV Cache 计算器。
    直接调用 llm_model（如 Qwen3ForCausalLM），与 model_stream.py 保持一致，
    不再手动剥离 backbone 或 hook decoder layer，避免与 transformers 版本耦合。
    """
    def __init__(self, llm_model: nn.Module):
        self.device = llm_model.device
        self.llm_model = llm_model

    def step(self, inputs_embeds: torch.Tensor, past_key_values, past_seq_len: int):
        batch_size, chunk_len, _ = inputs_embeds.shape
        total_len = past_seq_len + chunk_len
        attention_mask = torch.ones((batch_size, total_len), device=self.device, dtype=torch.long)

        with torch.no_grad():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
        return outputs.past_key_values, total_len

def print_param_status(model, stage_name):
    """专门针对全量微调：检查词表中特定 ID 对应的权重行"""
    print(f"--- Full SFT Special Tokens Check: {stage_name} ---")
    
    target_tokens = {
        "<A>": model.token_A_id,
        "</A>": model.token_A_end_id,
        "<W>": model.token_W_id
    }

    try:
        embed_table = model.llm_model.model.embed_tokens.weight.data
        print(f"  [Input Embedding] Table Shape: {embed_table.shape}")
        for name, tid in target_tokens.items():
            if tid < embed_table.shape[0]:
                row = embed_table[tid]
                print(f"    Token {name} (ID:{tid}): Mean={row.mean().item():.8f}, L2={row.norm().item():.4f}")
            else:
                print(f"    ❌ Token {name} (ID:{tid}) is OUT OF BOUNDS!")
    except Exception as e:
        pass

    try:
        head_table = None
        if hasattr(model.llm_model, 'lm_head'):
            head_table = model.llm_model.lm_head.weight.data
        elif hasattr(model.llm_model, 'get_output_embeddings'):
            head_table = model.llm_model.get_output_embeddings().weight.data
            
        if head_table is not None:
            print(f"  [Output Head] Table Shape: {head_table.shape}")
            for name, tid in target_tokens.items():
                if tid < head_table.shape[0]:
                    row = head_table[tid]
                    print(f"    Token {name} (ID:{tid}): Mean={row.mean().item():.8f}, L2={row.norm().item():.4f}")
    except Exception as e:
        pass