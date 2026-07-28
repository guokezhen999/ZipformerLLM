import torch
from typing import List, Optional, Dict
from .basic_model import SpeechLLM

class SpeechLLMStream(SpeechLLM):
    """
    流式 ASR 推理模型
    可处理多个连续的音频片段，每次推理基于历史上下文生成文本。
    序列结构: Prompt + <A>Audio1</A>Text1 + <A>Audio2</A>Text2 ...
    """
    
    def _truncate_kv_cache(self, past_key_values, remove_len=1):
        """
        安全地截断 KV Cache 的最后 N 个 token，抹除模型的这段记忆。
        兼容最新的 DynamicCache 对象和传统的 Tuple 结构。
        """
        if past_key_values is None:
            return None
        
        # 1. 适配 transformers >= 4.36 的 Cache 对象 (如 DynamicCache)
        if hasattr(past_key_values, "crop") and hasattr(past_key_values, "get_seq_length"):
            seq_len = past_key_values.get_seq_length()
            if seq_len > remove_len:
                past_key_values.crop(seq_len - remove_len)
            return past_key_values
            
        # 2. 适配传统 Tuple 结构: tuple of (K_tensor, V_tensor)
        if isinstance(past_key_values, tuple):
            new_past = []
            for layer_past in past_key_values:
                K, V = layer_past[0], layer_past[1]
                # 标准形状通常为 (batch_size, num_heads, seq_len, head_dim)
                # 假设 seq_len 位于维度 2
                if K.size(2) > remove_len:  
                    new_K = K[:, :, :-remove_len, :]
                    new_V = V[:, :, :-remove_len, :]
                else:
                    new_K, new_V = K, V
                new_past.append((new_K, new_V))
            return tuple(new_past)
            
        return past_key_values

    def _get_kv_seq_length(self, past_key_values) -> int:
        """获取当前 KV Cache 的序列长度。"""
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        if isinstance(past_key_values, tuple):
            return past_key_values[0][0].size(2)
        return 0

    def generate(
        self,
        audio_embeds: torch.Tensor,
        audio_lengths: torch.Tensor,
        prompts: List[str],
        segments: List[List[Dict]],
        generation_config: Optional[dict] = None,
        punct_kv_mode: int = 1,
    ) -> List[List[str]]:
        """
        流式推理函数
        Args:
            audio_embeds: (B, T, D) 音频 Embedding 序列
            audio_lengths: (B,) 音频有效长度
            prompts: List[str], 每个样本的初始 Prompt
            segments: List[List[Dict]], 每个样本包含一个切片列表。每个 Dict 为片段信息 {'start_idx': int, 'end_idx': int}
            generation_config: 生成配置参数 (如 beaming, sampling 等)
            punct_kv_mode: 遇到终结标点时的 KV Cache 处理策略
                0 - 不做任何处理
                1 - 仅移除末尾标点的 KV Cache (默认)
                2 - 清除除 Prompt 之外的所有 KV Cache
        Returns:
            List[List[str]]: 每个样本每个片段生成的文本列表
        """
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 200,
                "do_sample": False,
                "num_beams": 1,
                "pad_token_id": self.llm_model.config.pad_token_id,
                "eos_token_id": self.token_W_id,
                "top_p": None,
                "top_k": None,
                "temperature": None
            }

        # 1. 准备 Embedder
        if hasattr(self.llm_model, "get_input_embeddings"):
            embedder = self.llm_model.get_input_embeddings()
        elif hasattr(self.llm_model.model, "embed_tokens"):
            embedder = self.llm_model.model.embed_tokens
        else:
            embedder = self.llm_model.model.model.embed_tokens

        batch_size = audio_embeds.size(0)
        dtype = audio_embeds.dtype
        device = self.device

        # 2. 准备特殊 Token Embeddings (保持为 2D 形状 [1, D])
        if self.use_lora:
            emb_A = self.special_token_input_patch[0].unsqueeze(0).to(device=device, dtype=dtype)
            emb_A_end = self.special_token_input_patch[1].unsqueeze(0).to(device=device, dtype=dtype)
            emb_W = self.special_token_input_patch[2].unsqueeze(0).to(device=device, dtype=dtype)
        else:
            special_ids = torch.tensor([self.token_A_id, self.token_A_end_id, self.token_W_id], device=device)
            s_embs = embedder(special_ids).to(dtype=dtype)
            emb_A, emb_A_end, emb_W = s_embs[0].unsqueeze(0), s_embs[1].unsqueeze(0), s_embs[2].unsqueeze(0)

        # 3. 编码 Prompts
        prompt_data = self.llm_tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        prompt_ids_batch = prompt_data.input_ids
        prompt_mask_batch = prompt_data.attention_mask

        batch_generated_texts = []

        # 逐个样本处理
        for b in range(batch_size):
            sample_generated_texts = []
            past_key_values = None
            
            # A. 初始化历史: Prompt Embedding
            valid_p_len = prompt_mask_batch[b].sum()
            curr_p_ids = prompt_ids_batch[b, :valid_p_len]
            curr_p_embed = embedder(curr_p_ids.unsqueeze(0)).squeeze(0) # (T_prompt, D)
            prompt_kv_len = None  # 首次推理后记录 prompt 的 KV 长度

            # B. 获取全音频 Embedding
            full_audio = audio_embeds[b]
            max_audio_len = audio_lengths[b].item()

            # 遍历该样本的所有切片
            sample_segments = segments[b]
            for seg_idx, seg in enumerate(sample_segments):
                is_last_chunk = (seg_idx == len(sample_segments) - 1)
                s_idx, e_idx = seg['start_idx'], seg['end_idx']
                s_idx = max(0, min(s_idx, max_audio_len))
                e_idx = max(s_idx, min(e_idx, max_audio_len))
                
                # 音频切片 Embedding
                audio_slice = full_audio[s_idx:e_idx] # (T_audio, D)

                # 构建本次生成的输入
                if past_key_values is None:
                    # 首个片段：Prompt + <A> + Audio + </A>
                    prefix_embeds = torch.cat([curr_p_embed, emb_A, audio_slice, emb_A_end], dim=0).unsqueeze(0)
                else:
                    # 后续片段： <A> + Audio + </A>
                    prefix_embeds = torch.cat([emb_A, audio_slice, emb_A_end], dim=0).unsqueeze(0)

                # --- 核心：手写自回归解码循环 ---
                curr_embeds = prefix_embeds
                curr_past = past_key_values
                generated_ids = []
                max_tokens = generation_config.get("max_new_tokens", 200)

                for step in range(max_tokens):
                    with torch.no_grad():
                        outputs = self.llm_model(
                            inputs_embeds=curr_embeds,
                            past_key_values=curr_past,
                            use_cache=True
                        )
                    
                    # 1. 更新历史 Cache
                    curr_past = outputs.past_key_values
                    
                    # 2. 获取预测的下一个 token 的 Logits (Shape: [Batch, Vocab_Size])
                    next_token_logits = outputs.logits[:, -1, :]

                    # 3. 应用重复惩罚后贪心解码
                    repetition_penalty = generation_config.get("repetition_penalty", 1.0)
                    if repetition_penalty != 1.0 and len(generated_ids) > 0:
                        prev_ids = torch.tensor(generated_ids, device=device)
                        score = next_token_logits[0].index_select(0, prev_ids)
                        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                        next_token_logits[0].scatter_(0, prev_ids, score)
                    max_logit_val, max_token_id_tensor = torch.max(next_token_logits, dim=-1)
                    token_id_int = max_token_id_tensor.item()

                    # ==============================================================
                    # 4. 遇到终止符 <W> 则拦截并清理记忆
                    # ==============================================================
                    if token_id_int == self.token_W_id:
                        if punct_kv_mode != 0 and len(generated_ids) > 0:
                            prev_char = self.llm_tokenizer.decode([generated_ids[-1]]).strip()
                            is_end_punct = prev_char and prev_char[-1] in ["。", "？", "！", ".", "?", "!"]

                            if is_end_punct:
                                if punct_kv_mode == 1:
                                    # 仅移除末尾标点的 KV Cache
                                    curr_past = self._truncate_kv_cache(curr_past, remove_len=1)
                                elif punct_kv_mode == 2 and prompt_kv_len is not None:
                                    # 清除除 Prompt 之外的所有 KV Cache
                                    curr_kv_len = self._get_kv_seq_length(curr_past)
                                    if curr_kv_len > prompt_kv_len:
                                        curr_past = self._truncate_kv_cache(curr_past, remove_len=curr_kv_len - prompt_kv_len)
                        break
                        
                    # 只有不是 <W> 时，才真正追加到结果中
                    generated_ids.append(token_id_int)                    
                    
                    # ==============================================================
                    # 5. 将刚刚生成的 Token ID 转化为 Embedding，供下一步输入
                    # ==============================================================
                    next_token_tensor = torch.tensor([[token_id_int]], device=device)
                    curr_embeds = embedder(next_token_tensor).to(dtype=dtype)

                # 记录 prompt 的 KV 长度（仅在首个片段推理完成后记录一次）
                if prompt_kv_len is None:
                    prompt_kv_len = self._get_kv_seq_length(curr_past) - len(generated_ids)
                    if prompt_kv_len < 0:
                        prompt_kv_len = 0
                        
                # 循环结束，赋值回外层变量，供下一个音频切片使用
                past_key_values = curr_past
                # ---------------------------------------------------

                # 移除末尾的 EOS token (<W>) 以获得干净文本 (防守型冗余检查)
                if len(generated_ids) > 0 and generated_ids[-1] == self.token_W_id:
                     text_ids = generated_ids[:-1]
                else:
                     text_ids = generated_ids

                text = self.llm_tokenizer.decode(text_ids, skip_special_tokens=True)
                sample_generated_texts.append(text)

            batch_generated_texts.append(sample_generated_texts)

        return batch_generated_texts

    