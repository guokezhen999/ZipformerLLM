import torch
from typing import Optional, Tuple, Any

from .model_asr import StreamingSpeechLLM

class SpeechLLMASRStream(StreamingSpeechLLM):
    """
    流式 ASR 推理模型（真正增量接口 - 单条数据版）
    开放外部循环的增量音频特征块提取接口，用于模拟真实麦克风录入、递增特征编码。
    每次只处理一条音频数据（batch_size = 1），在指定断点自动对接 LLM 文本自回归翻译，
    并流转 KV Cache 维持连续对话。
    """
    def generate(
        self,
        audio_chunk_embed: torch.Tensor,
        past_key_values: Optional[Any] = None,
        past_seq_len: int = 0,
        prompt: Optional[str] = None,
        is_new_segment: bool = True,
        is_segment_end: bool = False,
        generation_config: Optional[dict] = None,
        first_token_eos_penalty: float = 1.0
    ) -> Tuple[str, Any, int]:
        """
        处理单条音频特征块（由外部流式 encoder 输出的 audio_embeds chunk）。
        
        Args:
            audio_chunk_embed: (1, ChunkLen, D) 当前步进送入的音频特征 chunk。
            past_key_values: 当前样本的 KV Cache。初次录音或释放后传入 None。
            past_seq_len: 当前 KV Cache 的长度，首次传 0。
            prompt: 在初次开启新录音（没有 cache 时）的初始前置文本。
            is_new_segment: 代表当前 chunk 是否是一个新翻译 Segment 的开端 (需要附带 <A>)。
                            首次开启新录音时应该为 True。
            is_segment_end: 代表当前 chunk 是否是这个翻译 Segment 的结尾。
                            若是，打入 </A> 并触发自回归，以 <W> 收尾。
            first_token_eos_penalty: 触发自回归后解码第一个 token 时，将 eos(<W>) 的 logit 除以该值以降低空输出概率。
                            1.0 (默认) 表示不做任何处理，大于 1.0 时有效。
        
        Returns:
            out_text: (str) 当 is_segment_end=True 时生成的文本，否则返回空字符串 ""。
            out_cache: (Any) 更新后的 KV Cache 状态。
            out_len: (int) 更新后的 KV Cache 序列长度状态。
        """
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 200,
                "eos_token_id": self.token_W_id,
            }
        repetition_penalty = generation_config.get("repetition_penalty", 1.0)
        repetition_penalty_window = generation_config.get("repetition_penalty_window", 0)

        # 准备 Embedder
        if hasattr(self.llm_model, "get_input_embeddings"):
            embedder = self.llm_model.get_input_embeddings()
        elif hasattr(self.llm_model.model, "embed_tokens"):
            embedder = self.llm_model.model.embed_tokens
        else:
            embedder = self.llm_model.model.model.embed_tokens

        device = self.device
        dtype = audio_chunk_embed.dtype

        # 准备特殊 Token
        if self.use_lora:
            emb_A = self.special_token_input_patch[0].unsqueeze(0).to(device=device, dtype=dtype)
            emb_A_end = self.special_token_input_patch[1].unsqueeze(0).to(device=device, dtype=dtype)
            emb_W = self.special_token_input_patch[2].unsqueeze(0).to(device=device, dtype=dtype)
        else:
            special_ids = torch.tensor([self.token_A_id, self.token_A_end_id, self.token_W_id], device=device)
            s_embs = embedder(special_ids).to(dtype=dtype)
            emb_A, emb_A_end, emb_W = s_embs[0].unsqueeze(0), s_embs[1].unsqueeze(0), s_embs[2].unsqueeze(0)

        # 1. 组装当前 Chunk 需要注入的前缀 Embedding
        prefix_embeds_list = []
        
        if is_new_segment:
            if past_key_values is None and prompt is not None:
                # 绝对开头：首先加入 Prompt
                prompt_data = self.llm_tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
                p_embed = embedder(prompt_data.input_ids) # (1, T_prompt, D)
                prefix_embeds_list.append(p_embed.to(dtype))
            
            # 每个新翻译 Segment 必定以 <A> 开始
            prefix_embeds_list.append(emb_A.unsqueeze(0)) # (1, 1, D)
            
        prefix_embeds_list.append(audio_chunk_embed) # (1, Chunk_Len, D)
        inputs_embeds = torch.cat(prefix_embeds_list, dim=1) # (1, Seq_len, D)
        
        # 2. 更新音频特征到 KV Cache
        if past_key_values is None:
            with torch.no_grad():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    return_dict=True
                )
            past_key_values = outputs.past_key_values
            past_seq_len = inputs_embeds.size(1)
        else:
            past_key_values, past_seq_len = self.kv_calculator.step(inputs_embeds, past_key_values, past_seq_len)
        
        text = ""
        
        # 3. 如果到达当前段尾，触发自回归生成文本
        if is_segment_end:
            emb_A_end_input = emb_A_end.unsqueeze(0) # (1, 1, D)
            attention_mask = torch.ones((1, past_seq_len + 1), device=device, dtype=torch.long)
            
            # 喂入 </A>
            with torch.no_grad():
                outputs = self.llm_model(
                    inputs_embeds=emb_A_end_input,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                )
            past_key_values = outputs.past_key_values
            past_seq_len += 1
            
            # 循环生成 Text (自回归解码)
            next_token_logits = outputs.logits[:, -1, :]
            if first_token_eos_penalty > 1.0:
                next_token_logits[:, self.token_W_id] = next_token_logits[:, self.token_W_id] / first_token_eos_penalty
            next_token_id = torch.argmax(next_token_logits, dim=-1) # (1,)
            token_id_int = next_token_id.item()
            
            generated_ids = []
            max_tokens = generation_config.get("max_new_tokens", 200)
            eos_token_id = self.llm_tokenizer.eos_token_id
            
            # 检查第一个 token 是否是 <W> 或 LLM 自带的 EOS
            if token_id_int != self.token_W_id and token_id_int != eos_token_id:
                generated_ids.append(token_id_int)
                
                for _ in range(max_tokens - 1):
                    curr_embeds = embedder(next_token_id.unsqueeze(0)).to(dtype)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
                    
                    with torch.no_grad():
                        outputs = self.llm_model(
                            inputs_embeds=curr_embeds,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            use_cache=True,
                            return_dict=True
                        )
                    past_key_values = outputs.past_key_values
                    past_seq_len += 1
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    # Apply repetition penalty (windowed)
                    if repetition_penalty != 1.0 and len(generated_ids) > 0:
                        window = generated_ids if repetition_penalty_window <= 0 else generated_ids[-repetition_penalty_window:]
                        for prev_id in set(window):
                            if next_token_logits[0, prev_id] > 0:
                                next_token_logits[0, prev_id] /= repetition_penalty
                            else:
                                next_token_logits[0, prev_id] *= repetition_penalty
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    token_id_int = next_token_id.item()
                    
                    # <W> 或 LLM 原生 EOS 都视为停止信号
                    if token_id_int == self.token_W_id or token_id_int == eos_token_id:
                        break
                    generated_ids.append(token_id_int)
            
            text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        return text, past_key_values, past_seq_len
