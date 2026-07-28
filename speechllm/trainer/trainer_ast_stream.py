import torch
from torch import nn
import lightning.pytorch as pl
import logging
import torch.utils.data as data_utils

from ..dataset import get_sampler, DatasetForStreamASTCollate
from ..dataset.shar_pool import SyncExhaustDataLoader
from .basic_trainer import SpeechLLMLightning

class SpeechLLMLightningStreamAST(SpeechLLMLightning):
    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__(model_config, train_ds, val_ds)

        if self.val_ds is not None and hasattr(self.model_config, "train"):
            val_decode_chunk_num = self.model_config.train.get("val_decode_chunk", 1)
            if hasattr(self.val_ds, "set_decode_chunk_num"):
                self.val_ds.set_decode_chunk_num(val_decode_chunk_num)
            
            ast_val_decode_chunk_num = self.model_config.train.get("ast_val_decode_chunk", val_decode_chunk_num)
            if hasattr(self.val_ds, "set_ast_decode_chunk_num"):
                self.val_ds.set_ast_decode_chunk_num(ast_val_decode_chunk_num)

    def train_dataloader(self):
        # 此时 self.trainer 已经自动包含了正确的分布式信息
        num_workers = self.model_config.train.get("dataloader_num_workers", 4)
        collate_fn = DatasetForStreamASTCollate(self.llm_tokenizer)

        if self.sampler_type == 'shar_pool_sampler':
            self._prepare_shar_pool_train_cuts()
            train_sampler = get_sampler(
                self.sampler_type,
                self.train_ds,
                config=self.model_config.data,
                shuffle=True,
            )
            persistent_workers = False
        elif self.sampler_type == 'stateless_sampler':
            train_sampler = get_sampler(
                self.sampler_type,
                config=self.stateless_sampler_config,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )
            persistent_workers = True
        else:
            train_sampler = get_sampler(
                self.sampler_type,
                self.train_ds,
                config=self.model_config.data,
                shuffle=True,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )
            persistent_workers = True

        if self.saved_sampler_state is not None:
            if self.sampler_type == 'shar_pool_sampler':
                logging.info("Skipping sampler state restoration for shar_pool_sampler.")
            else:
                logging.info("Restoring sampler state in train_dataloader...")
                try:
                    train_sampler.load_state_dict(self.saved_sampler_state)
                    logging.info("Sampler state restored successfully.")
                except Exception as e:
                    logging.error(f"Failed to restore sampler state: {e}")
            self.saved_sampler_state = None

        prefetch_factor = self.model_config.train.get("prefetch_factor", 2)
        loader = data_utils.DataLoader(
            self.train_ds,
            sampler=train_sampler,
            batch_size=None,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        if self.sampler_type == 'shar_pool_sampler':
            loader = SyncExhaustDataLoader(loader)
        return loader

    def val_dataloader(self):
        collate_fn = DatasetForStreamASTCollate(self.llm_tokenizer)
        val_sampler = get_sampler(
            "dynamic_bucket_sampler",
            self.val_ds,
            config=self.model_config.data,
            shuffle=False,
            rank=self.trainer.global_rank,   
            world_size=self.trainer.world_size 
        )
        num_workers = self.model_config.train.get("dataloader_num_workers", 4)
        return data_utils.DataLoader(
            self.val_ds,
            sampler=val_sampler,
            batch_size=None,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

    def training_step(self, batch, batch_idx):
        batch_features, audio_lengths, targets, prompt_ids, prompt_lens = batch
        # 更新样本计数
        batch_size = batch_features.shape[0]
        self.total_processed_samples += batch_size
        self.epoch_processed_samples += batch_size

        # 计算 encoder 和 connector 的输出 embed
        speech_embeds, speech_embeds_length = self.get_audio_embeds(batch_features, audio_lengths)

        # 拼接为 LLM 的输入
        input_embeds, attention_mask, labels = self.merge_input_ids_with_speech_features_stream(
            speech_embeds, speech_embeds_length, targets, prompt_ids, prompt_lens
        )

        if input_embeds is None:
            return None

        outputs = self.llm_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("total_samples", float(self.total_processed_samples), on_step=True, on_epoch=False, prog_bar=False)
        if self.sampler_type not in ("stateless_sampler", "shar_pool_sampler"):
            self.log("epoch_samples", float(self.epoch_processed_samples), on_step=True, on_epoch=False, prog_bar=False)
            
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_features, audio_lengths, targets, prompt_ids, prompt_lens = batch
        batch_size = batch_features.shape[0]
        device = self.device

        # 按 prompt 内容分类：含 "Translate" 的是 AST，否则是 ASR
        asr_indices = []
        ast_indices = []
        for b in range(batch_size):
            p_len = prompt_lens[b]
            p_ids = prompt_ids[b][:p_len]
            prompt_text = self.llm_tokenizer.decode(p_ids, skip_special_tokens=False)
            if "Translate" in prompt_text:
                ast_indices.append(b)
            else:
                asr_indices.append(b)

        # 整个 batch 过一次 encoder（共享）
        speech_embeds, speech_embeds_length = self.get_audio_embeds(batch_features, audio_lengths)

        # --- 整个 batch 一起算 val_loss（和原来一样） ---
        inputs_embeds, attention_mask, labels = self.merge_input_ids_with_speech_features_stream(
            speech_embeds, speech_embeds_length, targets, prompt_ids, prompt_lens
        )
        if inputs_embeds is None:
            return None

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # --- ASR 子 batch ---
        if asr_indices:
            idx = asr_indices
            sub_embeds, sub_attn, sub_labels = self.merge_input_ids_with_speech_features_stream(
                speech_embeds[idx], speech_embeds_length[idx],
                [targets[i] for i in idx],
                prompt_ids[idx], prompt_lens[idx]
            )
            if sub_embeds is not None:
                asr_out = self.llm_model(inputs_embeds=sub_embeds, attention_mask=sub_attn, labels=sub_labels)
                self.log("val_asr_loss", asr_out.loss, prog_bar=True, sync_dist=True, batch_size=len(idx))

        # --- AST 子 batch ---
        if ast_indices:
            idx = ast_indices
            sub_embeds, sub_attn, sub_labels = self.merge_input_ids_with_speech_features_stream(
                speech_embeds[idx], speech_embeds_length[idx],
                [targets[i] for i in idx],
                prompt_ids[idx], prompt_lens[idx]
            )
            if sub_embeds is not None:
                ast_out = self.llm_model(inputs_embeds=sub_embeds, attention_mask=sub_attn, labels=sub_labels)
                self.log("val_ast_loss", ast_out.loss, prog_bar=True, sync_dist=True, batch_size=len(idx))

        return val_loss

    def merge_input_ids_with_speech_features_stream(self, speech_embeds, speech_embeds_length, targets, prompt_ids, prompt_lens):
        """
        构建流式训练的 Input Embeddings 和 Labels
        序列结构: Prompt + <A>Audio1</A>Text1 + <A>Audio2</A>Text2 (输入中彻底拔除 <W>)
        预测目标: ...    +      ...        T1 + <W> +   ...    T2 + <W>

        优化：
        1. 将所有 token ids 收集起来，一次性调用 embedder。
        2. 预计算每个样本的总序列长度，一次性分配 [B, max_seq_len, D] 的 buffer，
           用 index copy 填充，避免逐 segment torch.cat 产生 CUDA 碎片。
        """
        batch_size = speech_embeds.shape[0]
        dtype = speech_embeds.dtype
        device = self.device

        embedder = getattr(self.llm_model.model, "embed_tokens", None)
        if embedder is None:
            embedder = self.llm_model.model.model.embed_tokens

        if getattr(self, "finetune_special_tokens", False) or getattr(self, "use_lora", False):
            emb_A = self.special_token_input_patch[0].unsqueeze(0).to(dtype=dtype)
            emb_A_end = self.special_token_input_patch[1].unsqueeze(0).to(dtype=dtype)
        else:
            special_ids = torch.tensor([self.token_A_id, self.token_A_end_id], device=device)
            s_embs = embedder(special_ids).to(dtype=dtype)
            emb_A, emb_A_end = s_embs[0:1], s_embs[1:2]

        embed_dim = embedder.weight.size(1)

        # ====== 第一步：收集所有 token ids + 预计算序列长度（单次遍历）======
        all_id_chunks = []   # 每个元素是一个 1D tensor
        all_id_lens = []     # 对应长度

        prompt_chunk_indices = []  # 每个样本的 prompt 在 all_id_chunks 中的索引
        seg_chunk_indices = []     # 按 (b, seg_idx) 顺序展平，-1 表示空文本
        seq_lens_list = []         # 每个样本的总 token 数
        seg_audio_lens = []        # [batch_size][num_segs] 缓存 (s_idx, e_idx, audio_len)

        for b in range(batch_size):
            # --- 收集 prompt ids ---
            p_len = prompt_lens[b].item() if isinstance(prompt_lens[b], torch.Tensor) else prompt_lens[b]
            p_ids = prompt_ids[b][:p_len]
            prompt_chunk_indices.append(len(all_id_chunks))
            all_id_chunks.append(p_ids)
            all_id_lens.append(p_len)

            total = p_len  # 序列长度累加器
            max_audio_len = speech_embeds_length[b].item()
            b_audio_lens = []

            # --- 收集 segment ids + 计算 audio 长度 ---
            for seg in targets[b]:
                s_idx = max(0, min(seg['start_idx'], max_audio_len))
                e_idx = max(s_idx, min(seg['end_idx'], max_audio_len))
                audio_len = e_idx - s_idx
                b_audio_lens.append((s_idx, e_idx, audio_len))

                # <A> + audio_slice + </A>
                total += 1 + audio_len + 1

                # text tokens
                txt_len = seg['input_len']
                if txt_len > 0:
                    seg_chunk_indices.append(len(all_id_chunks))
                    all_id_chunks.append(seg['input_ids'][:txt_len])
                    all_id_lens.append(txt_len)
                    total += txt_len
                else:
                    seg_chunk_indices.append(-1)

            # 收尾的 dummy token
            if len(targets[b]) > 0:
                total += 1

            seg_audio_lens.append(b_audio_lens)
            seq_lens_list.append(total)

        # 一次性 embed + 同时切分出 labels 用的 id tensor
        if all_id_chunks:
            all_ids_cat = torch.cat(all_id_chunks, dim=0).to(device, dtype=torch.long)
            all_embeds_flat = embedder(all_ids_cat).to(dtype=dtype)
            all_embeds_split = list(torch.split(all_embeds_flat, all_id_lens))
            # 切分出与 embeds 对齐的 id tensor，第三步直接用，避免重复 .to(device)
            all_ids_split = list(torch.split(all_ids_cat, all_id_lens))
        else:
            all_embeds_split = []
            all_ids_split = []

        if not seq_lens_list or max(seq_lens_list) == 0:
            return None, None, None

        max_seq_len = max(seq_lens_list)

        # ====== 第二步：一次性分配 buffer，用 index copy 填充 ======
        inputs_embeds = torch.zeros(batch_size, max_seq_len, embed_dim, dtype=dtype, device=device)
        labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long, device=device)

        seg_cursor = 0
        for b in range(batch_size):
            pos = 0  # 当前写入位置

            # --- Prompt ---
            p_idx = prompt_chunk_indices[b]
            p_len = all_id_lens[p_idx]
            inputs_embeds[b, pos:pos + p_len] = all_embeds_split[p_idx]
            # labels 已经是 -100，不需要写
            pos += p_len

            full_audio = speech_embeds[b]

            for seg_idx in range(len(targets[b])):
                s_idx, e_idx, audio_len = seg_audio_lens[b][seg_idx]
                chunk_idx = seg_chunk_indices[seg_cursor]

                # --- <A> ---
                inputs_embeds[b, pos:pos + 1] = emb_A
                # 【借位法】非第一个 segment 时，<A> 位置的 label 设为 <W>
                if seg_idx > 0:
                    labels[b, pos] = self.token_W_id
                # else: 保持 -100
                pos += 1

                # --- Audio slice ---
                if audio_len > 0:
                    inputs_embeds[b, pos:pos + audio_len] = full_audio[s_idx:e_idx]
                    # labels 保持 -100
                pos += audio_len

                # --- </A> ---
                inputs_embeds[b, pos:pos + 1] = emb_A_end
                # labels 保持 -100
                pos += 1

                # --- Text tokens（直接复用 all_ids_split，无需再 .to(device)）---
                if chunk_idx >= 0:
                    txt_len = all_id_lens[chunk_idx]
                    if txt_len > 0:
                        inputs_embeds[b, pos:pos + txt_len] = all_embeds_split[chunk_idx]
                        labels[b, pos:pos + txt_len] = all_ids_split[chunk_idx]
                        pos += txt_len

                seg_cursor += 1

            # --- 收尾 dummy token ---
            if len(targets[b]) > 0:
                # inputs_embeds[b, pos] 已经是 0（dummy embedding）
                labels[b, pos] = self.token_W_id
                pos += 1

        # ====== Attention mask ======
        seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.long)
        attention_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)

        return inputs_embeds, attention_mask, labels
