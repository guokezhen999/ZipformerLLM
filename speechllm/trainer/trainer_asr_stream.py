import torch
from torch import nn
import lightning.pytorch as pl
import logging
import torch.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence

from ..dataset import get_sampler, DatasetForStreamASRCollate
from ..dataset.shar_pool import SyncExhaustDataLoader
from .basic_trainer import SpeechLLMLightning

class SpeechLLMLightningStreamASR(SpeechLLMLightning):
    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__(model_config, train_ds, val_ds)

        if self.val_ds is not None and hasattr(self.model_config, "train"):
            val_decode_chunk_num = self.model_config.train.get("val_decode_chunk", 1)
            if hasattr(self.val_ds, "set_decode_chunk_num"):
                self.val_ds.set_decode_chunk_num(val_decode_chunk_num)

    def train_dataloader(self):
        # 此时 self.trainer 已经自动包含了正确的分布式信息
        num_workers = self.model_config.train.get("dataloader_num_workers", 4)
        collate_fn = DatasetForStreamASRCollate(self.llm_tokenizer)

        if self.sampler_type == 'shar_pool_sampler':
            self._prepare_shar_pool_train_cuts()
            train_sampler = get_sampler(
                self.sampler_type,
                self.train_ds,
                config=self.model_config.data,
                shuffle=True,
            )
            # epoch 会重建 dataloader；避免 persistent_workers 持有旧 CutSet
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
        # shar_pool：各 rank batch 数可能不同，必须同步停 epoch，否则 DDP 会 NCCL timeout
        if self.sampler_type == 'shar_pool_sampler':
            loader = SyncExhaustDataLoader(loader)
        return loader

    def val_dataloader(self):
        collate_fn = DatasetForStreamASRCollate(self.llm_tokenizer)
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
        # 计算 encoder 和 connector 的输出 embed
        speech_embeds, speech_embeds_length = self.get_audio_embeds(batch_features, audio_lengths)

        # 拼接为 LLM 的输入
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
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, batch_size=batch_features.shape[0])
        return val_loss

    def merge_input_ids_with_speech_features_stream(self, speech_embeds, speech_embeds_length, targets, prompt_ids, prompt_lens):
        """
        构建流式训练的 Input Embeddings 和 Labels
        序列结构: Prompt + <A>Audio1</A>Text1 + <A>Audio2</A>Text2 (输入中彻底拔除 <W>)
        预测目标: ...    +      ...        T1 + <W> +   ...    T2 + <W>

        优化：将所有 prompt 和 segment 的 token ids 收集起来，一次性调用 embedder，
        避免循环内多次 CUDA kernel launch。
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

        # ====== 第一步：收集所有需要 embed 的 token ids ======
        all_id_chunks = []
        all_id_lens = []

        prompt_chunk_indices = []
        for b in range(batch_size):
            p_len = prompt_lens[b].item() if isinstance(prompt_lens[b], torch.Tensor) else prompt_lens[b]
            p_ids = prompt_ids[b][:p_len]
            prompt_chunk_indices.append(len(all_id_chunks))
            all_id_chunks.append(p_ids)
            all_id_lens.append(p_len)

        seg_chunk_indices = []
        for b in range(batch_size):
            for seg in targets[b]:
                txt_len = seg['input_len']
                if txt_len > 0:
                    seg_chunk_indices.append(len(all_id_chunks))
                    all_id_chunks.append(seg['input_ids'][:txt_len])
                    all_id_lens.append(txt_len)
                else:
                    seg_chunk_indices.append(-1)

        if all_id_chunks:
            all_ids_cat = torch.cat(all_id_chunks, dim=0).to(device, dtype=torch.long)
            all_embeds_flat = embedder(all_ids_cat).to(dtype=dtype)
            all_embeds_split = list(torch.split(all_embeds_flat, all_id_lens))
        else:
            all_embeds_split = []

        # ====== 第二步：组装序列（不再调用 embedder）======
        batch_seq_embeds = []
        batch_seq_labels = []
        seg_cursor = 0

        for b in range(batch_size):
            curr_p_embed = all_embeds_split[prompt_chunk_indices[b]]
            curr_p_len = all_id_lens[prompt_chunk_indices[b]]
            curr_p_label = torch.full((curr_p_len,), -100, dtype=torch.long, device=device)

            current_embeds = [curr_p_embed]
            current_labels = [curr_p_label]

            full_audio = speech_embeds[b]
            max_audio_len = speech_embeds_length[b].item()

            for seg_idx, seg in enumerate(targets[b]):
                s_idx = max(0, min(seg['start_idx'], max_audio_len))
                e_idx = max(s_idx, min(seg['end_idx'], max_audio_len))
                audio_slice = full_audio[s_idx:e_idx]

                chunk_idx = seg_chunk_indices[seg_cursor]
                if chunk_idx >= 0:
                    txt_embed = all_embeds_split[chunk_idx]
                    txt_ids = seg['input_ids'][:seg['input_len']].to(device, dtype=torch.long)
                else:
                    txt_embed = torch.empty((0, embed_dim), device=device, dtype=dtype)
                    txt_ids = torch.empty((0,), device=device, dtype=torch.long)
                seg_cursor += 1

                chunk_input_emb = torch.cat([emb_A, audio_slice, emb_A_end], dim=0)
                chunk_input_lab = torch.full((chunk_input_emb.size(0),), -100, dtype=torch.long, device=device)

                if seg_idx > 0:
                    chunk_input_lab[0] = self.token_W_id

                chunk_target_emb = txt_embed
                chunk_target_lab = txt_ids
                
                current_embeds.extend([chunk_input_emb, chunk_target_emb])
                current_labels.extend([chunk_input_lab, chunk_target_lab])

            if len(targets[b]) > 0:
                dummy_emb = torch.zeros((1, embed_dim), dtype=dtype, device=device)
                final_lab = torch.tensor([self.token_W_id], dtype=torch.long, device=device)
                current_embeds.append(dummy_emb)
                current_labels.append(final_lab)

            if current_embeds:
                batch_seq_embeds.append(torch.cat(current_embeds, dim=0))
                batch_seq_labels.append(torch.cat(current_labels, dim=0))
        
        if not batch_seq_embeds:
            return None, None, None

        inputs_embeds = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=0.0)
        labels = pad_sequence(batch_seq_labels, batch_first=True, padding_value=-100)

        seq_lens = torch.tensor([s.size(0) for s in batch_seq_embeds], device=device)
        max_len = inputs_embeds.size(1)
        attention_mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        
        return inputs_embeds, attention_mask, labels
