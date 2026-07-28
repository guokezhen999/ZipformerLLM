import torch
from torch import nn
from torch.optim import AdamW
import lightning.pytorch as pl
import logging
import torch.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
import os
from collections import defaultdict

from ..dataset import get_sampler
from ..module import get_connector, get_llm, get_encoder
from ..utils import get_lr_schedule

class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__()
        
        # model_config is expected to be a dict or Addict-like object with structure:
        # { 'data': ..., 'model': { 'zipformer': ..., 'adapter': ..., 'llm': ... }, 'train': ... }
        torch.manual_seed(model_config.train.seed)
        self.save_hyperparameters(ignore=["train_ds", "val_ds"])
        self.model_config = model_config
        self.train_ds = train_ds
        self.val_ds = val_ds
        
        # Data / Experiment Configs
        self.exp_name = model_config.train.exp_name
        self.exp_dir = model_config.train.exp_dir

        # sampler 设置
        self.sampler_type = model_config.data.sampler_type
        if self.sampler_type == 'stateless_sampler':
            # 构造 stateless_sampler 所需要的数据格式
            train_paths = model_config.data.get('train_dataset', [])
            weights = model_config.data.get('train_dataset_weights', [1.0] * len(train_paths))
            cuts_data = [(p, float(w)) for p, w in zip(train_paths, weights)]

            self.stateless_sampler_config = model_config.data
            self.stateless_sampler_config.cuts_data = cuts_data
            self.stateless_sampler_config.seed = model_config.train.seed
            self.stateless_sampler_config.dataloader_num_workers = model_config.train.get('dataloader_num_workers', 4)

        # shar_pool preassigned：启动时加载本 rank 的固定 shard 列表，每 epoch 只 shuffle 顺序
        self._preassigned_shards = None
        if self.sampler_type == 'shar_pool_sampler':
            assignment_file = model_config.data.get('shard_assignment_file', None)
            if not assignment_file:
                raise ValueError(
                    "[shar_pool_sampler] 必须提供 data.shard_assignment_file（preassigned 模式）"
                )
            from ..dataset.shar_pool import parse_shar_dirs
            shar_dirs = parse_shar_dirs(model_config.data.get('train_shar_dirs', None))
            if not shar_dirs:
                raise ValueError(
                    "[shar_pool_sampler] 必须提供 data.train_shar_dirs，用于解析 assignment 中的 shard 路径"
                )
            # global_rank 在 __init__ 时尚不可用，延迟到 train_dataloader / setup 再加载
            self._shar_assignment_file = assignment_file
            self._shar_dirs = shar_dirs
            self._preassigned_shards = None
            self._preassigned_cut_count = None

        # LLM Configs
        llm_conf = model_config.model.llm
        self.llm_name = llm_conf.model_name
        self.use_lora = llm_conf.enable_lora
        
        # Zipformer Configs
        self.finetune_encoder = model_config.model.zipformer.get('finetune', False)
        
        # Adapter Configs
        adapter_conf = model_config.model.adapter
        self.adapter_name = adapter_conf.name
        
        # 模型初始化
        self.audio_encoder = get_encoder(model_config.model.zipformer)
        
        if not self.finetune_encoder:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False

        self.connector = get_connector(self.adapter_name, adapter_conf)
        
        # 根据 adapter.finetune 决定是否冻结 connector
        self.finetune_connector = adapter_conf.get('finetune', True)
        if not self.finetune_connector:
            for p in self.connector.parameters():
                p.requires_grad = False
        
        # Get special tokens from config or use defaults
        self.special_tokens_dict = llm_conf.get('special_tokens', {
            "<A>": "<|vision_start|>", 
            "</A>": "<|vision_end|>", 
            "<W>": "<|im_end|>"
        })
        special_tokens_list = list(self.special_tokens_dict.keys())
        
        self.llm_tokenizer, self.llm_model = get_llm(
            llm_conf.model_name, 
            use_lora=llm_conf.enable_lora, 
            lora_r=llm_conf.get('lora_r', 8), 
            lora_alpha=llm_conf.get('lora_alpha', 16),
            finetune=llm_conf.finetune,
            gradient_checkpointing=llm_conf.get('gradient_checkpointing', False),
            special_tokens=special_tokens_list,
            enable_dropout=llm_conf.get('enable_dropout', True),
            dropout_rate=llm_conf.get('dropout_rate', 0.05),
        )
        
        # 保存特殊 Token ID
        self.special_token_ids = {}
        for token in special_tokens_list:
            tid = self.llm_tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, list): tid = tid[0]
            self.special_token_ids[token] = tid
            
        # Backward compatibility for common tokens
        self.token_A_id = self.special_token_ids.get("<A>")
        self.token_A_end_id = self.special_token_ids.get("</A>")
        self.token_W_id = self.special_token_ids.get("<W>")

        # Training Hyperparameters
        train_conf = model_config.train
        self.max_lr = train_conf.get('max_lr', 1e-3)
        self.lr_connector = train_conf.get('lr_connector', self.max_lr)
        self.lr_llm = train_conf.get('lr_llm', self.max_lr)
        self.lr_encoder = train_conf.get('lr_encoder', self.max_lr)
        
        self.finetune_special_tokens = train_conf.get('finetune_special_tokens', True)
        self.lr_special_tokens = train_conf.get('lr_special_tokens', self.lr_llm)

        # 初始化增加的特殊 token 的权重
        self._init_special_tokens()

        # Scheduler-specific parameters
        scheduler_conf = train_conf.get('scheduler', {})
        self.warmup_steps = scheduler_conf.get('warmup_steps', train_conf.get('warmup_steps', 1000))
        
        self.max_epochs = train_conf.get('max_epochs', -1)
        self.weight_decay = train_conf.get('weight_decay', 0)
        self.weight_decay_connector = train_conf.get('weight_decay_connector', self.weight_decay)
        self.weight_decay_special_tokens = train_conf.get('weight_decay_special_tokens', 0.0)
        self.weight_decay_encoder = train_conf.get('weight_decay_encoder', self.weight_decay)
                
        self.max_token_length = train_conf.get('max_token_length', None)

        self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id
                
        self.pretrained_model_path = train_conf.get('resume_from_checkpoint', None)
        
        self.saved_sampler_state = None
        self.total_processed_samples = 0
        self.epoch_processed_samples = 0

    @property
    def _is_rank0(self):
        """判断当前进程是否为 rank 0（用于控制日志只在主进程打印）"""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _log_rank0(self, msg, level=logging.INFO):
        """只在 rank 0 打印日志"""
        if self._is_rank0:
            logging.log(level, msg)

    def load_pretrained_model(self, ckpt=None):
        path = ckpt if ckpt is not None else self.pretrained_model_path
        if not path:
            return
            
        # 如果传入的是 DeepSpeed checkpoint 文件夹（如 last.ckpt），自动寻找其中的模型权重文件
        if os.path.isdir(path):
            mp_rank_path = os.path.join(path, "checkpoint", "mp_rank_00_model_states.pt")
            if os.path.exists(mp_rank_path):
                self._log_rank0(f"检测到 DeepSpeed 检查点目录 {path}，自动定位到权重文件: {mp_rank_path}")
                path = mp_rank_path
            else:
                raise FileNotFoundError(f"DeepSpeed 检查点目录 {path} 中未找到 {mp_rank_path}")

        state_dict = torch.load(path, map_location='cpu', weights_only=False)
            
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
            
        # 1. 提取并加载 Audio Encoder
        audio_encoder_dict = {k.replace("audio_encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("audio_encoder.")}
        if audio_encoder_dict:
            self.audio_encoder.load_state_dict(audio_encoder_dict, strict=False)
            self._log_rank0(f"从 {path} 成功加载 audio_encoder (共 {len(audio_encoder_dict)} 个权重)")
        else:
            self._log_rank0(f"检查点 {path} 中未找到 'audio_encoder' 权重", logging.WARNING)
            
        # 2. 提取并加载 Connector
        connector_dict = {k.replace("connector.", "", 1): v for k, v in state_dict.items() if k.startswith("connector.")}
        if connector_dict:
            self.connector.load_state_dict(connector_dict, strict=False)
            self._log_rank0(f"从 {path} 成功加载 connector (共 {len(connector_dict)} 个权重)")
        else:
            self._log_rank0(f"检查点 {path} 中未找到 'connector' 权重", logging.WARNING)
            
        # 3. 提取并加载 LLM Model
        llm_dict = {k.replace("llm_model.", "", 1): v for k, v in state_dict.items() if k.startswith("llm_model.")}
        llm_alt_dict = {k.replace("llm.", "", 1): v for k, v in state_dict.items() if k.startswith("llm.")}
        
        if len(llm_alt_dict) > len(llm_dict):
            llm_dict = llm_alt_dict
            self._log_rank0(f"使用 'llm.' 前缀加载 LLM 权重 (on_save_checkpoint 格式)")
        
        if llm_dict:
            expected_keys = set(self.llm_model.state_dict().keys())
            loaded_keys = set(llm_dict.keys())
            coverage = len(loaded_keys & expected_keys) / len(expected_keys) if expected_keys else 0
            
            if coverage < 0.5:
                self._log_rank0(
                    f"检查点中 LLM 权重不完整 (覆盖率 {coverage:.1%}, {len(loaded_keys & expected_keys)}/{len(expected_keys)})，"
                    f"可能是因为 stage1 训练时 LLM 被冻结。跳过 LLM 权重加载，使用预训练初始权重。",
                    logging.WARNING
                )
            else:
                self.llm_model.load_state_dict(llm_dict, strict=False)
                self._log_rank0(f"从 {path} 成功加载 llm_model 权重 (共 {len(llm_dict)} 个权重, 覆盖率 {coverage:.1%})")
        else:
            self._log_rank0(f"检查点 {path} 中未找到 LLM 权重，LLM 将使用预训练初始权重", logging.WARNING)
            
        # 4. 加载特殊 Token 权重
        if "special_token_input_patch" in state_dict:
            if hasattr(self, "special_token_input_patch"):
                old_val = self.special_token_input_patch.data.clone()
                self.special_token_input_patch.data.copy_(state_dict["special_token_input_patch"])
                new_val = self.special_token_input_patch.data
                diff = torch.abs(old_val - new_val).sum().item()
                self._log_rank0(f"成功加载 special_token_input_patch, L1 Diff: {diff:.6f}, "
                                f"Mean: {new_val.mean().item():.8f}, Norm: {new_val.norm().item():.4f}")
                if diff == 0:
                    self._log_rank0("⚠️ special_token_input_patch 加载后数值未变化", logging.WARNING)
            else:
                self._log_rank0("检查点含 special_token_input_patch，但当前模型无此属性，跳过加载", logging.WARNING)
        else:
            self._log_rank0(f"检查点 {path} 中未找到 special_token_input_patch", logging.WARNING)

        if "special_token_output_patch" in state_dict:
            if hasattr(self, "special_token_output_patch"):
                old_val_out = self.special_token_output_patch.data.clone()
                self.special_token_output_patch.data.copy_(state_dict["special_token_output_patch"])
                new_val_out = self.special_token_output_patch.data
                diff_out = torch.abs(old_val_out - new_val_out).sum().item()
                self._log_rank0(f"成功加载 special_token_output_patch, L1 Diff: {diff_out:.6f}, "
                                f"Mean: {new_val_out.mean().item():.8f}, Norm: {new_val_out.norm().item():.4f}")
            else:
                self._log_rank0("检查点含 special_token_output_patch，但当前模型无此属性，跳过加载", logging.WARNING)
        else:
            self._log_rank0(f"检查点 {path} 中未找到 special_token_output_patch", logging.WARNING)
                    
        # 5. 加载 Sampler State
        if "sampler_state" in state_dict:
            self.saved_sampler_state = state_dict["sampler_state"]
            self._log_rank0(f"从 {path} 加载了 sampler_state")
        elif "all_sampler_states" in state_dict:
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            all_states = state_dict["all_sampler_states"]
            if rank < len(all_states):
                self.saved_sampler_state = all_states[rank]
                self._log_rank0(f"从 {path} 加载了 Rank {rank} 的 sampler_state")
        else:
            self._log_rank0(f"检查点 {path} 中没有 sampler_state 或 all_sampler_states")
             
    def on_save_checkpoint(self, checkpoint) -> None:
        if "state_dict" not in checkpoint:
            checkpoint["state_dict"] = {}
        
        checkpoint["state_dict"]["audio_encoder"] = self.audio_encoder.state_dict()
        checkpoint["state_dict"]["connector"] = self.connector.state_dict()
        if self.use_lora:
            from peft import get_peft_model_state_dict
            checkpoint["state_dict"]["llm_lora"] = get_peft_model_state_dict(self.llm_model)
        else:
            checkpoint["state_dict"]["llm_model"] = self.llm_model.state_dict()
        
        if self.sampler_type != 'shar_pool_sampler' and hasattr(self.trainer.train_dataloader, "sampler"):
            sampler = self.trainer.train_dataloader.sampler
            if hasattr(sampler, "state_dict"):
                local_sampler_state = sampler.state_dict()
                
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    all_states = [None] * world_size
                    torch.distributed.all_gather_object(all_states, local_sampler_state)
                    checkpoint["all_sampler_states"] = all_states
                else:
                    checkpoint["sampler_state"] = local_sampler_state
        
        checkpoint["total_processed_samples"] = self.total_processed_samples
        
        if "optimizer_states" not in checkpoint and self.trainer.optimizers:
            logging.info("Explicitly adding optimizer states to checkpoint.")
            checkpoint["optimizer_states"] = [opt.state_dict() for opt in self.trainer.optimizers]

        if "lr_schedulers" not in checkpoint and self.trainer.lr_scheduler_configs:
            logging.info("Explicitly adding lr scheduler states to checkpoint.")
            checkpoint["lr_schedulers"] = [
                {"scheduler": config.scheduler.state_dict()}
                for config in self.trainer.lr_scheduler_configs
            ]

    def on_load_checkpoint(self, checkpoint) -> None:
        """从上一个 epoch 结束的检查点恢复，进入下一个 epoch。

        Lightning 2.x 的 ``current_epoch`` 来自
        ``loops['fit_loop']['epoch_progress'].current.completed``，
        仅修改顶层 ``checkpoint['epoch']`` 不会生效。
        ``save_on_train_epoch_end`` 保存时 ``completed`` 尚未 +1，需要在此补上，
        并清空本 epoch 内的 batch 进度，避免卡在已结束的 epoch。
        """
        if "epoch" in checkpoint:
            checkpoint["epoch"] += 1

        if "loops" in checkpoint and "fit_loop" in checkpoint["loops"]:
            fit_loop = checkpoint["loops"]["fit_loop"]
            # 丢掉已结束 epoch 内的 batch 进度，从下一个 epoch 的 batch 0 开始
            fit_loop.pop("epoch_loop.batch_progress", None)

            ep = fit_loop.get("epoch_progress")
            if isinstance(ep, dict):
                for section in ("current", "total"):
                    prog = ep.get(section)
                    if not isinstance(prog, dict) or "completed" not in prog:
                        continue
                    # 正常 epoch 结束保存时: processed == completed + 1
                    target = prog.get("processed", prog["completed"] + 1)
                    if prog["completed"] < target:
                        prog["completed"] = target
                    else:
                        prog["completed"] += 1
                        for k in ("ready", "started", "processed"):
                            if k in prog:
                                prog[k] = max(prog[k], prog["completed"])
                next_epoch = ep.get("current", {}).get("completed")
                logging.info(
                    f"End-of-epoch resume: advanced epoch_progress -> "
                    f"next current_epoch={next_epoch}"
                )

        state_dict = checkpoint.get("state_dict", {})

        if "all_sampler_states" in checkpoint:
            rank = self.global_rank
            all_states = checkpoint["all_sampler_states"]
            if rank < len(all_states):
                self.saved_sampler_state = all_states[rank]
                logging.info(f"成功从检查点根部恢复 Rank {rank} 的 sampler 状态")
        elif "sampler_state" in checkpoint:
            self.saved_sampler_state = checkpoint["sampler_state"]
            logging.info("成功从检查点根部加载单卡 sampler 状态")
        elif "all_sampler_states" in state_dict:
            rank = self.global_rank
            all_states = state_dict["all_sampler_states"]
            if rank < len(all_states):
                self.saved_sampler_state = all_states[rank]
                logging.info(f"成功从 state_dict 恢复 Rank {rank} 的 sampler 状态")
        elif "sampler_state" in state_dict:
            self.saved_sampler_state = state_dict["sampler_state"]
            logging.info("成功从 state_dict 加载单卡 sampler 状态")

        if "total_processed_samples" in checkpoint:
            self.total_processed_samples = checkpoint["total_processed_samples"]
            logging.info(f"从检查点恢复总样本计数: {self.total_processed_samples}")

        # ZeRO-2 下 Lightning.load_model_state_dict 是空操作，权重只靠 DeepSpeed 静默 load。
        # 实测 resume 后 val_loss 对不上 ckpt，这里强制从 mp_rank 文件再灌一遍并打日志，避免“假恢复”。
        self._reload_weights_from_deepspeed_ckpt()

    def _reload_weights_from_deepspeed_ckpt(self) -> None:
        """从 DeepSpeed ckpt 目录的 mp_rank_00_model_states.pt 显式加载 module 权重。"""
        ckpt_path = getattr(self, "_resume_ckpt_path", None) or getattr(self.trainer, "ckpt_path", None)
        if not ckpt_path:
            return
        if not os.path.isdir(ckpt_path):
            self._log_rank0(f"ckpt_path={ckpt_path} 不是目录，跳过显式权重重载", logging.WARNING)
            return

        mp_rank_path = os.path.join(ckpt_path, "checkpoint", "mp_rank_00_model_states.pt")
        if not os.path.exists(mp_rank_path):
            self._log_rank0(f"未找到 {mp_rank_path}，跳过显式权重重载", logging.WARNING)
            return

        # 重载前记录若干参数，便于确认数值真的变了
        probe_before = {}
        with torch.no_grad():
            if hasattr(self, "special_token_input_patch"):
                probe_before["special_token_input_patch"] = (
                    float(self.special_token_input_patch.data.float().norm()),
                    float(self.special_token_input_patch.data.float().mean()),
                )
            for name, p in self.connector.named_parameters():
                probe_before[f"connector.{name}"] = (float(p.data.float().norm()), float(p.data.float().mean()))
                break
            for name, p in self.llm_model.named_parameters():
                if "embed_tokens.weight" in name or name.endswith("norm.weight"):
                    probe_before[f"llm_model.{name}"] = (float(p.data.float().norm()), float(p.data.float().mean()))
                    break

        self._log_rank0(f"显式从 DeepSpeed ckpt 重载权重: {mp_rank_path}")
        self.load_pretrained_model(mp_rank_path)

        max_delta = 0.0
        with torch.no_grad():
            for key, (norm0, mean0) in probe_before.items():
                if key == "special_token_input_patch" and hasattr(self, "special_token_input_patch"):
                    t = self.special_token_input_patch.data.float()
                elif key.startswith("connector."):
                    name = key[len("connector."):]
                    t = dict(self.connector.named_parameters())[name].data.float()
                elif key.startswith("llm_model."):
                    name = key[len("llm_model."):]
                    t = dict(self.llm_model.named_parameters())[name].data.float()
                else:
                    continue
                delta = abs(float(t.norm()) - norm0)
                max_delta = max(max_delta, delta)
                self._log_rank0(
                    f"[ckpt reload probe] {key}: "
                    f"before norm={norm0:.6f} mean={mean0:.8f} -> "
                    f"after norm={float(t.norm()):.6f} mean={float(t.mean()):.8f} "
                    f"|Δnorm|={delta:.6f}"
                )

        if max_delta < 1e-5:
            self._log_rank0(
                "显式重载前后参数几乎不变：DeepSpeed load_checkpoint 已写入相同权重 "
                f"(max|Δnorm|={max_delta:.2e})"
            )
        else:
            self._log_rank0(
                f"显式重载改变了权重 (max|Δnorm|={max_delta:.6f})："
                "说明 DeepSpeed 静默恢复未把 module 写成 ckpt 内容，已用 mp_rank 文件纠正",
                logging.WARNING,
            )

    def setup(self, stage=None):
        """在 configure_optimizers 之前加载预训练权重，确保 DeepSpeed 分片时拿到的是正确的参数值"""
        # resume 自 DeepSpeed ckpt 时不要先灌 stage1，避免“假恢复”后仍像 stage1
        if getattr(self, "_resume_ckpt_path", None):
            self._log_rank0(
                f"检测到 resume ckpt ({self._resume_ckpt_path})，跳过 setup 中的 pretrained 加载"
            )
            return
        if stage == "fit" and self.pretrained_model_path is not None:
            if not getattr(self, '_pretrained_loaded', False):
                self._log_rank0(f"从预训练模型加载权重: {self.pretrained_model_path}")
                self.load_pretrained_model(self.pretrained_model_path)
                self._pretrained_loaded = True
            else:
                self._log_rank0("预训练权重已加载，跳过重复加载")

    def on_train_start(self) -> None:
        if not self.finetune_encoder:
            self.audio_encoder.eval()

    @staticmethod
    def _should_weight_decay(name):
        """判断参数是否应该加 weight decay，只对 weight 矩阵加，不对 bias / LayerNorm / Embedding 加"""
        no_decay_keywords = ("bias", "layernorm", "layer_norm", "ln_", "norm", "embedding")
        name_lower = name.lower()
        return not any(kw in name_lower for kw in no_decay_keywords)

    def _split_params_by_decay(self, module, lr, wd, prefix=""):
        """将 module 的可训练参数按是否需要 weight decay 分成两组，同时记录参数名"""
        decay_params = []
        decay_names = []
        no_decay_params = []
        no_decay_names = []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            full_name = f"{prefix}.{name}" if prefix else name
            if self._should_weight_decay(name) and wd > 0:
                decay_params.append(param)
                decay_names.append(full_name)
            else:
                no_decay_params.append(param)
                no_decay_names.append(full_name)
        
        groups = []
        if decay_params:
            groups.append({"params": decay_params, "lr": lr, "weight_decay": wd, "_names": decay_names})
        if no_decay_params:
            groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0, "_names": no_decay_names})
        return groups

    def configure_optimizers(self):
        opt = []
        
        # 1. Connector: 按 decay / no_decay 分组
        if self.finetune_connector:
            opt.extend(self._split_params_by_decay(
                self.connector, self.lr_connector, self.weight_decay_connector, prefix="connector"
            ))
            
        # 2. LLM: 按 decay / no_decay 分组
        llm_groups = self._split_params_by_decay(
            self.llm_model, self.lr_llm, self.weight_decay, prefix="llm_model"
        )
        opt.extend(llm_groups)
            
        # 3. Special tokens: input_patch 和 output_patch 分开
        if self.finetune_special_tokens and hasattr(self, 'special_token_input_patch'):
            opt.append({
                "params": [self.special_token_input_patch],
                "lr": self.lr_special_tokens,
                "weight_decay": self.weight_decay_special_tokens,
                "_names": ["special_token_input_patch"],
            })
            opt.append({
                "params": [self.special_token_output_patch],
                "lr": self.lr_special_tokens,
                "weight_decay": self.weight_decay_special_tokens,
                "_names": ["special_token_output_patch"],
            })
             
        # 4. Encoder (如果微调)
        if self.finetune_encoder:
            opt.extend(self._split_params_by_decay(
                self.audio_encoder, self.lr_encoder, self.weight_decay_encoder, prefix="audio_encoder"
            ))

        # Rank 0 打印优化器参数组详情
        if self._is_rank0:
            logging.info(f"{'='*80}")
            logging.info(f"Optimizer: {len(opt)} parameter groups")
            logging.info(f"{'='*80}")
            for i, group in enumerate(opt):
                num_params = sum(p.numel() for p in group["params"])
                names = group.get("_names", [])
                logging.info(f"  Group {i}: lr={group['lr']}, wd={group.get('weight_decay', 0.0)}, "
                             f"num_tensors={len(group['params'])}, num_params={num_params:,}")
                for n in names:
                    logging.info(f"    - {n}")
            logging.info(f"{'='*80}")

        # 传给 AdamW 前移除辅助字段 _names
        for group in opt:
            group.pop("_names", None)

        optimizer = AdamW(opt)
    
        def get_processed_samples():
            return self.total_processed_samples

        if self.sampler_type not in ("stateless_sampler", "shar_pool_sampler"):
            world_size = self.trainer.world_size if self.trainer.world_size > 0 else 1
            total_samples_on_this_rank = len(self.train_ds) // world_size
            scheduler = get_lr_schedule(
                optimizer,
                self.model_config.train,
                self.warmup_steps,
                total_samples_on_this_rank,
                get_processed_samples
            )
        else:
            # 按步训练：LR 依赖 max_steps / step_decay，不依赖 epoch 样本数
            scheduler = get_lr_schedule(
                optimizer,
                self.model_config.train,
                self.warmup_steps,
                None,
                get_processed_samples
            )
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _ensure_preassigned_shards(self):
        """按当前 global_rank 加载预分配 shard（只做一次）。"""
        if self._preassigned_shards is not None:
            return
        from ..dataset.shar_pool import load_shard_assignment, count_cuts_in_shards

        rank = self.trainer.global_rank
        self._preassigned_shards = load_shard_assignment(
            assignment_file=self._shar_assignment_file,
            rank=rank,
            shar_dirs=self._shar_dirs,
        )
        if not self._preassigned_shards:
            raise RuntimeError(
                f"[shar_pool_sampler] Rank {rank} got 0 shards from "
                f"{self._shar_assignment_file}"
            )
        self._preassigned_cut_count = count_cuts_in_shards(self._preassigned_shards)
        logging.info(
            f"[Preassign] Rank {rank}: {len(self._preassigned_shards)} shards, "
            f"~{self._preassigned_cut_count} cuts"
        )

    def _prepare_shar_pool_train_cuts(self):
        """每个 epoch：shuffle 本 rank 预分配 shard 并注入 dataset.cuts。"""
        from ..dataset.shar_pool import create_cuts_from_assigned_shards

        self._ensure_preassigned_shards()
        epoch = int(self.trainer.current_epoch)
        seed = int(self.model_config.train.get("seed", 42))
        cuts = create_cuts_from_assigned_shards(
            self._preassigned_shards,
            epoch=epoch,
            seed=seed,
            shuffle=True,
        )
        self.train_ds.set_cuts(cuts, estimated_len=self._preassigned_cut_count)

    def on_train_epoch_start(self):
        self.epoch_processed_samples = 0
        if not self.finetune_encoder:
            self.audio_encoder.eval()

    def _init_special_tokens(self):
        # 只要开启微调特殊 Token（默认开启），或者使用了 LoRA，就使用 Patch 机制
        if self.finetune_special_tokens or self.use_lora:
            hidden_size = self.llm_model.config.hidden_size
            num_tokens = len(self.special_tokens_dict)
            self.special_token_input_patch = nn.Parameter(torch.zeros(num_tokens, hidden_size))
            self.special_token_output_patch = nn.Parameter(torch.zeros(num_tokens, hidden_size))
            
            # 只有设置了微调特殊 token 才可以被梯度更新
            self.special_token_input_patch.requires_grad = self.finetune_special_tokens
            self.special_token_output_patch.requires_grad = self.finetune_special_tokens
            
            nn.init.normal_(self.special_token_input_patch, mean=0.0, std=0.02)
            nn.init.normal_(self.special_token_output_patch, mean=0.0, std=0.02)
            
            with torch.no_grad():
                input_w = self.llm_model.get_input_embeddings().weight
                output_w = self.llm_model.get_output_embeddings().weight
                
                for i, (token, init_token) in enumerate(self.special_tokens_dict.items()):
                    if init_token == "eos_token":
                        init_id = self.llm_tokenizer.eos_token_id
                    else:
                        init_id = self.llm_tokenizer.convert_tokens_to_ids(init_token)
                    
                    if init_id != self.llm_tokenizer.unk_token_id and init_id is not None:
                        self.special_token_input_patch.data[i].copy_(input_w[init_id])
                        self.special_token_output_patch.data[i].copy_(output_w[init_id])
                    else:
                        logging.warning(f"Special token {token} initialization token {init_token} not found, using random initialization.")

            def output_patch_hook(module, input, output):
                hidden_states = input[0]
                # patch_logits shape: [batch, seq_len, num_tokens]
                patch_logits = torch.matmul(hidden_states, self.special_token_output_patch.to(hidden_states.dtype).T)
                for i, token in enumerate(self.special_tokens_dict.keys()):
                    tid = self.special_token_ids[token]
                    output[..., tid] = patch_logits[..., i]
                return output

            self.llm_model.get_output_embeddings().register_forward_hook(output_patch_hook)
        
        else:
            input_embeds = self.llm_model.get_input_embeddings()
            output_embeds = self.llm_model.get_output_embeddings()
            
            with torch.no_grad():
                for token, init_token in self.special_tokens_dict.items():
                    tid = self.special_token_ids[token]
                    if tid is None or tid == self.llm_tokenizer.unk_token_id:
                        continue
                        
                    if init_token == "eos_token":
                        init_id = self.llm_tokenizer.eos_token_id
                    else:
                        init_id = self.llm_tokenizer.convert_tokens_to_ids(init_token)

                    if init_id != self.llm_tokenizer.unk_token_id and init_id is not None:
                        input_embeds.weight[tid] = input_embeds.weight[init_id].clone()
                        if output_embeds is not None:
                            output_embeds.weight[tid] = output_embeds.weight[init_id].clone()
                    else:
                        nn.init.normal_(input_embeds.weight[tid], mean=0.0, std=0.02)
                        if output_embeds is not None:
                            nn.init.normal_(output_embeds.weight[tid], mean=0.0, std=0.02)

    def get_audio_embeds(self, batch_features, audio_lengths):
        context_manager = torch.no_grad() if not self.finetune_encoder else torch.enable_grad()
        with context_manager:
            speech_encoder_input = batch_features.to(dtype=next(self.audio_encoder.parameters()).dtype)
            speech_embeds, params_lengths = self.audio_encoder(
                speech_encoder_input, 
                audio_lengths
            )
            params_lengths = torch.clamp(params_lengths - 2, min=0)
            speech_embeds = speech_embeds.to(torch.bfloat16)
        
        params_lengths = (params_lengths + self.connector.pooling_factor - 1) // self.connector.pooling_factor
        speech_embeds = self.connector(speech_embeds)
        return speech_embeds, params_lengths
