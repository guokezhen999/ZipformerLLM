from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import lightning.pytorch as pl

import torch.utils.data as data_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback, DeviceStatsMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch
import json
import argparse
import logging
import os

from speechllm.utils import load_config, CustomProgressBar, ValidateOnTrainEpochEnd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if torch.version.cuda and tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 6):
    torch.backends.cuda.enable_cudnn_sdp(False)
    logging.info(f"PyTorch {torch.__version__} < 2.6.0, disabled cuDNN SDP to avoid stride mismatch warning")
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, force=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Config file path')
    parser.add_argument('--train_type', type=str, default='streaming_asr', choices=['streaming_asr', 'streaming_ast'],
                        help='Training type: streaming_asr or streaming_ast')
    parser.add_argument('--exp_dir', required=False, type=str, help='')
    parser.add_argument('--exp_name', required=False, type=str, help='Experiment name')
    parser.add_argument('--train_dataset', type=str, help='Path to training dataset JSON file')
    parser.add_argument('--valid_dataset', type=str, help='Path to validation dataset JSON file')
    parser.add_argument('--train_shar_dirs', type=str, help='Comma-separated Shar dirs for shar_pool_sampler')
    parser.add_argument('--shard_assignment_file', type=str, help='Preassigned shard JSON for shar_pool_sampler')
    parser.add_argument('--pretrained_model_path', required=False, type=str, default=None, help='the pretrained connector model path')
    parser.add_argument('--ckpt_path', required=False, type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--ds_stage', type=int, default=2, choices=[0, 1, 2, 3], help='DeepSpeed ZeRO stage (0=disabled, 1/2/3=ZeRO stage). Default: 2')
    args, options = parser.parse_known_args()
    
    if args.train_type == 'streaming_asr':
        from speechllm.trainer import SpeechLLMLightningStreamASR
        from speechllm.dataset import DatasetForStreamASR
        ModelClass = SpeechLLMLightningStreamASR
        DatasetClass = DatasetForStreamASR
    elif args.train_type == 'streaming_ast':
        from speechllm.trainer import SpeechLLMLightningStreamAST
        from speechllm.dataset import DatasetForStreamAST
        ModelClass = SpeechLLMLightningStreamAST
        DatasetClass = DatasetForStreamAST
    else:
        raise ValueError(f"Invalid train_type: {args.train_type}")

    logging.info(f"Using training type: {args.train_type}")
    
    model_config = load_config(args.config)

    # Overwrite config with args if provided
    if args.exp_dir:
        model_config.train.exp_dir = args.exp_dir
    if args.exp_name:
        model_config.train.exp_name = args.exp_name
    if args.train_dataset:
        model_config.data.train_dataset = args.train_dataset
    if args.valid_dataset:
        model_config.data.valid_dataset = args.valid_dataset
    if args.train_shar_dirs:
        model_config.data.train_shar_dirs = args.train_shar_dirs
    if args.shard_assignment_file:
        model_config.data.shard_assignment_file = args.shard_assignment_file
    if args.pretrained_model_path is not None and not args.ckpt_path:
        model_config.train.resume_from_checkpoint = args.pretrained_model_path
    elif args.ckpt_path and args.pretrained_model_path:
        logging.warning(
            "同时指定了 --pretrained_model_path 与 --ckpt_path："
            "resume 时忽略 pretrained，只从 DeepSpeed ckpt 恢复权重"
        )
    
    run_dir = os.path.join(model_config.train.exp_dir, model_config.train.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=tb_dir, name=".")

    checkpoint_dir_name = model_config.train.get("checkpoint_dir_name", "checkpoints")
    checkpoint_dir = os.path.join(run_dir, checkpoint_dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(model_config.train.seed)

    # 根据 ds_stage 选择训练策略
    if args.ds_stage == 0:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False)
        logging.info("Using DDP strategy (DeepSpeed disabled)")
    else:
        strategy = DeepSpeedStrategy(stage=args.ds_stage)
        logging.info(f"Using DeepSpeed ZeRO stage {args.ds_stage}")

    # 从 adapter 配置自动注入 pooling_factor 到 data 配置
    adapter_pooling_factor = model_config.model.adapter.get('pooling_factor', 4)
    model_config.data.pooling_factor = adapter_pooling_factor

    sampler_type = model_config.data.sampler_type
    train_manifest = model_config.data.get('train_dataset', None)
    if sampler_type == 'shar_pool_sampler' and not train_manifest:
        train_manifest = []

    train_dataset = DatasetClass(
        manifest_paths=train_manifest,
        mode='train',
        config=model_config.data
    )
    val_dataset = DatasetClass(
        manifest_paths=model_config.data.valid_dataset, 
        mode='valid',
        config=model_config.data
    )
    
    model = ModelClass(model_config, train_dataset, val_dataset)
    if args.ckpt_path:
        model._resume_ckpt_path = args.ckpt_path
        logging.info(f"Resume from DeepSpeed ckpt: {args.ckpt_path}")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    step_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-step-{step}-{val_loss:.4f}', 
        save_top_k=1,         
        monitor="val_loss",   
        mode="min",
        save_last=True        
    )

    max_epochs = model_config.train.get('max_epochs', 3)
    max_steps = model_config.train.get('max_steps', -1)
    val_interval = model_config.train.get('val_interval', 2000)
    save_interval = model_config.train.get('save_interval', 5000)

    # 基础回调：按步验证由 val_check_interval 触发；epoch 结束验证由 ValidateOnTrainEpochEnd 补齐
    base_callbacks = [
        step_checkpoint, 
        lr_monitor, 
        CustomProgressBar(),
        ValidateOnTrainEpochEnd(),
    ]

    # 按步训练：stateless / shar_pool(preassigned)；后者仍有 epoch 概念用于重洗 shard
    use_step_training = sampler_type in ('stateless_sampler', 'shar_pool_sampler')

    if use_step_training:
        logging.info(
            f"检测到 {sampler_type}，使用按步训练模式: max_steps={max_steps}, "
            f"val_check_interval={val_interval} (按步验证) + epoch 结束验证"
            + (", reload_dataloaders_every_n_epochs=1" if sampler_type == 'shar_pool_sampler' else "")
        )
        
        periodic_step_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='step-{step}', 
            every_n_train_steps=save_interval,    
            save_top_k=-1,        
        )

        epoch_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='epoch-{epoch:02d}-step-{step}', 
            every_n_epochs=1,    
            save_top_k=-1,        
            save_on_train_epoch_end=True 
        )

        trainer_kwargs = dict(
            accelerator='gpu',
            max_steps=max_steps,
            max_epochs=-1,
            devices=model_config.train.gpus,
            strategy=strategy,
            use_distributed_sampler=False,
            limit_train_batches=None,
            limit_val_batches=1.0,
            log_every_n_steps=model_config.train.get('log_interval', 100),
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            val_check_interval=val_interval,
            callbacks=base_callbacks + [periodic_step_checkpoint, epoch_checkpoint],
            fast_dev_run=False,
            logger=logger,
            accumulate_grad_batches=model_config.train.grad_accumulate_steps,
            precision="bf16-mixed",
            num_sanity_val_steps=0,
        )
        # preassigned：每个 epoch 结束重建 dataloader，重新 shuffle 本 rank 的 shard
        if sampler_type == 'shar_pool_sampler':
            trainer_kwargs['reload_dataloaders_every_n_epochs'] = 1

        trainer = Trainer(**trainer_kwargs)
    else:
        # 2. 针对普通 Sampler 的 Trainer 配置 (按 Epoch 训练)
        logging.info(
            f"使用常规训练模式: max_epochs={max_epochs}, "
            f"val_check_interval={val_interval} (按步验证) + epoch 结束验证"
        )

        epoch_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='epoch-{epoch}-{step}', 
            every_n_epochs=1,    
            save_top_k=-1,        
            save_on_train_epoch_end=True 
        )

        trainer = Trainer(
            accelerator='gpu',
            max_epochs=max_epochs,
            devices=model_config.train.gpus,
            strategy=strategy,
            use_distributed_sampler=False,
            limit_train_batches=None,
            limit_val_batches=1.0,
            log_every_n_steps=model_config.train.get('log_interval', 100),
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            val_check_interval=val_interval,
            callbacks=base_callbacks + [epoch_checkpoint],
            fast_dev_run=False,
            logger=logger,
            accumulate_grad_batches=model_config.train.grad_accumulate_steps,
            precision="bf16-mixed",
            num_sanity_val_steps=0,
        )
    
    trainer.fit(model, ckpt_path=args.ckpt_path)
