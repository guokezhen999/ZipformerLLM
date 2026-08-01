import os
import logging
import argparse

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy

from speechllm.utils import load_config, CustomProgressBar
from speechllm.dataset import DatasetForStreamAST

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--grpo_mode",
        type=str,
        default="multi_sglang",
        choices=["multi_sglang", "local_sglang"],
        help="GRPO trainer variant: "
             "multi_sglang = N SGLang instances on separate GPUs, "
             "local_sglang = SGLang co-located on same GPU as training, "
    )
    parser.add_argument("--exp_dir", required=False, type=str)
    parser.add_argument("--exp_name", required=False, type=str)
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--valid_dataset", type=str)
    parser.add_argument("--pretrained_model_path", required=False, type=str, default=None)
    parser.add_argument("--ckpt_path", required=False, type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for SGLang generate")
    parser.add_argument("--offload", action="store_true", help="Enable DeepSpeed optimizer offloading to CPU")
    args, _ = parser.parse_known_args()

    # ---- 根据 grpo_mode 选择 Trainer 类 ----
    if args.grpo_mode == "multi_sglang":
        from speechllm.trainer.trainer_ast_grpo_multi_sglang import SpeechLLMLightningASTGRPOMultiSGLang
        ActorClass = SpeechLLMLightningASTGRPOMultiSGLang
    elif args.grpo_mode == "local_sglang":
        from speechllm.trainer.trainer_ast_grpo_local_sglang import SpeechLLMLightningASTGRPOLocalSGLang
        ActorClass = SpeechLLMLightningASTGRPOLocalSGLang
    else:
        raise ValueError(f"Unknown grpo_mode: {args.grpo_mode}")

    logging.info(f"Using GRPO mode: {args.grpo_mode} -> {ActorClass.__name__}")

    model_config = load_config(args.config)

    if args.exp_dir:
        model_config.train.exp_dir = args.exp_dir
    if args.exp_name:
        model_config.train.exp_name = args.exp_name
    if args.train_dataset:
        model_config.data.train_dataset = args.train_dataset
    if args.valid_dataset:
        model_config.data.valid_dataset = args.valid_dataset
    if args.pretrained_model_path is not None:
        model_config.train.resume_from_checkpoint = args.pretrained_model_path
    model_config.train.debug = args.debug

    run_dir = os.path.join(model_config.train.exp_dir, model_config.train.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=tb_dir, name=".")

    checkpoint_dir_name = model_config.train.get("checkpoint_dir_name", "checkpoints")
    checkpoint_dir = os.path.join(run_dir, checkpoint_dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(model_config.train.seed)

    train_dataset = DatasetForStreamAST(
        manifest_paths=model_config.data.train_dataset,
        mode="train",
        config=model_config.data,
    )
    val_dataset = DatasetForStreamAST(
        manifest_paths=model_config.data.valid_dataset,
        mode="valid",
        config=model_config.data,
    )

    actor = ActorClass(model_config, train_dataset, val_dataset)

    max_steps = model_config.train.get("max_steps", -1)
    max_epochs = model_config.train.get("max_epochs", 3)
    val_interval = model_config.train.get("val_interval", 2000)
    save_interval = model_config.train.get("save_interval", 5000)

    step_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-step-{step}-{val_reward_bleu:.4f}",
        save_top_k=1,
        monitor="val_reward_bleu",
        mode="max",
        save_last=True,
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step-{step}",
        every_n_train_steps=save_interval,
        save_top_k=-1,
    )

    callbacks = [
        step_checkpoint,
        periodic_checkpoint,
        DeviceStatsMonitor(),
        LearningRateMonitor(logging_interval="step"),
        CustomProgressBar(),
    ]

    trainer = Trainer(
        accelerator="gpu",
        max_steps=max_steps if max_steps > 0 else -1,
        max_epochs=max_epochs if max_steps <= 0 else -1,
        devices=model_config.train.gpus,
        strategy=DeepSpeedStrategy(stage=2, offload_optimizer=args.offload),
        use_distributed_sampler=False,
        limit_val_batches=1.0,
        log_every_n_steps=model_config.train.get("log_interval", 100),
        val_check_interval=val_interval,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=model_config.train.grad_accumulate_steps,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.fit(actor, ckpt_path=args.ckpt_path)
