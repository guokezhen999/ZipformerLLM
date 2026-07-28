import json
import os
from addict import Dict
from lightning.pytorch.callbacks import TQDMProgressBar, Callback


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    # verify main keys
    required_keys = ['data', 'model', 'train']
    for key in required_keys:
        if key not in config_dict:
            raise ValueError(f"Missing required config section: {key}")
            
    return Dict(config_dict)


class ValidateOnTrainEpochEnd(Callback):
    """在每个 train epoch 结束时再跑一次验证。

    Lightning 在 ``val_check_interval`` 为整数时，只用
    ``(batch_idx+1) % interval == 0`` 判断，会覆盖 ``is_last_batch``，
    因此 epoch 最后一步通常不会验证。本回调补上 epoch 结束验证；
    若最后一步已经因 interval 触发过验证，则跳过避免重复。
    """

    def __init__(self):
        super().__init__()
        self._validated_on_last_batch = False

    def on_train_epoch_start(self, trainer, pl_module):
        self._validated_on_last_batch = False

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        # 嵌在 train epoch 内的验证：若正好落在最后一步，标记以免 epoch end 再跑
        try:
            if trainer.fit_loop.epoch_loop.batch_progress.is_last_batch:
                self._validated_on_last_batch = True
        except (AttributeError, RuntimeError):
            pass

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.enable_validation:
            return
        if trainer.val_dataloaders is None:
            return
        if self._validated_on_last_batch:
            self._validated_on_last_batch = False
            return

        # 与 TrainingEpochLoop.on_advance_end 中的验证路径保持一致
        trainer.validating = True
        first_loop_iter = trainer._logger_connector._first_loop_iter
        try:
            from lightning.pytorch.trainer import call
            call._call_lightning_module_hook(trainer, "on_validation_model_zero_grad")
            trainer.fit_loop.epoch_loop.val_loop.run()
        finally:
            trainer.training = True
            trainer._logger_connector._first_loop_iter = first_loop_iter


class CustomProgressBar(TQDMProgressBar):
    """
    自定义进度条，显示已处理样本数/总样本数。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        if not self.trainer.is_global_zero:
            bar.disable = True
        return bar
    
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not self.trainer.is_global_zero:
            bar.disable = True
        return bar

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        return metrics

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        # 优先显示精准的 Step 进度 (Step X/TotalSteps)
        max_steps = getattr(trainer, "max_steps", -1)
        if max_steps and max_steps > 0:
            self.train_progress_bar.set_description(
                f"Step {trainer.global_step}/{max_steps} (Epoch {trainer.current_epoch})"
            )
        elif hasattr(pl_module, 'train_ds'):
            total_samples = len(pl_module.train_ds)
            sampler_type = getattr(pl_module, 'sampler_type', None)
            if trainer.world_size > 1 and sampler_type != 'shar_pool_sampler':
                total_samples = total_samples // trainer.world_size
            
            processed = getattr(pl_module, 'epoch_processed_samples', 0)
            self.train_progress_bar.set_description(
                f"Epoch {trainer.current_epoch}: {processed}/{total_samples} samples"
            )
