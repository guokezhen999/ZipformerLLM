from torch.optim.lr_scheduler import LambdaLR

def get_epoch_linear_schedule(optimizer, num_warmup_steps, max_epochs, get_current_epoch_fn, last_epoch=-1):
    """
    Step-based Warmup + Epoch-based Linear Decay.
    """
    def lr_lambda(current_step: int):
        # 1. Warmup factor (based on steps)
        if current_step < num_warmup_steps:
             warmup_factor = float(current_step) / float(max(1, num_warmup_steps))
        else:
             warmup_factor = 1.0
             
        # 2. Decay factor (based on epochs)
        current_epoch = get_current_epoch_fn()
        # Linear decay: 1 - epoch / max_epochs
        # Using max(0.0, ...) to ensure non-negative
        decay_factor = max(0.0, 1.0 - current_epoch / max_epochs)
        
        return min(warmup_factor, decay_factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_step_decay_schedule(optimizer, num_warmup_steps, lr_decay_step, lr_decay_factor, min_lr_ratio=0.0, last_epoch=-1):
    """
    Step-based Warmup + Step-based Discrete Decay.
    每个 lr_decay_step 步，学习率乘以 lr_decay_factor。
    """
    import math
    def lr_lambda(current_step: int):
        # 1. Warmup factor (based on steps)
        if current_step < num_warmup_steps:
             warmup_factor = float(current_step) / float(max(1, num_warmup_steps))
        else:
             warmup_factor = 1.0
             
        # 2. Step Decay factor
        # decay_count = (current_step - warmup_steps) // lr_decay_step
        # if current_step < warmup_steps, decay_factor = 1.0
        if current_step < num_warmup_steps:
            decay_factor = 1.0
        else:
            decay_count = (current_step - num_warmup_steps) // lr_decay_step
            decay_factor = lr_decay_factor ** decay_count
            
        return max(min_lr_ratio, min(warmup_factor, decay_factor))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_step_cosine_schedule(optimizer, num_warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1):
    """
    Step-based Warmup + Step-based Continuous Cosine Annealing Decay.
    使用总步数进行 Cosine 衰减。
    """
    import math
    def lr_lambda(current_step: int):
        # 1. Warmup factor (based on steps)
        if current_step < num_warmup_steps:
             warmup_factor = float(current_step) / float(max(1, num_warmup_steps))
        else:
             warmup_factor = 1.0
             
        # 2. Continuous Decay factor (based on steps)
        # 计算进度: [0.0, 1.0]
        # 注意：衰减通常是从 warmup 结束开始，或者从 step 0 开始。这里采用全局 step 进度。
        progress = min(1.0, float(current_step) / float(max(1, total_steps)))
        
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        decay_factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor
        
        return min(warmup_factor, decay_factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_sample_cosine_schedule(optimizer, num_warmup_steps, total_training_samples, get_processed_samples_fn, min_lr_ratio=0.0, last_epoch=-1):
    """
    Step-based Warmup + Sample-based Continuous Cosine Annealing Decay.
    使用已处理的样本数占总样本数的比例来计算衰减，实现更平滑的曲线。
    """
    import math
    def lr_lambda(current_step: int):
        # 1. Warmup factor (based on steps)
        if current_step < num_warmup_steps:
             warmup_factor = float(current_step) / float(max(1, num_warmup_steps))
        else:
             warmup_factor = 1.0
             
        # 2. Continuous Decay factor (based on samples)
        processed_samples = get_processed_samples_fn()
        # 计算全局进度: [0.0, 1.0]
        progress = min(1.0, float(processed_samples) / float(max(1, total_training_samples)))
        
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        decay_factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor
        
        return min(warmup_factor, decay_factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_lr_schedule(
    optimizer,
    train_conf, 
    num_warmup_steps,
    total_samples_on_this_rank,
    get_processed_samples_fn,
):
    import logging
    scheduler_conf = train_conf.get('scheduler', {})
    lr_scheduler_type = scheduler_conf.get('name', train_conf.get('lr_scheduler', 'sample_cosine'))
    min_lr_ratio = scheduler_conf.get('min_lr_ratio', train_conf.get('min_lr_ratio', 0.01))
    max_epochs = train_conf.get('max_epochs', 0)

    if lr_scheduler_type == 'step_decay':
        decay_step = scheduler_conf.get('decay_step', train_conf.get('lr_decay_step', 2000))
        decay_factor = scheduler_conf.get('decay_factor', train_conf.get('lr_decay_factor', 0.5))
        logging.info(f"Using step-based LR decay: step={decay_step}, factor={decay_factor}")
        return get_step_decay_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            lr_decay_step=decay_step,
            lr_decay_factor=decay_factor,
            min_lr_ratio=min_lr_ratio
        )
    elif lr_scheduler_type == 'step_cosine':
        # 基于总步数的 Cosine 衰减
        # 如果配置了 max_steps 则使用 max_steps，否则试图估计
        max_steps = train_conf.get('max_steps', -1)
        if max_steps == -1:
            # 估计总步数: (样本总数 / world_size / grad_accum) * epochs
            grad_accum = train_conf.get('grad_accumulate_steps', 1)
            max_steps = (total_samples_on_this_rank // grad_accum) * max_epochs
        
        logging.info(f"Using step-based cosine decay: total_steps={max_steps}")
        return get_step_cosine_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            total_steps=max_steps,
            min_lr_ratio=min_lr_ratio
        )
    elif lr_scheduler_type == 'sample_cosine':
        # Default to continuous cosine annealing decay (Sample-based)
        total_training_samples = total_samples_on_this_rank * max_epochs
        return get_sample_cosine_schedule(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            total_training_samples=total_training_samples,
            get_processed_samples_fn=get_processed_samples_fn,
            min_lr_ratio=min_lr_ratio
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")