import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
from speechllm.zipformer.model import EncoderModel
from speechllm.zipformer.zipformer import Zipformer2
from speechllm.zipformer.subsampling import Conv2dSubsampling
from speechllm.zipformer.scaling import ScheduledFloat
from speechllm.icefall.utils import AttributeDict

def _to_int_tuple(s):
    if isinstance(s, int):
        return (s,)
    if isinstance(s, (list, tuple)):
        return tuple(map(int, s))
    if isinstance(s, str):
        return tuple(map(int, s.split(",")))
    return (int(s),)

def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.get("feature_dim", 80),
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed

def get_zipformer_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.get("downsampling_factor")),
        num_encoder_layers=_to_int_tuple(params.get("num_encoder_layers")),
        encoder_dim=_to_int_tuple(params.get("encoder_dim")),
        encoder_unmasked_dim=_to_int_tuple(params.get("encoder_unmasked_dim")),
        query_head_dim=int(params.get("query_head_dim", 32)),
        pos_head_dim=int(params.get("pos_head_dim", 4)),
        value_head_dim=int(params.get("value_head_dim", 12)),
        pos_dim=int(params.get("pos_dim", 48)),
        num_heads=_to_int_tuple(params.get("num_heads")),
        feedforward_dim=_to_int_tuple(params.get("feedforward_dim")),
        cnn_module_kernel=_to_int_tuple(params.get("cnn_module_kernel")),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=bool(params.get("causal", False)),
        chunk_size=_to_int_tuple(params.get("chunk_size", -1)),
        left_context_frames=_to_int_tuple(params.get("left_context_frames", -1)),
    )
    return encoder

def get_encoder(params: AttributeDict) -> nn.Module:
    encoder_embed = get_encoder_embed(params)
    encoder = get_zipformer_model(params)

    model = EncoderModel(
        encoder_embed=encoder_embed,
        encoder=encoder
    )

    # 尝试从未训练好的音频编码器加载权重
    audio_encoder_path = params.get("path", None)
    if audio_encoder_path and os.path.exists(audio_encoder_path):
        logging.info(f"正在从 {audio_encoder_path} 加载预训练音频编码器")
        checkpoint = torch.load(audio_encoder_path, map_location="cpu", weights_only=False)
        
        if "model_avg" in checkpoint:
            state_dict = checkpoint["model_avg"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.") or k.startswith("encoder_embed."):
                new_state_dict[k] = v
        
        if len(new_state_dict) == 0:
            logging.warning(f"在检查点 {audio_encoder_path} 中未找到编码器相关 key。找到的 key 包括: {list(state_dict.keys())[:5]}...")
        else:
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if len(missing) > 0:
                logging.warning(f"加载音频编码器时缺失 key: {missing}")
            if len(unexpected) > 0:
                logging.info(f"加载音频编码器时发现意外 key (如果不涉及编码器则可安全忽略): {unexpected}")
            
            logging.info(f"成功将 {len(new_state_dict)} 个 key 加载到音频编码器。")
    elif audio_encoder_path:
        logging.warning(f"警告: 未找到音频编码器路径 {audio_encoder_path}。")

    return model
