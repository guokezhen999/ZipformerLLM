# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from speechllm.icefall.utils import make_pad_mask

class EncoderModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module, # Conv2dSubsampling
        encoder: nn.Module,       # Zipformer
    ):
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder

    # 训练只用这个！
    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, T, C)
        x_lens: (N,)
        """
        # 1. 前端卷积
        x, x_lens = self.encoder_embed(x, x_lens)
        
        # 2. 生成标准的 Padding Mask (训练用)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # 3. Encoder 计算
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)
        return encoder_out, encoder_out_lens
    
class StreamingEncoderWrapper(nn.Module):
    """
    专门用于流式推理的包装器。
    它负责管理 State，负责拼接 Mask，负责处理 Context Overlap。
    """
    def __init__(self, model: EncoderModel):
        super().__init__()
        # 直接复用训练好的模块
        self.encoder = model.encoder
        self.encoder_embed = model.encoder_embed
        
        # 读取配置
        self.chunk_size = self.encoder.chunk_size[0]
        self.left_context_len = self.encoder.left_context_frames[0]

    def get_init_states(self, batch_size=1, device="cpu"):
        """生成初始状态列表"""
        # 1. Zipformer 内部状态
        states = self.encoder.get_init_states(batch_size, device)
        
        # 2. 前端卷积缓存
        embed_states = self.encoder_embed.get_init_states(batch_size, device)
        states.append(embed_states)
        
        # 3. 计数器 (用来造 Mask)
        processed_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        states.append(processed_lens)
        
        return states

    def forward(self, x: torch.Tensor, states: List[torch.Tensor]):
        """
        流式推理 Forward。
        x: (N, T, C) -> 这里的 T = chunk_size + pad_length (包含上下文)
        states: 状态列表
        """
        # --- 1. 准备数据 ---
        N = x.size(0)
        # 这里不需要 x_lens 参数，因为流式输入是固定长度
        # 但底层接口可能需要，我们造一个假的
        x_lens = torch.tensor([x.size(1)] * N, device=x.device)
        
        # --- 2. 解包状态 ---
        # 倒数第1个是计数器
        processed_lens = states[-1]
        # 倒数第2个是卷积缓存
        cached_embed_left_pad = states[-2]
        # 剩下的是 Zipformer 状态
        zipformer_states = states[:-2]

        # --- 3. 前端卷积 (Streaming) ---
        # 调用底层的 streaming_forward
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=x,
            x_lens=x_lens,
            cached_left_pad=cached_embed_left_pad,
        )
        # 此时 x 的长度应该是 chunk_size

        # --- 4. 构造复杂的 Mask (核心逻辑) ---
        src_key_padding_mask = torch.zeros(N, self.chunk_size, dtype=torch.bool, device=x.device)
        
        # 构造历史 Mask
        hist_mask = torch.arange(self.left_context_len, device=x.device).expand(N, self.left_context_len)
        hist_mask = (processed_lens.unsqueeze(1) <= hist_mask).flip(1)
        
        # 拼接
        src_key_padding_mask = torch.cat([hist_mask, src_key_padding_mask], dim=1)

        # --- 5. Encoder 推理 (Streaming) ---
        x = x.permute(1, 0, 2)
        (
            encoder_out,
            encoder_out_lens,
            new_zipformer_states,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=zipformer_states,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)

        # --- 6. 更新状态 ---
        new_processed_lens = processed_lens + x_lens
        
        new_states = new_zipformer_states + [
            new_cached_embed_left_pad,
            new_processed_lens,
        ]

        return encoder_out, new_states

class StreamingEncoderAdapter(nn.Module):
    """
    流式推理适配器。
    它包装了 StreamingEncoderWrapper 和 Connector (Projector)。
    """
    def __init__(self, model: EncoderModel, connector: nn.Module):
        super().__init__()
        self.streaming_encoder = StreamingEncoderWrapper(model)
        self.connector = connector

    def get_init_states(self, batch_size=1, device="cpu"):
        """生成初始状态列表"""
        return self.streaming_encoder.get_init_states(batch_size, device)

    def forward(self, x: torch.Tensor, states: List[torch.Tensor]):
        """
        流式推理 Forward。
        x: (N, T, C)
        states: 状态列表
        """
        # 1. Encoder 推理
        encoder_out, new_states = self.streaming_encoder(x, states)
        connector_dtype = next(self.connector.parameters()).dtype
        if encoder_out.dtype != connector_dtype:
            encoder_out = encoder_out.to(dtype=connector_dtype)
        # 2. Connector (Projector) 推理
        # encoder_out: (N, T, C)
        projector_out = self.connector(encoder_out)
        
        return projector_out, new_states