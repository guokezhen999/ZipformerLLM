import torch
from torch import nn

def get_connector(adapter_name, adapter_conf):
    if adapter_name == 'pooling-adapter':
        return PoolingAdapter(
            input_dim=adapter_conf['input_dim'], 
            hidden_dim=adapter_conf['hidden_dim'], 
            output_dim=adapter_conf['llm_dim'], 
            num_layers=adapter_conf.get('num_layers', 2),
            pooling=adapter_conf.get('pooling', 'cat'),
            pooling_factor=adapter_conf['pooling_factor'],
            dropout=adapter_conf.get('dropout', 0.1),
        )
    else:
        raise NotImplementedError(f"Adapter {adapter_name} not implemented or supported in simplified connector.")

class ConcatPooling(nn.Module):
    """
    将序列维度进行拼接降采样。
    输入: (batch, seq_len, dim) -> 输出: (batch, seq_len // factor, dim * factor)
    """
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        # x: (B, L, D)
        batch, seq_len, dim = x.shape
        # 确保序列长度可以被 factor 整除，多余的截断
        new_len = seq_len // self.factor
        x = x[:, :new_len * self.factor, :]
        # 重新排列并拼接特征维度
        x = x.reshape(batch, new_len, dim * self.factor)
        return x

class PoolingAdapter(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers: int = 2,
        activation: str = "relu",
        pooling: str = "cat",
        pooling_factor: int = 5,
        dropout: float = 0.1,  # <--- 新增参数：默认 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim else input_dim
        self.num_layers = num_layers
        self.activation = activation
        self.pooling = pooling
        self.pooling_factor = pooling_factor

        if num_layers == 1:
            self.hidden_dim = output_dim

        # 第一层降维/特征融合
        if pooling == "cat":
            self.preprocess = nn.Sequential(
                ConcatPooling(pooling_factor), 
                nn.Linear(self.input_dim * self.pooling_factor, self.hidden_dim)
            )
        else:
            self.preprocess = nn.Sequential(
                nn.AvgPool1d(pooling_factor, stride=pooling_factor), 
                nn.Linear(input_dim, self.hidden_dim)
            )
        
        # 构建后续的隐藏层
        layers = []
        for _ in range(self.num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))  # <--- 注入 Dropout
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        # 最后一层输出层
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))      # <--- 注入 Dropout
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.projector = nn.Sequential(*layers)
    
    def forward(self, audio_signal):
        outputs = self.preprocess(audio_signal)
        outputs = self.projector(outputs)
        return outputs