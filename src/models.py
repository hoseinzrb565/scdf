import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        device: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_sku_idx: int,
        sku_embedding_size: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.sku_embedding = nn.Embedding(num_sku_idx, sku_embedding_size)

        self.lstm = nn.LSTM(
            input_size + sku_embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        sku_idx: torch.Tensor,
    ) -> torch.Tensor:
        # x: (bs, length, input_size)
        # sku_idx: (bs)
        length = x.size(1)

        sku_embedded = self.sku_embedding.forward(sku_idx)  # (bs, sku_embedding_size)

        sku_embedded_expanded = sku_embedded.unsqueeze(1).expand(
            -1, length, -1,
        )  # (bs, length, sku_embedding_size)
        x = torch.concat(
            [x, sku_embedded_expanded], dim=-1,
        )  # (bs, length, input_size+sku_embedding_size)

        hidden, _ = self.lstm.forward(x)  # (bs, length, hidden_size)
        return self.fc.forward(hidden)  # (bs, length, output_size)

