from typing import Literal, Set

import numpy as np
import torch
from torch import nn

from src.models import LSTM
from src.utils import softmax_scale


class Client:
    def __init__(
        self,
        cidx: int,
        sku_idx: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        model: LSTM,
        loss_fn: nn.Module,
        eval_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        nidx2score: dict[int, float] | None = None,
        nidx2num_sampled: dict[int, int] | None = None,
    ) -> None:
        self.cidx = cidx
        self.sku_idx = sku_idx
        self.x_train = x_train  # (n_sku_ids, n_train_weeks, input_size)
        self.y_train = y_train  # (n_sku_ids, n_train_weeks, output_size)
        self.x_val = x_val  # (n_sku_ids, n_weeks, input_size)
        self.y_val = y_val  # (n_sku_ids, n_weeks, output_size)
        self.model = model
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.optimizer = optimizer
        self.nidx2score = nidx2score
        self.nidx2num_sampled = nidx2num_sampled

    def train_model(
        self,
        epochs: int,
    ) -> None:
        self.model.train()
        for _ in range(epochs):
            yhat = self.model.forward(x=self.x_train, sku_idx=self.sku_idx)
            # yhat: (n_skus, n_train_weeks, output_size)

            loss = self.loss_fn.forward(yhat, torch.log1p(self.y_train))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def eval_model(
        self,
        model: LSTM,
        split: Literal["train", "val"],
        loss_or_eval_fn: Literal["loss", "eval"],
    ) -> float:
        if split == "train":
            x = self.x_train
            y = self.y_train
        elif split == "val":
            x = self.x_val
            y = self.y_val

        if loss_or_eval_fn == "loss":
            fn = self.loss_fn
        elif loss_or_eval_fn == "eval":
            fn = self.eval_fn

        model.eval()

        yhat = model.forward(x=x, sku_idx=self.sku_idx)
        yhat_val = yhat[:, -y.size(1) :, :]
        # yhat and yhat_val: (n_sku_ids, n_weeks, output_size)

        score = fn.forward(yhat_val, torch.log1p(y))

        return score.item()

    def sample_neighbors(
        self,
        size: int,
    ) -> Set[int]:
        assert self.nidx2score is not None

        all_nidxs = list(self.nidx2score.keys())
        scores = softmax_scale(np.array([self.nidx2score[nidx] for nidx in all_nidxs]))
        sampled_nidxs = np.random.choice(all_nidxs, size=size, replace=False, p=scores)

        return set(sampled_nidxs)

    def add_twohop_dac(
        self,
        clients: list["Client"],
    ) -> None:
        assert self.nidx2num_sampled is not None
        assert self.nidx2score is not None

        # Find all neighbors of self
        nidxs = {
            nidx
            for nidx, num_sampled in self.nidx2num_sampled.items()
            if num_sampled > 0
        }
        nidx2neighbor = {
            neighbor.cidx: neighbor for neighbor in clients if neighbor.cidx in nidxs
        }

        # Find two-hop neighbors and their closest common neighbor with self
        nnidx2closest_nidx: dict[int, int] = {}
        for nidx, neighbor in nidx2neighbor.items():
            assert neighbor.nidx2num_sampled is not None

            nnidxs = {
                nnidx
                for nnidx, num_sampled in neighbor.nidx2num_sampled.items()
                if num_sampled > 0 and nnidx not in nidxs and nnidx != self.cidx
            }

            for nnidx in nnidxs:
                if nnidx not in nnidx2closest_nidx:
                    nnidx2closest_nidx[nnidx] = nidx
                    continue

                closest_nidx = nnidx2closest_nidx[nnidx]
                if self.nidx2score[closest_nidx] < self.nidx2score[nidx]:
                    nnidx2closest_nidx[nnidx] = nidx

        # Update scores of self for two-hop neighbors.
        # For each two-hop neighbor, Use the score of the closest common neighbor with self as a proxy.
        for nnidx, closest_nidx in nnidx2closest_nidx.items():
            closest_neighbor = nidx2neighbor[closest_nidx]
            assert closest_neighbor.nidx2score is not None
            self.nidx2score[nnidx] = closest_neighbor.nidx2score[nnidx]
