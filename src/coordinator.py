import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Set

import numpy as np
import torch
from client import Client
from models import LSTM
from preprocessing import data_preprocessing
from torch import nn
from torchmetrics.regression import MeanAbsolutePercentageError
from utils import Timer, compute_model_size, weighted_avg

Algorithm = Literal["local", "gossip", "fedavg", "dac", "dac2"]
Device = Literal["cpu", "cuda"]


@dataclass
class Client_Logs:
    algorithm: Algorithm
    cidx: int
    sampled_nidxs: List[Set[int]] = field(default_factory=list)
    downloaded_bytes: List[int] = field(default_factory=list)
    train_evals: List[float] = field(default_factory=list)
    val_evals: List[float] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)
    agg_times: List[float] = field(default_factory=list)


class Coordinator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device: Device = args.device
        self.algorithm: Algorithm = args.algorithm
        self.num_layers = int(args.num_layers)
        self.hidden_size = int(args.hidden_size)
        self.sku_embedding_size = int(args.sku_embedding_size)
        self.lr = float(args.lr)
        self.epochs = int(args.epochs)
        self.test_ratio = float(args.test_ratio)
        self.num_rounds = int(args.num_rounds)
        self.num_warmup_epochs = int(args.num_warmup_epochs)
        self.client_logs: list[Client_Logs] = []

        if args.model == "lstm":
            self.model = LSTM
        else:
            msg = "Invalid model"
            raise ValueError(msg)

        if args.loss_fn == "mse":
            self.loss_fn = nn.MSELoss
        else:
            msg = "Invalid loss_fn"
            raise ValueError(msg)

        if args.eval_fn == "mape":
            self.eval_fn = MeanAbsolutePercentageError
        else:
            msg = "Invalid eval_fn"
            raise ValueError(msg)

        if args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD
        else:
            msg = "Invalid optimizer"
            raise ValueError(msg)

        if self.algorithm in ["gossip", "dac", "dac2", "fedavg"]:
            self.num_neighbors = int(args.num_neighbors)

        self.agg_func: Callable[[], None]
        if self.algorithm == "gossip":
            self.agg_func = self.agg_gossip
        elif self.algorithm == "fedavg":
            self.agg_func = self.agg_fedavg
        elif self.algorithm == "dac":
            self.agg_func = self.agg_dac
        elif self.algorithm == "dac2":
            self.agg_func = self.agg_dac2
        elif self.algorithm == "local":
            self.agg_func = self.agg_local
        else:
            msg = "Unknown algorithm"
            raise ValueError(msg)

        self.initialize_clients()

    def initialize_clients(self) -> None:
        df, input_columns, output_columns = data_preprocessing()
        client_indices: list[int] = sorted(df["store_idx"].unique())
        sku_indices: list[int] = sorted(df["sku_idx"].unique())
        sorted_weeks = sorted(df["week"].unique())
        split_week_idx = int((1 - self.test_ratio) * len(sorted_weeks))
        self.clients: list[Client] = []

        for cidx in client_indices:
            client_df = df[df["store_idx"] == cidx]
            client_df = client_df.sort_values(["sku_idx", "week"])
            client_in_df = client_df.pivot_table(
                index=["sku_idx", "week"],
                values=input_columns,
            )
            client_out_df = client_df.pivot_table(
                index=["sku_idx", "week"],
                values=output_columns,
            )

            # x: (n_skus, n_weeks, input_cols)
            # y: (n_skus, n_weeks, output_cols)
            x = client_in_df.to_xarray().to_array().values.transpose((1, 2, 0))
            y = client_out_df.to_xarray().to_array().values.transpose((1, 2, 0))
            x = torch.from_numpy(x.astype(np.float64)).to(self.device, torch.float)
            y = torch.from_numpy(y.astype(np.float64)).to(self.device, torch.float)
            x_train, y_train = x[:, :split_week_idx, :], y[:, :split_week_idx, :]
            x_val, y_val = x, y[:, split_week_idx:, :]
            client_sku_idx = np.sort(client_df["sku_idx"].unique())
            client_sku_idx = torch.from_numpy(client_sku_idx).to(self.device, torch.int)

            model = self.model(
                device=self.device,
                input_size=len(input_columns),
                output_size=len(output_columns),
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                sku_embedding_size=self.sku_embedding_size,
                num_sku_idx=len(sku_indices),
            ).to(self.device)
            loss_fn = self.loss_fn().to(self.device)
            eval_fn = self.eval_fn().to(self.device)
            optimizer = self.optimizer(model.parameters(), lr=self.lr)

            nidx2score, nidx2num_sampled = None, None
            if self.algorithm in ["gossip", "dac", "dac2"]:
                neighbor_indices = set(client_indices) - {cidx}
                nidx2score = {nidx: 0.0 for nidx in neighbor_indices}
                nidx2num_sampled = {nidx: 0 for nidx in neighbor_indices}

            client = Client(
                cidx=cidx,
                sku_idx=client_sku_idx,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                model=model,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                optimizer=optimizer,
                nidx2score=nidx2score,
                nidx2num_sampled=nidx2num_sampled,
            )
            client_logs = Client_Logs(
                algorithm=self.algorithm,
                cidx=cidx,
            )
            self.clients.append(client)
            self.client_logs.append(client_logs)

    def train(self) -> None:
        # Local warm-up.
        for t in range(self.num_warmup_epochs):
            # Local learning.
            for cidx, client in enumerate(self.clients):
                with Timer() as timer:
                    client.train_model(epochs=self.epochs)
                self.client_logs[cidx].training_times.append(timer.elapsed_time)

        # Federated learning.
        for t in range(self.num_rounds):
            # Global aggregation.
            self.agg_func()

            # Evaluation.
            for cidx, client in enumerate(self.clients):
                train_eval = client.eval_model(
                    model=client.model,
                    split="train",
                    loss_or_eval_fn="eval",
                )
                val_eval = client.eval_model(
                    model=client.model,
                    split="val",
                    loss_or_eval_fn="eval",
                )

                self.client_logs[cidx].train_evals.append(train_eval)
                self.client_logs[cidx].val_evals.append(val_eval)

            # Local learning.
            for cidx, client in enumerate(self.clients):
                with Timer() as timer:
                    client.train_model(epochs=self.epochs)
                self.client_logs[cidx].training_times.append(timer.elapsed_time)

    def agg_local(self) -> None:
        pass

    def agg_fedavg(self) -> None:
        sds: list[dict[str, Any]] = []
        weights: list[float] = []

        # Clients upload their models.
        for cidx, client in enumerate(self.clients):
            model = client.model
            sd = model.state_dict()
            sds.append(sd)
            weights.append(1.0)

            model_size = compute_model_size(model)
            self.client_logs[cidx].downloaded_bytes.append(model_size) # Server

        # Server aggregates the models.
        avg_sd = weighted_avg(sds, weights)

        # Clients download the aggregated model.
        for cidx, client in enumerate(self.clients):
            model = client.model
            model.load_state_dict(avg_sd)

            model_size = compute_model_size(model)
            self.client_logs[cidx].downloaded_bytes.append(model_size) # Client

    def agg_gossip(self) -> None:
        # Don't change models before end of round,
        # as we're pretending that this is a parallel loop.
        cidx2avg_sd: dict[int, dict[str, Any]] = {}
        for cidx, client in enumerate(self.clients):
            assert client.nidx2score is not None
            assert client.nidx2num_sampled is not None

            # Sample neighbors.
            sampled_nidxs = client.sample_neighbors(size=self.num_neighbors)

            # Aggregate the models.
            downloaded_bytes = 0
            with Timer() as timer:
                sds: list[dict[str, Any]] = []
                weights: list[float] = []
                for nidx in sampled_nidxs:
                    model = self.clients[nidx].model

                    sds.append(model.state_dict())
                    weights.append(1.0)

                    model_size = compute_model_size(model)
                    downloaded_bytes += model_size

                cidx2avg_sd[cidx] = weighted_avg(sds, weights)

            self.client_logs[cidx].sampled_nidxs.append(sampled_nidxs)
            self.client_logs[cidx].agg_times.append(timer.elapsed_time)
            self.client_logs[cidx].downloaded_bytes.append(downloaded_bytes)

        # Update the clients' models.
        for cidx, client in enumerate(self.clients):
            client.model.load_state_dict(cidx2avg_sd[cidx])

    def agg_dac(self) -> None:
        # Don't change models before end of round,
        # as we're pretending that this is a parallel loop.
        cidx2avg_sd: dict[int, dict[str, Any]] = {}
        for cidx, client in enumerate(self.clients):
            assert client.nidx2score is not None
            assert client.nidx2num_sampled is not None

            # Sample neighbors.
            sampled_nidxs = client.sample_neighbors(size=self.num_neighbors)

            self.client_logs[cidx].sampled_nidxs.append(sampled_nidxs)

            # Aggregate the models, weighted by training eval.
            downloaded_bytes = 0
            with Timer() as timer:
                sds: list[dict[str, Any]] = []
                weights: list[float] = []
                for nidx in sampled_nidxs:
                    model = self.clients[nidx].model

                    test_loss = client.eval_model(
                        model=model,
                        split="train",
                        loss_or_eval_fn="loss",
                    )
                    score = 1.0 / test_loss

                    sds.append(model.state_dict())
                    weights.append(1.0)
                    client.nidx2num_sampled[nidx] += 1
                    client.nidx2score[nidx] = score

                    model_size = compute_model_size(model)
                    downloaded_bytes += model_size

                cidx2avg_sd[cidx] = weighted_avg(sds, weights)

            self.client_logs[cidx].sampled_nidxs.append(sampled_nidxs)
            self.client_logs[cidx].agg_times.append(timer.elapsed_time)
            self.client_logs[cidx].downloaded_bytes.append(downloaded_bytes)

        # Update the clients' models.
        for cidx, client in enumerate(self.clients):
            client.model.load_state_dict(cidx2avg_sd[cidx])

    def agg_dac2(self) -> None:
        # Do DAC aggregation.
        self.agg_dac()

        # Add two-hop scores.
        for cidx, client in enumerate(self.clients):
            with Timer() as timer:
                client.add_twohop_dac(clients=self.clients)

            self.client_logs[cidx].agg_times[-1] += timer.elapsed_time
