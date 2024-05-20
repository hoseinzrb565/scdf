import argparse
import dataclasses
import json
import os

import torch

from src.coordinator import Client_Logs, Coordinator

parser = argparse.ArgumentParser()

parser.add_argument("--num_seeds", type=int, default=1)
parser.add_argument(
    "--algorithm",
    type=str,
    choices=["local", "gossip", "fedavg", "dac", "dac2"],
    default="local",
)
parser.add_argument("--num_neighbors", type=int, default=5)
parser.add_argument("--num_warmup_epochs", type=int, default=0)
parser.add_argument("--num_rounds", type=int, default=100)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--loss_fn", type=str, choices=["mse"], default="mse")
parser.add_argument("--eval_fn", type=str, choices=["mape"], default="mape")
parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="sgd")
parser.add_argument("--model", type=str, choices=["lstm"], default="lstm")
parser.add_argument("--hidden_size", type=int, default=16)
parser.add_argument("--sku_embedding_size", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cuda" if torch.cuda.is_available() else "cpu",
)

args = parser.parse_args()

run_logs: list[list[Client_Logs]] = []
for run_idx in range(int(args.num_seeds)):
    # Train
    coordinator = Coordinator(args)
    coordinator.train()

    # Save logs
    run_path = f"logs/{args.algorithm}/run_{run_idx}/"
    os.makedirs(run_path, exist_ok=True)
    for cidx, client_logs in enumerate(coordinator.client_logs):
        with open(run_path + f"{cidx}.json", "w") as f:
            json.dump(dataclasses.asdict(obj=client_logs), f, default=str)
