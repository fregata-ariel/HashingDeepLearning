from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import time
from collections.abc import Iterator, Sequence
from typing import Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from mongoose_reformer.reformer_lib.generative_tools import TrainingWrapper
from mongoose_reformer.reformer_lib.reformer_pytorch import ReformerLM_tune


class TrainArgs(Protocol):
    dataset: str
    seq_len: int
    min_seq_len: int
    ntokens: int
    emsize: int
    nhid: int
    nlayers: int
    nhead: int
    bucket_size_list: list[int]
    n_hashes_list: list[int]
    attn_type_list: list[str]
    dropout: float
    batch_size: int
    full_attn_thres: int
    lr_main: float
    lr_tri: float
    tri_alpha: float
    use_full_attn: bool
    log: bool
    epochs: int
    train_batches: int
    eval_batches: int
    print_loss: int
    note: str
    scheduler_hashes: int
    thresh: float
    local_rank: int
    device: str
    distributed: bool
    gpu: int
    world_size: int


Array: TypeAlias = npt.NDArray[np.generic]
Batch: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
BatchIterator: TypeAlias = Iterator[Batch]

seed = 17
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="synthetic")
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--min_seq_len", type=int, default=4)
parser.add_argument("--ntokens", type=int, default=16)
parser.add_argument("--emsize", type=int, default=256)
parser.add_argument("--nhid", type=int, default=256)
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument(
    "--bucket_size_list", nargs="+", type=int, default=[64, 64]
)
parser.add_argument(
    "--n_hashes_list", nargs="+", type=int, default=[1, 1]
)
parser.add_argument(
    "--attn_type_list", nargs="+", default=["triplet", "triplet"]
)
parser.add_argument("--dropout", type=float, default=0.05)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--full_attn_thres", type=int, default=0)
parser.add_argument("--lr_main", type=float, default=1e-3)
parser.add_argument("--lr_tri", type=float, default=1e-3)
parser.add_argument("--tri_alpha", type=float, default=1.0)
parser.add_argument("--use_full_attn", action="store_true")
parser.add_argument("--log", action="store_false")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--train_batches", type=int, default=5000)
parser.add_argument("--eval_batches", type=int, default=500)
parser.add_argument("--print_loss", type=int, default=500)
parser.add_argument("--note", type=str, default="")
parser.add_argument("--scheduler_hashes", type=int, default=10)
parser.add_argument("--thresh", type=float, default=0.01)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--device", choices=["auto", "cpu", "cuda"], default="auto"
)

GRADIENT_ACCUMULATE_EVERY = 1
args = cast(TrainArgs, parser.parse_args())

if args.device == "auto":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
elif args.device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but CUDA is unavailable"
        )
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

APEX_AVAILABLE = importlib.util.find_spec("apex") is not None
print("training device:", device)
print("APEX available:", APEX_AVAILABLE)

args.distributed = False
if "WORLD_SIZE" in os.environ:
    args.distributed = int(os.environ["WORLD_SIZE"]) > 1
args.gpu = 0
args.world_size = 1

if args.distributed:
    rank = int(os.environ["RANK"])
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)
        torch.distributed.init_process_group(backend="nccl")
    else:
        torch.distributed.init_process_group(backend="gloo")
    args.world_size = torch.distributed.get_world_size()

writer: SummaryWriter | None = None
log_file: str | None = None
if args.log and args.local_rank == 0:
    log_dir = f"./log_paper/{args.dataset}"
    log_file = (
        log_dir
        + "/{}_bucket{}_hash{}_seq{}_bz{}_token{}_lr{}_alpha{}_layer{}_note{}.txt".format(
            "_".join(args.attn_type_list),
            "_".join(str(x) for x in args.bucket_size_list),
            "_".join(str(x) for x in args.n_hashes_list),
            args.seq_len,
            args.batch_size,
            args.ntokens,
            args.lr_tri,
            args.tri_alpha,
            args.nlayers,
            args.note,
        )
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as handle:
        print("args: ", args, file=handle)
    print("args: ", args)
    run_file = (
        "./run_paper/{}/{}_bucket{}_hash{}_seq{}_bz{}_token{}_lr{}_alpha{}_layer{}_note{}".format(
            args.dataset,
            "_".join(args.attn_type_list),
            "_".join(str(x) for x in args.bucket_size_list),
            "_".join(str(x) for x in args.n_hashes_list),
            args.seq_len,
            args.batch_size,
            args.ntokens,
            args.lr_tri,
            args.tri_alpha,
            args.nlayers,
            args.note,
        )
    )
    writer = SummaryWriter(run_file)


def save_model(
    model: nn.Module,
    optimizer: Optimizer,
    name: str,
    iteration: int,
) -> None:
    with open(
        os.path.join(
            "synthetic", f"model_{name}_{iteration}.pt"
        ),
        "wb",
    ) as handle:
        torch.save(model.state_dict(), handle)
    with open(
        os.path.join(
            "synthetic", f"optimizer_{name}_{iteration}.pt"
        ),
        "wb",
    ) as handle:
        torch.save(optimizer.state_dict(), handle)


def _pad_to_multiple_of(
    array: Array, multiple: int, axis: int
) -> Array:
    """Pad an ndarray to a multiple on one axis."""
    pad_len = (
        np.ceil(array.shape[axis] / float(multiple))
        * multiple
    )
    pad_widths = [(0, 0)] * len(array.shape)
    pad_widths[axis] = (
        0,
        int(pad_len - array.shape[axis]),
    )
    return cast(
        Array,
        np.pad(
            array,
            pad_widths,
            mode="constant",
            constant_values=array.dtype.type(0),
        ),
    )


def sequence_copy_inputs(
    vocab_size: int,
    batch_size: int,
    train_length: int,
    eval_min_length: int,
    eval_max_length: int,
    reverse: bool = False,
    pad_to_multiple: int = 32,
) -> tuple[BatchIterator, BatchIterator]:
    """Create infinite train/eval streams for the synthetic copy task."""

    def random_minibatches(
        length_list: Sequence[int],
    ) -> BatchIterator:
        while True:
            length = random.choice(length_list)
            if length % 2 != 0:
                raise ValueError("copy-task sequence length must be even")
            w_length = (length // 2) - 1
            w = np.random.randint(
                low=1,
                high=vocab_size - 1,
                size=(batch_size, w_length),
            )
            zero = np.zeros([batch_size, 1], np.int32)
            loss_weights = np.concatenate(
                [
                    np.zeros((batch_size, w_length + 2)),
                    np.ones((batch_size, w_length)),
                ],
                axis=1,
            )
            if reverse:
                x = np.concatenate(
                    [zero, w, zero, np.flip(w, axis=1)],
                    axis=1,
                )
            else:
                x = np.concatenate(
                    [zero, w, zero, w], axis=1
                )

            padded_x = _pad_to_multiple_of(
                cast(Array, x), pad_to_multiple, 1
            )
            padded_weights = _pad_to_multiple_of(
                cast(Array, loss_weights),
                pad_to_multiple,
                1,
            )
            x_tensor = torch.from_numpy(
                np.asarray(padded_x)
            ).to(device=device, dtype=torch.long)
            weight_tensor = torch.from_numpy(
                np.asarray(padded_weights)
            ).to(device=device, dtype=torch.long)
            yield x_tensor, x_tensor, weight_tensor

    train_lengths = [
        2 * (index + 2)
        for index in range(train_length - 1)
    ]
    eval_lengths = [
        2 * (index + 1)
        for index in range(eval_min_length, eval_max_length)
    ]
    return (
        random_minibatches(train_lengths),
        random_minibatches(eval_lengths),
    )


def _tunable_model(
    model: TrainingWrapper,
) -> ReformerLM_tune:
    wrapped = model.net.net
    if not isinstance(wrapped, ReformerLM_tune):
        raise TypeError("expected TrainingWrapper(ReformerLM_tune)")
    return wrapped


def train(
    model: nn.Module,
    base_model: TrainingWrapper,
    train_loader: BatchIterator,
    epoch: int,
    optimizer_main: Optimizer,
    optimizer_tri: Optimizer | None,
) -> None:
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch in range(args.train_batches):
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            data, _target, loss_mask = next(train_loader)
            if "triplet" in args.attn_type_list:
                loss = cast(
                    torch.Tensor,
                    model(
                        data,
                        loss_weight=loss_mask,
                        return_loss=True,
                        calc_triplet=True,
                    ),
                )
                tunable = _tunable_model(base_model)
                tri_loss = tunable.get_triplet_loss()
                if isinstance(tri_loss, torch.Tensor):
                    torch.autograd.backward(tri_loss)
                    tunable.clear_triplet_loss()
            else:
                loss = cast(
                    torch.Tensor,
                    model(
                        data,
                        loss_weight=loss_mask,
                        return_loss=True,
                        calc_triplet=False,
                    ),
                )
            torch.autograd.backward(loss)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.5
        )
        optimizer_main.step()
        optimizer_main.zero_grad()

        if (
            "triplet" in args.attn_type_list
            and optimizer_tri is not None
        ):
            optimizer_tri.step()
            optimizer_tri.zero_grad()

        total_loss += float(loss.item())
        log_interval = args.print_loss
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            message = (
                "| epoch {:3d} | {:5d}/{:5d} batches | "
                "lr {:02.5f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}"
            ).format(
                epoch,
                batch,
                args.train_batches,
                args.lr_main,
                elapsed * 1000 / log_interval,
                cur_loss,
                math.exp(cur_loss),
            )
            if args.local_rank == 0:
                print(message)
            if (
                args.log
                and args.local_rank == 0
                and log_file is not None
            ):
                with open(
                    log_file, "a", encoding="utf-8"
                ) as handle:
                    print(message, file=handle)
            total_loss = 0.0
            if (
                args.log
                and args.local_rank == 0
                and writer is not None
            ):
                step = epoch * args.train_batches + batch
                writer.add_scalar(
                    "Loss/train", cur_loss, step
                )
                writer.add_scalar(
                    "Loss/train_pp",
                    math.exp(cur_loss),
                    step,
                )
            start_time = time.time()


def evaluate(
    eval_model: nn.Module,
    data_source: BatchIterator,
    data_batches: int,
    epoch: int,
) -> float:
    del epoch
    eval_model.eval()
    total_loss = 0.0
    counter = 0.0
    with torch.no_grad():
        for _ in range(data_batches):
            data, _target, loss_mask = next(data_source)
            loss = cast(
                torch.Tensor,
                eval_model(
                    data,
                    loss_weight=loss_mask,
                    return_loss=True,
                    calc_triplet=False,
                ),
            )
            valid_token = float(
                torch.sum(loss_mask).item()
            )
            total_loss += valid_token * float(loss.item())
            counter += valid_token
    return total_loss / counter


def main() -> None:
    train_loader, eval_loader = sequence_copy_inputs(
        args.ntokens,
        args.batch_size,
        args.seq_len // 2,
        args.min_seq_len,
        args.seq_len // 2,
    )

    language_model = ReformerLM_tune(
        dim=args.nhid,
        emb_dim=args.emsize,
        depth=args.nlayers,
        max_seq_len=args.seq_len,
        num_tokens=args.ntokens,
        heads=args.nhead,
        bucket_size_list=args.bucket_size_list,
        fixed_position_emb=True,
        n_hashes_list=args.n_hashes_list,
        ff_chunks=1,
        ff_mult=1,
        attn_chunks=1,
        layer_dropout=0.0,
        ff_dropout=args.dropout,
        post_attn_dropout=args.dropout,
        lsh_dropout=args.dropout,
        weight_tie=False,
        causal=True,
        n_local_attn_heads=0,
        use_full_attn=args.use_full_attn,
        reverse_thres=9999999999,
        full_attn_thres=args.full_attn_thres,
        num_mem_kv=0,
        attn_type_list=args.attn_type_list,
        store_stats=args.log,
        pkm_num_keys=0,
        scheduler_hashes=args.scheduler_hashes,
        thresh=args.thresh,
    )

    base_model = TrainingWrapper(language_model)
    base_model.to(device)

    params_main: list[nn.Parameter] = []
    params_tri: list[nn.Parameter] = []
    for name, parameter in base_model.named_parameters():
        if "rotation" in name:
            params_tri.append(parameter)
        else:
            params_main.append(parameter)

    optimizer_main = torch.optim.Adam(
        params_main
        if "triplet" in args.attn_type_list
        else list(base_model.parameters()),
        lr=args.lr_main,
    )
    optimizer_tri: Optimizer | None = None
    if "triplet" in args.attn_type_list:
        optimizer_tri = torch.optim.Adam(
            params_tri, lr=args.lr_tri
        )

    train_model: nn.Module = base_model
    if args.distributed:
        if device.type == "cuda":
            train_model = DDP(
                base_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )
        else:
            train_model = DDP(base_model)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train(
            train_model,
            base_model,
            train_loader,
            epoch,
            optimizer_main,
            optimizer_tri,
        )
        val_loss = evaluate(
            train_model,
            eval_loader,
            args.eval_batches,
            epoch,
        )

        if (
            args.log
            and args.local_rank == 0
            and writer is not None
        ):
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar(
                "Loss/val_pp", math.exp(val_loss), epoch
            )

        message = (
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid loss {:5.2f} | valid ppl {:8.2f}"
        ).format(
            epoch,
            time.time() - epoch_start_time,
            val_loss,
            math.exp(val_loss),
        )
        print("-" * 89)
        print(message)
        print("-" * 89)

        if (
            args.log
            and args.local_rank == 0
            and log_file is not None
        ):
            with open(
                log_file, "a", encoding="utf-8"
            ) as handle:
                print(message, file=handle)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
