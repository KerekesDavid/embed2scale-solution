import os
import pathlib
import random
import time

import numpy as np
import torch
import wandb
from torch.nn import functional as F
from tqdm import tqdm

from decoder import LinearDecoder
from embedding_datasets import (
    EmbeddingDataset,
    MergedEmbeddingDataset,
    SingleFileEmbeddingDataset,
)
from encoder import FFNEncoder


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    exp_dir: str,
    epochs: int,
    dataloader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_weights: dict[str, float],
    device: torch.device,
    eval_interval: int = 5,
    checkpoint_interval: int = 5,
) -> None:
    loss_weights = {k: torch.tensor(v, device=device) for k, v in loss_weights.items()}
    best_mse = float("inf")

    for epoch in tqdm(range(epochs), desc="Training", smoothing=0):
        loss_sums = {k: 0.0 for k in loss_weights.keys()}
        loss_sums["MSE"] = 0.0
        if epoch % eval_interval == 0:
            metrics = evaluate(model, val_loader, loss_weights, device)
            if metrics["MSE"] < best_mse:
                best_mse = metrics["MSE"]
                save_model(model, epoch, exp_dir, is_best=True)
        if epoch % checkpoint_interval == 0:
            save_model(model, epoch, exp_dir)

        # Train for one epoch
        model.train()
        for batch_idx, data in enumerate(dataloader):
            image, target = data["image"], data["target"]
            image = {modality: value.to(device) for modality, value in image.items()}
            target = {modality: value.to(device) for modality, value in target.items()}

            logits = model(image)
            loss, loss_items = compute_losses(logits, target, loss_weights, device)
            for k in loss_sums.keys():
                loss_sums[k] += loss_items[k]

            optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            del loss, logits, image, target, data, batch_idx
        tqdm.write(
            f"Epoch {epoch} | Training losses: \t"
            + "  ".join(
                (f"{k}: {i/len(dataloader):8.3g}" for k, i in loss_sums.items())
            )
        )
        wandb.log(
            {"train_loss_" + k: v / len(dataloader) for k, v in loss_sums.items()},
            commit=False,
        )
        wandb.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]})

    metrics = evaluate(model, val_loader, loss_weights, device)
    if metrics["MSE"] < best_mse:
        best_mse = metrics["MSE"]
        save_model(model, epoch, exp_dir, is_best=True)
    # save last model
    save_model(model, epoch, exp_dir, is_final=True)


def save_model(
    model: torch.nn.Module,
    epoch: int,
    exp_dir: str,
    is_final: bool = False,
    is_best: bool = False,
):
    suffix = "best" if is_best else f"{epoch}_final" if is_final else f"{epoch}"
    checkpoint_path = os.path.join(exp_dir, f"checkpoint_{suffix}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    tqdm.write(f"Epoch {epoch} | Checkpoint saved at {checkpoint_path}")


def compute_losses(
    logits: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    loss_weights: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_items = {}
    mse = torch.tensor(0.0, device=device)
    for k in target.keys():
        t_mse = F.mse_loss(logits[k], target[k])
        weighted_t_mse = t_mse * loss_weights[k]
        mse += weighted_t_mse

        loss_items[k] = weighted_t_mse.item()

    loss_items["MSE"] = mse.item()

    return mse, loss_items


@torch.no_grad()
def evaluate(model, data_loader, loss_weights, device, model_ckpt_path=None):
    if model_ckpt_path is not None:
        model_dict = torch.load(model_ckpt_path, map_location=device)
        model.load_state_dict(model_dict)

        tqdm.write(f"Loaded model from {model_ckpt_path} for evaluation")

    model.eval()

    mses = {}

    for data in tqdm(data_loader, desc="Evaluating"):
        image, target = data["image"], data["target"]
        image = {k: v.to(device) for k, v in image.items()}
        target = {k: v.to(device) for k, v in target.items()}

        logits = model(image)
        for k in target.keys():
            mse = F.mse_loss(logits[k], target[k])
            if k in mses:
                mses[k] += mse
            else:
                mses[k] = mse

    mses = {k: mse.item() / len(data_loader) for k, mse in mses.items()}

    metrics = {f"{k}": mse for k, mse in mses.items()}
    metrics["MSE"] = sum([loss_weights[k] * v for k, v in mses.items()])
    wandb.log({"eval_loss_" + k: v for k, v in metrics.items()}, commit=False)
    tqdm.write(
        "Evaluation losses: \t"
        + "  ".join((f"{k}: {metric:8.3g}" for k, metric in metrics.items()))
    )

    return metrics


def fix_seeds(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(
    batch: dict[str, dict[str, torch.Tensor] | torch.Tensor],
) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    out_batch = {}
    out_batch["image"] = {
        modality: torch.stack([x["image"][modality] for x in batch])
        for modality in batch[0]["image"].keys()
    }

    if "target" in batch[0]:
        out_batch["target"] = {
            modality: torch.stack([x["target"][modality] for x in batch])
            for modality in batch[0]["target"].keys()
        }

    if "metadata" in batch[0]:
        out_batch["metadata"] = [x["metadata"] for x in batch]

    return out_batch


def extract(
    model: torch.nn.Module,
    checkpoint_name: pathlib.Path,
    experiment_directory: pathlib.Path,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
):
    checkpoint_path = experiment_directory / checkpoint_name

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        model_dict = torch.load(checkpoint_path, map_location=device)
        encoder_dict = {
            k[8:]: v for k, v in model_dict.items() if k.startswith("encoder")
        }
        model.load_state_dict(encoder_dict)
        print(f"Loaded model from checkpoint: {checkpoint_path}")

        model.eval()

        embeddings_path = experiment_directory / "embeddings"
        embeddings_path.mkdir(exist_ok=True)

        for batch in tqdm(dataloader, desc="Extracting"):
            image = batch["image"]
            metadatas = batch["metadata"]
            image = {k: v.to(device) for k, v in image.items()}

            embeddings = model(image)
            for batch_index, md in enumerate(metadatas):
                file_name = md["file_name"]
                np.savez_compressed(
                    embeddings_path / (file_name + "_embedding.npz"),
                    embeddings[batch_index].numpy(force=True).astype(np.float16),
                )


def main() -> None:
    seed = 1337
    batch_size = 32
    val_batch_size = 8
    num_workers = 8
    val_num_workers = 8
    data_directory = pathlib.Path("/geoinfo_proj/Shared/embed2scale-embeddings")
    experiment_dir_prefix = "experiments"

    lr = 1e-4
    loss_weights = {
        "copernicus": 1.0,
        "dofa": 1.0,
        "croma": 0.5,
        "prithvi": 2.0,
        "scalemae": 0.75,
    }
    n_epochs = 160
    lr_milestones = [0.6, 0.9]

    experiment_name = (
        "linear_embed_cdfpy"
        + f"_{lr}_{n_epochs}_"
        + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    )
    experiment_directory = pathlib.Path(experiment_dir_prefix) / experiment_name
    experiment_directory.mkdir(exist_ok=True)

    wandb.init(
        project="embed2scale",
        name=experiment_name,
        resume="allow",
        config={
            "seed": seed,
            "lr": lr,
            "loss_weights": loss_weights,
            "n_epochs": n_epochs,
            "lr_milestones": lr_milestones,
            "batch_size": batch_size,
        },
    )

    fix_seeds(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        torch.cpu.set_device(device)

    encoder = FFNEncoder(
        input_size=[[4, 768], [4, 1536], [4, 1024], [1, 1536], [4, 1536]],
        input_dropout=torch.nn.Dropout(0.1),
        encoder_weights=loss_weights,
    )

    decoder = LinearDecoder(encoder)
    decoder.to(device)

    train_dataset = MergedEmbeddingDataset(
        datasets=[
            SingleFileEmbeddingDataset(
                path=data_directory
                / "copernicus-fm/copernicusfm_concat_temporal_raw_4x768_eval.npz",
                dataset_key="copernicus",
            ),
            SingleFileEmbeddingDataset(
                path=data_directory / "dofa/dofa_temporal_raw_4x1536_eval.npz",
                dataset_key="dofa",
            ),
            EmbeddingDataset(
                root_paths={"croma": data_directory / "croma-eval-embeddings"}
            ),
            SingleFileEmbeddingDataset(
                path=data_directory / "prithvi2/prithvi2_1x1536_eval.npz",
                dataset_key="prithvi",
            ),
            SingleFileEmbeddingDataset(
                path=data_directory / "scalemae/scalemae_temporal_data_eval_4x1536.npz",
                dataset_key="scalemae",
            ),
        ]
    )
    # train_dataset = torch.utils.data.Subset(train_dataset, [i for i in range(64)])
    val_dataset = MergedEmbeddingDataset(
        datasets=[
            SingleFileEmbeddingDataset(
                path=data_directory
                / "copernicus-fm/copernicusfm_concat_temporal_raw_4x768_dev.npz",
                dataset_key="copernicus",
            ),
            SingleFileEmbeddingDataset(
                path=data_directory / "dofa/dofa_temporal_raw_4x1536_dev.npz",
                dataset_key="dofa",
            ),
            EmbeddingDataset(
                root_paths={"croma": data_directory / "croma-dev-embeddings"}
            ),
            SingleFileEmbeddingDataset(
                path=data_directory / "prithvi2/prithvi2_1x1536_dev.npz",
                dataset_key="prithvi",
            ),
            SingleFileEmbeddingDataset(
                path=data_directory / "scalemae/scalemae_temporal_data_dev_4x1536.npz",
                dataset_key="scalemae",
            ),
        ]
    )
    # val_dataset = torch.utils.data.Subset(val_dataset, [i for i in range(64)])
    print(f"Length of training dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")

    # get train val data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        params=decoder.parameters(),
        lr=lr,
        weight_decay=0.05,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        [int(len(train_loader) * n_epochs * r) for r in lr_milestones],
        gamma=0.1,
    )

    train(
        model=decoder,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        exp_dir=experiment_directory,
        epochs=n_epochs,
        dataloader=train_loader,
        val_loader=val_loader,
        loss_weights=loss_weights,
        device=device,
    )

    extract(encoder, "checkpoint_best.pth", experiment_directory, train_dataset, device)

    wandb.finish()


if __name__ == "__main__":
    main()
