import sys
import pathlib
import random

import numpy as np
import torch
from tqdm import tqdm

from embedding_datasets import (
    EmbeddingDataset,
    MergedEmbeddingDataset,
    SingleFileEmbeddingDataset,
)
from encoder import FFNEncoder


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
    checkpoint_name: str,
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
        if "model" in model_dict:
            model_dict = model_dict["model"]
        encoder_dict = {
            k[8:]: v for k, v in model_dict.items() if k.startswith("encoder")
        }
        model.load_state_dict(encoder_dict)
        print(f"Loaded model from checkpoint: {checkpoint_path}")

        model.eval()

        embeddings_dir = experiment_directory / "all_checkpoint_extracts"
        embeddings_dir.mkdir(exist_ok=True)
        embeddings_path = embeddings_dir / pathlib.Path(checkpoint_name).stem
        embeddings_path.mkdir(exist_ok=True)
        embeddings_path = embeddings_path / "embeddings"
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
    data_directory = pathlib.Path("./data/")
    experiment_directory = pathlib.Path(sys.argv[1])
    assert experiment_directory.exists(), f"{experiment_directory} doesn't exist"
    loss_weights = {
        "copernicus": 1.0,
        "dofa": 1.0,
        "croma": 0.5,
        "prithvi": 2.0,
        "scalemae": 0.75,
    }

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
    encoder.to(device)

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

    checkpoints = sorted(experiment_directory.glob("*.pth"))

    for checkpoint_path in checkpoints:
        extract(
            encoder, checkpoint_path.name, experiment_directory, train_dataset, device
        )


if __name__ == "__main__":
    main()
