import numpy as np
import sklearn.decomposition
import pathlib
import pandas as pd
import torch
import sys

from tqdm import tqdm

from pangaea.datasets.e2s_embeddings import EmbeddingDataset


def main():
    model_paths = {"linear": pathlib.Path(sys.argv[1])}
    output_path = pathlib.Path(sys.argv[2])

    assert model_paths["linear"].exists()
    assert output_path.parent.exists()

    output_path.mkdir(exist_ok=True)

    dataset = EmbeddingDataset(
        bands={"linear": []},
        dataset_name="EmbeddingDataset",
        root_paths=model_paths,
        as_tensor=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=16)

    for data in tqdm(dataloader, desc="Computing embeddings"):
        embedding = data["image"]["linear"][3]
        file_id = data["metadata"]["file_name"]

        embedding_mean = embedding.mean((2, 3))
        # todo: argmax???
        embedding_std = embedding.std((2, 3))
        embedding = torch.concat((embedding_mean, embedding_std))

        np.savez_compressed(
            output_path / (file_id + "_embedding.npz"),
            embedding.numpy(force=True).astype(np.float16),
        )


if __name__ == "__main__":
    main()
