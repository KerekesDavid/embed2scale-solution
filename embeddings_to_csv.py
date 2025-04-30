import pathlib
import sys

import pandas as pd
import torch
from tqdm import tqdm

from embedding_datasets import EmbeddingDataset


def create_submission_from_dict(emb_dict):
    """Assume dictionary has format {hash-id0: embedding0, hash-id1: embedding1, ...}"""
    df_submission = pd.DataFrame.from_dict(emb_dict, orient="index")

    # Reset index with name 'id'
    df_submission.index.name = "id"
    df_submission.reset_index(drop=False, inplace=True)

    return df_submission


def main():
    model_paths = {"linear": pathlib.Path(sys.argv[1])}
    output_path = pathlib.Path(sys.argv[2])

    assert model_paths["linear"].exists()
    assert output_path.parent.exists()

    output_path.parent.mkdir(exist_ok=True)

    dataset = EmbeddingDataset(
        root_paths=model_paths,
        as_tensor=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=16)

    embed_dict = {}
    for data in tqdm(dataloader, desc="Computing embeddings"):
        embedding = data["image"]["linear"][0].numpy()
        file_id = data["metadata"]["file_name"]

        embed_dict[file_id] = embedding.flatten()

    submission_df = create_submission_from_dict(embed_dict)
    submission_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
