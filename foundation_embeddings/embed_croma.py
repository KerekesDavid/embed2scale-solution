import pathlib
import sys

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from challenge_dataset import (
    S1GRD_MEAN,
    S1GRD_STD,
    S2L2A_MEAN,
    S2L2A_STD,
    E2SChallengeDataset,
    collate_fn,
)
from use_croma import PretrainedCROMA


def main():
    data_path = pathlib.Path(sys.argv[1])
    embeddings_path = pathlib.Path(sys.argv[2])

    assert data_path.exists()
    assert embeddings_path.parent.exists()

    mean_data = S2L2A_MEAN + S1GRD_MEAN
    std_data = S2L2A_STD + S1GRD_STD

    data_transform = transforms.Compose(
        [transforms.Normalize(mean=mean_data, std=std_data)]
    )

    modalities = ["s2l2a", "s1"]

    embeddings_path.mkdir(exist_ok=True)

    dataset_e2s = E2SChallengeDataset(
        data_path,
        modalities=modalities,
        dataset_name="bands",
        transform=data_transform,
        concat=False,
        output_file_name=True,
        shift_s2_channels=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        torch.cpu.set_device(device)

    model = PretrainedCROMA(
        pretrained_path="./pretrained_models/CROMA_large.pt",
        size="large",
        modality="both",
        image_resolution=264,
    ).to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset_e2s, batch_size=8, num_workers=16, collate_fn=collate_fn
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch["data"]
            file_name = batch["file_name"]

            outputs = []

            for temp_index in range(data["s1"].shape[1]):
                outputs.append(
                    model(
                        SAR_images=data["s1"][:, temp_index, ...].to(device),
                        optical_images=data["s2l2a"][:, temp_index, ...].to(device),
                    )["joint_GAP"]
                )

            outputs = torch.stack(outputs, dim=2)

            for batch_index, fn in enumerate(file_name):
                np.savez_compressed(
                    embeddings_path / (fn + "_embedding.npz"),
                    outputs[batch_index].numpy(force=True).astype(np.float16),
                )


if __name__ == "__main__":
    main()
