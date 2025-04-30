import torch
from challenge_dataset import E2SChallengeDataset
from challenge_dataset import collate_fn
from challenge_dataset import (
    S1GRD_MEAN_SSL4EO,
    S2L2A_MEAN_SSL4EO,
    S1GRD_STD_SSL4EO,
    S2L2A_STD_SSL4EO,
    S2L2A_MEAN,
    S2L2A_STD,
    S1GRD_MEAN,
    S1GRD_STD,
)
from use_croma import PretrainedCROMA
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    mean_data = S2L2A_MEAN + S1GRD_MEAN
    std_data = S2L2A_STD + S1GRD_STD

    data_transform = transforms.Compose(
        [
            # Add additional transformation here
            transforms.Normalize(mean=mean_data, std=std_data)
        ]
    )

    embeddings_path = Path("./work-dir/croma-eval-embeddings/")
    # path_to_data = (
    #     "/home/david/KTH/embed2scale-pangaea/data/SSL4EO-S12-downstream/data_dev"
    # )
    path_to_data = "/geoinfo_proj/Shared/SSL4EO-S12-downstream/data_eval"
    modalities = ["s2l2a", "s1"]

    embeddings_path.mkdir(exist_ok=True)

    dataset_e2s = E2SChallengeDataset(
        path_to_data,
        modalities=modalities,
        dataset_name="bands",
        transform=data_transform,
        concat=False,
        output_file_name=True,
        shift_s2_channels=False,
    )

    device = "cuda:0"  # use a GPU
    # device = "cpu"
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
