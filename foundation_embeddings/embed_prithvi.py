import os
import sys
import time

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from challenge_dataset import (
    S1GRD_MEAN_SSL4EO,
    S1GRD_STD_SSL4EO,
    S2L1C_MEAN_SSL4EO,
    S2L1C_STD_SSL4EO,
    E2SChallengeDataset,
)
from prithvi_mae import PrithviMAE

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    try:
        model = PrithviMAE()

        # Load manually downloaded pre-trained weights
        checkpoint_path = "./pretrained_models/Prithvi_EO_V2_600M.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.encoder.load_state_dict(
            checkpoint, strict=False
        )  # Set strict=False in case of missing keys
        model.eval()  # Set to evaluation mode
        model.to(device)
        print("model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading model: {e}")
        exit()

    path_to_data = sys.argv[1]
    output_file = sys.argv[2]

    # Script Parameters
    os.mkdir(os.path.dirname(output_file), exist_ok=True)
    save_temporal_embeddings = True
    print(f"Using data path: {path_to_data}")
    print(f"Temporal embeddings will be saved to (if enabled): {output_file}")

    # Embedding dimensions
    MODALITIES_TO_PROCESS = ["s1", "s2l1c"]
    N_SEASONS = 4
    n_samples_to_process = None

    # Data Preparation
    mean_s1 = S1GRD_MEAN_SSL4EO
    std_s1 = S1GRD_STD_SSL4EO
    mean_s2l1c = S2L1C_MEAN_SSL4EO
    std_s2l1c = S2L1C_STD_SSL4EO
    data_transform = transforms.Compose(
        [transforms.Normalize(mean=mean_s1 + mean_s2l1c, std=std_s1 + std_s2l1c)]
    )
    data_preprocess = transforms.Resize((256, 256), antialias=True)

    temporal_embeddings = {}  # Dict: file_name -> np.array(1, 1536)

    print("Initializing dataset for temporal embedding generation...")
    try:
        dataset_e2s = E2SChallengeDataset(
            path_to_data,
            modalities=MODALITIES_TO_PROCESS,
            dataset_name="bands",
            transform=data_transform,
            concat=False,
            output_file_name=True,
            shift_s2_channels=True,
            seasons=N_SEASONS,
        )
        print(f"Length of dataset: {len(dataset_e2s)}")
    except Exception as e:
        print(f"FATAL: Failed to initialize dataset: {e}")
        exit()

    indices_to_process = list(range(len(dataset_e2s)))
    if n_samples_to_process is not None:
        indices_to_process = indices_to_process[
            : min(n_samples_to_process, len(dataset_e2s))
        ]
    print(f"Will process {len(indices_to_process)} samples for temporal embeddings.")

    print("Starting temporal embedding generation...")
    failed_files_gen = []
    start_time_proc = time.time()

    for idx in tqdm(
        indices_to_process, desc="Processing samples for temporal embeddings"
    ):
        if idx > 10:
            break
        try:
            data_file_name = dataset_e2s[idx]
            data_dict = data_file_name["data"]
            file_name = data_file_name["file_name"]

            img_all_t = []
            with torch.no_grad():
                for t in range(N_SEASONS):  # Loop through time steps
                    embeddings_modalities_t = {}  # Embeddings for this time step t

                    for m in MODALITIES_TO_PROCESS:  # Loop through modalities
                        if m not in data_dict:
                            continue  # Skip if modality missing
                        tensor = data_dict[m].to(device)  # Full tensor [1, 4, C, H, W]
                        if m == "s2l1c":
                            C1 = tensor[:, t, 2:5, :, :]
                            C2 = tensor[:, t, 8:9, :, :]
                            C3 = tensor[:, t, 11:, :, :]
                            img_t = torch.cat([C1, C2, C3], dim=1)
                        else:
                            continue
                        img_resized_t = data_preprocess(img_t)
                    img_all_t.append(img_resized_t)
                img_all_t = torch.stack(img_all_t)
                img_all_t = img_all_t.permute(1, 2, 0, 3, 4)
                temporal_coords, location_coords = [], []

                feature_list = model.forward_features(
                    img_all_t, temporal_coords, location_coords
                )  # 12x1025x768
                stacked_tensor = torch.stack(feature_list)[11:, :, :]
                stacked_tensor = stacked_tensor.squeeze()
                features_mean = stacked_tensor.mean(dim=0)
                features_std = stacked_tensor.std(dim=0)
                combined_embedding = torch.cat(([features_mean, features_std]), dim=0)
                combined_embedding = combined_embedding.unsqueeze(0)
                temporal_embeddings[file_name] = combined_embedding.cpu().numpy()

        except Exception as e:
            current_file_name = (
                f"Index_{idx}" if "file_name" not in locals() else file_name
            )
            print(f"\nERROR processing sample {current_file_name}: {e}")
            failed_files_gen.append(current_file_name)

    # Save intermediate temporal embeddings
    if save_temporal_embeddings:
        print(f"Saving temporal embeddings to {output_file}...")
        np.savez_compressed(
            output_file,
            embeddings=temporal_embeddings,
            failed_files=np.array(failed_files_gen, dtype=object),
        )  # Use object array for strings
        print("Temporal embeddings saved.")

