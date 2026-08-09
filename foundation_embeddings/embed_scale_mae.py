import os
import sys
import time

import numpy as np
import torch
from torchgeo.models import ScaleMAE
from torchvision import transforms
from tqdm import tqdm

from challenge_dataset import (
    S1GRD_MEAN,
    S1GRD_STD,
    S2L1C_MEAN,
    S2L1C_STD,
    SSL4EODownstreamDataset,
)

# --- Main Execution Block ---
if __name__ == "__main__":
    # device Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    try:
        model = ScaleMAE()
        checkpoint_path = "./pretrained_models/scalemae-vitlarge-800.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()  # Set to evaluation mode
        model.to(device)
        print("model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading model: {e}")
        exit()

    path_to_data = sys.argv[1]
    output_file = sys.argv[2]

    # Script Parameters
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_temporal_embeddings = True
    print(f"Using data path: {path_to_data}")
    print(f"Temporal embeddings will be saved to (if enabled): {output_file}")

    # Embedding dimensions
    MODALITIES_TO_PROCESS = ["s1", "s2l1c"]
    N_SEASONS = 4
    OUT_DIM = 768
    COMBINED_DIM_PER_T = OUT_DIM * len(MODALITIES_TO_PROCESS)

    n_samples_to_process = None

    # Data Preparation
    mean_s1 = S1GRD_MEAN
    std_s1 = S1GRD_STD
    mean_s2l1c = S2L1C_MEAN
    std_s2l1c = S2L1C_STD
    data_transform = transforms.Compose(
        [transforms.Normalize(mean=mean_s1 + mean_s2l1c, std=std_s1 + std_s2l1c)]
    )
    data_preprocess = transforms.Resize((224, 224), antialias=True)

    temporal_embeddings = {}  # Dict: file_name -> np.array(4, 1536)

    print("Initializing dataset for temporal embedding generation...")
    try:
        dataset_e2s = SSL4EODownstreamDataset(
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
        try:
            data_file_name = dataset_e2s[idx]
            data_dict = data_file_name["data"]
            file_name = data_file_name["file_name"]
            embeddings_all_t = []  # Store [emb_t0, emb_t1, emb_t2, emb_t3]

            with torch.no_grad():
                for t in range(N_SEASONS):  # Loop through time steps
                    embeddings_modalities_t = {}  # Embeddings for this time step t

                    for m in MODALITIES_TO_PROCESS:  # Loop through modalities
                        if m not in data_dict:
                            continue  # Skip if modality missing
                        tensor = data_dict[m].to(device)  # Full tensor [1, 4, C, H, W]
                        if m == "s1":
                            img_temp = tensor[:, t, :, :, :]
                            C1 = img_temp[:, 0:1, :, :]
                            img_t = torch.cat([img_temp, C1], dim=1)
                        else:
                            C1 = tensor[:, t, 7:8, :, :]
                            C2 = tensor[:, t, 3:4, :, :]
                            C3 = tensor[:, t, 2:3, :, :]
                            img_t = torch.cat([C1, C2, C3], dim=1)

                        img_resized_t = data_preprocess(img_t)
                        features = model.forward_features(img_resized_t)
                        features_t = features.mean(dim=1)
                        embeddings_modalities_t[m] = features_t.cpu().numpy().flatten()

                    # Concatenate modalities for this time step t
                    if (
                        "s1" in embeddings_modalities_t
                        and "s2l1c" in embeddings_modalities_t
                    ):
                        combined_embedding_t = np.concatenate(
                            [
                                embeddings_modalities_t["s1"],
                                embeddings_modalities_t["s2l1c"],
                            ]
                        )
                        embeddings_all_t.append(combined_embedding_t)
                    else:
                        # Handle missing modality for a time step -> append zeros or skip file
                        print(
                            f"\nWarning: Missing S1 or S2L1C for time {t} in {file_name}. Appending zeros."
                        )
                        embeddings_all_t.append(np.zeros(COMBINED_DIM_PER_T))

            # After processing all time steps, check if we got 4 embeddings
            if len(embeddings_all_t) == N_SEASONS:
                temporal_embeddings[file_name] = np.stack(
                    embeddings_all_t, axis=0
                )  # Shape (4, 1536)
            else:
                print(
                    f"\nWarning: Did not generate {N_SEASONS} time steps for {file_name}. Skipping file."
                )
                failed_files_gen.append(file_name)

        except Exception as e:
            current_file_name = (
                f"Index_{idx}" if "file_name" not in locals() else file_name
            )
            print(f"\nERROR processing sample {current_file_name}: {e}")
            failed_files_gen.append(current_file_name)

    end_time_proc = time.time()
    print(
        f"\nFinished temporal embedding generation in {end_time_proc - start_time_proc:.2f} seconds."
    )
    print(
        f"Successfully generated {len(temporal_embeddings)} temporal embeddings. Failed: {len(failed_files_gen)}"
    )
    if failed_files_gen:
        print("First 10 failed files/indices:", failed_files_gen[:10])

    # Save temporal embeddings
    if save_temporal_embeddings:
        print(f"Saving temporal embeddings to {output_file}...")
        np.savez_compressed(
            output_file,
            embeddings=temporal_embeddings,
            failed_files=np.array(failed_files_gen, dtype=object),
        )  # Use object array for strings
        print("Temporal embeddings saved in", output_file)
