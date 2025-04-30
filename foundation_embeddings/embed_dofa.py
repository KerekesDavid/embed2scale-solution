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

if __name__ == "__main__":
    # --- Configuration ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- DOFA Model Loading ---
    print("Loading DOFA model...")
    try:
        dofa_model = torch.hub.load(
            "zhu-xlab/DOFA",
            "vit_base_dofa",
            pretrained=True,
        )
        dofa_model.eval()
        dofa_model.to(device)
        print("DOFA model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading DOFA model: {e}")
        exit()

    # --- Constants ---
    S1_WAVELENGTHS = [5.405, 5.405]
    S2L1C_WAVELENGTHS = [
        0.443,
        0.490,
        0.560,
        0.665,
        0.705,
        0.740,
        0.783,
        0.842,
        0.865,
        0.945,
        1.375,
        1.610,
        2.190,
    ]
    WAVELENGTHS_MAP = {"s1": S1_WAVELENGTHS, "s2l1c": S2L1C_WAVELENGTHS}
    MODALITIES_TO_PROCESS = ["s1", "s2l1c"]
    N_SEASONS = 4
    DOFA_DIM = 768
    COMBINED_DIM_PER_T = DOFA_DIM * len(MODALITIES_TO_PROCESS)  # 1536

    path_to_data = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    n_samples_to_process = None

    print(f"Using data path: {path_to_data}")
    print(f"Raw temporal embeddings will be saved to: {output_file}")

    # --- Data Preparation ---
    mean_s1 = S1GRD_MEAN_SSL4EO
    std_s1 = S1GRD_STD_SSL4EO
    mean_s2l1c = S2L1C_MEAN_SSL4EO
    std_s2l1c = S2L1C_STD_SSL4EO
    data_transform = transforms.Compose(
        [transforms.Normalize(mean=mean_s1 + mean_s2l1c, std=std_s1 + std_s2l1c)]
    )
    dofa_preprocess = transforms.Resize((224, 224), antialias=True)

    # --- Generate Temporal DOFA Embeddings (4x1536) ---
    temporal_embeddings = {}  # Dict: file_name -> np.array(4, 1536)

    print("Initializing dataset...")
    try:
        # Load S1 and S2 separately
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

    print("Starting temporal embedding generation...")
    failed_files_gen = []
    start_time_proc = time.time()

    for idx in tqdm(list(range(len(dataset_e2s))), desc="Processing samples"):
        file_name = "Unknown"
        try:
            data_file_name_dict = dataset_e2s[idx]
            data_dict = data_file_name_dict["data"]
            file_name = data_file_name_dict["file_name"]

            embeddings_all_t = []  # Store [emb_t0, emb_t1, emb_t2, emb_t3]

            with torch.no_grad():
                for t in range(N_SEASONS):  # Loop through time steps
                    embeddings_modalities_t = {}  # Embeddings for this time step t

                    for m in (
                        MODALITIES_TO_PROCESS
                    ):  # Loop through modalities ('s1', 's2l1c')
                        if m not in data_dict or data_dict[m] is None:
                            print(
                                f"\nWarning: Modality '{m}' missing for time {t} in {file_name}. Skipping file."
                            )
                            embeddings_all_t = []  # Reset for this file
                            break  # Stop processing this file

                        # Full tensor [1, N_SEASONS, C, H, W] e.g. [1, 4, 2, 264, 264] for s1
                        tensor_modality = data_dict[m].to(device)
                        # Select time step t -> [1, C, H, W]
                        img_t = tensor_modality[:, t, :, :, :]

                        # Preprocess and check wavelengths
                        img_resized_t = dofa_preprocess(img_t)  # [1, C, 224, 224]
                        wavelengths = WAVELENGTHS_MAP.get(m)
                        if (
                            wavelengths is None
                            or len(wavelengths) != img_resized_t.shape[1]
                        ):
                            print(
                                f"\nWarning: Wavelength mismatch for {m} ({len(wavelengths)} channels) vs image ({img_resized_t.shape[1]} channels) for {file_name}, time {t}. Skipping file."
                            )
                            embeddings_all_t = []
                            break  # Stop processing this file

                        # Extract features
                        features_t = dofa_model.forward_features(
                            img_resized_t, wave_list=wavelengths
                        )  # [1, 768]
                        embeddings_modalities_t[m] = (
                            features_t.cpu().numpy().flatten()
                        )  # (768,)

                    if not embeddings_all_t and len(embeddings_modalities_t) != len(
                        MODALITIES_TO_PROCESS
                    ):
                        # This check catches breaks from the inner modality loop
                        if file_name not in failed_files_gen:
                            failed_files_gen.append(file_name)
                        break  # Stop processing time steps for this file

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
                        )  # Shape (1536,)
                        embeddings_all_t.append(combined_embedding_t)
                    else:
                        print(
                            f"\nWarning: Missing S1 or S2L1C embedding after processing time {t} in {file_name}. Skipping file."
                        )
                        embeddings_all_t = []  # Reset
                        if file_name not in failed_files_gen:
                            failed_files_gen.append(file_name)
                        break  # Stop processing time steps

            # After processing all time steps, check if we got 4 embeddings
            if len(embeddings_all_t) == N_SEASONS:
                temporal_embeddings[file_name] = np.stack(
                    embeddings_all_t, axis=0
                )  # Shape (4, 1536)
            elif file_name not in failed_files_gen:
                print(
                    f"\nWarning: Did not generate {N_SEASONS} time steps for {file_name} (got {len(embeddings_all_t)}). Skipping file."
                )
                failed_files_gen.append(file_name)

        except Exception as e:
            current_id = file_name if file_name != "Unknown" else f"Index_{idx}"
            print(f"\nERROR processing sample {current_id}: {e}")
            import traceback

            traceback.print_exc()
            if current_id != "Unknown" and current_id not in failed_files_gen:
                failed_files_gen.append(current_id)

    end_time_proc = time.time()
    print(
        f"\nFinished DOFA temporal embedding generation in {end_time_proc - start_time_proc:.2f} seconds."
    )
    print(
        f"Successfully generated {len(temporal_embeddings)} embeddings (shape 4x{COMBINED_DIM_PER_T}). Failed: {len(failed_files_gen)}"
    )
    if failed_files_gen:
        print("Failed files/indices:", failed_files_gen)

    # Save intermediate temporal embeddings
    print(f"Saving DOFA temporal embeddings to {output_file}...")
    np.savez_compressed(
        output_file,
        embeddings=temporal_embeddings,
        failed_files=np.array(failed_files_gen, dtype=object),
    )
    print("DOFA Temporal embeddings saved.")
