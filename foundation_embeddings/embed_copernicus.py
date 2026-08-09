import numpy as np
import torch
from torchvision import transforms

from challenge_dataset import (
    S1GRD_MEAN,
    S1GRD_STD,
    S2L1C_MEAN,
    S2L1C_STD,
    SSL4EODownstreamDataset,
)

from tqdm import tqdm
import os
import sys
import time

copernicus_fm_repo_path = "./Copernicus-FM/Copernicus-FM"

# --- Add Copernicus-FM src to Python path ---
sys.path.append(copernicus_fm_repo_path)
sys.path.append(copernicus_fm_repo_path + "/src")
from src.model_vit import vit_base_patch16


if __name__ == "__main__":
    # --- Device Configuration ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Note: MPS support might vary
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Copernicus-FM Model Loading ---
    print("Loading Copernicus-FM model...")
    # Path relative to where the script is run (assuming setup instructions followed)
    copernicus_weights_path = (
        "./pretrained_models/CopernicusFM_ViT_base_varlang_e100.pth"
    )
    if not os.path.exists(copernicus_weights_path):
        print(f"Copernicus-FM weights not found at '{copernicus_weights_path}'")
        print("Ensure you ran the wget command from the setup instructions.")
        exit()

    try:
        # Initialize model as per official example
        copernicus_model = vit_base_patch16(num_classes=10, global_pool=False)
        check_point = torch.load(
            copernicus_weights_path, map_location="cpu"
        )  # Load to CPU first
        # Handle potential nesting in checkpoint file
        state_dict = check_point.get("model", check_point)
        msg = copernicus_model.load_state_dict(state_dict, strict=False)
        print("Copernicus-FM Load Message:", msg)
        copernicus_model.eval()
        copernicus_model.to(device)
        print("Copernicus-FM model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading Copernicus-FM model: {e}")
        exit()

    # --- Constants (Matching Copernicus-FM expectations) ---
    S1_WAVELENGTHS_NM = [50000000.0, 50000000.0]
    S2L1C_WAVELENGTHS_NM = [
        443,
        492,
        560,
        665,
        704,
        740,
        783,
        833,
        865,
        945,
        1374,
        1614,
        2190,
    ]  # L1C nm
    ALL_WAVELENGTHS_NM = S1_WAVELENGTHS_NM + S2L1C_WAVELENGTHS_NM

    S1_BANDWIDTHS_NM = [100.0, 100.0]
    S2L1C_BANDWIDTHS_NM = [
        21,
        66,
        36,
        31,
        15,
        15,
        20,
        106,
        21,
        20,
        31,
        91,
        175,
    ]  # L1C nm
    ALL_BANDWIDTHS_NM = S1_BANDWIDTHS_NM + S2L1C_BANDWIDTHS_NM

    N_CHANNELS = len(ALL_WAVELENGTHS_NM)  # 2 + 13 = 15
    N_SEASONS = 4
    MODALITIES_TO_LOAD = ["s1", "s2l1c"]  # Load both S1 and S2L1C
    COPERNICUS_DIM = 768  # Output dimension of the ViT embedding
    COPERNICUS_KERNEL_SIZE = 16  # Expected patch size for the ViT ('spectral' mode)
    INPUT_MODE = "spectral"  # We are using S1+S2 spectral data

    print(f"Expecting concatenated input with {N_CHANNELS} channels.")
    assert N_CHANNELS == 15, "Channel count mismatch"

    # --- Script Parameters ---
    path_to_data = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    n_samples_to_process = None

    print(f"Using data path: {path_to_data}")
    print(f"Raw temporal embeddings will be saved to: {output_file}")

    # --- Data Preparation ---
    # Normalization uses combined SSL4EO stats, applied AFTER dataset concatenates
    mean_combined = S1GRD_MEAN + S2L1C_MEAN
    std_combined = S1GRD_STD + S2L1C_STD
    data_transform = transforms.Compose(
        [transforms.Normalize(mean=mean_combined, std=std_combined)]
    )
    # Copernicus-FM uses ViT, requires fixed input size (e.g., 224x224)
    copernicus_preprocess = transforms.Resize((224, 224), antialias=True)

    # --- Generate Temporal Copernicus-FM Embeddings (4x768) ---
    temporal_embeddings = {}  # Dict: file_name -> np.array(4, 768)

    print("Initializing dataset...")
    try:
        # Load S1+S2 CONCATENATED
        dataset_e2s = SSL4EODownstreamDataset(
            path_to_data,
            modalities=MODALITIES_TO_LOAD,
            dataset_name="bands",
            transform=data_transform,
            concat=True,
            output_file_name=True,
            shift_s2_channels=True,
            seasons=N_SEASONS,
        )
        print(f"Length of dataset: {len(dataset_e2s)}")
    except FileNotFoundError:
        print(
            f"FATAL: Dataset not found at '{path_to_data}'. Please check 'path_to_data' variable."
        )
        exit()
    except Exception as e:
        print(f"FATAL: Failed to initialize dataset: {e}")
        exit()

    print("Starting temporal embedding generation...")
    failed_files_gen = []
    start_time_proc = time.time()

    dummy_meta_tensor = torch.full(
        (1, 4), float("nan"), device=device, dtype=torch.float32
    )

    for idx in tqdm(list(range(len(dataset_e2s))), desc="Processing samples"):
        file_name = "Unknown"
        try:
            data_file_dict = dataset_e2s.__getitem__(idx)
            # Expected shape: [1, N_SEASONS, N_CHANNELS, H, W] e.g., [1, 4, 15, 264, 264]
            data_tensor = data_file_dict["data"]
            file_name = data_file_dict["file_name"]

            # Basic shape check
            if not isinstance(data_tensor, torch.Tensor) or data_tensor.ndim != 5:
                print(
                    f"\nWarning: Invalid data format for index {idx}, file {file_name}. Expected 5D Tensor, got {type(data_tensor)}. Skipping."
                )
                failed_files_gen.append(file_name)
                continue
            if data_tensor.shape[1] != N_SEASONS or data_tensor.shape[2] != N_CHANNELS:
                print(
                    f"\nWarning: Unexpected data tensor shape for {file_name}: {data_tensor.shape}. Expected [1, {N_SEASONS}, {N_CHANNELS}, H, W]. Skipping."
                )
                failed_files_gen.append(file_name)
                continue

            embeddings_all_t = []  # Store [emb_t0, emb_t1, emb_t2, emb_t3]

            with torch.no_grad():
                for t in range(N_SEASONS):  # Loop through time steps
                    # Select time step t -> [1, N_CHANNELS, H, W] e.g. [1, 15, 264, 264]
                    img_t = data_tensor[:, t, :, :, :]

                    # Check if img_t is valid before moving to device
                    if img_t is None or not isinstance(img_t, torch.Tensor):
                        print(
                            f"\nWarning: Invalid image tensor for time step {t} in {file_name}. Skipping file."
                        )
                        embeddings_all_t = []  # Reset
                        if file_name not in failed_files_gen:
                            failed_files_gen.append(file_name)
                        break  # Stop processing time steps for this file

                    img_t = img_t.to(device)

                    # Resize for ViT
                    img_resized_t = copernicus_preprocess(
                        img_t
                    )  # Shape [1, 15, 224, 224]

                    if img_resized_t.shape[1] != N_CHANNELS:
                        print(
                            f"\nWarning: Channel mismatch after resize for {file_name}, time {t}. Got {img_resized_t.shape[1]}, expected {N_CHANNELS}. Skipping file."
                        )
                        embeddings_all_t = []  # Reset
                        if file_name not in failed_files_gen:
                            failed_files_gen.append(file_name)
                        break  # Break inner loop (skip this file)

                    logit, embed = copernicus_model(
                        img_resized_t,
                        dummy_meta_tensor,
                        ALL_WAVELENGTHS_NM,
                        ALL_BANDWIDTHS_NM,
                        None,
                        INPUT_MODE,
                        COPERNICUS_KERNEL_SIZE,
                    )

                    # embed shape should be [1, 768]
                    if embed.shape == (1, COPERNICUS_DIM):
                        embedding_t = embed.cpu().numpy().flatten()  # Shape (768,)
                        embeddings_all_t.append(embedding_t)
                    else:
                        print(
                            f"\nWarning: Unexpected embedding shape from Copernicus-FM for {file_name}, time {t}. Got {embed.shape}, expected {(1, COPERNICUS_DIM)}. Skipping file."
                        )
                        embeddings_all_t = []  # Reset
                        if file_name not in failed_files_gen:
                            failed_files_gen.append(file_name)
                        break  # Break inner loop

            # After processing all time steps, check if we got 4 embeddings
            if len(embeddings_all_t) == N_SEASONS:
                temporal_embeddings[file_name] = np.stack(
                    embeddings_all_t, axis=0
                )  # Shape (4, 768)
            # Only add to failed list if break occurred in time loop and not already added
            elif (
                file_name not in failed_files_gen and len(embeddings_all_t) != N_SEASONS
            ):
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
        f"\nFinished Copernicus-FM temporal embedding generation in {end_time_proc - start_time_proc:.2f} seconds."
    )
    print(
        f"Successfully generated {len(temporal_embeddings)} embeddings (shape 4x{COPERNICUS_DIM}). Failed: {len(failed_files_gen)}"
    )
    if failed_files_gen:
        print("Failed files/indices:", failed_files_gen)

    print(f"Saving Copernicus-FM temporal embeddings to {output_file}...")
    np.savez_compressed(
        output_file,
        embeddings=temporal_embeddings,
        failed_files=np.array(failed_files_gen, dtype=object),
    )
    print("Copernicus-FM Temporal embeddings saved.")
