import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import concurrent.futures


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        root_paths: dict[str, str],
        as_tensor: bool = True,
        extra_target_paths: list[str] = [],
    ):
        self.root_paths = root_paths
        self.as_tensor = as_tensor
        self.extra_target_paths = extra_target_paths

        for p in root_paths.values():
            assert os.path.exists(p), f"No directory named {p}"

        sample_list_per_model = {
            m: sorted(glob.glob(os.path.join(p, "embeddings", "*.npz")))
            for m, p in self.root_paths.items()
        }

        self.models = sorted(root_paths.keys())

        self.samples = {}
        for model, samples_list in sample_list_per_model.items():
            for s in samples_list:
                scene_id = os.path.basename(s).split("_")[0]
                if scene_id in self.samples:
                    self.samples[scene_id][model].append(s)
                else:
                    self.samples[scene_id] = {k: [s] for k in self.root_paths.keys()}

        self.target_samples = [
            sorted(glob.glob(os.path.join(p, "*.npz"))) for p in self.extra_target_paths
        ]

        for sps in self.target_samples:
            assert len(sps) == len(self.samples)

        self.index_to_id = sorted(self.samples.keys())
        self.executor = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.executor:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        s2e_id = self.index_to_id[idx]
        sample_paths_per_model = self.samples[s2e_id]
        extra_target_paths = [sp[idx] for sp in self.target_samples]

        embeddings = {}

        for model, sample_paths in sample_paths_per_model.items():
            results = self.executor.map(np.load, sample_paths)
            data_list = list(results)
            del results
            if self.as_tensor:
                data_list = [torch.from_numpy(d["arr_0"]) for d in data_list]
                embeddings[model] = torch.concat(data_list).type(torch.float32)
                del data_list
            else:
                embeddings[model] = [d["arr_0"] for d in data_list]

        target = embeddings

        results = self.executor.map(np.load, extra_target_paths)
        extra_target_data = list(results)
        extra_target_data = [torch.from_numpy(d["arr_0"]) for d in extra_target_data]
        if extra_target_data:
            target["extra"] = extra_target_data

        output = {
            "image": embeddings,
            "target": target,
            "metadata": {"file_name": s2e_id},
        }

        return output


class SingleFileEmbeddingDataset(Dataset):
    def __init__(
        self,
        path: str,
        dataset_key: str,
    ):
        self.path = path
        self.dataset_key = dataset_key

        data_dict = np.load(self.path, allow_pickle=True)
        self.data = data_dict["embeddings"].item()
        assert not data_dict["failed_files"]
        self.idx_to_id = sorted(list(self.data.keys()))

    def __len__(self):
        return len(self.idx_to_id)

    def __getitem__(self, idx):
        s2e_id = self.idx_to_id[idx]

        embeddings = {
            self.dataset_key: torch.from_numpy(self.data[s2e_id]).transpose(
                0, 1
            )  # t c -> c t
        }
        target = {
            self.dataset_key: torch.from_numpy(self.data[s2e_id]).transpose(
                0, 1
            )  # t c -> c t
        }

        output = {
            "image": embeddings,
            "target": target,
            "metadata": {"file_name": s2e_id},
        }

        return output


class MergedEmbeddingDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset],
    ):
        self.datasets = datasets

        for d in self.datasets:
            assert len(d) == len(self.datasets[0])

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        output = {
            "image": {},
            "target": {},
            "metadata": {},
        }

        for ds in self.datasets:
            data = ds[idx]
            if "filename" in output["metadata"]:
                assert data["metadata"]["filename"] == output["metadata"]["file_name"]
            for k in data.keys():
                output[k] = data[k] | output[k]
        return output
