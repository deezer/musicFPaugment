import json
import os
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from training.parameters import (
    dcase_2017_dev_path,
    dcase_2017_eval_path,
    dcase_2018_dev_path,
    dcase_2018_eval_path,
    dcase_2020_dev_path,
    dcase_2020_eval_path,
)


def get_dcase2017(path: str) -> Any:
    """
    Get DCASE 2017 filenames from metadata.
    """
    dataset_metas = os.path.join(path, "meta.txt")
    data = pd.read_csv(dataset_metas, sep="\t", header=None, engine="python")
    data = data.rename(columns={0: "filename", 1: "scene_label", 2: "location"})

    return data


def get_dcase2018_2020(path: str, type: str = "dev") -> Any:
    """
    Get DCASE 2018 or 2020 datasets.
    """
    if type == "dev":
        dataset_metas = os.path.join(path, "meta.csv")
        data = pd.read_csv(dataset_metas, sep=",|\t", header=0, engine="python")
        data = data.rename(columns={"identifier": "location", "source_label": "device"})

    elif type == "eval":
        if "2020" in path:
            dataset_metas = os.path.join(path, "evaluation_setup/fold1_test.csv")
            data = pd.read_csv(dataset_metas, sep=",|\t", header=0, engine="python")

        else:
            dataset_metas = os.path.join(path, "evaluation_setup/test.txt")
            data = pd.read_csv(dataset_metas, sep=",|\t", header=None, engine="python")
            data = data.rename(columns={0: "filename"})

    return data


def get_dcase_union() -> Any:
    """
    Get union of DCASE datasets.
    """
    dcase_dev_2017 = NoiseDataset(name="dcase2017", type="dev")
    dcase_dev_2017.dataset["filename"] = (
        dcase_2017_dev_path + "/" + dcase_dev_2017.dataset["filename"]
    )

    dcase_eval_2017 = NoiseDataset(name="dcase2017", type="eval")
    dcase_eval_2017.dataset["filename"] = (
        dcase_2017_eval_path + "/" + dcase_eval_2017.dataset["filename"]
    )

    dcase_dev_2018 = NoiseDataset(name="dcase2018", type="dev")
    device_a = [
        {"column": "device", "operator": "equals", "value": "a"},
    ]
    dcase_dev_2018.filter_dataset(device_a)
    dcase_dev_2018.dataset.drop(columns=["device"], inplace=True)
    dcase_dev_2018.dataset["filename"] = (
        dcase_2018_dev_path + "/" + dcase_dev_2018.dataset["filename"]
    )

    dcase_eval_2018 = NoiseDataset(name="dcase2018", type="eval")
    dcase_eval_2018.dataset["filename"] = (
        dcase_2018_eval_path + "/" + dcase_eval_2018.dataset["filename"]
    )

    dcase_dev_2020 = NoiseDataset(name="dcase2020", type="dev")
    dcase_dev_2020.filter_dataset(device_a)
    dcase_dev_2020.dataset.drop(columns=["device"], inplace=True)
    dcase_dev_2020.dataset["filename"] = (
        dcase_2020_dev_path + "/" + dcase_dev_2020.dataset["filename"]
    )

    # We will need to remove these
    dcase_eval_2020 = NoiseDataset(name="dcase2020", type="eval")
    dcase_eval_2020.dataset["filename"] = (
        dcase_2020_eval_path + "/" + dcase_eval_2020.dataset["filename"]
    )

    data = (
        pd.concat(
            [
                dcase_dev_2017.dataset,
                dcase_eval_2017.dataset,
                dcase_dev_2018.dataset,
                dcase_eval_2018.dataset,
                dcase_dev_2020.dataset,
                dcase_eval_2020.dataset,
            ]
        )
        .reset_index()
        .drop(columns=["index"])
    )

    return data


class NoiseDataset:
    """
    Custom pytorch Dataset class.
    Used to load Background noise dataset.
    """

    def __init__(self, name: str, type: str = "dev") -> None:
        """
        name (str):
            Dataset name, possible choices: "dcase2017",  "dcase2018", "dcase2020", "union".
        type (str, optional):
            "eval" or "dev".
        """

        self.name = name
        self.type = type
        if self.name == "dcase2017":
            if self.type == "dev":
                self.path = dcase_2017_dev_path
            elif self.type == "eval":
                self.path = dcase_2017_eval_path
            self.dataset = get_dcase2017(self.path)

        if self.name == "dcase2018":
            if self.type == "dev":
                self.path = dcase_2018_dev_path
            elif self.type == "eval":
                self.path = dcase_2018_eval_path
            self.dataset = get_dcase2018_2020(self.path, type=self.type)

        if self.name == "dcase2020":
            if self.type == "dev":
                self.path = dcase_2020_dev_path
            if self.type == "eval":
                self.path = dcase_2020_eval_path
            self.dataset = get_dcase2018_2020(self.path, type=self.type)

        if self.name == "union":
            self.path = "union"
            self.dataset = get_dcase_union()

        self.size = len(self.dataset)
        self.columns = self.dataset.columns

    def balance_dataset(self) -> None:
        """
        Used to balance dataset pandas dataframe.
        """
        balanced_dataset = self.dataset.groupby("scene_label")
        balanced_dataset = balanced_dataset.apply(
            lambda x: x.sample(balanced_dataset.size().min(), random_state=42)
        ).reset_index(drop=True)
        self.dataset = balanced_dataset

    def drop(self, column: str, label: Any) -> None:
        """
        Drop rows from self.dataset pandas Dataframe where "column" has "label" value
        """
        self.dataset = self.dataset[self.dataset[column] != str(label)]

    def filter_dataset(self, conditions: List[Dict[str, Any]]) -> None:
        """
        Filter self.dataset pandas dataframe based on conditions.

        How to write a condition:
           conditions = [
                {
                    "column" : "scene_label",
                    'operator' : 'equals',
                    'value' : "bus"
                },
            ]
        """

        filtered_dataset = self.dataset

        for condition in conditions:
            if condition["operator"] == "equals":
                filtered_dataset = filtered_dataset.loc[
                    filtered_dataset[condition["column"]] == condition["value"]
                ]

            elif condition["operator"] == "superior":
                filtered_dataset = filtered_dataset.loc[
                    filtered_dataset[condition["column"]] > condition["value"]
                ]

            elif condition["operator"] == "inferior":
                filtered_dataset = filtered_dataset.loc[
                    filtered_dataset[condition["column"]] < condition["value"]
                ]

        self.dataset = filtered_dataset

    def get_classes(self) -> Any:
        """
        Get dataset classes.
        """
        self.classes = self.dataset.scene_label.unique()
        return self.classes

    def get_nb_samples_per_class(self) -> Any:
        """
        Number of samples per class.
        """
        self.samples_per_class = (
            self.dataset.groupby("scene_label").count()["filename"].to_dict()
        )
        return self.samples_per_class

    def get_file_paths_per_class(self) -> Dict[str, List[str]]:
        self.dataset = self.dataset.dropna()

        file_paths: Dict[str, List[str]] = {}

        for _, row in self.dataset.iterrows():
            scene_label = row["scene_label"]
            filename = row["filename"]

            if scene_label in file_paths:
                file_paths[scene_label].append(filename)
            else:
                file_paths[scene_label] = [filename]

        self.file_paths = file_paths

        return self.file_paths

    def train_val_test_split(
        self, num_val: int = 20, num_test: int = 100
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        train_paths = {}
        val_paths = {}
        test_paths = {}

        # Iterate through each scene label
        for scene_label, filenames in self.file_paths.items():
            # Shuffle the filenames to randomize the split
            random.shuffle(filenames)

            # Split the filenames
            val_paths[scene_label] = filenames[:num_val]
            test_paths[scene_label] = filenames[num_val : num_test + num_val]
            train_paths[scene_label] = filenames[num_test + num_val :]

        return train_paths, val_paths, test_paths

    def sample_from_class(self, scene_label: str, n: int) -> List[Any]:
        """
        Sample n random files from a scene label.
        """
        files = list(
            self.dataset.loc[self.dataset["scene_label"] == str(scene_label)][
                "filename"
            ]
        )
        samples = random.sample(files, n)
        return samples


if __name__ == "__main__":
    noiseDataset = NoiseDataset("union")
    noiseDataset.drop("scene_label", "metro")
    noiseDataset.get_file_paths_per_class()
    train_paths, val_paths, test_paths = noiseDataset.train_val_test_split()

    with open("training/splits/train.json", "w") as json_file:
        json.dump(train_paths, json_file, indent=4)

    with open("training/splits/val.json", "w") as json_file:
        json.dump(val_paths, json_file, indent=4)

    with open("training/splits/test.json", "w") as json_file:
        json.dump(test_paths, json_file, indent=4)

    """import torchaudio  # type: ignore
    from augment_fp import AugmentFP
    import json

    with open("training/splits/train.json", "r") as f:
        noise_paths = json.load(f)
    target_sr = 16000
    af = AugmentFP(noise_paths, target_sr)

    duration = 3
    waveform, sr = torchaudio.load(
        "/workspace/src/mp3s/277a0d95c102a299f3cc8022a22bfbfb.mp3"
    )
    waveform = waveform.mean(axis=0)
    waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    nb_samples_segment = target_sr * duration
    start = random.randint(0, waveform.shape[0] - nb_samples_segment)
    waveform = waveform[start : start + nb_samples_segment].unsqueeze(0)

    aug = af(waveform)"""
