import csv
import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import AugmentationDataset
from training.parameters import (
    DURATION,
    N_SEGMENTS,
    TRAIN_BUFFER_SIZE,
    VAL_BUFFER_SIZE,
    WAVEFORM_SAMPLING_RATE,
)


def get_parser() -> Any:
    # TODO
    return []


def get_length(values: Any) -> int:
    return len(str(max(values)))


def get_id(value: str) -> int:
    return int(value.split("_")[1])


def read_file(
    tsv_file: str,
) -> Tuple[Dict[int, Dict[str, Any]], DefaultDict[Any, Dict[Any, Any]], Dict[str, int]]:
    tracks: Dict[int, Dict[str, Any]] = {}
    tags: DefaultDict[Any, Dict[Any, Any]] = defaultdict(dict)

    # For statistics
    artist_ids = set()
    albums_ids = set()

    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            track_id = get_id(row[0])
            tracks[track_id] = {
                "artist_id": get_id(row[1]),
                "album_id": get_id(row[2]),
                "path": row[3],
                "duration": float(row[4]),
                "tags": row[5:],  # raw tags, not sure if will be used
            }
            tracks[track_id].update(
                {category: set() for category in ["genre", "instrument", "mood/theme"]}
            )

            artist_ids.add(get_id(row[1]))
            albums_ids.add(get_id(row[2]))

            for tag_str in row[5:]:
                category, tag = tag_str.split("---")

                if tag not in tags[category]:
                    tags[category][tag] = set()

                tags[category][tag].add(track_id)

                if category not in tracks[track_id]:
                    tracks[track_id][category] = set()

                tracks[track_id][category].update(set(tag.split(",")))

    print(
        "Reading: {} tracks, {} albums, {} artists".format(
            len(tracks), len(albums_ids), len(artist_ids)
        )
    )

    extra = {
        "track_id_length": get_length(tracks.keys()),
        "artist_id_length": get_length(artist_ids),
        "album_id_length": get_length(albums_ids),
    }
    return tracks, tags, extra


def get_jamendo_data(
    dataset_path: str, num_val: int
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    train_path = dataset_path + "data/splits/split-0/autotagging-train.tsv"
    train_split, _, _ = read_file(train_path)

    val_path = dataset_path + "data/splits/split-0/autotagging-validation.tsv"
    val_split, _, _ = read_file(val_path)

    test_path = dataset_path + "data/splits/split-0/autotagging-test.tsv"
    test_split, _, _ = read_file(test_path)

    # Train, validation, and test set IDs
    train_ids = list(train_split.keys())
    val_ids = list(val_split.keys())
    test_ids = list(test_split.keys())

    random.Random(4).shuffle(val_ids)

    audio_path = dataset_path + "raw_30s/audio/"

    train_paths = []
    val_paths = []

    for _, key in enumerate(train_ids):
        train_paths.append(audio_path + train_split[key]["path"])

    for idx, key in enumerate(val_ids):
        if idx < num_val:
            val_paths.append(audio_path + val_split[key]["path"])
        else:
            train_paths.append(audio_path + val_split[key]["path"])

    for _, key in enumerate(test_ids):
        train_paths.append(audio_path + test_split[key]["path"])

    return (np.array(train_paths), np.array(val_paths))


def get_data_loaders(
    model_duration_seconds: float = DURATION,
    sampling_frequency: int = WAVEFORM_SAMPLING_RATE,
    val_steps: int = 64,
    mono: bool = True,
    batch_size: int = 8,
    run_val: bool = True,
    dataset_path: str = "/workspace/mtg-jamendo-dataset/",
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Get train, val, and test datasets that can be fed into torch dataloader.
    ---
    Args:
        experiment (str):
            MSD experiment being run.
        percentage (Optional[float]):
            Percentage of train set used for training.
        dataset_path (str):
            Path to Jamendo dataset as described in:
            https://github.com/MTG/mtg-jamendo-dataset
    Returns:
        Tuple[SupervisedDataset, SupervisedDataset]:
            Train, val, and test sets in torch format :)
    """

    train_ids, val_ids = get_jamendo_data(dataset_path, num_val=val_steps * batch_size)

    print(f"\nNumber of training tracks: {len(train_ids)}")
    print(f"Number of validation tracks: {len(val_ids)}\n")

    # Create loaders
    train_loader: DataLoader[Any] = DataLoader(
        AugmentationDataset(
            train_ids,
            sampling_frequency=sampling_frequency,
            mono=mono,
            n_segments=N_SEGMENTS,
            model_duration_seconds=model_duration_seconds,
            buffer_size=TRAIN_BUFFER_SIZE,
            noise_split="train",
        ),
        batch_size=batch_size,
    )
    val_dataset = AugmentationDataset(
        val_ids,
        sampling_frequency=sampling_frequency,
        mono=mono,
        n_segments=1,
        model_duration_seconds=model_duration_seconds,
        buffer_size=VAL_BUFFER_SIZE,
        noise_split="val",
    )
    val_dataset.tf_dataloader = (
        val_dataset.tf_dataloader.take(val_steps * batch_size)
        .cache("/tmp/validation_set_cache")
        .repeat()
    )
    if run_val:
        print("--- Runing the ds_test to cache in memory before training ---")
        _ = [
            None
            for _ in tqdm(
                val_dataset.tf_dataloader.take((val_steps * batch_size * 2) + 1)
            )
        ]

    val_loader: DataLoader[Any] = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
