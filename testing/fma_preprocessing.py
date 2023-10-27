import ast
import glob
import os
import random
from typing import List

import pandas as pd


def preprocessing_fma_large() -> List[str]:
    test_mp3s = glob.glob("/workspace/fma/fma_large/" + "/*/*.mp3", recursive=True)

    text_file = open("/workspace/src/testing/dataset/fma_large_to_remove.txt", "r")
    mp3s_to_remove = text_file.read().split("\n")
    mp3s_to_remove = ["/workspace/fma/" + mp3_to_rem for mp3_to_rem in mp3s_to_remove]

    test_mp3s = [test_mp3 for test_mp3 in test_mp3s if test_mp3 not in mp3s_to_remove]
    return test_mp3s


def load(filepath: str) -> pd.DataFrame:
    filename = os.path.basename(filepath)

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    elif "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        try:
            tracks["set", "subset"] = tracks["set", "subset"].astype(  # type: ignore
                "category", categories=SUBSETS, ordered=True
            )
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True)  # type: ignore
            )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks

    else:
        raise ValueError(f"File {filename} cannot be opened.")


def get_file_path(input_str: str) -> str:
    input_str = str(input_str).zfill(6)  # Ensure 6 digits with leading zeros if needed
    return f"/workspace/fma/fma_large/{input_str[:3]}/{input_str}.mp3"


class TestSet:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset

    def get_samples_per_class(self) -> pd.DataFrame:
        return self.dataset.groupby("top_genre", dropna=False).count()

    def remove_short_tracks(self) -> None:
        self.dataset = self.dataset[self.dataset["duration"] > 12]

    def sample_queries(self) -> List[str]:
        random.seed(31)

        result = self.dataset["top_genre"].value_counts()
        samples = []

        for genre, count in result.items():
            if count < 900:
                files = self.dataset[self.dataset["top_genre"] == genre]
                samples.extend(files["file_paths"].tolist())

            if count > 900:
                files = self.dataset[self.dataset["top_genre"] == genre].sample(
                    n=885, random_state=42
                )
                samples.extend(files["file_paths"].tolist())

        random.shuffle(samples)

        return samples

    def remove_exceptions(self, samples: List[str]) -> List[str]:
        samples_to_remove = [
            77,
            2476,
            2979,
            3023,
            4165,
            7945,
        ]

        for index in samples_to_remove:
            samples.pop(index)

        return samples


if __name__ == "__main__":
    tracks = load("/workspace/fma/fma_metadata/tracks.csv")
    genres = load("/workspace/fma/fma_metadata/genres.csv")

    genre_data_csv = tracks["track"][["genre_top", "duration"]]
    genre_dataset = pd.DataFrame(
        {
            "track_id": genre_data_csv.index,
            "top_genre": genre_data_csv.genre_top,
            "duration": genre_data_csv.duration,
        }
    )

    genre_dataset["file_paths"] = genre_dataset["track_id"].apply(get_file_path)
    genre_dataset.drop(columns=["track_id"], inplace=True)

    preprocessed_fma = pd.read_csv(
        "/workspace/src/testing/dataset/fma_large_preprocessed.txt", header=None
    )
    preprocessed_fma = preprocessed_fma.rename(columns={0: "file_paths"})

    final_dataset = preprocessed_fma.merge(genre_dataset, on="file_paths", how="left")

    final_dataset.to_csv(
        "/workspace/src/testing/dataset/fma_large_preprocessed.csv", index=False
    )
