import argparse
import json
import os
import pickle
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio  # type: ignore
from tqdm import tqdm

from augmentation import AugmentFP
from testing.fma_preprocessing import TestSet
from testing.parameters import (
    WAVEFORM_SAMPLING_RATE,
    queries_paths,
    test_pipelines_parameters,
)


def generate_clean_queries(
    md5s: List[str],
    save_path: str,
    sr: int = WAVEFORM_SAMPLING_RATE,
    duration: int = 8,
    burn_in: int = 0,
    save: bool = False,
) -> None:
    random.seed(42)

    for i in tqdm(range(len(md5s))):
        if os.path.isfile(md5s[i]):
            waveform, origin_sr = torchaudio.load(md5s[i])
            waveform = waveform.mean(axis=0)

            if origin_sr != sr:
                resample = torchaudio.transforms.Resample(origin_sr, sr)
                waveform = resample(waveform)

            try:
                nb_samples_segment = sr * duration  # samples per extract

                start = random.randrange(
                    burn_in, waveform.shape[0] - nb_samples_segment - burn_in
                )

                waveform = waveform[start : start + nb_samples_segment]

                if save:
                    with open(
                        save_path + "/" + md5s[i].split("/")[-1].split(".")[0] + ".pkl",
                        "wb",
                    ) as handle:
                        pickle.dump(np.array(waveform), handle)
            except:
                print(md5s[i] + " is " + str(waveform.shape[0]) + "long")
        else:
            print(md5s[i] + "A file is missing")


def generate_augmented_queries(
    save_path: str,
    parameters: Dict[str, float],
    save: bool = False,
    device: str = "cpu",
) -> None:
    with open("/workspace/src/training/splits/train.json", "r") as f:
        noise_paths = json.load(f)

    aug_pipeline = AugmentFP(noise_paths, WAVEFORM_SAMPLING_RATE, parameters=parameters)
    aug_pipeline.augmentation_pipeline.freeze_parameters(42)
    aug_pipeline.augmentation_pipeline = aug_pipeline.augmentation_pipeline.to(device)
    os.makedirs(save_path, exist_ok=True)

    clean_queries = os.listdir(queries_paths["cleans"])

    for query in tqdm(clean_queries):
        try:
            with open(queries_paths["cleans"] + "/" + str(query), "rb") as f:
                clean_audio = pickle.load(f)

            clean_audio = torch.tensor(clean_audio).unsqueeze(0).to(device)

            augmented_audio = aug_pipeline(clean_audio)

            if save:
                with open(save_path + "/" + str(query), "wb") as handle:
                    pickle.dump(np.array(augmented_audio.T[:, 0].to("cpu")), handle)
        except:
            print("error with : ", query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries", help="redis cluster to select", default="augmented"
    )

    args = parser.parse_args()

    if str(args.queries) == "cleans":
        fma_large = pd.read_csv(
            "/workspace/src/testing/dataset/fma_large_preprocessed.csv"
        )
        testset = TestSet(fma_large)
        testset.remove_short_tracks()
        selected_queries = testset.sample_queries()
        selected_queries = testset.remove_exceptions(selected_queries)

        generate_clean_queries(
            selected_queries,
            queries_paths["cleans"],
            save=False,
        )

    if str(args.queries) == "augmented":
        for (
            pipeline_type,
            test_augmentation_pipeline,
        ) in test_pipelines_parameters.items():
            print(str(pipeline_type))
            generate_augmented_queries(
                queries_paths[str(pipeline_type)],
                test_augmentation_pipeline,
                save=False,
            )
