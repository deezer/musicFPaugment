import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from afp.dejavu.dejavu import Dejavu
from afp.dejavu.file_recognizer import FileRecognizer
from testing.metrics import F1score, Precision, Recall, psnr
from testing.parameters import afp_db_paths, afp_settings, queries_paths


def create_fp_database(files: List[str]):
    djv = Dejavu(afp_db_paths["dejavu"], afp_settings["dejavu"], "set")
    djv.fingerprint_directory(files)


def compute_accuracy(audio_paths, djv, djv2):
    recognizer1 = FileRecognizer(djv)
    recognizer2 = FileRecognizer(djv2)

    tp_no_denoising = 0
    tp_denoising = 0
    tp_mix = 0

    for i in tqdm(range(len(audio_paths))):
        gt = audio_paths[i].split("/")[-2]

        results1 = recognizer1.recognize_file(audio_paths[i])
        results2 = recognizer2.recognize_file(audio_paths[i])

        if results1["match"]:
            retrieved_md51 = results1["results"][0]["song_name"].decode("utf-8")
            md51_nb_matches = results1["results"][0]["nb_matches_with_offset"]
            if str(retrieved_md51) == str(gt):
                tp_no_denoising += 1
        else:
            retrieved_md51 = ""
            md51_nb_matches = 0

        # Denoising:
        if results2["match"]:
            retrieved_md52 = results2["results"][0]["song_name"].decode("utf-8")
            md52_nb_matches = results2["results"][0]["nb_matches_with_offset"]
            if str(retrieved_md52) == str(gt):
                tp_denoising += 1
        else:
            retrieved_md52 = ""
            md52_nb_matches = 0

        # Mix:
        if md51_nb_matches >= md52_nb_matches:
            pred_mix = retrieved_md51
            if results1["match"]:
                message = "MATCH"
            else:
                message = "NOMATCH"
        else:
            pred_mix = retrieved_md52
            if results2["match"]:
                message = "MATCH"
            else:
                message = "NOMATCH"

        if message == "MATCH" and str(gt) == str(pred_mix):
            tp_mix += 1

    accuracy_no_den = tp_no_denoising / len(audio_paths)
    accuracy_den = tp_denoising / len(audio_paths)
    accuracy_mix = tp_mix / len(audio_paths)

    return {
        "No Denoising": accuracy_no_den,
        "With Denoising": accuracy_den,
        "Mix Pipeline": accuracy_mix,
    }


def compute_peaks_metrics(
    queries_augmented: List[str],
    djv_no_den: Dejavu,
    djv_den: Dejavu,
) -> Dict[str, float]:
    precision = Precision()
    recall = Recall()
    f1_score = F1score()

    prec = 0.0
    prec_den = 0.0
    rec = 0.0
    rec_den = 0.0
    f1 = 0.0
    f1_den = 0.0

    psnr_no_den_spec = 0.0
    psnr_den_spec = 0.0

    psnr_no_den_wav = 0.0
    psnr_den_wav = 0.0

    for i in tqdm(range(len(queries_augmented))):
        query = queries_augmented[i].split("/")[-1]
        query_clean = os.path.join(queries_paths["cleans"], query)

        m_clean, sgram_clean = djv_no_den.generate_fingerprints(
            query_clean, get_masks=True
        )
        m_augmented, sgram_augmented = djv_no_den.generate_fingerprints(
            queries_augmented[i], get_masks=True
        )
        m_denoised, sgram_denoised = djv_den.generate_fingerprints(
            queries_augmented[i], get_masks=True
        )

        mask_clean = torch.tensor(m_clean).T.unsqueeze(0)
        mask_augmented = torch.tensor(m_augmented).T.unsqueeze(0)
        mask_denoised = torch.tensor(m_denoised).T.unsqueeze(0)

        # waveform_clean = torch.tensor(w_clean).unsqueeze(0)
        # waveform_augmented = torch.tensor(w_augmented).unsqueeze(0)
        # waveform_denoised = torch.tensor(w_denoised).unsqueeze(0)

        sgram_clean = torch.tensor(sgram_clean).unsqueeze(0)
        sgram_augmented = torch.tensor(sgram_augmented).unsqueeze(0)
        sgram_denoised = torch.tensor(sgram_denoised).unsqueeze(0)

        prec += precision(mask_augmented, mask_clean)
        prec_den += precision(mask_denoised, mask_clean)

        rec += recall(mask_augmented, mask_clean)
        rec_den += recall(mask_denoised, mask_clean)

        f1 += f1_score(mask_augmented, mask_clean)
        f1_den += f1_score(mask_denoised, mask_clean)

        psnr_no_den_spec += psnr(sgram_augmented, sgram_clean).item()
        psnr_den_spec += psnr(sgram_denoised, sgram_clean).item()

        psnr_no_den_wav += psnr(sgram_augmented, sgram_clean).item()
        psnr_den_wav += psnr(sgram_denoised, sgram_clean).item()

    prec /= len(queries_augmented)
    prec_den /= len(queries_augmented)
    rec /= len(queries_augmented)
    rec_den /= len(queries_augmented)
    f1 /= len(queries_augmented)
    f1_den /= len(queries_augmented)
    psnr_no_den_spec /= len(queries_augmented)
    psnr_den_spec /= len(queries_augmented)
    psnr_no_den_wav /= len(queries_augmented)
    psnr_den_wav /= len(queries_augmented)

    return {
        "precision_no_den": prec,
        "recall_no_den": rec,
        "f1_score_no_den": f1,
        "psnr_no_den_spec": psnr_no_den_spec,
        "psnr_no_den_wav": psnr_no_den_spec,
        "prec_den": prec_den,
        "rec_den": rec_den,
        "f1_den": f1_den,
        "psnr_den_spec": psnr_den_spec,
        "psnr_den_wav": psnr_den_wav,
    }


def identification_rate_results(model="unet"):
    results = {}
    for augmentation, query_path in queries_paths.items():
        print("augmentation:", augmentation)

        queries = [os.path.join(query_path, query) for query in os.listdir(query_path)]

        djv_no_den = Dejavu(afp_db_paths["dejavu"], afp_settings["dejavu"], "set")
        djv_den = Dejavu(
            afp_db_paths["dejavu"],
            afp_settings["dejavu"],
            "set",
            denoising=True,
            denoising_model=model,
        )

        results[str(augmentation)] = compute_accuracy(
            queries,
            djv_no_den,
            djv_den,
        )
        print(results[str(augmentation)])
    with open("/workspace/src/testing/results/accuracy_dejavu_{}.json".format(model), "w") as json_file:
        json.dump(results, json_file, indent=4)


def peaks_metrics_results(denoising_model="unet") -> None:
    results = {}

    for augmentation, query_path in queries_paths.items():
        if augmentation == "cleans":
            continue

        print("augmentation:", augmentation)

        queries_augmented = [
            os.path.join(query_path, query) for query in os.listdir(query_path)
        ][:5]

        djv_no_den = Dejavu(afp_db_paths["dejavu"], afp_settings["dejavu"], "set")
        djv_den = Dejavu(
            afp_db_paths["dejavu"],
            afp_settings["dejavu"],
            "set",
            denoising=True,
            denoising_model=denoising_model,
        )

        results[str(augmentation)] = compute_peaks_metrics(
            queries_augmented, djv_no_den, djv_den
        )

        print(results[str(augmentation)])

    with open(
        "/workspace/src/testing/results/peaks_metrics_dejavu_{}.json".format(denoising_model), "w"
    ) as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", help="", default="index")
    parser.add_argument("--model", help="", default="unet")

    args = parser.parse_args()

    if str(args.action) == "index":
        fma_large = pd.read_csv(
            "/workspace/src/testing/dataset/fma_large_preprocessed.csv"
        )
        tracks_to_index = list(fma_large["file_paths"])
        create_fp_database(tracks_to_index)

    if str(args.action) == "identification_rate":
        if str(args.model) == "unet":
            identification_rate_results("unet")
        if str(args.model) == "demucs":
            identification_rate_results("demucs")

    if str(args.action) == "peaks_metrics":
        if str(args.model) == "unet":
            peaks_metrics_results("unet")
        if str(args.model) == "demucs":
            peaks_metrics_results("demucs")
