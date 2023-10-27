import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from afp.audfprint.audfprint_match import Matcher
from afp.audfprint.hash_table import HashTable
from afp.audfprint.peak_extractor import Audfprint_peaks
from testing.metrics import F1score, Precision, Recall, psnr
from testing.parameters import afp_db_paths, afp_settings, queries_paths


def create_fp_database(files: List[str], dbpath: str) -> None:
    hash_tab = HashTable()

    analyzer = Audfprint_peaks(afp_settings["audfprint"])
    analyzer.shifts = 1

    for filename in tqdm(files):
        try:
            _, _ = analyzer.ingest(hash_tab, filename)
        except:
            print("error with ", filename)
    hash_tab.save(dbpath)


def compute_accuracy(
    files: List[str],
    dbpath: str,
    analyzer1: Audfprint_peaks,
    analyzer2: Audfprint_peaks,
) -> Dict[str, float]:
    hash_tab = HashTable(dbpath)
    matcher = Matcher()

    acc_no_den = 0
    acc_den = 0
    acc_mix = 0

    for _, filename in tqdm(enumerate(files)):
        gt = filename.split("/")[-1].split(".")[0]
        msgs1 = matcher.file_match_to_msgs(analyzer1, hash_tab, filename)
        msgs2 = matcher.file_match_to_msgs(analyzer2, hash_tab, filename)

        pred1 = msgs1[1].split("/")[-1].split(".")[0]

        if msgs1[0] == "MATCH" and str(gt) == str(pred1):
            acc_no_den += 1

        pred2 = msgs2[1].split("/")[-1].split(".")[0]

        if msgs2[0] == "MATCH" and str(gt) == str(pred2):
            acc_den += 1

        if msgs1[2] >= msgs2[2]:
            pred_mix = pred1
            if msgs1[0] == "MATCH":
                message = "MATCH"
            else:
                message = "NOMATCH"
        else:
            pred_mix = pred2
            if msgs2[0] == "MATCH":
                message = "MATCH"
            else:
                message = "NOMATCH"

        if message == "MATCH" and str(gt) == str(pred_mix):
            acc_mix += 1

    accuracy_no_den = acc_no_den / len(files)
    accuracy_den = acc_den / len(files)
    accuracy_mix = acc_mix / len(files)

    return {
        "No Denoising": accuracy_no_den,
        "With Denoising": accuracy_den,
        "Mix Pipeline": accuracy_mix,
    }


def compute_peaks_metrics(
    queries_augmented: List[str],
    analyzer_no_den: Audfprint_peaks,
    analyzer_den: Audfprint_peaks,
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
    
    for i in tqdm(range(len(queries_augmented))):
        query = queries_augmented[i].split("/")[-1]
        query_clean = os.path.join(queries_paths["cleans"], query)

        m_clean, w_clean, sgram_clean = analyzer_no_den.wavfile2peaks(
            query_clean, get_masks_waveforms=True
        )
        m_augmented, w_augmented, sgram_augmented = analyzer_no_den.wavfile2peaks(
            queries_augmented[i], get_masks_waveforms=True
        )
        m_denoised, w_denoised, sgram_denoised = analyzer_den.wavfile2peaks(
            queries_augmented[i], get_masks_waveforms=True
        )

        mask_clean = torch.tensor(m_clean).T.unsqueeze(0)
        mask_augmented = torch.tensor(m_augmented).T.unsqueeze(0)
        mask_denoised = torch.tensor(m_denoised).T.unsqueeze(0)

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

    prec /= len(queries_augmented)
    prec_den /= len(queries_augmented)
    rec /= len(queries_augmented)
    rec_den /= len(queries_augmented)
    f1 /= len(queries_augmented)
    f1_den /= len(queries_augmented)
    psnr_no_den_spec /= len(queries_augmented)
    psnr_den_spec /= len(queries_augmented)

    return {
        "precision_no_den": prec,
        "recall_no_den": rec,
        "f1_score_no_den": f1,
        "psnr_no_den_spec": psnr_no_den_spec,
        "prec_den": prec_den,
        "rec_den": rec_den,
        "f1_den": f1_den,
        "psnr_den_spec": psnr_den_spec,
    }


def identification_rate_results(denoising_model="unet") -> None:
    results = {}
    for augmentation, query_path in queries_paths.items():
        print("augmentation:", augmentation)

        queries = [os.path.join(query_path, query) for query in os.listdir(query_path)]
        analyzer_no_den = Audfprint_peaks(afp_settings["audfprint"])
        analyzer_no_den.shifts = 4

        analyzer_den = Audfprint_peaks(
            afp_settings["audfprint"], denoising=True, denoising_model=denoising_model
        )
        analyzer_den.shifts = 4

        results[str(augmentation)] = compute_accuracy(
            queries,
            afp_db_paths["audfprint"],
            analyzer_no_den,
            analyzer_den,
        )
        print(results[str(augmentation)])

    with open(
        "/workspace/src/testing/results/accuracy_audfprint_{}.json".format(denoising_model), "w"
    ) as json_file:
        json.dump(results, json_file, indent=4)


def peaks_metrics_results(denoising_model="unet") -> None:
    results = {}

    for augmentation, query_path in queries_paths.items():
        if augmentation == "cleans":
            continue

        print("augmentation:", augmentation)

        queries_augmented = [
            os.path.join(query_path, query) for query in os.listdir(query_path)
        ]

        analyzer_no_den = Audfprint_peaks(afp_settings["audfprint"])
        analyzer_den = Audfprint_peaks(
            afp_settings["audfprint"], denoising=True, denoising_model=denoising_model
        )

        results[str(augmentation)] = compute_peaks_metrics(
            queries_augmented, analyzer_no_den, analyzer_den
        )

        print(results[str(augmentation)])

    with open(
        "/workspace/src/testing/results/peaks_metrics_audfprint_{}.json".format(denoising_model), "w"
    ) as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", help="", default="identification_rate")
    parser.add_argument("--model", help="", default="unet")

    args = parser.parse_args()

    if str(args.action) == "index":
        fma_large = pd.read_csv(
            "/workspace/src/testing/dataset/fma_large_preprocessed.csv"
        )
        tracks_to_index = list(fma_large["file_paths"])
        create_fp_database(
            tracks_to_index, "/workspace/src/afp/audfprint/fp_database.pklz"
        )

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
