import hashlib
from operator import itemgetter
from typing import List, Tuple

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import torch
from dejavu.variables import (
    CONNECTIVITY_MASK,
    FINGERPRINT_REDUCTION,
    MAX_HASH_TIME_DELTA,
    MIN_HASH_TIME_DELTA,
    PEAK_NEIGHBORHOOD_SIZE,
)
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (
    binary_erosion,
    generate_binary_structure,
    iterate_structure,
)

from testing.parameters import afp_settings
from training.unet import UNet
from training.utils import set_gpus

device = set_gpus()
unet = UNet(1, 1, rate=0.05).to(device)
weights_path = "/workspace/src/training/checkpoints/unet_lr_0.001_BS_128/best_epoch.pt"
unet.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
unet.eval()


def fingerprint(
    channel_samples: List[int],
    Fs: float = afp_settings["dejavu"]["samplerate"],
    wsize: int = afp_settings["dejavu"]["n_fft"],
    n_hop: int = afp_settings["dejavu"]["n_hop"],
    fan_value: int = afp_settings["dejavu"]["fan_value"],
    amp_min: int = afp_settings["dejavu"]["amp_min"],
    denoising: bool = False,
    denoising_model: str = "unet",
    get_masks: bool = "False",
) -> List[Tuple[str, int]]:
    """
    FFT the channel, log transform output, find local maxima, then return locally sensitive hashes.

    :param channel_samples: channel samples to fingerprint.
    :param Fs: audio sampling rate.
    :param wsize: FFT windows size.
    :param wratio: ratio by which each sequential window overlaps the last and the next window.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list of hashes with their corresponding offsets.
    """
    # FFT the signal and extract frequency components
    if denoising:
        assert denoising_model in ["unet", "demucs"]

    arr2D = mlab.specgram(
        np.array(channel_samples),
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=n_hop,
    )[0]

    arr2D /= arr2D.max()

    if denoising is True and denoising_model == "unet":
        arr2D = torch.tensor(arr2D).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            arr2D = unet(arr2D).cpu()
        arr2D = arr2D**2
        arr2D = np.array(arr2D.squeeze())

    specgram = arr2D.copy()
    arr2D = 10 * np.log(np.maximum(arr2D, np.max(arr2D) / 1e6))
    arr2D = arr2D - np.mean(arr2D)

    # Apply log transform since specgram function returns linear array. 0s are excluded to avoid np warning.
    # dejavu:
    # arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))

    local_maxima, peak_mask = get_2D_peaks(arr2D, plot=False, amp_min=amp_min)
    # print('nb peaks: ', len(local_maxima))
    # return hashes
    if get_masks is True:
        return generate_hashes(local_maxima, fan_value=fan_value), peak_mask, specgram

    return generate_hashes(local_maxima, fan_value=fan_value)


def get_2D_peaks(
    arr2D: np.array,
    plot: bool = False,
    amp_min: int = afp_settings["dejavu"]["amp_min"],
) -> List[Tuple[List[int], List[int]]]:
    """
    Extract maximum peaks from the spectogram matrix (arr2D).

    :param arr2D: matrix representing the spectogram.
    :param plot: for plotting the results.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    # Original code from the repo is using a morphology mask that does not consider diagonal elements
    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
    # is to change from the current diamond figure to a just a normal square one:
    #       F   T   F           T   T   T
    #       T   T   T   ==>     T   T   T
    #       F   T   F           T   T   T
    # In my local tests time performance of the square mask was ~3 times faster
    # respect to the diamond one, without hurting accuracy of the predictions.
    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
    # That being said, we generate the mask by using the following function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)

    #  And then we apply dilation using the following function
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
    #  change it by the following code:
    #  neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter mask
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D

    # Applying erosion, the dejavu documentation does not talk about this step.
    background = arr2D == 0
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )

    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background
    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    # print(len(freqs))
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    peak_coordinates = list(zip(freqs_filter, times_filter))
    peak_mask = np.zeros(detected_peaks.shape)

    for coord in peak_coordinates:
        peak_mask[coord] = 1

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots(figsize=(16, 16))
        im = ax.imshow(arr2D)
        ax.scatter(times_filter, freqs_filter)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()
        plt.colorbar(im)
        plt.savefig("44100Hz.png")
    # print(len(peak_coordinates))
    return peak_coordinates, peak_mask


def generate_hashes(
    peaks: List[Tuple[int, int]],
    fan_value: int = afp_settings["dejavu"]["fan_value"],
) -> List[Tuple[str, int]]:
    """
    Hash list structure:
       sha1_hash[0:FINGERPRINT_REDUCTION]    time_offset
        [(e05b341a9b77a51fd26, 32), ... ]

    :param peaks: list of peak frequencies and times.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :return: a list of hashes with their corresponding offsets.
    """
    # frequencies are in the first position of the tuples
    idx_freq = 0
    # times are in the second position of the tuples
    idx_time = 1

    # if PEAK_SORT:
    peaks.sort(key=itemgetter(1))

    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1

                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1(
                        f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode("utf-8")
                    )

                    hashes.append((h.hexdigest()[0:FINGERPRINT_REDUCTION], t1))

    # print('number of hashes: ', len(hashes))
    return hashes
