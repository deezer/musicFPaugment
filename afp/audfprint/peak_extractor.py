# coding=utf-8
"""
Class to do the analysis of wave files into hash constellations.
2014-09-20 Dan Ellis dpwe@ee.columbia.edu
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.signal  # type: ignore
import torch
import torchaudio  # type: ignore
from torchaudio.transforms import Resample  # type: ignore

import afp.audfprint.stft as stft
from afp.audfprint.hash_table import HashTable
from testing.parameters import WAVEFORM_SAMPLING_RATE
from training.model import Demucs
from training.unet import UNet
from training.utils import set_gpus

device = set_gpus()
unet = UNet(1, 1, rate=0.05).to(device)
weights_path = "/workspace/src/training/checkpoints/unet_lr_0.001_BS_128/best_epoch.pt"
unet.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
unet.eval()

demucs = Demucs().to(device)
weights_path = (
    "/workspace/src/training/checkpoints/demucs_lr_0.0005_BS_128/best_epoch.pt"
)
demucs.load_state_dict(
    torch.load(weights_path, map_location=device)["model_state_dict"]
)
demucs.eval()


def landmarks2hashes(
    landmarks_list: List[Tuple[int, int, int, int]]
) -> npt.NDArray[np.int32]:
    """
    Convert a list of (time, bin1, bin2, dtime) landmarks into a list of (time, hash) pairs
    where the hash combines the three remaining values.
    """
    landmarks = np.array(landmarks_list)
    # Deal with special case of empty landmarks.
    if landmarks.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    hashes = np.zeros((landmarks.shape[0], 2), dtype=np.int32)
    hashes[:, 0] = landmarks[:, 0]
    hashes[:, 1] = (
        ((landmarks[:, 1] & 255) << 12)
        | (((landmarks[:, 2] - landmarks[:, 1]) & 63) << 6)
        | (landmarks[:, 3] & 63)
    )
    return hashes


def locmax(vec: npt.NDArray[np.float32], indices: bool = False) -> npt.NDArray[Any]:
    """
    Return a boolean vector of which points in vec are local maxima. End points are peaks if larger than single neighbors.
    If indices=True, return the indices of the True values instead of the boolean vector.
    """
    nbr = np.zeros(len(vec) + 1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[:-1])
    maxmask = nbr[:-1] & ~nbr[1:]
    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask


class Audfprint_peaks(object):
    """
    A class to wrap up all the parameters associated with the analysis of soundfiles into fingerprints.
    """

    # Optimization: cache pre-calculated Gaussian profile
    __sp_width = None
    __sp_len = None
    __sp_vals = np.zeros([])

    def __init__(
        self, params: Dict[str, Any], denoising: bool = False, denoising_model=None
    ) -> None:
        self.density = params["density"]
        self.target_sr = params["samplerate"]
        self.n_fft = params["n_fft"]
        self.n_hop = params["n_hop"]
        self.shifts = params["shifts"]
        # how wide to spreak peaks
        self.f_sd = params["freq-sd"]
        # Maximum number of local maxima to keep per frame
        self.maxpksperframe = params["pks-per-frame"]

        # Used for hash construction:
        # Limit the num of pairs we'll make from each peak (Fanout)
        self.maxpairsperpeak = 3
        # min time separation (traditionally 1, upped 2014-08-04)
        self.mindt = 2
        # max lookahead in time (LIMITED TO <64 IN LANDMARK2HASH)
        self.targetdt = 63
        # Values controlling peaks2landmarks
        # +/- 31 bins in freq (LIMITED TO -32..31 IN LANDMARK2HASH)
        self.targetdf = 31
        self.denoising = denoising
        self.denoising_model = denoising_model

        if self.denoising == True:
            assert self.denoising_model in ["demucs", "unet"]

    def spreadpeaksinvector(
        self, vector: npt.NDArray[np.float32], width: float = 4.0
    ) -> npt.NDArray[np.float32]:
        """
        Create a blurred version of vector, where each of the local maxes is spread by a gaussian with SD <width>.
        """
        npts = len(vector)
        peaks = locmax(vector, indices=True)
        return self.spreadpeaks(
            [(p, v) for p, v in zip(peaks, vector[peaks])], npoints=npts, width=width
        )

    def spreadpeaks(
        self,
        peaks: List[Tuple[int, float]],
        npoints: Optional[int] = None,
        width: float = 4.0,
        base: Optional[npt.NDArray[np.float32]] = None,
    ) -> npt.NDArray[np.float32]:
        """
        Generate a vector consisting of the max of a set of Gaussian bumps.
        ---
        Args:
            peaks (List[Tuple[int, float]]):
                List of (index, value) pairs giving the center point and height of each gaussian.
            npoints (Optional[int]):
                The length of the output vector (needed if base not provided).
            width (float):
                The half-width of the Gaussians to lay down at each point.
            base (Optional[npt.NDArray[np.float32]]):
                Optional initial lower bound to place Gaussians above.

        Returns:
            npt.NDArray[np.float32]]
                The maximum across all the scaled Gaussians.
        """
        if base is None and npoints is not None:
            vec = np.zeros(npoints, dtype=np.float32)
        elif base is not None:
            npoints = len(base)
            vec = np.copy(base)
        else:
            raise ValueError("Please provide arguments npoints or base!")

        if width != self.__sp_width or npoints != self.__sp_len:
            # Need to calculate new vector
            self.__sp_width = width
            self.__sp_len = npoints
            self.__sp_vals = np.exp(
                -0.5 * ((np.arange(-npoints, npoints + 1) / width) ** 2)
            )
        # Now the actual function
        for pos, val in peaks:
            vec = np.maximum(
                vec, val * self.__sp_vals[np.arange(npoints) + npoints - pos]
            )
        return vec

    def _decaying_threshold_fwd_prune(
        self, sgram: npt.NDArray[np.float32], a_dec: float
    ) -> npt.NDArray[np.float32]:
        """
        Forward pass of findpeaks. Initial threshold envelope based on peaks in first 10 frames.
        """
        (srows, scols) = np.shape(sgram)
        sthresh = self.spreadpeaksinvector(
            np.max(sgram[:, : np.minimum(10, scols)], axis=1), self.f_sd
        )
        # Store sthresh at each column, for debug
        # thr = np.zeros((srows, scols))
        peaks = np.zeros((srows, scols), dtype=np.float32)
        # optimization of mask update
        __sp_pts = len(sthresh)
        __sp_v = self.__sp_vals

        for col in range(scols):
            s_col = sgram[:, col]
            # Find local magnitude peaks that are above threshold
            sdmaxposs = np.nonzero(locmax(s_col) * (s_col > sthresh))[0]
            # Work down list of peaks in order of their absolute value
            # above threshold
            valspeaks = sorted(zip(s_col[sdmaxposs], sdmaxposs), reverse=True)
            for val, peakpos in valspeaks[: self.maxpksperframe]:
                sthresh = np.maximum(
                    sthresh,
                    val * __sp_v[(__sp_pts - peakpos) : (2 * __sp_pts - peakpos)],
                )
                peaks[peakpos, col] = 1
            sthresh *= a_dec
        return peaks

    def _decaying_threshold_bwd_prune_peaks(
        self,
        sgram: npt.NDArray[np.float32],
        peaks: npt.NDArray[np.float32],
        a_dec: float,
    ) -> npt.NDArray[np.float32]:
        """
        Backwards pass of findpeaks.
        """
        scols = np.shape(sgram)[1]
        # Backwards filter to prune peaks
        sthresh = self.spreadpeaksinvector(sgram[:, -1], self.f_sd)
        for col in range(scols, 0, -1):
            pkposs = np.nonzero(peaks[:, col - 1])[0]
            peakvals = sgram[pkposs, col - 1]
            for val, peakpos in sorted(zip(peakvals, pkposs), reverse=True):
                if val >= sthresh[peakpos]:
                    # Setup the threshold
                    sthresh = self.spreadpeaks(
                        [(peakpos, val)], base=sthresh, width=self.f_sd
                    )
                    # Delete any following peak (threshold should, but be sure)
                    if col < scols:
                        peaks[peakpos, col] = 0
                else:
                    # delete the peak
                    peaks[peakpos, col - 1] = 0
            sthresh = a_dec * sthresh
        return peaks

    def find_peaks(
        self, d: npt.NDArray[np.float32]
    ) -> Tuple[List[Tuple[int, int]], npt.NDArray[np.float32]]:
        """
        Find the local peaks in the spectrogram as basis for fingerprints. Returns a list of (time_frame, freq_bin) pairs.
        ---
        Args:
            d (npt.NDArray[np.int32]):
                Input waveform as 1D vector

        Returns:
            List[Tuple[int, int]]:
                Ordered list of landmark peaks found in STFT.  First value of each pair is the time index (in STFT frames,
                i.e., units of n_hop/sr secs), second is the FFT bin (in units of sr/n_fft Hz).
            npt.NDArray[np.float32]:
                Masks.
        """
        if len(d) == 0:
            return [], np.array([])

        # Take spectrogram
        mywin = np.hanning(self.n_fft + 2)[1:-1]

        sgram = np.abs(
            stft.stft(d, n_fft=self.n_fft, hop_length=self.n_hop, window=mywin)
        )

        sgram /= np.max(sgram)

        if self.denoising and self.denoising_model == "unet":
            sgram = torch.tensor(sgram).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                sgram = unet(sgram).cpu()
            sgram = np.array(sgram.squeeze())

        spec = sgram.copy()
        sgrammax = np.max(sgram)

        if sgrammax > 0.0:
            sgram = np.log(np.maximum(sgram, np.max(sgram) / 1e6))
            sgram = sgram - np.mean(sgram)
        else:
            # The sgram is identically zero, i.e., the input signal was identically
            # zero.  Not good, but let's let it through for now.
            print("find_peaks: Warning: input signal is identically zero.")

        # sgram /= np.max(np.abs(sgram))
        # High-pass filter onset emphasis
        # [:-1,] discards top bin (nyquist) of sgram so bins fit in 8 bits

        sgram = np.array(
            [scipy.signal.lfilter([1, -1], [1, -(0.98**1)], s_row) for s_row in sgram]
        )[
            :-1,
        ]

        # Prune to keep only local maxima in spectrum that appear above an online,
        # decaying threshold
        # masking envelope decay constant
        a_dec = 1 - 0.01 * (self.density * np.sqrt(self.n_hop / 352.8) / 35)

        peaks = self._decaying_threshold_fwd_prune(sgram, a_dec)

        # Further prune these peaks working backwards in time, to remove small peaks
        # that are closely followed by a large peak
        peaks = self._decaying_threshold_bwd_prune_peaks(sgram, peaks, a_dec)

        peaks_mask = peaks.copy()
        # build a list of peaks we ended up with
        scols = np.shape(sgram)[1]
        pklist = []
        for col in range(scols):
            for bin_ in np.nonzero(peaks[:, col])[0]:
                pklist.append((col, bin_))

        return pklist, peaks_mask, spec

    def peaks2landmarks(
        self, pklist: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Take a list of local peaks in spectrogram and form them into pairs as landmarks.
        pklist is a column-sorted list of (col, bin) pairs as created by findpeaks().
        Return a list of (col, peak, peak2, col2-col) landmark descriptors.
        """
        # Form pairs of peaks into landmarks
        landmarks = []
        if len(pklist) > 0:
            # Find column of the final peak in the list
            scols = pklist[-1][0] + 1
            # Convert (col, bin) list into peaks_at[col] lists
            peaks_at = [[] for _ in range(scols)]
            for col, bin_ in pklist:
                peaks_at[col].append(bin_)

            # Build list of landmarks <starttime F1 endtime F2>
            for col in range(scols):
                for peak in peaks_at[col]:
                    pairsthispeak = 0
                    for col2 in range(
                        col + self.mindt, min(scols, col + self.targetdt)
                    ):
                        if pairsthispeak < self.maxpairsperpeak:
                            for peak2 in peaks_at[col2]:
                                if abs(peak2 - peak) < self.targetdf:
                                    # and abs(peak2-peak) + abs(col2-col) > 2 ):
                                    if pairsthispeak < self.maxpairsperpeak:
                                        # We have a pair!
                                        landmarks.append((col, peak, peak2, col2 - col))
                                        pairsthispeak += 1
        return landmarks

    def wavfile2peaks(
        self,
        filename: str,
        shifts: Optional[int] = None,
        get_masks_waveforms: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Read a soundfile and return its landmark peaks as a list of (time, bin) pairs.
        If specified, resample to sr first. shifts > 1 causes hashes to be extracted from multiple shifts of
        waveform, to reduce frame effects.
        """
        # try:
        # [d, sr] = librosa.load(filename, sr=self.target_sr)
        if filename.split(".")[-1] == "pkl":
            with open(
                filename,
                "rb",
            ) as f:
                d = pickle.load(f)
                d = torch.tensor(d)

                if self.denoising and self.denoising_model == "demucs":
                    with torch.no_grad():
                        d = (
                            demucs(d.unsqueeze(0).to(device))
                            .squeeze(0)
                            .squeeze(0)
                            .cpu()
                        )

                resampling = Resample(WAVEFORM_SAMPLING_RATE, self.target_sr)
                d = resampling(d)
                waveform = d
                sr = self.target_sr

        elif filename.split(".")[-1] == "mp3":
            # TODO: handle denoising with mp3 files.
            d, sr = torchaudio.load(filename)
            resampling = Resample(sr, self.target_sr)
            d = d.mean(axis=0)
            d = resampling(d)
            sr = self.target_sr

        if d.max().to(float) == 0:
            print("error with filename: ", filename)

        # except Exception as e:
        #     message = "wavfile2peaks: Error reading " + filename
        #     # if self.fail_on_error:
        #     #     print(e)
        #     #     raise IOError(message)
        #     print(message, "skipping")
        #     d = []
        #     sr = self.target_sr

        # Store duration in a global because it's hard to handle
        dur = len(d) / sr

        if shifts is None or shifts < 2:
            peaks, peaks_mask, sgram = self.find_peaks(d)
        else:
            # Calculate hashes with optional part-frame shifts
            peaklists = []
            for shift in range(shifts):
                shiftsamps = int(shift / self.shifts * self.n_hop)
                peaklists.append(self.find_peaks(d[shiftsamps:])[0])

            peaks = peaklists

        # instrumentation to track total amount of sound processed
        self.soundfiledur = dur

        if get_masks_waveforms:
            return peaks_mask, waveform, sgram
        # self.soundfiletotaldur += dur
        # self.soundfilecount += 1
        return peaks

    def wavfile2hashes(self, filename: str) -> npt.NDArray[Any]:
        """
        Read a soundfile and return its fingerprint hashes as a list of (time, hash) pairs.
        If specified, resample to sr first. shifts > 1 causes hashes to be extracted from
        multiple shifts of waveform, to reduce frame effects.
        """
        peaks = self.wavfile2peaks(filename, self.shifts)

        if len(peaks) == 0:
            return np.hstack([]).astype(np.int32)

        # Did we get returned a list of lists of peaks due to shift?
        if isinstance(peaks[0], list):
            peaklists = peaks
            qhashes = []
            for peaklist in peaklists:
                qhashes.append(landmarks2hashes(self.peaks2landmarks(peaklist)))
            query_hashes = np.concatenate(qhashes)

        else:
            query_hashes = landmarks2hashes(self.peaks2landmarks(peaks))

        # Remove duplicates by merging each row into a single value.
        hashes_hashes = ((query_hashes[:, 0].astype(np.uint64)) << 32) + query_hashes[
            :, 1
        ].astype(np.uint64)
        unique_hash_hash = np.sort(np.unique(hashes_hashes))
        unique_hashes = np.hstack(
            [
                (unique_hash_hash >> 32)[:, np.newaxis],
                (unique_hash_hash & ((1 << 32) - 1))[:, np.newaxis],
            ]
        ).astype(np.int32)

        return unique_hashes

    def ingest(self, hashtable: HashTable, filename: str) -> Tuple[float, int]:
        """
        Read an audio file and add it to the database.
        ---
        Args:
            hashtable (HashTable):
                The hash table to add to.
            filename (str):
                Name of the soundfile to add.

        Returns:
            float:
                The duration of the track.
            int:
                The number of hashes it mapped into.
        """
        hashes = self.wavfile2hashes(filename)
        hashtable.store(filename, hashes)

        return self.soundfiledur, len(hashes)
