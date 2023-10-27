# Inspired by tornado
# https://www.tornadoweb.org/en/stable/_modules/tornado/util.html#ObjectDict

import math
import os
import typing
from pathlib import Path
from typing import Any, List, Optional, Text, Tuple, Union

import librosa  # type: ignore
import torch
import torchaudio  # type: ignore
from torch import Tensor

_ObjectDictBase = typing.Dict[str, typing.Any]
SUPPORTED_EXTENSIONS = (".wav",)


def is_multichannel(samples: Tensor) -> bool:
    return samples.shape[1] > 1


def calculate_rms(samples: Tensor) -> Tensor:
    """
    Calculates the root mean square.

    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples), dim=-1, keepdim=False))


def convert_decibels_to_amplitude_ratio(decibels: float) -> float:
    return 10 ** (decibels / 20)


def convert_frequencies_to_mels(f: torch.Tensor) -> torch.Tensor:
    """
    Convert f hertz to m mels

    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def convert_mels_to_frequencies(m: torch.Tensor) -> Any:
    """
    Convert m mels to f hertz

    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


class ObjectDict(_ObjectDictBase):
    """
    Make a dictionary behave like an object, with attribute-style access.

    Here are some examples of how it can be used:

    o = ObjectDict(my_dict)
    # or like this:
    o = ObjectDict(samples=samples, sample_rate=sample_rate)

    # Attribute-style access
    samples = o.samples

    # Dict-style access
    samples = o["samples"]
    """

    def __getattr__(self, name):
        # type: (str) -> typing.Any
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        # type: (str, typing.Any) -> None
        self[name] = value


def find_audio_files_in_paths(
    paths: List[str],
    filename_endings: Tuple[str] = SUPPORTED_EXTENSIONS,
    traverse_subdirectories: bool = True,
    follow_symlinks: bool = True,
) -> List[str]:
    """
    Return a list of paths to all audio files with the given extension(s) contained in the list or in its directories.
    Also traverses subdirectories by default.
    """

    file_paths = []

    for p in paths:
        if str(p).lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = os.path.abspath(p)
            file_paths.append(file_path)
        elif os.path.isdir(p):
            file_paths += find_audio_files(
                p,
                filename_endings=filename_endings,
                traverse_subdirectories=traverse_subdirectories,
                follow_symlinks=follow_symlinks,
            )
    return file_paths


def find_audio_files(
    root_path: str,
    filename_endings: Tuple[str] = SUPPORTED_EXTENSIONS,
    traverse_subdirectories: bool = True,
    follow_symlinks: bool = True,
) -> List[str]:
    """
    Return a list of paths to all audio files with the given extension(s) in a directory.
    Also traverses subdirectories by default.
    """
    file_paths = []

    for root, _, filenames in os.walk(root_path, followlinks=follow_symlinks):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.lower().endswith(filename_endings):
                file_paths.append(file_path)
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


AudioFile = Union[Path, Text, dict]


class Audio:
    """
    Audio IO with on-the-fly resampling

    Parameters
    ----------
    sample_rate: int
        Target sample rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.

    Usage
    -----
    >>> audio = Audio(sample_rate=16000)
    >>> samples = audio("/path/to/audio.wav")

    # on-the-fly resampling
    >>> original_sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * original_sample_rate)
    >>> samples = audio({"samples": two_seconds_stereo, "sample_rate": original_sample_rate})
    >>> assert samples.shape[1] == 2 * 16000
    """

    @staticmethod
    def is_valid(file: AudioFile) -> bool:
        if isinstance(file, dict):
            if "samples" in file:
                samples = file["samples"]
                if len(samples.shape) != 2 or samples.shape[0] > samples.shape[1]:
                    raise ValueError(
                        "'samples' must be provided as a (channel, time) torch.Tensor."
                    )

                sample_rate = file.get("sample_rate", None)
                if sample_rate is None:
                    raise ValueError(
                        "'samples' must be provided with their 'sample_rate'."
                    )
                return True

            elif "audio" in file:
                return True

            else:
                # TODO improve error message
                raise ValueError("either 'audio' or 'samples' key must be provided.")

        return True

    @staticmethod
    def rms_normalize(samples: Tensor) -> Tensor:
        """
        Power-normalize samples

        Parameters
        ----------
        samples : (..., time) Tensor
            Single (or multichannel) samples or batch of samples

        Returns
        -------
        samples: (..., time) Tensor
            Power-normalized samples
        """
        rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
        return samples / (rms + 1e-8)

    @staticmethod
    def get_audio_metadata(file_path: Union[Path, str]) -> Tuple[Tensor, Tensor]:
        """Return (num_samples, sample_rate)."""
        info = torchaudio.info(file_path)
        # Deal with backwards-incompatible signature change.
        # See https://github.com/pytorch/audio/issues/903 for more information.
        if type(info) is tuple:
            si, _ = info
            num_samples = si.length
            sample_rate = si.rate
        else:
            num_samples = info.num_frames
            sample_rate = info.sample_rate
        return num_samples, sample_rate

    def get_num_samples(self, file: AudioFile) -> Any:
        """Number of samples (in target sample rate)

        :param file: audio file

        """

        self.is_valid(file)

        if isinstance(file, dict):
            if "samples" in file:
                num_samples = file["samples"].shape[1]
                sample_rate = file["sample_rate"]

            else:
                num_samples, sample_rate = self.get_audio_metadata(file["audio"])

        #  file = str or Path
        else:
            num_samples, sample_rate = self.get_audio_metadata(file)

        return math.floor(num_samples * self.sample_rate / sample_rate)

    def __init__(self, sample_rate: int, mono: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(self, samples: Tensor, sample_rate: int) -> Tensor:
        """
        Downmix and resample

        Parameters
        ----------
        samples : (channel, time) Tensor
            Samples.
        sample_rate : int
            Original sample rate.

        Returns
        -------
        samples : (channel, time) Tensor
            Remixed and resampled samples
        """

        # downmix to mono
        if self.mono and samples.shape[0] > 1:
            samples = samples.mean(dim=0, keepdim=True)

        # resample
        if self.sample_rate != sample_rate:
            tmp = samples.numpy()
            if self.mono:
                # librosa expects mono audio to be of shape (n,), but we have (1, n).
                tmp = librosa.core.resample(
                    tmp[0], orig_sr=sample_rate, target_sr=self.sample_rate
                )[None]
            else:
                tmp = librosa.core.resample(
                    tmp.T, orig_sr=sample_rate, target_sr=self.sample_rate
                ).T

            samples = torch.tensor(tmp)

        return samples

    def __call__(
        self, file: AudioFile, sample_offset: int = 0, num_samples: Optional[int] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        file : AudioFile
            Audio file.
        sample_offset : int, optional
            Start loading at this `sample_offset` sample. Defaults ot 0.
        num_samples : int, optional
            Load that many samples. Defaults to load up to the end of the file.

        Returns
        -------
        samples : (time, channel) torch.Tensor
            Samples

        """

        self.is_valid(file)

        original_samples = None

        if isinstance(file, dict):
            # file = {"samples": torch.Tensor, "sample_rate": int, [ "channel": int ]}
            if "samples" in file:
                original_samples = file["samples"]
                original_sample_rate = file["sample_rate"]
                original_total_num_samples = original_samples.shape[1]
                channel = file.get("channel", None)

            # file = {"audio": str or Path, [ "channel": int ]}
            else:
                audio_path = str(file["audio"])
                (
                    original_total_num_samples,
                    original_sample_rate,
                ) = self.get_audio_metadata(audio_path)
                channel = file.get("channel", None)

        else:
            audio_path = str(file)
            original_total_num_samples, original_sample_rate = self.get_audio_metadata(
                audio_path
            )
            channel = None

        original_sample_offset = round(
            sample_offset * original_sample_rate / self.sample_rate
        )
        if num_samples is None:
            original_num_samples = original_total_num_samples - original_sample_offset
        else:
            original_num_samples = round(
                num_samples * original_sample_rate / self.sample_rate
            )

        if original_sample_offset + original_num_samples > original_total_num_samples:
            raise ValueError(
                f"Sample offset {original_sample_offset} -- number of samples {original_num_samples} -- total number of samples {original_total_num_samples}."
            )

        if original_samples is None:
            try:
                original_data, _ = torchaudio.load(
                    audio_path,
                    frame_offset=original_sample_offset,
                    num_frames=original_num_samples,
                )
            except TypeError:
                raise Exception(
                    "It looks like you are using an unsupported version of torchaudio."
                    " If you have 0.6 or older, please upgrade to a newer version."
                )

        else:
            original_data = original_samples[
                :,
                original_sample_offset : original_sample_offset + original_num_samples,
            ]

        if channel is not None:
            original_data = original_data[channel - 1 : channel, :]

        result = self.downmix_and_resample(original_data, original_sample_rate)

        if num_samples is not None:
            # If there is an off-by-one error in the length (e.g. due to resampling), fix it.
            if result.shape[-1] > num_samples:
                result = result[:, :num_samples]
            elif result.shape[-1] < num_samples:
                diff = num_samples - result.shape[-1]
                result = torch.nn.functional.pad(result, (0, diff))

        return result
