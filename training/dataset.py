import json
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore
import tensorflow_io as tfio  # type: ignore
import torch
import torchaudio  # type: ignore
from torch.utils.data import IterableDataset

from augmentation import AugmentFP


def yield_ids(song_path: str) -> Iterator[Dict[str, Any]]:
    for idx in np.random.permutation(len(song_path)):
        yield {"idx": idx, "song_path": song_path[idx]}


@tf.function(experimental_relax_shapes=True)  # type: ignore
def load_audio(
    data: Dict[str, Any],
    sr: int,
    mono: bool,
    do_norm: bool,
    max_dur_in_minutes: float = 10.0,
) -> Dict[str, Any]:
    def py_load_audio(
        song_path: tf.string, max_n_samples: int, sr: tf.int32
    ) -> Tuple[npt.NDArray[np.float64], int]:
        try:
            x, sr_in = torchaudio.load(
                song_path.numpy().decode("utf-8"),
                channels_first=False,
                num_frames=max_n_samples,
            )
        except:
            sr_in = sr
            x = torch.zeros(max_n_samples, 2)
        if mono:
            x = torch.mean(x, dim=1).unsqueeze(-1)
        return x.numpy(), sr_in

    max_n_samples = tf.cast(max_dur_in_minutes * 60 * 44100, tf.int64)
    audio, sr_in = tf.py_function(
        py_load_audio,
        [data["song_path"], max_n_samples, sr],
        (tf.float32, tf.int32),
    )
    if sr_in != sr:
        audio = tfio.audio.resample(
            audio, tf.cast(sr_in, tf.int64), tf.cast(sr, tf.int64)
        )

    if do_norm:
        audio = tf.where(
            tf.reduce_max(tf.abs(audio), keepdims=True) != 0,
            x=tf.divide(audio, tf.reduce_max(tf.abs(audio), keepdims=True)),
            y=audio,
        )
    return {
        "audio": audio,
        "idx": data["idx"],
        "duration": float(len(audio)) / float(sr),
    }


def chunk_audio(
    audio: npt.NDArray[np.float32],
    dur: float,
    sr: int,
    step_per_cent: float,
    perm: List[List[int]],
) -> tf.Tensor:
    def seconds_to_samples(x: float, sr: int) -> tf.Tensor:
        return tf.cast(x * tf.cast(sr, tf.float32), tf.int32)

    x = tf.signal.frame(
        tf.transpose(audio, perm=perm[0]),
        frame_length=seconds_to_samples(dur, sr),
        frame_step=seconds_to_samples(dur * step_per_cent, sr),
    )
    return tf.transpose(x, perm=perm[1])


def select_no_silence_frames(
    audio: tf.Tensor,
    segments: tf.Tensor,
    dbs_threshold: float = -7.5,
) -> Tuple[tf.Tensor, tf.Tensor]:
    rms_ref = lambda x: tf.math.sqrt(tf.reduce_mean(tf.math.pow(x, 2)))
    rms_segments = lambda x: tf.math.sqrt(
        tf.reduce_mean(tf.math.pow(x, 2), axis=[-1, -2])
    )
    py_mask_audio_fn = lambda x, mask: tf.boolean_mask(x, mask)
    # non "silence" segments
    dbs = 10 * tf.math.log(rms_segments(segments) / rms_ref(audio))  # type: ignore
    # "silence" = segments with rms values below dbs_threshold
    mask = tf.math.greater(dbs, dbs_threshold)
    indices = tf.py_function(
        py_mask_audio_fn, [tf.range(tf.shape(segments)[0]), mask], (tf.int32)
    )
    segments = tf.py_function(py_mask_audio_fn, [segments, mask], (tf.float32))
    indices_tmp = tf.random.shuffle(tf.range(tf.shape(indices)[0]))
    segments = tf.gather(segments, indices_tmp, axis=0)
    indices = tf.gather(indices, indices_tmp, axis=0)
    return (segments, indices)


def get_segments(
    data: Dict[str, Any],
    duration: float,
    sr: int,
    n_segments: int,
    step_per_cent: float = 1.0,
) -> Dict[str, Any]:
    _ = data.pop("duration")
    segments = chunk_audio(
        data["audio"], duration, sr, step_per_cent, perm=[[1, 0], [1, 2, 0]]
    )
    segments, _ = select_no_silence_frames(data["audio"], segments)
    data["audio"] = tf.random.shuffle(segments)[:n_segments]
    if "class" in data:
        d = tf.shape(data["audio"])[0]
        data["class"] = tf.tile(tf.expand_dims(data["class"], axis=0), [d])
        data["label"] = tf.tile(tf.expand_dims(data["label"], axis=0), [d])
        data["task"] = tf.tile(tf.expand_dims(data["task"], axis=0), [d])
    if "idx" in data:
        d = tf.shape(data["audio"])[0]
        data["idx"] = tf.tile(tf.expand_dims(data["idx"], axis=0), [d])
    return data


def augment_audio(
    data: Dict[str, Any],
    sr: int,
    noise_paths: Dict[str, List[str]],
) -> Dict[str, Any]:
    def augment_wrapper(
        input_data: tf.Tensor,
        sr: tf.Tensor,
    ) -> tf.Tensor:
        af = AugmentFP(noise_paths, sr.numpy())
        input_audio = torch.tensor(input_data.numpy()).T
        aug = af(input_audio)

        return aug.T.numpy()

    aug = tf.py_function(
        augment_wrapper,
        [data["audio"], sr],
        tf.float32,
    )
    return {"clean_audio": data["audio"], "aug_audio": aug}


class AugmentationDataset(IterableDataset):  # type: ignore
    """
    Dataset for Denoising Experiments
    """

    def __init__(
        self,
        paths: npt.NDArray[Any],
        sampling_frequency: int,
        mono: bool = True,
        n_segments: int = 1,
        model_duration_seconds: float = 3.0,
        do_norm: bool = True,
        buffer_size: int = 32,
        noise_split: str = "train",
    ) -> None:
        """
        Args:
            paths (np.array):
                Paths to audios.
            sampling_frequency (int):
                Self-explanatory.
            mono (bool):
                Self-explanatory.
            n_segments (int):
                Number of segments to use when training.
            model_duration_seconds (float):
                Training audio duration in seconds.
            do_norm (bool):
                Normalize waveform?
            buffer_size (int):
                TF buffer size for parrallelization.
        """

        if noise_split == "train":
            with open("training/splits/train.json", "r") as f:
                self.noise_paths = json.load(f)

        if noise_split == "val":
            with open("training/splits/val.json", "r") as f:
                self.noise_paths = json.load(f)

        if noise_split == "test":
            with open("training/splits/test.json", "r") as f:
                self.noise_paths = json.load(f)

        self.song_paths = paths
        self.sampling_frequency = sampling_frequency

        taxonomy = {
            "song_path": tf.TensorSpec(shape=(), dtype=tf.string),
            "idx": tf.TensorSpec(shape=(), dtype=tf.int32),
        }
        # self.af = AugmentFP(self.noise_paths, self.sampling_frequency)

        self.tf_dataloader = (
            tf.data.Dataset.from_generator(
                yield_ids,
                output_signature=taxonomy,
                args=[self.song_paths],
            )
            .map(
                lambda x: load_audio(x, self.sampling_frequency, mono, do_norm),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda x: get_segments(
                    x, model_duration_seconds, self.sampling_frequency, n_segments
                )
            )
            .filter(lambda x: tf.greater_equal(tf.shape(x["audio"])[0], 1))
            .unbatch()
            .shuffle(buffer_size, reshuffle_each_iteration=True)
            .repeat()
            .map(
                lambda x: augment_audio(
                    x,
                    self.sampling_frequency,
                    self.noise_paths,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get audio and accompanying tags.
        ---
        Returns:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                clean_audio:
                    Clean song audio of size [batch_size, num_channels, duration * sample_rate].
                aug_audio:
                    Augmented song audio of size [batch_size, num_channels, duration * sample_rate].
        """
        for data in self.tf_dataloader.as_numpy_iterator():
            yield (torch.Tensor(data["clean_audio"]), torch.Tensor(data["aug_audio"]))
