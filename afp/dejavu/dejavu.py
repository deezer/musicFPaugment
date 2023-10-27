import os
import pickle
from hashlib import sha1
from itertools import groupby
from time import time
from typing import Dict, List, Tuple

# from dejavu.third_party import wavio
import torch
import torchaudio
from dejavu.fingerprint import fingerprint
from dejavu.postgres_database import PostgreSQLDatabase
from dejavu.variables import (
    FINGERPRINTED_CONFIDENCE,
    FINGERPRINTED_HASHES,
    HASHES_MATCHED,
    INPUT_CONFIDENCE,
    INPUT_CONFIDENCE_2,
    INPUT_HASHES,
    OFFSET,
    OFFSET_SECS,
    SONG_ID,
    SONG_NAME,
    TOPN,
)
from torchaudio.transforms import Resample
from tqdm import tqdm

from testing.parameters import WAVEFORM_SAMPLING_RATE, afp_settings
from training.model import Demucs
from training.utils import set_gpus

device = set_gpus()

demucs = Demucs().to(device)
weights_path = (
    "/workspace/src/training/checkpoints/demucs_lr_0.0005_BS_128/best_epoch.pt"
)
demucs.load_state_dict(
    torch.load(weights_path, map_location=device)["model_state_dict"]
)
demucs.eval()


def unique_hash(file_path: str, block_size: int = 2**20) -> str:
    """Small function to generate a hash to uniquely generate
    a file. Inspired by MD5 version here:
    http://stackoverflow.com/a/1131255/712997

    Works with large files.

    :param file_path: path to file.
    :param block_size: read block size.
    :return: a hash in an hexagesimal string form.
    """
    s = sha1()
    with open(file_path, "rb") as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            s.update(buf)
    return s.hexdigest().upper()


def read(
    filename: str, denoising: bool = False, denoising_model: str = "unet"
) -> Tuple[List[List[int]], int, str]:
    """
    Reads any file supported by pydub (ffmpeg) and returns the data contained
    within. If file reading fails due to input being a 24-bit wav file,
    wavio is used as a backup.

    Can be optionally limited to a certain amount of seconds from the start
    of the file by specifying the `limit` parameter. This is the amount of
    seconds from the start of the file.

    :param filename: file to be read.
    :param limit: number of seconds to limit.
    :return: tuple list of (channels, sample_rate, content_file_hash).
    """
    # pydub does not support 24-bit wav files, use wavio when this occurs
    # try:
    ## Here : load audios in mono with torchaudio
    if denoising is True:
        assert denoising_model in ["demucs", "unet"]

    sr = afp_settings["dejavu"]["samplerate"]
    resampling = Resample(WAVEFORM_SAMPLING_RATE, sr)

    if filename.split(".")[-1] == "pkl":
        with open(
            filename,
            "rb",
        ) as f:
            audio = pickle.load(f)
            audio = torch.tensor(audio).squeeze(0)

            if denoising is True and denoising_model == "demucs":
                with torch.no_grad():
                    audio = (
                        demucs(audio.unsqueeze(0).to(device))
                        .squeeze(0)
                        .squeeze(0)
                        .cpu()
                    )

            channels = [resampling(audio) * 32767]

    elif filename.split(".")[-1] == "mp3":
        audio, _ = torchaudio.load(filename)
        audio = audio.mean(axis=0)
        channels = [resampling(audio) * 32767]

    return channels, sr, unique_hash(filename)


class Dejavu:
    def __init__(
        self, config, settings, state="set", denoising=False, denoising_model=None
    ):
        self.config = config
        self.settings = settings

        # Setup database:
        self.db = PostgreSQLDatabase(**config.get("database", {}))
        self.denoising = denoising
        self.denoising_model = denoising_model

        if self.denoising is True:
            assert self.denoising_model in ["unet", "demucs"]

        if state == "set":
            self.db.setup()
        elif state == "clear":
            self.db.empty()

        self.__load_fingerprinted_audio_hashes()

    def __load_fingerprinted_audio_hashes(self) -> None:
        """
        Keeps a dictionary with the hashes of the fingerprinted songs, in that way is possible to check
        whether or not an audio file was already processed.
        """
        # get songs previously indexed
        self.songs = self.db.get_songs()
        self.songhashes_set = set()  # to know which ones we've computed before
        for song in self.songs:
            song_hash = song["file_sha1"]
            self.songhashes_set.add(song_hash)

    def fingerprint_directory(
        self, mp3_path_list: list, nprocesses: int = None
    ) -> None:
        """
        Given a directory and a set of extensions it fingerprints all files that match each extension specified.

        :param path_list: list of files to be fingerprinted
        :param nprocesses: amount of processes to fingerprint the files within the directory.
        """
        # Try to use the maximum amount of processes if not given.
        # try:
        #     nprocesses = nprocesses or multiprocessing.cpu_count()
        # except NotImplementedError:
        #     nprocesses = 1
        # else:
        #    nprocesses = 1 if nprocesses <= 0 else nprocesses

        # pool = multiprocessing.Pool(nprocesses)

        # filenames_to_fingerprint = []

        # for filename in mp3_path_list:
        #     # don't refingerprint already fingerprinted files
        #     if unique_hash(filename) in self.songhashes_set:
        #         print(f"{filename} already fingerprinted, continuing...")
        #         continue

        #   filenames_to_fingerprint.append(filename)

        # Prepare _fingerprint_worker input
        worker_input = list(zip(mp3_path_list, [None] * len(mp3_path_list)))

        # Send off our tasks
        # iterator = pool.imap_unordered(Dejavu._fingerprint_worker, worker_input)

        # Loop till we have all of them
        # while True:
        #     try:
        #         song_name, hashes, file_hash = next(iterator)
        #     except multiprocessing.TimeoutError:
        #         continue
        #     except StopIteration:
        #         break
        #     except Exception:
        #         print("Failed fingerprinting")
        #         # Print traceback because we can't reraise it here
        #         traceback.print_exc(file=sys.stdout)
        #     else:
        #         sid = self.db.insert_song(song_name, file_hash, len(hashes))
        #         self.db.insert_hashes(sid, hashes)
        #         self.db.set_song_fingerprinted(sid)
        #         self.__load_fingerprinted_audio_hashes()
        for i in tqdm(range(len(worker_input))):
            if unique_hash(worker_input[i][0]) in self.songhashes_set:
                # print(f"{worker_input[i][0]} already fingerprinted, continuing...")
                continue

            song_name, hashes, file_hash = self._fingerprint_worker(worker_input[i])
            sid = self.db.insert_song(song_name, file_hash, len(hashes))
            self.db.insert_hashes(sid, hashes)
            self.db.set_song_fingerprinted(sid)
            self.__load_fingerprinted_audio_hashes()

        # pool.close()
        # pool.join()

    @staticmethod
    def _fingerprint_worker(arguments):
        # Pool.imap sends arguments as tuples so we have to unpack
        # them ourself.
        try:
            file_name, limit = arguments
        except ValueError:
            pass

        song_name, extension = os.path.splitext(os.path.basename(file_name))

        fingerprints, file_hash = Dejavu.get_file_fingerprints(
            file_name, print_output=False
        )

        return song_name, fingerprints, file_hash

    @staticmethod
    def get_file_fingerprints(file_name: str, print_output: bool = False):
        channels, fs, file_hash = read(file_name)
        fingerprints = set()
        channel_amount = len(channels)
        for channeln, channel in enumerate(channels, start=1):
            if print_output:
                print(
                    f"Fingerprinting channel {channeln}/{channel_amount} for {file_name}"
                )

            hashes = fingerprint(channel, Fs=fs)
            if print_output:
                print(f"Finished channel {channeln}/{channel_amount} for {file_name}")

            fingerprints |= set(hashes)

        return fingerprints, file_hash

    def generate_fingerprints(
        self, samples: List[int], get_masks: bool = False
    ) -> Tuple[List[Tuple[str, int]], float]:
        """
        Generate the fingerprints for the given sample data (channel).

        :param samples: list of ints which represents the channel info of the given audio file.
        :param Fs: sampling rate which defaults to 8000.
        :return: a list of tuples for hash and its corresponding offset, together with the generation time.
        """
        Fs = self.settings["samplerate"]
        t = time()

        if get_masks is False:
            hashes = fingerprint(
                samples,
                Fs=Fs,
                denoising=self.denoising,
                denoising_model=self.denoising_model,
                get_masks=get_masks,
            )
        if get_masks is True:
            channels, _, _ = read(
                samples, denoising=self.denoising, denoising_model=self.denoising_model
            )

            hashes, peak_mask, specgram = fingerprint(
                channels[0],
                Fs=Fs,
                denoising=self.denoising,
                denoising_model=self.denoising_model,
                get_masks=get_masks,
            )

            return peak_mask, specgram

        fingerprint_time = time() - t
        return hashes, fingerprint_time

    def find_matches(
        self, hashes: List[Tuple[str, int]]
    ) -> Tuple[List[Tuple[int, int]], Dict[str, int], float]:
        """
        Finds the corresponding matches on the fingerprinted audios for the given hashes.

        :param hashes: list of tuples for hashes and their corresponding offsets
        :return: a tuple containing the matches found against the db, a dictionary which counts the different
         hashes matched for each song (with the song id as key), and the time that the query took.

        """
        t = time()
        matches, dedup_hashes = self.db.return_matches(hashes)
        query_time = time() - t

        return matches, dedup_hashes, query_time

    def align_matches(
        self,
        matches: List[Tuple[int, int]],
        dedup_hashes: Dict[str, int],
        queried_hashes: int,
        topn: int = TOPN,
    ) -> List[Dict[str, any]]:
        """
        Finds hash matches that align in time with other matches and finds
        consensus about which hashes are "true" signal from the audio.

        :param matches: matches from the database
        :param dedup_hashes: dictionary containing the hashes matched without duplicates for each song
        (key is the song id).
        :param queried_hashes: amount of hashes sent for matching against the db
        :param topn: number of results being returned back.
        :return: a list of dictionaries (based on topn) with match information.
        """
        # count offset occurrences per song and keep only the maximum ones.
        sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))
        counts = [
            (*key, len(list(group)))
            for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))
        ]
        songs_matches = sorted(
            [
                max(list(group), key=lambda g: g[2])
                for key, group in groupby(counts, key=lambda count: count[0])
            ],
            key=lambda count: count[2],
            reverse=True,
        )

        songs_result = []
        for song_id, offset, _ in songs_matches[
            0:topn
        ]:  # consider topn elements in the result
            song = self.db.get_song_by_id(song_id)

            song_name = song.get(SONG_NAME, None)
            song_hashes = song.get("total_hashes", None)
            nseconds = round(
                float(offset) / self.settings["samplerate"] * self.settings["n_hop"],
                5,
            )
            hashes_matched = dedup_hashes[song_id]

            song = {
                SONG_ID: song_id,
                SONG_NAME: song_name.encode("utf8"),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: song_hashes,
                HASHES_MATCHED: hashes_matched,
                # Percentage regarding hashes matched vs hashes from the input.
                INPUT_CONFIDENCE: round(hashes_matched / queried_hashes, 2),
                # Percentage regarding hashes matched vs hashes fingerprinted in the db.
                INPUT_CONFIDENCE_2: round(songs_matches[0][2] / queried_hashes, 2),
                "nb_matches_with_offset": songs_matches[0][2],
                FINGERPRINTED_CONFIDENCE: round(hashes_matched / song_hashes, 2),
                OFFSET: offset,
                OFFSET_SECS: nseconds,
                "file_sha1": song.get("file_sha1", None).encode("utf8"),
            }

            songs_result.append(song)

        return songs_result
