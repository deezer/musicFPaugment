import abc
from time import time
from typing import Dict, List, Tuple

import numpy as np
from dejavu.dejavu import read
from dejavu.variables import MIN_HASHES

from testing.parameters import afp_settings


class BaseRecognizer(object, metaclass=abc.ABCMeta):
    def __init__(self, dejavu):
        self.dejavu = dejavu
        self.Fs = afp_settings["dejavu"]["samplerate"]

    def _recognize(self, *data) -> Tuple[List[Dict[str, any]], int, int, int]:
        fingerprint_times = []
        hashes = set()  # to remove possible duplicated fingerprints we built a set.

        for channel in data:
            fingerprints, fingerprint_time = self.dejavu.generate_fingerprints(channel)

            fingerprint_times.append(fingerprint_time)
            hashes |= set(fingerprints)

        matches, dedup_hashes, query_time = self.dejavu.find_matches(hashes)

        t = time()
        final_results = self.dejavu.align_matches(matches, dedup_hashes, len(hashes))

        align_time = time() - t

        return final_results, np.sum(fingerprint_times), query_time, align_time

    @abc.abstractmethod
    def recognize(self) -> Dict[str, any]:
        pass  # base class does nothing


class FileRecognizer(BaseRecognizer):
    def __init__(self, dejavu):
        super().__init__(dejavu)

    def recognize_file(self, filename: str) -> Dict[str, any]:
        channels, self.Fs, _ = read(
            filename,
            denoising=self.dejavu.denoising,
            denoising_model=self.dejavu.denoising_model,
        )

        t = time()
        matches, fingerprint_time, query_time, align_time = self._recognize(*channels)
        t = time() - t

        if len(matches):
            if matches[0]["nb_matches_with_offset"] > MIN_HASHES:
                is_match = True
            else:
                is_match = False
        else:
            is_match = False

        # if matches[0]['nb_matches_with_offset'] <= MIN_HASHES:
        #     matches = [{}]

        results = {
            "total_time": t,
            "fingerprint_time": fingerprint_time,
            "query_time": query_time,
            "align_time": align_time,
            "results": matches,
            "match": is_match,
        }
        return results

    def recognize(self, filename: str) -> Dict[str, any]:
        return self.recognize_file(filename)
