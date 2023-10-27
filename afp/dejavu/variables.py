# SETTINGS:
# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
# DEFAULT_FS = 11025 #44100

# Size of the FFT window, affects frequency granularity
# DEFAULT_WINDOW_SIZE = 512 #4096

# Ratio by which each sequential window overlaps the last and the
# next window. Higher overlap will allow a higher granularity of offset
# matching, but potentially more fingerprints.
# DEFAULT_OVERLAP_RATIO = 0.5

# Degree to which a fingerprint can be paired with its neighbors. Higher values will
# cause more fingerprints, but potentially better accuracy.
# DEFAULT_FAN_VALUE = 5  # 15 was the original value.

CONNECTIVITY_MASK = 2
PEAK_NEIGHBORHOOD_SIZE = 10  # 20 was the original value.
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 20

OFFSET = "offset"
OFFSET_SECS = "offset_seconds"


SONG_ID = "song_id"
SONG_NAME = "song_name"
INPUT_HASHES = "input_total_hashes"
FINGERPRINTED_HASHES = "fingerprinted_hashes_in_db"
HASHES_MATCHED = "hashes_matched_in_input"
INPUT_CONFIDENCE = "input_confidence"
INPUT_CONFIDENCE_2 = "input_confidence_2"
FINGERPRINTED_CONFIDENCE = "fingerprinted_confidence"
OFFSET = "offset"
OFFSET_SECS = "offset_seconds"

TOPN = 1
MIN_HASHES = 1
RESULTS = "results"
