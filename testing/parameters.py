WAVEFORM_SAMPLING_RATE = 8000

queries_paths = {
    "cleans": "/workspace/queries/cleans",
    "bn_m10": "/workspace/queries/bn_m10",
    "bn_m5": "/workspace/queries/bn_m5",
    "bn_0": "/workspace/queries/bn_0",
    "bn_p5": "/workspace/queries/bn_p5",
    "bn_p10": "/workspace/queries/bn_p10",
    "reverb": "/workspace/queries/reverb",
    "recording_device": "/workspace/queries/recording_device",
    "full_light": "/workspace/queries/full_light",
    "full_hard": "/workspace/queries/full_hard",
    "default_parameters": "/workspace/queries/default_parameters",
}

afp_settings = {
    "audfprint": {
        "density": 20,
        "pks-per-frame": 5,
        "freq-sd": 30,
        "shifts": 1,
        "samplerate": 8000,
        "n_fft": 512,
        "n_hop": 256,
    },
    "dejavu": {
        "samplerate": 8000,
        "n_fft": 512,
        "n_hop": int(0.5 * 512),
        "fan_value": 3,  # Degree to which a fingerprint can be paired with its neighbors. Higher values will cause more fingerprints, but potentially better accuracy.
        "amp_min": 50,
        "peak_neighb_size": 10,
    },
}

afp_db_paths = {
    "audfprint": "/workspace/src/afp/audfprint/fp_database.pklz",
    "dejavu": {
        "database": {
            "host": "db_fma",
            "user": "postgres",
            "password": "password",
            "database": "dejavu_fma",
        },
    },
}


## Background noise:

bn_m10_params = {
    "proba_cutoff_freq1": 0,
    "proba_snr_in_db": 1,
    "proba_ir_response": 0,
    "proba_gain_in_db": 0,
    "proba_percentile_threshold": 0,
    "proba_cutoff_freq2": 0,
    "proba_cutoff_freq3": 0,
    "min_snr_in_db": -10,
    "max_snr_in_db": -10,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 0.1,
    "min_gain_in_db": 0,
    "max_gain_in_db": 0.1,
    "max_percentile_threshold": 0.1,
    "min_cutoff_freq2": 0,
    "max_cutoff_freq2": 0.1,
    "min_cutoff_freq3": 0,
    "max_cutoff_freq3": 0.1,
}
bn_m5_params = {
    "proba_cutoff_freq1": 0,
    "proba_snr_in_db": 1,
    "proba_ir_response": 0,
    "proba_gain_in_db": 0,
    "proba_percentile_threshold": 0,
    "proba_cutoff_freq2": 0,
    "proba_cutoff_freq3": 0,
    "min_snr_in_db": -5,
    "max_snr_in_db": -5,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 0.1,
    "min_gain_in_db": 0,
    "max_gain_in_db": 0.1,
    "max_percentile_threshold": 0.1,
    "min_cutoff_freq2": 0,
    "max_cutoff_freq2": 0.1,
    "min_cutoff_freq3": 0,
    "max_cutoff_freq3": 0.1,
}

bn_0_params = {
    "proba_cutoff_freq1": 0,
    "proba_snr_in_db": 1,
    "proba_ir_response": 0,
    "proba_gain_in_db": 0,
    "proba_percentile_threshold": 0,
    "proba_cutoff_freq2": 0,
    "proba_cutoff_freq3": 0,
    "min_snr_in_db": 0,
    "max_snr_in_db": 0,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 0.1,
    "min_gain_in_db": 0,
    "max_gain_in_db": 0.1,
    "max_percentile_threshold": 0.1,
    "min_cutoff_freq2": 0,
    "max_cutoff_freq2": 0.1,
    "min_cutoff_freq3": 0,
    "max_cutoff_freq3": 0.1,
}

bn_p5_params = {
    "proba_cutoff_freq1": 0,
    "proba_snr_in_db": 1,
    "proba_ir_response": 0,
    "proba_gain_in_db": 0,
    "proba_percentile_threshold": 0,
    "proba_cutoff_freq2": 0,
    "proba_cutoff_freq3": 0,
    "min_snr_in_db": 5,
    "max_snr_in_db": 5,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 0.1,
    "min_gain_in_db": 0,
    "max_gain_in_db": 0.1,
    "max_percentile_threshold": 0.1,
    "min_cutoff_freq2": 0,
    "max_cutoff_freq2": 0.1,
    "min_cutoff_freq3": 0,
    "max_cutoff_freq3": 0.1,
}

bn_p10_params = {
    "proba_cutoff_freq1": 0,
    "proba_snr_in_db": 1,
    "proba_ir_response": 0,
    "proba_gain_in_db": 0,
    "proba_percentile_threshold": 0,
    "proba_cutoff_freq2": 0,
    "proba_cutoff_freq3": 0,
    "min_snr_in_db": 10,
    "max_snr_in_db": 10,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 0.1,
    "min_gain_in_db": 0,
    "max_gain_in_db": 0.1,
    "max_percentile_threshold": 0.1,
    "min_cutoff_freq2": 0,
    "max_cutoff_freq2": 0.1,
    "min_cutoff_freq3": 0,
    "max_cutoff_freq3": 0.1,
}


## Reverb alone:

reverb_params = {
    "proba_cutoff_freq1": 0,
    "proba_snr_in_db": 0,
    "proba_ir_response": 1,
    "proba_gain_in_db": 0,
    "proba_percentile_threshold": 0,
    "proba_cutoff_freq2": 0,
    "proba_cutoff_freq3": 0,
    "min_snr_in_db": 0,
    "max_snr_in_db": 0,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 0.1,
    "min_gain_in_db": 0,
    "max_gain_in_db": 0.1,
    "max_percentile_threshold": 0.1,
    "min_cutoff_freq2": 0,
    "max_cutoff_freq2": 0.1,
    "min_cutoff_freq3": 0,
    "max_cutoff_freq3": 0.1,
}


## Recording device:

recording_device_params = {
    "proba_cutoff_freq1": 0,
    "proba_snr_in_db": 0,
    "proba_ir_response": 0,
    "proba_gain_in_db": 1,
    "proba_percentile_threshold": 1,
    "proba_cutoff_freq2": 1,
    "proba_cutoff_freq3": 1,
    "min_snr_in_db": 0,
    "max_snr_in_db": 0,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 0.1,
    "min_gain_in_db": -5.0,
    "max_gain_in_db": 5.0,
    "max_percentile_threshold": 0.01,
    "min_cutoff_freq2": 3000,
    "max_cutoff_freq2": 3999,
    "min_cutoff_freq3": 30,
    "max_cutoff_freq3": 150,
}

## Full pipelines:
light_parameters = {
    "proba_cutoff_freq1": 1,
    "proba_snr_in_db": 1,
    "proba_ir_response": 1,
    "proba_gain_in_db": 1,
    "proba_percentile_threshold": 1,
    "proba_cutoff_freq2": 1,
    "proba_cutoff_freq3": 1,
    "min_cutoff_freq1": 0,
    "max_cutoff_freq1": 30,
    "min_snr_in_db": 0,
    "max_snr_in_db": 5,
    "min_gain_in_db": -0.5,
    "max_gain_in_db": 0.5,
    "max_percentile_threshold": 0.0001,
    "min_cutoff_freq2": 3500,
    "max_cutoff_freq2": 3999,
    "min_cutoff_freq3": 0,
    "max_cutoff_freq3": 20,
}


hard_parameters = {
    "proba_cutoff_freq1": 1,
    "proba_snr_in_db": 1,
    "proba_ir_response": 1,
    "proba_gain_in_db": 1,
    "proba_percentile_threshold": 1,
    "proba_cutoff_freq2": 1,
    "proba_cutoff_freq3": 1,
    "min_cutoff_freq1": 0,
    "max_cutoff_freq1": 150,
    "min_snr_in_db": -5,
    "max_snr_in_db": 0,
    "min_gain_in_db": -5,
    "max_gain_in_db": 5,
    "max_percentile_threshold": 0.01,
    "min_cutoff_freq2": 3000,
    "max_cutoff_freq2": 3500,
    "min_cutoff_freq3": 30,
    "max_cutoff_freq3": 150,
}

default_parameters = {
    "proba_cutoff_freq1": 1,
    "proba_snr_in_db": 1,
    "proba_ir_response": 1,
    "proba_gain_in_db": 1,
    "proba_percentile_threshold": 1,
    "proba_cutoff_freq2": 1,
    "proba_cutoff_freq3": 1,
    "min_cutoff_freq1": 0.0,
    "max_cutoff_freq1": 150.0,
    "min_snr_in_db": -10,
    "max_snr_in_db": 10,
    "min_gain_in_db": -5.0,
    "max_gain_in_db": 5.0,
    "max_percentile_threshold": 0.01,
    "min_cutoff_freq2": 3000.0,
    "max_cutoff_freq2": 3999.0,
    "min_cutoff_freq3": 30.0,
    "max_cutoff_freq3": 150.0,
}

test_pipelines_parameters = {
    "bn_m10": bn_m10_params,
    "bn_m5": bn_m5_params,
    "bn_0": bn_0_params,
    "bn_p5": bn_p5_params,
    "bn_p10": bn_p10_params,
    "reverb": reverb_params,
    "recording_device": recording_device_params,
    "full_light": light_parameters,
    "full_hard": hard_parameters,
    "default_parameters": default_parameters,
}
