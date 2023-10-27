import io
import json
import tempfile

import streamlit as st
import torch
import torchaudio
from PIL import Image
from torchaudio.transforms import Resample

from augmentation import AugmentFP

device = "cuda:1"


def load_audio():
    EXAMPLES = {
        "Clean 1": "/workspace/src/streamlit_app/examples/1_clean.wav",
        "Clean 2": "/workspace/src/streamlit_app/examples/2_clean.wav",
        "Clean 3": "/workspace/src/streamlit_app/examples/3_clean.wav",
        "Clean 4": "/workspace/src/streamlit_app/examples/4_clean.wav",
        "Clean 5": "/workspace/src/streamlit_app/examples/5_clean.wav",
        "Clean 6": "/workspace/src/streamlit_app/examples/6_clean.wav",
        "Clean 7": "/workspace/src/streamlit_app/examples/7_clean.wav",
        "Clean 8": "/workspace/src/streamlit_app/examples/8_clean.wav",
        "Clean 9": "/workspace/src/streamlit_app/examples/9_clean.wav",
        "Clean 10": "/workspace/src/streamlit_app/examples/10_clean.wav",
    }

    example_names = list(EXAMPLES.keys())
    selected_example_names = st.multiselect("Select from example(s)", example_names)

    audio_tensors = []
    sample_rates = []

    resample = Resample(44100, 16000)

    if selected_example_names:
        for example_name in selected_example_names:
            example_path = EXAMPLES[example_name]
            st.audio(example_path, format="wav")
            audio_tensor, sample_rate = torchaudio.load(example_path)
            audio_tensor = resample(audio_tensor)
            audio_tensors.append(audio_tensor)
            sample_rates.append(16000)
        return torch.stack(audio_tensors), sample_rates

    audio_files = st.file_uploader(
        "Upload audio", type=["mp3", "wav"], accept_multiple_files=True
    )

    if audio_files:
        for audio_file in audio_files:
            st.audio(audio_file, format="wav")
            if audio_file.type == "audio/x-wav":
                audio_tensor, sample_rate = torchaudio.load(
                    io.BytesIO(audio_file.read()), format=str("wav")
                )
            else:
                audio_tensor, sample_rate = torchaudio.load(
                    io.BytesIO(audio_file.read()), format=str("mp3")
                )
                audio_tensor = audio_tensor.mean(axis=0).unsqueeze(0)

            audio_tensor = resample(audio_tensor)
            audio_tensors.append(audio_tensor)
            sample_rates.append(16000)
        return torch.stack(audio_tensors), sample_rates
    else:
        return None, None


def process_audio(audios, sample_rates, model):
    # Instantiate the AudioToAudio model
    # model = train_augmentation_pipeline.to(device)
    if audios.shape[0] == 1:
        aug_audios = model(audios.squeeze(0).to(device)).cpu()
        aug_audios = aug_audios.unsqueeze(0)
    else:
        aug_audios = model.batch_augment(audios.to(device)).cpu()
    outputs = []
    i = 0
    for aug_audio in aug_audios:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, aug_audio, sample_rate=sample_rates[i])
            output_audio_bytes = f.read()

        outputs.append(output_audio_bytes)
        i += 1

    return outputs


def define_model():
    st.markdown("<h4>Loudspeakers</h4>", unsafe_allow_html=True)

    min_cutoff_freq1, max_cutoff_freq1 = st.slider(
        "-3dB cutoff freq (Hz)", min_value=0, max_value=300, step=1, value=(0, 150)
    )

    st.markdown("<h4>Room</h4>", unsafe_allow_html=True)

    reverb = st.checkbox("Reverb", value=True)

    st.markdown("<h4>Background Noise</h4>", unsafe_allow_html=True)

    min_snr, max_snr = st.slider(
        "SNR (dB)", min_value=-20, max_value=20, step=1, value=(-10, 10)
    )

    st.markdown("<h4>Recording Device</h4>", unsafe_allow_html=True)

    min_gain, max_gain = st.slider(
        "Gain (dB)", min_value=-10, max_value=10, step=1, value=(-5, 5)
    )
    max_percentile_threshold = st.slider(
        "Clipping", min_value=0.0, max_value=1.0, step=0.01, value=0.01
    )
    min_low_pass_filter, max_low_pass_filter = st.slider(
        "Low pass filter: -3dB cutoff frequency (Hz)",
        min_value=2000,
        max_value=3999,
        step=1,
        value=(3000, 3999),
    )
    min_high_pass_filter, max_high_pass_filter = st.slider(
        "High pass filter: -3dB cutoff frequency (Hz)",
        min_value=0,
        max_value=300,
        step=1,
        value=(30, 150),
    )

    original_parameters = {
        "proba_cutoff_freq1": 1.0,
        "proba_snr_in_db": 1.0,
        "proba_ir_response": int(reverb),
        "proba_gain_in_db": 1.0,
        "proba_percentile_threshold": 1.0,
        "proba_cutoff_freq2": 1.0,
        "proba_cutoff_freq3": 1.0,
        "min_cutoff_freq1": min_cutoff_freq1,
        "max_cutoff_freq1": max_cutoff_freq1,
        "min_snr_in_db": min_snr,
        "max_snr_in_db": max_snr,
        "min_gain_in_db": min_gain,
        "max_gain_in_db": max_gain,
        "max_percentile_threshold": max_percentile_threshold,
        "min_cutoff_freq2": min_low_pass_filter,
        "max_cutoff_freq2": max_low_pass_filter,
        "min_cutoff_freq3": min_high_pass_filter,
        "max_cutoff_freq3": max_high_pass_filter,
    }

    with open("training/splits/train.json", "r") as f:
        noise_paths = json.load(f)

    pipeline = AugmentFP(noise_paths, 16000, original_parameters)
    pipeline.augmentation_pipeline = pipeline.augmentation_pipeline.to(device)

    return pipeline


st.markdown(
    "<h1 style='text-align: center;'>Music Augmentation Pipeline</h1>",
    unsafe_allow_html=True,
)

img = Image.open("/workspace/src/images/SourcesOfNoise.png")
st.image(img, use_column_width=True)

# Load the input audio file
st.markdown("<h2>Parameters</h2>", unsafe_allow_html=True)

model = define_model()

st.header("Input audio")


input_audios, sample_rates = load_audio()

if input_audios is not None:
    # Process the input audio file
    output_audio_bytes = process_audio(input_audios, sample_rates, model)
    # Play the output audio file
    st.header("Augmented audios")
    for i in range(len(output_audio_bytes)):
        st.audio(output_audio_bytes[i], format="wav")
