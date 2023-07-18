import torchaudio


def resample_audio(input_path, target_sr=16000):

    # Set audiobackend to soundfile (catched torchaudio error)
    torchaudio.set_audio_backend("soundfile")
    # Load audio file from input_path
    waveform, original_sr = torchaudio.load(input_path)
    # Resample to new sample rate (ouput as a Tensor)
    resampled_waveform = torchaudio.transforms.Resample(original_sr, target_sr)(waveform)
    return resampled_waveform
