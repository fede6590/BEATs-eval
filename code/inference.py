# Python Built-Ins:
import json
# import logging
# import sys
import os

# External Dependencies:
import torch
import torchaudio

# Local Dependencies:
from BEATs import BEATs, BEATsConfig

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    """Load saved model from file"""
    print(f"Executing model_fn from inference.py {model_dir} …")
    # Prueba
    env = os.environ
    model_name_1 = f"/opt/ml/model/code/{env['MODEL_NAME']}"
    print(f'prueba nombre: {model_name_1}')
    
    try:
        checkpoint = torch.load(model_dir)
    except:
        print(f'failure with dir: {model_dir}')

    try:
        checkpoint = torch.load(model_name_1)
    except:
        print(f'failure with dir: {model_name_1}')


    print("A..")
    cfg = BEATsConfig(checkpoint['cfg'])
    print("b..")
    model = BEATs(cfg)
    print("c..")
    model.load_state_dict(checkpoint['model'])
    print("d..")
    model.eval()
    print("e..")

    return model


def input_fn(request_body, request_content_type):
    print("Receiving endpoint data …")

    """Validate, de-serialize and pre-process requests"""
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    # Set audiobackend to soundfile (catched torchaudio error)
    torchaudio.set_audio_backend("soundfile")
    # Load audio file from input_path
    waveform, original_sr = torchaudio.load(data)
    # Resample to new sample rate (ouput as a Tensor)
    data = torchaudio.transforms.Resample(original_sr, 16000)(waveform)
    return data


def predict_fn(input_object, model):
    print("Inference function")

    """Execute the model on input data"""
    with torch.no_grad():
        prediction = model(input_object)
    return prediction


def output_fn(predictions, content_type):
    print("Output data …")
    """Post-process and serialize model output to API response"""
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)