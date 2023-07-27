# BEATs

Missing folders:
- "checkpoints": for model and tokenizer weights
- "files": got all the audio files no matter where or in which subfolder (ONLY AUDIO FILES)


model.tar.gz
├─ code/
│ ├── inference.py
│ └── requirements.txt
└── yolov8l.pt

1- Rename model for practicity: BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
to BEATs
2- $ tar -czvf model.tar.gz code/ BEATs.pt
3- Upload tar to S3


2023-07-25T17:52:36,211 [INFO ] W-9015-model_1.0-stdout MODEL_LOG - Executing model_fn from inference.py /opt/ml/model …