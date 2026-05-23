# Checkpoints para Hugging Face

Repo: **charleskulkauski/fiap-gynecology-models**

Gerado por `python scripts/train_and_export_hf_models.py`.

## Estrutura

```
hf_models/
├── manifest.json
├── deployment/
│   ├── best.pt                      # YOLO medical (instrumentos)
│   ├── best_combined.pt             # YOLO combined (instrumento + bleeding)
│   └── best_instrument_bleeding.pt  # YOLO GynSurg instruments (laparoscopia)
├── yolo/
│   ├── medical_fiap-tech-challenge/best.pt
│   ├── bleeding_fiap-tech-challenge/best.pt
│   ├── combined_medical_bleeding/best.pt
│   └── gynsurg_instruments_detection/best.pt
└── action_events/
    └── gynsurg_action_3sec_round2/
        ├── action/best.pt
        ├── bleeding/best.pt
        └── smoke/best.pt
```

## Download (deploy)

```bash
pip install huggingface_hub
python scripts/download_hf_assets.py --models
```

## Upload

```bash
cd hf_models
hf upload charleskulkauski/fiap-gynecology-models .
```
