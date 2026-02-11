# internal-tool

FastAPI service for parsing floor plan files (`pdf`, `dxf`, `png/jpg`) with:
- core extraction pipeline
- supervised phase-1 training/inference scaffold
- unsupervised image parsing flow

## Requirements
- Python `3.12` (64-bit)

## Setup
```powershell
py -3.12-64 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run
```powershell
python -m uvicorn app.main:app --reload
```

## Test
```powershell
python -m pytest -q
python -m pytest -q tests/test_basic.py
python -m pytest -q tests/test_supervised_pipeline.py
python -m pytest -q tests/test_unsupervised_image_parser.py
```

## Supervised Phase 1
Dataset layout:
```text
dataset_root/
  images/
    sample_1.png
  annotations/
    sample_1.json
```

Annotation format:
```json
{
  "file_name": "sample_1.png",
  "width": 320,
  "height": 240,
  "objects": [
    {"label": "room", "bbox": [30, 40, 110, 150]}
  ]
}
```

Validate dataset:
```powershell
curl -X POST "http://127.0.0.1:8000/model/validate-supervised-dataset" `
  -H "Content-Type: application/json" `
  -d "{\"dataset_dir\":\"C:\\\\path\\\\to\\\\dataset_root\"}"
```

Train model:
```powershell
curl -X POST "http://127.0.0.1:8000/model/train-supervised" `
  -H "Content-Type: application/json" `
  -d "{\"dataset_dir\":\"C:\\\\path\\\\to\\\\dataset_root\",\"output_path\":\"app/models/supervised_baseline.json\",\"min_samples\":1}"
```

Infer on image:
```powershell
curl -X POST "http://127.0.0.1:8000/parse/image-supervised?model_path=app/models/supervised_baseline.json" `
  -F "file=@C:\\path\\to\\sample.png"
```

## Unsupervised Flow
Train:
```powershell
curl -X POST "http://127.0.0.1:8000/model/train-unsupervised" `
  -H "Content-Type: application/json" `
  -d "{\"image_dir\":\"C:\\\\path\\\\to\\\\images\",\"output_path\":\"app/models/unsupervised_codebook.json\",\"clusters\":32}"
```

Infer:
```powershell
curl -X POST "http://127.0.0.1:8000/parse/image-unsupervised?model_path=app/models/unsupervised_codebook.json" `
  -F "file=@C:\\path\\to\\sample.png"
```
