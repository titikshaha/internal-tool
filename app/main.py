"""
Builtattic Internal Tool API.
Supports PDF (with OCR fallback), DXF, and image files.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

from app.services.pdf_extractor import extract_from_pdf, extract_from_image
from app.services.dxf_extractor import extract_from_dxf
from app.services.unsupervised_image_parser import (
    train_codebook,
    parse_image_to_json,
    DEFAULT_MODEL_PATH,
)
from app.services.supervised_pipeline import (
    train_supervised_model,
    parse_image_supervised,
    validate_supervised_dataset,
    DEFAULT_SUPERVISED_MODEL_PATH,
)

SUPPORTED_EXTENSIONS = ["pdf", "dxf", "png", "jpg", "jpeg"]
SUPPORTED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UnsupervisedTrainRequest(BaseModel):
    image_dir: str = Field(..., description="Directory path containing unlabeled training images.")
    output_path: str = Field(
        default=DEFAULT_MODEL_PATH,
        description="Path where trained model JSON should be written.",
    )
    clusters: int = Field(default=32, ge=2, le=512)


class SupervisedTrainRequest(BaseModel):
    dataset_dir: str = Field(
        ...,
        description="Dataset root containing annotations/*.json and images/",
    )
    output_path: str = Field(
        default=DEFAULT_SUPERVISED_MODEL_PATH,
        description="Path where supervised model JSON should be written.",
    )
    min_samples: int = Field(default=1, ge=1, le=100000)


class SupervisedValidateRequest(BaseModel):
    dataset_dir: str = Field(
        ...,
        description="Dataset root containing annotations/*.json and images/",
    )

app = FastAPI(
    title="Builtattic Internal Tool",
    description="Internal tool for floor plan parsing and image analysis workflows",
    version="2.0.0",
)

# CORS for main app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve a minimal internal HTML UI for parsing floor plans."""
    metadata = {
        "service": "Builtattic Internal Tool",
        "version": "2.0.0",
        "supported_formats": SUPPORTED_EXTENSIONS,
        "features": ["ocr_fallback", "dxf_parsing", "image_input"],
    }

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        html = """
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Builtattic Internal Tool</title>
            <style>
                :root {
                    --bg: #f3f4f6;
                    --card: #ffffff;
                    --text: #111827;
                    --muted: #6b7280;
                    --border: #d1d5db;
                    --btn: #111827;
                    --btn-text: #ffffff;
                }
                body {
                    margin: 0;
                    font-family: "Segoe UI", Tahoma, sans-serif;
                    background: var(--bg);
                    color: var(--text);
                }
                .wrap {
                    min-height: 100vh;
                    display: grid;
                    place-items: center;
                    padding: 20px;
                }
                .card {
                    width: min(720px, 100%);
                    background: var(--card);
                    border: 1px solid var(--border);
                    border-radius: 12px;
                    padding: 20px;
                }
                h1 {
                    margin: 0 0 16px;
                    font-size: 20px;
                }
                .row {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                    flex-wrap: wrap;
                }
                input[type="file"] {
                    flex: 1;
                }
                button {
                    border: 0;
                    background: var(--btn);
                    color: var(--btn-text);
                    padding: 10px 14px;
                    border-radius: 8px;
                    cursor: pointer;
                }
                pre {
                    margin: 16px 0 0;
                    background: #f9fafb;
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    padding: 12px;
                    overflow: auto;
                    max-height: 50vh;
                    font-size: 12px;
                }
                .status {
                    margin-top: 10px;
                    color: var(--muted);
                    font-size: 13px;
                }
            </style>
        </head>
        <body>
            <div class="wrap">
                <div class="card">
                    <h1>Builtattic Internal Tool</h1>
                    <div class="row">
                        <input type="file" id="file" name="file" />
                        <button type="button" onclick="upload()">Analyze</button>
                    </div>
                    <div id="status" class="status"></div>
                    <pre id="result">No result yet.</pre>
                </div>
            </div>

            <script>
            async function upload() {
                const input = document.getElementById("file");
                const status = document.getElementById("status");
                if (!input.files || input.files.length === 0) {
                    status.textContent = "Select a file.";
                    return;
                }
                const fd = new FormData();
                fd.append("file", input.files[0]);

                status.textContent = "Analyzing...";
                const res = await fetch("/parse", { method: "POST", body: fd });
                const txt = await res.text();
                try {
                    const payload = JSON.parse(txt);
                    document.getElementById("result").textContent = JSON.stringify(payload, null, 2);
                    status.textContent = res.ok ? "Done." : "Request failed.";
                } catch (err) {
                    document.getElementById("result").textContent = txt;
                    status.textContent = res.ok ? "Done." : "Request failed.";
                }
            }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    return JSONResponse(content=metadata)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "pdf_parser": "ok",
        "dxf_parser": "ok",
        "ocr_available": False,
    }

    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        health_status["ocr_available"] = True
    except Exception:
        health_status["ocr_available"] = False

    return health_status


@app.post("/parse")
async def parse_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Parse a floor plan file (PDF, DXF, PNG, JPG).
    Returns extracted walls, rooms, dimensions, doors, windows.
    """
    if not file.filename:
        logger.warning("Parse request with no filename")
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split(".")[-1]
    logger.info(f"Parsing file: {file.filename} (type: {ext})")

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported file type: {ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        logger.info(f"Saved to temp file: {tmp_path}, size: {len(content)} bytes")

    try:
        if ext == "pdf":
            logger.info("Starting PDF extraction...")
            result = extract_from_pdf(tmp_path)
            logger.info(
                f"PDF extraction complete: {len(result.get('walls', []))} walls, "
                f"{len(result.get('dimensions', []))} dimensions, "
                f"{len(result.get('rooms', []))} rooms, "
                f"OCR used: {result.get('ocr_used', False)}"
            )
        elif ext == "dxf":
            logger.info("Starting DXF extraction...")
            result = extract_from_dxf(tmp_path)
            logger.info(
                f"DXF extraction complete: {len(result.get('walls', []))} walls, "
                f"{len(result.get('dimensions', []))} dimensions"
            )
        else:
            logger.info("Starting image extraction...")
            result = extract_from_image(tmp_path)
            logger.info(
                f"Image extraction complete: {len(result.get('walls', []))} walls, "
                f"{len(result.get('dimensions', []))} dimensions, "
                f"{len(result.get('rooms', []))} rooms, "
                f"OCR used: {result.get('ocr_used', False)}"
            )

        result["filename"] = file.filename
        return result

    except Exception as e:
        logger.error(f"Extraction failed for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug(f"Cleaned up temp file: {tmp_path}")


@app.post("/model/train-unsupervised")
async def train_unsupervised_model(payload: UnsupervisedTrainRequest) -> Dict[str, Any]:
    """
    Train a no-label image parser model using raw image folders and save a JSON codebook.
    """
    try:
        return train_codebook(
            image_dir=payload.image_dir,
            output_path=payload.output_path,
            clusters=payload.clusters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unsupervised training failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@app.post("/model/train-supervised")
async def train_supervised(payload: SupervisedTrainRequest) -> Dict[str, Any]:
    """
    Train phase-1 supervised baseline from labeled dataset annotations.
    """
    try:
        return train_supervised_model(
            dataset_dir=payload.dataset_dir,
            output_path=payload.output_path,
            min_samples=payload.min_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Supervised training failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@app.post("/model/validate-supervised-dataset")
async def validate_supervised(payload: SupervisedValidateRequest) -> Dict[str, Any]:
    """
    Validate supervised dataset structure and annotations before training.
    """
    try:
        return validate_supervised_dataset(payload.dataset_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Supervised dataset validation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")


@app.post("/parse/image-unsupervised")
async def parse_image_unsupervised(
    file: UploadFile = File(...),
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Parse image into unlabeled structural JSON using a trained unsupervised codebook.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split(".")[-1]
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {ext}. Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = parse_image_to_json(tmp_path, model_path=model_path)
        result["filename"] = file.filename
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unsupervised image parsing failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/parse/image-supervised")
async def parse_image_with_supervised_model(
    file: UploadFile = File(...),
    model_path: str = DEFAULT_SUPERVISED_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Parse image with phase-1 supervised model and return normalized detections JSON.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split(".")[-1]
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {ext}. Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = parse_image_supervised(tmp_path, model_path=model_path)
        result["filename"] = file.filename
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Supervised image parsing failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/parse/multi")
async def parse_multi_page(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Parse a multi-page PDF, returning data for each page.
    For DXF and image files, behaves the same as /parse.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split(".")[-1]

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if ext == "dxf":
            result = extract_from_dxf(tmp_path)
            result["filename"] = file.filename
            return result

        if ext in ["png", "jpg", "jpeg"]:
            result = extract_from_image(tmp_path)
            result["filename"] = file.filename
            return result

        import fitz

        doc = fitz.open(tmp_path)
        pages_data = []

        for page_num in range(len(doc)):
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as page_tmp:
                single_page_doc.save(page_tmp.name)
                single_page_doc.close()

                page_result = extract_from_pdf(page_tmp.name)
                page_result["page_number"] = page_num + 1
                pages_data.append(page_result)
                os.unlink(page_tmp.name)

        doc.close()

        return {
            "filename": file.filename,
            "total_pages": len(pages_data),
            "pages": pages_data,
            "summary": {
                "total_walls": sum(len(p["walls"]) for p in pages_data),
                "total_dimensions": sum(len(p["dimensions"]) for p in pages_data),
                "total_rooms": sum(len(p["rooms"]) for p in pages_data),
                "ocr_used_on_pages": [
                    p["page_number"] for p in pages_data if p.get("ocr_used", False)
                ],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
