"""
Oneiros FastAPI backend.

Start with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

# PyTorch + XGBoost both link OpenMP on macOS; loading both in one process
# without this can segfault ("Python quit unexpectedly") when XGBoost trains.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XGBOOST_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import asyncio
import io
import json
import os
import tempfile
import uuid
from typing import Any

import torch
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from compiler import build_model
from datasets import get_loaders, get_dataset_info, list_datasets, make_custom_loaders
from trainer import run_training_async
from ai import chat as ai_chat
from plotter import generate_plot, CHART_DEFS, PALETTES
from xgboost_trainer import train_xgboost_subprocess, REGRESSION_OBJECTIVES, _xgb_hint
from edf_processor import (
    load_edf,
    apply_edf_pipeline,
    generate_edf_plot,
    EDF_STEP_META,
)
from hf_hub import hf_status, hf_search, hf_model_info, hf_validate_model
from cv_dataset import (
    load_image_folder,
    make_image_loaders,
    preview_augmentation,
    extract_zip_to_temp,
    AUG_STEP_META,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Oneiros API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory run state ───────────────────────────────────────────────────────

# run_id → {"request": TrainRequest, "stop": asyncio.Event}
pending_runs: dict[str, Any] = {}
active_runs: dict[str, dict[str, Any]] = {}

# Store trained models for export (run_id → {"model": nn.Module, "graph": dict})
trained_models: dict[str, dict[str, Any]] = {}

# Store trained XGBoost models for export (run_id → base64 model string)
xgb_models: dict[str, str] = {}


# ── Request / response models ─────────────────────────────────────────────────

class CompileRequest(BaseModel):
    nodes: list[dict]
    edges: list[dict]


class TrainConfig(BaseModel):
    dataset: str = "mnist"
    epochs: int = 10
    batchSize: int = 64
    learningRate: float = 0.001
    optimizer: str = "adam"
    lossFunction: str = "cross_entropy"
    # Schedulers
    scheduler: str = "none"
    schedulerStepSize: int = 5
    schedulerGamma: float = 0.5
    schedulerTMax: int = 10


class TrainRequest(BaseModel):
    graph: dict
    config: TrainConfig
    customDataset: dict | None = None
    cvDataset: dict | None = None  # {sessionId, augmentSteps}


class AIChatRequest(BaseModel):
    messages: list[dict]
    context: dict = {}
    apiKey: str = ""
    model: str = "gpt-4o"


# PlotRequest intentionally uses raw Request — dataset rows can contain
# any JSON scalar types and strict Pydantic coercion causes false 422s.


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/datasets")
async def datasets_list() -> dict:
    return {
        "datasets": [
            {"id": ds, **get_dataset_info(ds)}  # type: ignore[misc]
            for ds in list_datasets()
        ]
    }


@app.get("/api/plot/meta")
async def plot_meta() -> dict:
    """Return available chart types and palettes."""
    return {"chartDefs": CHART_DEFS, "palettes": PALETTES}


@app.post("/api/plot")
async def plot(request: Request) -> StreamingResponse:
    """Generate a seaborn/matplotlib chart and return PNG bytes."""
    from fastapi.responses import JSONResponse
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse({"error": f"Invalid JSON body: {exc}"}, status_code=400)

    rows: list[dict] = body.get("rows", [])
    config: dict = body.get("config", {})

    if not rows:
        return JSONResponse({"error": "No data rows provided."}, status_code=422)

    try:
        png_bytes = generate_plot(rows, config)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Plot failed: {exc}"}, status_code=500)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/xgboost/meta")
async def xgboost_meta() -> dict:
    """Return static metadata used by the XGBoost UI (regression objectives etc.)."""
    return {"regressionObjectives": REGRESSION_OBJECTIVES}


@app.post("/api/xgboost/train")
async def xgboost_train(request: Request) -> dict:
    """Train an XGBoost model and return metrics + feature importance."""
    from fastapi.responses import JSONResponse
    try:
        raw = await request.body()
        if len(raw) > 64 * 1024 * 1024:
            return JSONResponse(
                {
                    "error": f"Request body too large ({len(raw) // (1024 * 1024)} MB; max 64 MB).",
                    "hint": "Reduce rows or features in the Dataset pipeline.",
                },
                status_code=413,
            )
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"Invalid JSON: {exc}"}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not read request: {exc}"}, status_code=400)

    data   = body.get("data", {})
    config = body.get("config", {})

    if not isinstance(data, dict) or not isinstance(config, dict):
        return JSONResponse({"error": "Request body must include 'data' and 'config' objects."}, status_code=400)

    if not data.get("X_train"):
        return JSONResponse(
            {
                "error": "No training data provided. Load a CSV dataset and set a target column.",
                "hint": "Import data in Dataset tab, pick a target, and connect Source → Split in the pipeline.",
            },
            status_code=422,
        )

    try:
        result = train_xgboost_subprocess(data, config)
    except TimeoutError as exc:
        return JSONResponse(
            {"error": str(exc), "hint": "Lower n_estimators or use a smaller dataset."},
            status_code=504,
        )
    except ValueError as exc:
        hint = _xgb_hint(exc)
        return JSONResponse({"error": str(exc), "hint": hint}, status_code=422)
    except RuntimeError as exc:
        msg = str(exc)
        hint = _xgb_hint(exc)
        status = 503 if "not installed" in msg.lower() else 422
        return JSONResponse({"error": msg, "hint": hint}, status_code=status)
    except MemoryError as exc:
        return JSONResponse(
            {
                "error": f"Out of memory during training: {exc}",
                "hint": "Use fewer rows, fewer features, or lower n_estimators.",
            },
            status_code=422,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        hint = _xgb_hint(exc)
        return JSONResponse(
            {"error": f"XGBoost training failed: {exc}", "hint": hint},
            status_code=500,
        )

    run_id = str(uuid.uuid4())
    xgb_models[run_id] = result.pop("modelB64")
    result["runId"] = run_id
    return result


@app.get("/api/xgboost/{run_id}/export")
async def xgboost_export(run_id: str) -> StreamingResponse:
    """Download the trained XGBoost model as a JSON file."""
    model_b64 = xgb_models.get(run_id)
    if not model_b64:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Model not found — train first."}, status_code=404)

    import base64
    model_bytes = base64.b64decode(model_b64)
    return StreamingResponse(
        io.BytesIO(model_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="xgboost_model.json"'},
    )


@app.post("/api/compile")
async def compile_graph(req: CompileRequest) -> dict:
    graph = {"nodes": req.nodes, "edges": req.edges}
    try:
        _model, errors = build_model(graph)
    except (RuntimeError, ValueError) as exc:
        return {"success": False, "errors": [str(exc)]}
    except Exception as exc:
        return {"success": False, "errors": [f"Failed to build model: {exc}"]}
    if errors:
        return {"success": False, "errors": errors}
    return {"success": True, "errors": []}


@app.post("/api/ai/chat")
async def ai_chat_endpoint(req: AIChatRequest) -> dict:
    if not req.apiKey:
        return {
            "message": (
                "No API key set. Enter your OpenAI key in the AI panel settings "
                "(click the key icon). Your key is stored only in your browser."
            )
        }
    message = await ai_chat(req.messages, req.context, req.apiKey, req.model)
    return {"message": message}


@app.post("/api/train/start")
async def start_training(req: TrainRequest) -> dict:
    run_id = str(uuid.uuid4())
    stop_event = asyncio.Event()
    pending_runs[run_id] = req
    # Register the stop event immediately so /api/train/stop works even before
    # the WebSocket handler has had a chance to build the model or load data.
    active_runs[run_id] = {"stop": stop_event}
    return {"runId": run_id}


@app.post("/api/train/stop/{run_id}")
async def stop_training(run_id: str) -> dict:
    run = active_runs.get(run_id)
    if run:
        run["stop"].set()
        return {"status": "stopping"}
    return {"status": "not_found"}


# ── Export endpoints ──────────────────────────────────────────────────────────

@app.get("/api/export/{run_id}/weights")
async def export_weights(run_id: str, filename: str = "model.pt") -> StreamingResponse:
    """Download trained model weights as a PyTorch .pt file."""
    entry = trained_models.get(run_id)
    if not entry:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Model not found — train the model first."}, status_code=404)

    model: torch.nn.Module = entry["model"]
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/{run_id}/onnx")
async def export_onnx(run_id: str, filename: str = "model.onnx") -> StreamingResponse:
    """Export trained model as ONNX."""
    entry = trained_models.get(run_id)
    if not entry:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Model not found — train the model first."}, status_code=404)

    model: torch.nn.Module = entry["model"]
    graph: dict = entry["graph"]

    # Infer input shape from graph
    input_node = next((n for n in graph.get("nodes", []) if n.get("type") == "inputNode"), None)
    if input_node:
        d = input_node.get("data", {})
        dummy = torch.randn(1, int(d.get("channels", 1)), int(d.get("height", 28)), int(d.get("width", 28)))
    else:
        dummy = torch.randn(1, 1, 28, 28)

    model.eval()
    buf = io.BytesIO()
    try:
        torch.onnx.export(
            model,
            dummy,
            buf,
            export_params=True,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
    except Exception as exc:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"ONNX export failed: {exc}"}, status_code=500)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/{run_id}/full")
async def export_full(run_id: str, filename: str = "model_full.pt") -> StreamingResponse:
    """Download the full model (architecture + weights) via torch.save."""
    entry = trained_models.get(run_id)
    if not entry:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Model not found — train the model first."}, status_code=404)

    model: torch.nn.Module = entry["model"]
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── WebSocket training endpoint ───────────────────────────────────────────────

@app.websocket("/ws/training/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()

    req: TrainRequest | None = pending_runs.pop(run_id, None)  # type: ignore[assignment]
    if req is None:
        await websocket.send_json({"type": "error", "message": "Invalid or expired run ID"})
        await websocket.close()
        return

    # Build model
    try:
        model, errors = build_model(req.graph)
    except (RuntimeError, ValueError) as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"Failed to build model: {exc}"})
        await websocket.close()
        return
    if errors:
        await websocket.send_json({"type": "error", "message": "; ".join(errors)})
        await websocket.close()
        return

    # Load dataset
    try:
        if req.config.dataset == "custom":
            if not req.customDataset:
                await websocket.send_json({"type": "error", "message": "Custom dataset payload missing."})
                await websocket.close()
                return
            train_loader, val_loader = make_custom_loaders(
                req.customDataset,
                batch_size=req.config.batchSize,
            )
        elif req.config.dataset == "image_folder":
            if not req.cvDataset or not req.cvDataset.get("sessionId"):
                await websocket.send_json({"type": "error", "message": "Image dataset session missing. Upload a zip first."})
                await websocket.close()
                return
            train_loader, val_loader = make_image_loaders(
                session_id=req.cvDataset["sessionId"],
                augment_steps=req.cvDataset.get("augmentSteps", []),
                batch_size=req.config.batchSize,
            )
        else:
            train_loader, val_loader = get_loaders(
                req.config.dataset,
                batch_size=req.config.batchSize,
            )
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    queue: asyncio.Queue[dict] = asyncio.Queue()
    # Reuse the stop event that was pre-registered in /api/train/start so that
    # a stop request that arrives before the WebSocket is fully set up still works.
    run = active_runs.get(run_id)
    stop_event = run["stop"] if run else asyncio.Event()
    active_runs[run_id] = {"stop": stop_event}

    # Dry-run: forward one sample through the model before training starts.
    # This catches shape mismatches (wrong Input node size) immediately with a
    # clear error message rather than crashing silently at epoch 1.
    try:
        dummy_batch, _ = next(iter(train_loader))
        with torch.no_grad():
            model(dummy_batch[:1])
    except Exception as shape_exc:
        msg = str(shape_exc)
        hint = ""
        if "cannot be multiplied" in msg or "size mismatch" in msg.lower():
            if req.config.dataset == "image_folder":
                hint = " — Check that your Input node C/H/W match your image dataset shape, or add a Resize augmentation node."
            else:
                hint = " — Check that your Input node dimensions match your dataset (for a CSV with N features set channels=N, height=1, width=1)."
        elif "output size is too small" in msg.lower() or "calculated padded input size" in msg.lower():
            hint = " — A convolutional kernel is larger than the feature map. Add padding, reduce kernel size, or add fewer pooling layers."
        elif "expected input" in msg.lower() and "channel" in msg.lower():
            hint = " — Channel count mismatch. For pretrained backbones, set Input node channels=3 (RGB)."
        await websocket.send_json({"type": "error", "message": f"Shape mismatch: {msg}{hint}"})
        await websocket.close()
        active_runs.pop(run_id, None)
        return

    config_dict = req.config.model_dump()

    training_task = asyncio.create_task(
        run_training_async(model, train_loader, val_loader, config_dict, queue, stop_event)
    )

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if training_task.done():
                    break
                continue

            await websocket.send_json(msg)

            if msg.get("type") in ("complete", "stopped", "error"):
                # Store model if training completed successfully
                if msg.get("type") == "complete":
                    trained_models[run_id] = {"model": model, "graph": req.graph}
                break

        # Drain remaining messages
        while not queue.empty():
            msg = queue.get_nowait()
            try:
                await websocket.send_json(msg)
            except Exception:
                break

    except WebSocketDisconnect:
        stop_event.set()
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        stop_event.set()
        # Await the training task separately so that any exception it raises
        # (e.g. shape mismatch, CUDA OOM) gets forwarded to the client as an
        # error message rather than crashing the WebSocket handler silently.
        try:
            await training_task
        except Exception as task_exc:
            import traceback
            traceback.print_exc()
            try:
                await websocket.send_json({"type": "error", "message": f"Training error: {task_exc}"})
            except Exception:
                pass
        active_runs.pop(run_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


# ── EDF / MNE endpoints ───────────────────────────────────────────────────────

@app.get("/api/edf/meta")
async def edf_meta() -> dict:
    """Return supported EDF pipeline step types and their parameter schemas."""
    return {"steps": EDF_STEP_META}


@app.post("/api/edf/load")
async def edf_load(file: UploadFile = File(...)) -> dict:
    """Accept an EDF file upload, parse with MNE, return metadata + preview."""
    from fastapi.responses import JSONResponse
    if not file.filename or not file.filename.lower().endswith(".edf"):
        return JSONResponse({"error": "Only .edf files are supported."}, status_code=400)

    # Write to a temp file (MNE requires a real path)
    suffix = ".edf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = load_edf(tmp_path)
        # Rename the session's stored name to the original filename
        result["name"] = Path(file.filename).stem
        return result
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Failed to load EDF: {exc}"}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/api/edf/process")
async def edf_process(request: Request) -> dict:
    """Apply an EDF pipeline to a loaded session and return epoch data."""
    from fastapi.responses import JSONResponse
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse({"error": f"Invalid JSON: {exc}"}, status_code=400)

    session_id = body.get("sessionId", "")
    steps       = body.get("steps", [])

    if not session_id:
        return JSONResponse({"error": "sessionId is required."}, status_code=422)

    try:
        result = apply_edf_pipeline(session_id, steps)
        return result
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Pipeline failed: {exc}"}, status_code=500)


@app.post("/api/edf/plot")
async def edf_plot(request: Request) -> StreamingResponse:
    """Generate a waveform or PSD PNG for an EDF session."""
    from fastapi.responses import JSONResponse
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse({"error": f"Invalid JSON: {exc}"}, status_code=400)

    session_id = body.get("sessionId", "")
    config      = body.get("config", {})

    if not session_id:
        return JSONResponse({"error": "sessionId is required."}, status_code=422)

    try:
        png_bytes = generate_edf_plot(session_id, config)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Plot failed: {exc}"}, status_code=500)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# ── Hugging Face endpoints ────────────────────────────────────────────────────

@app.get("/api/hf/status")
async def huggingface_status() -> dict:
    """Check whether transformers / huggingface_hub are available."""
    return hf_status()


@app.get("/api/hf/search")
async def huggingface_search(q: str = "", limit: int = 20) -> dict:
    from fastapi.responses import JSONResponse
    result = hf_search(q, limit=limit)
    if "error" in result and "models" not in result:
        code = 503 if "not installed" in result["error"].lower() else 400
        return JSONResponse(result, status_code=code)
    return result


@app.get("/api/hf/model-info")
async def huggingface_model_info(model_id: str = "") -> dict:
    from fastapi.responses import JSONResponse
    result = hf_model_info(model_id)
    if "error" in result:
        code = 404 if "not found" in result["error"].lower() else 403 if result.get("gated") else 400
        return JSONResponse(result, status_code=code)
    return result


class HFValidateRequest(BaseModel):
    modelId: str
    trustRemoteCode: bool = False
    token: str | None = None


@app.post("/api/hf/validate")
async def huggingface_validate(body: HFValidateRequest) -> dict:
    from fastapi.responses import JSONResponse
    if body.token:
        os.environ["HF_TOKEN"] = body.token.strip()
    result = hf_validate_model(
        body.modelId,
        trust_remote_code=body.trustRemoteCode,
        token=body.token,
    )
    if not result.get("valid"):
        code = 403 if result.get("gated") else 422
        return JSONResponse(result, status_code=code)
    return result


# ── Computer Vision endpoints ─────────────────────────────────────────────────

@app.get("/api/cv/meta")
async def cv_meta() -> dict:
    """Return available augmentation step types and their parameter schemas."""
    return {"steps": AUG_STEP_META}


@app.post("/api/cv/load")
async def cv_load(file: UploadFile = File(...)) -> dict:
    """Accept a zip of image class folders, extract and parse, return metadata."""
    from fastapi.responses import JSONResponse
    if not file.filename or not file.filename.lower().endswith(".zip"):
        return JSONResponse({"error": "Only .zip files are supported. Zip your class folders: class_a/, class_b/, …"}, status_code=400)

    try:
        content = await file.read()
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read uploaded file: {exc}"}, status_code=400)

    try:
        extracted_path = extract_zip_to_temp(content, file.filename)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to extract zip: {exc}"}, status_code=500)

    try:
        result = load_image_folder(extracted_path)
        result["name"] = Path(file.filename).stem
        return result
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "hint": "Zip your class folders directly: class_a/img1.jpg, class_b/img2.png …"}, status_code=422)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": f"Failed to load image folder: {exc}"}, status_code=500)


@app.post("/api/cv/augment_preview")
async def cv_augment_preview(request: Request) -> StreamingResponse:
    """Apply augmentation pipeline to one image, return side-by-side before/after PNG."""
    from fastapi.responses import JSONResponse
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse({"error": f"Invalid JSON: {exc}"}, status_code=400)

    session_id  = body.get("sessionId", "")
    steps       = body.get("steps", [])
    image_index = int(body.get("imageIndex", 0))

    if not session_id:
        return JSONResponse({"error": "sessionId is required."}, status_code=422)

    try:
        png_bytes = preview_augmentation(session_id, steps, image_index)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": f"Preview failed: {exc}"}, status_code=500)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
