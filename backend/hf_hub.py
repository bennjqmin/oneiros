"""
Hugging Face Hub helpers — model search, metadata, and validation.

All public functions return dicts with either success fields or structured
``error`` / ``hint`` keys suitable for JSON API responses.
"""

from __future__ import annotations

import os
import re
from typing import Any

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def _normalize_model_id(model_id: str) -> str:
    return model_id.strip()


def _validate_model_id(model_id: str) -> str | None:
    """Return an error message if invalid, else None."""
    if not model_id:
        return "Model ID is required (e.g. bert-base-uncased or google/vit-base-patch16-224)."
    if len(model_id) > 200:
        return "Model ID is too long."
    if not _MODEL_ID_RE.match(model_id):
        return (
            "Invalid model ID format. Use owner/name (e.g. distilbert-base-uncased) "
            "with letters, numbers, dots, dashes, and underscores."
        )
    return None


def hf_status() -> dict[str, Any]:
    """Check whether optional HF dependencies are installed."""
    transformers_ok = False
    hub_ok = False
    transformers_err = ""
    hub_err = ""

    try:
        import transformers  # noqa: F401

        transformers_ok = True
    except ImportError as exc:
        transformers_err = str(exc)

    try:
        import huggingface_hub  # noqa: F401

        hub_ok = True
    except ImportError as exc:
        hub_err = str(exc)

    ready = transformers_ok and hub_ok
    return {
        "ready": ready,
        "transformers": transformers_ok,
        "huggingfaceHub": hub_ok,
        "hint": None
        if ready
        else "Install backend dependencies: pip install transformers huggingface_hub",
        "errors": {
            "transformers": transformers_err or None,
            "huggingfaceHub": hub_err or None,
        },
    }


def infer_output_features(config: Any) -> int:
    """Best-effort hidden size from a HF config object."""
    for attr in ("hidden_size", "d_model", "n_embd", "dim", "encoder_hidden_size"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if isinstance(val, (list, tuple)) and val:
                return int(val[-1])
            if val is not None:
                return int(val)
    if hasattr(config, "hidden_sizes"):
        hs = getattr(config, "hidden_sizes")
        if isinstance(hs, (list, tuple)) and hs:
            return int(hs[-1])
    return 768


def infer_model_kind(config: Any, pipeline_tag: str | None) -> str:
    tag = (pipeline_tag or "").lower()
    if "image" in tag or "vision" in tag:
        return "vision"
    if any(k in tag for k in ("text", "fill-mask", "token", "question", "summarization")):
        return "text"
    cfg_name = type(config).__name__.lower()
    if any(k in cfg_name for k in ("vit", "swin", "deit", "convnext", "resnet")):
        return "vision"
    if any(k in cfg_name for k in ("bert", "gpt", "roberta", "t5", "llama", "distil")):
        return "text"
    return "generic"


def hf_search(query: str, *, limit: int = 20) -> dict[str, Any]:
    q = query.strip()
    if not q:
        return {"error": "Search query is required.", "hint": "Try bert, vit, distilbert, or a task like text-classification."}

    status = hf_status()
    if not status["huggingfaceHub"]:
        return {"error": "huggingface_hub is not installed.", "hint": status.get("hint")}

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        return {"error": f"Cannot import huggingface_hub: {exc}", "hint": status.get("hint")}

    try:
        api = HfApi()
        models = api.list_models(search=q, limit=min(max(limit, 1), 50), sort="downloads", direction=-1)
        results = []
        for m in models:
            results.append(
                {
                    "id": m.id,
                    "pipelineTag": getattr(m, "pipeline_tag", None),
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "private": bool(getattr(m, "private", False)),
                    "gated": bool(getattr(m, "gated", False)),
                }
            )
        return {"query": q, "models": results}
    except Exception as exc:
        err = str(exc).lower()
        if "connection" in err or "network" in err or "resolve" in err:
            return {
                "error": "Could not reach Hugging Face Hub. Check your internet connection.",
                "hint": "If you are offline, import a model ID you already know is cached locally.",
            }
        return {"error": f"Search failed: {exc}"}


def hf_model_info(model_id: str) -> dict[str, Any]:
    model_id = _normalize_model_id(model_id)
    id_err = _validate_model_id(model_id)
    if id_err:
        return {"error": id_err}

    status = hf_status()
    if not status["huggingfaceHub"]:
        return {"error": "huggingface_hub is not installed.", "hint": status.get("hint")}

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as exc:
        return {"error": f"Cannot import huggingface_hub: {exc}", "hint": status.get("hint")}

    try:
        api = HfApi()
        info = api.model_info(model_id)
        return {
            "id": info.id,
            "pipelineTag": getattr(info, "pipeline_tag", None),
            "libraryName": getattr(info, "library_name", None),
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "private": bool(getattr(info, "private", False)),
            "gated": bool(getattr(info, "gated", False)),
            "tags": list(getattr(info, "tags", []) or [])[:12],
            "sha": getattr(info, "sha", None),
        }
    except RepositoryNotFoundError:
        return {
            "error": f"Model '{model_id}' was not found on Hugging Face Hub.",
            "hint": "Check spelling and owner/name format (e.g. google/vit-base-patch16-224).",
        }
    except GatedRepoError:
        return {
            "error": f"Model '{model_id}' is gated — accept the license on huggingface.co and set HF_TOKEN.",
            "hint": "Export HF_TOKEN with a read token, or paste your token in the Hugging Face tab.",
            "gated": True,
        }
    except Exception as exc:
        err = str(exc).lower()
        if "401" in err or "403" in err or "unauthorized" in err:
            return {
                "error": f"Access denied for '{model_id}'.",
                "hint": "This model may be private or gated. Set HF_TOKEN with read access.",
                "gated": True,
            }
        if "connection" in err or "network" in err:
            return {
                "error": "Could not reach Hugging Face Hub.",
                "hint": "Check your network connection and try again.",
            }
        return {"error": f"Failed to fetch model info: {exc}"}


def hf_validate_model(model_id: str, *, trust_remote_code: bool = False, token: str | None = None) -> dict[str, Any]:
    """Load config (not full weights) and infer output feature size."""
    model_id = _normalize_model_id(model_id)
    id_err = _validate_model_id(model_id)
    if id_err:
        return {"error": id_err, "valid": False}

    status = hf_status()
    if not status["ready"]:
        return {
            "error": "transformers and huggingface_hub are required on the backend.",
            "hint": status.get("hint"),
            "valid": False,
        }

    env_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        os.environ.setdefault("HF_TOKEN", env_token)

    try:
        from transformers import AutoConfig
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as exc:
        return {"error": f"Cannot import transformers: {exc}", "hint": status.get("hint"), "valid": False}

    try:
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            token=env_token,
        )
    except RepositoryNotFoundError:
        return {
            "error": f"Model '{model_id}' was not found.",
            "hint": "Verify the model ID on huggingface.co.",
            "valid": False,
        }
    except GatedRepoError:
        return {
            "error": f"Model '{model_id}' is gated.",
            "hint": "Accept the model license on Hugging Face and provide HF_TOKEN.",
            "valid": False,
            "gated": True,
        }
    except OSError as exc:
        msg = str(exc)
        lower = msg.lower()
        if "trust_remote_code" in lower or "remote code" in lower:
            return {
                "error": "This model requires trust_remote_code=True.",
                "hint": "Enable “Trust remote code” in the Hugging Face tab and validate again.",
                "valid": False,
                "needsTrustRemoteCode": True,
            }
        if "401" in lower or "403" in lower or "authorized" in lower:
            return {
                "error": "Authentication failed for this model.",
                "hint": "Set HF_TOKEN if the model is private or gated.",
                "valid": False,
            }
        return {"error": f"Could not load config: {msg}", "valid": False}
    except Exception as exc:
        err = str(exc)
        if "Connection" in err or "Network" in err:
            return {
                "error": "Network error while loading model config.",
                "hint": "Check your connection. Cached models may still work offline.",
                "valid": False,
            }
        return {"error": f"Validation failed: {err}", "valid": False}

    output_features = infer_output_features(config)
    model_kind = infer_model_kind(config, None)
    config_name = type(config).__name__

    return {
        "valid": True,
        "modelId": model_id,
        "outputFeatures": output_features,
        "modelKind": model_kind,
        "configClass": config_name,
        "trustRemoteCode": trust_remote_code,
        "architectures": list(getattr(config, "architectures", []) or [])[:4],
        "hint": None,
    }


def build_hf_wrapper(
    model_id: str,
    *,
    pooling: str = "mean",
    freeze: bool = False,
    trust_remote_code: bool = False,
    token: str | None = None,
) -> Any:
    """Load a Hugging Face AutoModel wrapped for Oneiros graph execution."""
    try:
        import torch.nn as nn
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Hugging Face models. "
            "Run: pip install transformers huggingface_hub"
        ) from exc

    env_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    class HFModelWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pooling = pooling
            load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
            if env_token:
                load_kwargs["token"] = env_token
            try:
                self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
            except OSError as exc:
                msg = str(exc)
                if "trust_remote_code" in msg.lower() or "remote code" in msg.lower():
                    raise RuntimeError(
                        f"Model '{model_id}' requires trust_remote_code=True."
                    ) from exc
                if "401" in msg or "403" in msg or "authorized" in msg.lower():
                    raise RuntimeError(
                        f"Access denied loading '{model_id}'. Set HF_TOKEN for gated/private models."
                    ) from exc
                raise RuntimeError(f"Failed to load '{model_id}': {msg}") from exc
            except Exception as exc:
                raise RuntimeError(f"Failed to load Hugging Face model '{model_id}': {exc}") from exc

            if freeze:
                for param in self.model.parameters():
                    param.requires_grad_(False)

        def _pool(self, outputs: Any) -> Any:
            import torch

            if self.pooling == "pooler":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    return pooler

            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is not None:
                if self.pooling == "cls":
                    return hidden[:, 0]
                if self.pooling == "mean":
                    return hidden.mean(dim=1)
                return hidden.reshape(hidden.size(0), -1)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output

            if isinstance(outputs, (tuple, list)) and outputs:
                first = outputs[0]
                if hasattr(first, "dim"):
                    t = first
                    return t.reshape(t.size(0), -1) if t.dim() > 2 else t

            raise RuntimeError(
                f"Could not extract features from '{model_id}' with pooling='{self.pooling}'."
            )

        def forward(self, x: Any) -> Any:
            if x.dim() == 4:
                outputs = self.model(pixel_values=x)
            elif x.dim() == 2:
                input_ids = x.long()
                attention_mask = (input_ids != 0).long()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                raise RuntimeError(
                    f"Hugging Face node expects 2D (batch, seq) or 4D (batch, C, H, W) input, "
                    f"got shape {tuple(x.shape)}."
                )
            return self._pool(outputs)

    return HFModelWrapper()
