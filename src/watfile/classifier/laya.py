"""Local typed-decision classifier backend over the Laya family.

Three runtimes, chosen by availability (override with LAYA_RUNTIME env):

  - "mlx"    — laya-mlx (Apple Silicon, MLX/GPU, ~13ms/decision)
  - "coreml" — laya-coreml (Apple Silicon, Core ML/ANE, ~5ms, 2.8x energy saving)
  - "torch"  — laya (PyTorch, CPU/GPU — works everywhere: Windows, Linux,
               Intel Mac; ~190-460ms/decision on CPU)

All three share the same shape: load(checkpoint).predict(state, questions)
-> result["answers"][key] with {"choice", "probabilities", ...}.

Checkpoints differ per runtime. Default per runtime:
  mlx    -> aac6fef/laya-typed-decisions-mlx   (1024-tok ctx, English)
  coreml -> aac6fef/laya-multilingual-coreml    (1024-tok ctx, multilingual)
  torch  -> convaiinnovations/laya subfolder=multilingual (multilingual)

Language coverage: choose the *multilingual* checkpoint when documents can be
non-English (inv vs Mietvertrag); Jev handles language itself.
"""

from __future__ import annotations

import os
import platform
from typing import Sequence

from .base import Classifier, Verdict

#: max chars of state fed to the local model when used bare (no MultiChunk
#: wrapper). ~1024-token checkpoints -> ~3.4k chars safe ceiling.
_MAX_STATE_CHARS = 3_400

_QUESTION_KEY = "category"

_DEFAULT_CHECKPOINTS = {
    "mlx": "aac6fef/laya-typed-decisions-mlx",
    "coreml": "aac6fef/laya-multilingual-coreml",
    "torch": "convaiinnovations/laya",
}
_DEFAULT_TORCH_SUBFOLDER = "multilingual"


def _detect_runtime() -> str:
    """Pick the best available runtime for this machine."""
    override = os.environ.get("LAYA_RUNTIME")
    if override in ("mlx", "coreml", "torch"):
        return override
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        for name in ("coreml", "mlx"):
            try:
                __import__(f"laya_{name}")
                return name
            except ImportError:
                continue
    try:
        import laya  # noqa: F401 — plain torch-based laya

        return "torch"
    except ImportError:
        pass
    raise ImportError(
        "no laya runtime available. Install one of:\n"
        "  pip install 'watfile[laya]'       # Apple Silicon: MLX (GPU)\n"
        "  pip install laya-coreml           # Apple Silicon: Core ML / ANE\n"
        "  pip install laya                  # any platform: PyTorch (CPU/GPU)"
    )


class LayaClassifier(Classifier):
    def __init__(self, model: str | None = None, runtime: str | None = None) -> None:
        self._runtime = runtime or _detect_runtime()
        self._model_name = (
            model
            or os.environ.get("LAYA_MODEL")
            or _DEFAULT_CHECKPOINTS[self._runtime]
        )
        self._agent = None  # lazy: first classify() downloads/loads the checkpoint

    @property
    def runtime(self) -> str:
        return self._runtime

    def _agent_or_load(self):
        if self._agent is None:
            if self._runtime == "mlx":
                import laya_mlx as laya

                self._agent = laya.load(self._model_name, dtype="float16")
            elif self._runtime == "coreml":
                import laya_coreml as laya

                self._agent = laya.load(self._model_name)
            else:  # torch
                import laya

                if self._model_name.startswith("convaiinnovations/"):
                    self._agent = laya.load(
                        self._model_name, subfolder=os.environ.get("LAYA_SUBFOLDER", _DEFAULT_TORCH_SUBFOLDER)
                    )
                else:
                    self._agent = laya.load(self._model_name)
        return self._agent

    def classify(self, text: str, categories: Sequence[str]) -> Verdict:
        state = text[:_MAX_STATE_CHARS]
        questions = {
            _QUESTION_KEY: {
                "type": "choice",
                "instructions": (
                    "Which category does this document belong to? "
                    "Judge by content, not filename."
                ),
                "criteria": list(categories),
            }
        }
        result = self._agent_or_load().predict(state, questions)
        answer = result["answers"][_QUESTION_KEY]
        probabilities = {c: float(p) for c, p in answer["probabilities"].items()}
        category = answer["choice"]
        return Verdict(
            category=category,
            confidence=probabilities.get(category, 0.0),
            probabilities=probabilities,
        )
