"""Per-format pipeline registry.

Maps ``TaskFormat`` values to their :class:`FormatPipeline` singletons.
Shared construction modules call :func:`get_pipeline` rather than
branching on the format enum directly.

Registration is lazy (first call imports the format packages) so the
``formats/`` sub-packages can freely import from the shared construction
modules without risking a circular import at package load.
"""

from __future__ import annotations

from draper.construction.formats.base import FormatPipeline
from draper.construction.schemas import TaskFormat

_REGISTRY: dict[TaskFormat, FormatPipeline] = {}
_INITIALIZED = False


def register(pipeline: FormatPipeline) -> None:
    """Install ``pipeline`` as the handler for its ``task_format``."""
    _REGISTRY[pipeline.task_format] = pipeline


def _ensure_initialized() -> None:
    """Import the format sub-packages on first use.

    Each sub-package's ``__init__`` calls :func:`register`. Deferring the
    import until lookup-time lets those sub-packages import from shared
    construction modules without a circular dependency.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    # Import for side effects: each sub-package registers its pipeline.
    from draper.construction.formats import copywriting  # noqa: F401

    _INITIALIZED = True


def get_pipeline(task_format: TaskFormat) -> FormatPipeline:
    """Return the registered :class:`FormatPipeline` for ``task_format``.

    Raises ``KeyError`` if no pipeline is registered — this is a
    programmer error (every ``TaskFormat`` value must have a pipeline).
    """
    _ensure_initialized()
    pipeline = _REGISTRY.get(task_format)
    if pipeline is None:
        msg = (
            f"No FormatPipeline registered for {task_format.value!r}. "
            f"Every TaskFormat value must have a formats/<name>/ package "
            f"that registers itself at import time."
        )
        raise KeyError(msg)
    return pipeline


def all_pipelines() -> list[FormatPipeline]:
    """Return all registered pipelines (for coverage tests)."""
    _ensure_initialized()
    return list(_REGISTRY.values())
