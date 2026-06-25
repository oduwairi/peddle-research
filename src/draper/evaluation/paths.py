"""Central path resolver for everything under ``data/eval/``.

One module owns every on-disk path. Every writer in the eval pipeline
(``scripts/eval.py``, ``scripts/diagnostics/agent_smoke.py``, future scripts)
goes through ``EvalPaths`` so the layout stays consistent and no new
one-off directory shapes can sneak in.

Layout (canonical):

    data/eval/
      inferences/<config>/<example_id>.json
      inferences_clean/<config>/<example_id>.json
      judgments/<judge>/<pair>/<example_id>.json
      learned_scores/<config>.parquet
      mauve_scores/<config>.parquet
      mauve_ref/<platform_or_all>.parquet
      reference_scores/<config>.parquet
      validation/<stream>_<judge>.json
      runs/<run_id>/
        manifest.json
        aggregates/...
        diagnostics/<kind>/...
        batches/<judge>/<pair>/...

Naming conventions:

- ``run_id``: ``YYYY-MM-DD-<slug>`` (e.g. ``2026-05-15-hook-v2``).
- ``config``: ``<base>[@<variant>]`` — base is ``[A-Za-z0-9_]+``,
  variant slug is ``[A-Za-z0-9_-]+``. ``@`` is reserved as the variant
  delimiter so iterations of an agent architecture (``A_pipe@hook-v2``)
  get their own preserved directory rather than overwriting the base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("data/eval")

_RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*$")
_CONFIG_RE = re.compile(r"^[A-Za-z0-9_]+(@[A-Za-z0-9_-]+)?$")


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` if it matches ``YYYY-MM-DD-<slug>``, else raise.

    ``slug`` is lowercase alphanumeric + dashes, must start with [a-z0-9].
    Rejects ``..`` and ``/`` (path-traversal guard) by construction.
    """
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"Invalid run_id {run_id!r}: expected YYYY-MM-DD-<slug> "
            "(e.g. '2026-05-15-hook-v2'). Slug must be lowercase "
            "alphanumeric + dashes."
        )
    return run_id


def validate_config_name(name: str) -> str:
    """Return ``name`` if it matches ``<base>[@<variant>]``, else raise.

    Examples: ``A``, ``B_pipe``, ``A_pipe@hook-v2``, ``C_pipe@no-rag``.
    Rejects path separators and unexpected characters.
    """
    if not _CONFIG_RE.match(name):
        raise ValueError(
            f"Invalid config name {name!r}: expected <base>[@<variant>] "
            "where base is [A-Za-z0-9_]+ and variant slug is [A-Za-z0-9_-]+. "
            "Use '@' to separate the variant suffix (e.g. 'A_pipe@hook-v2')."
        )
    return name


def split_variant(name: str) -> tuple[str, str | None]:
    """Split ``name`` into (base, variant_or_None).

    ``'A_pipe@hook-v2'`` → ``('A_pipe', 'hook-v2')``.
    ``'A_pipe'`` → ``('A_pipe', None)``.
    """
    validate_config_name(name)
    if "@" in name:
        base, variant = name.split("@", 1)
        return base, variant
    return name, None


def _pair_dir_name(pair: tuple[str, str]) -> str:
    """Return ``"<a>_vs_<b>"`` after validating both sides."""
    validate_config_name(pair[0])
    validate_config_name(pair[1])
    return f"{pair[0]}_vs_{pair[1]}"


@dataclass(frozen=True)
class EvalPaths:
    """Resolver for every path under ``data/eval/``.

    Construct once with the eval root (usually ``Path("data/eval")``) and
    call the methods to get specific paths. Methods do not create
    directories — callers ``mkdir(parents=True, exist_ok=True)`` as needed.
    """

    root: Path = DEFAULT_ROOT

    # ---- flat per-config caches (latest, overwritable) -------------------

    @property
    def inferences_root(self) -> Path:
        return self.root / "inferences"

    @property
    def inferences_clean_root(self) -> Path:
        return self.root / "inferences_clean"

    @property
    def judgments_root(self) -> Path:
        return self.root / "judgments"

    @property
    def learned_scores_root(self) -> Path:
        return self.root / "learned_scores"

    @property
    def mauve_scores_root(self) -> Path:
        return self.root / "mauve_scores"

    @property
    def mauve_ref_root(self) -> Path:
        return self.root / "mauve_ref"

    @property
    def reference_scores_root(self) -> Path:
        return self.root / "reference_scores"

    @property
    def validation_root(self) -> Path:
        return self.root / "validation"

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    @property
    def scenarios_path(self) -> Path:
        return self.root / "url_scenarios.jsonl"

    def inferences_dir(self, config: str) -> Path:
        validate_config_name(config)
        return self.inferences_root / config

    def inferences_clean_dir(self, config: str) -> Path:
        validate_config_name(config)
        return self.inferences_clean_root / config

    def judgments_dir(self, judge: str, pair: tuple[str, str]) -> Path:
        return self.judgments_root / judge.replace("/", "_") / _pair_dir_name(pair)

    def learned_scores_path(self, config: str) -> Path:
        validate_config_name(config)
        return self.learned_scores_root / f"{config}.parquet"

    def mauve_scores_path(self, config: str) -> Path:
        validate_config_name(config)
        return self.mauve_scores_root / f"{config}.parquet"

    def reference_scores_path(self, config: str) -> Path:
        validate_config_name(config)
        return self.reference_scores_root / f"{config}.parquet"

    def validation_path(self, stream: str, judge: str) -> Path:
        return self.validation_root / f"{stream}_{judge.replace('/', '_')}.json"

    # ---- per-run frozen artifacts ----------------------------------------

    def run_dir(self, run_id: str) -> Path:
        validate_run_id(run_id)
        return self.runs_root / run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "manifest.json"

    def aggregates_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "aggregates"

    def diagnostics_dir(self, run_id: str, kind: str) -> Path:
        if not re.match(r"^[a-z0-9][a-z0-9_-]*$", kind):
            raise ValueError(
                f"Invalid diagnostics kind {kind!r}: expected lowercase "
                "alphanumeric + dashes/underscores (e.g. 'agent_smoke')."
            )
        return self.run_dir(run_id) / "diagnostics" / kind

    def batches_dir(self, run_id: str, judge: str, pair: tuple[str, str]) -> Path:
        return self.run_dir(run_id) / "batches" / judge.replace("/", "_") / _pair_dir_name(pair)

    # ---- collision guards ------------------------------------------------

    def assert_run_id_free(self, run_id: str, *, force: bool) -> None:
        """Raise ``FileExistsError`` if ``runs/<run_id>/`` already has content.

        ``force=True`` skips the check. Subdirectories of an in-progress
        run (e.g. ``aggregates/`` written by one step, then ``batches/`` by
        another) are allowed — we only refuse if the run already has the
        artifact category being written. Specific writers call the finer-
        grained guards (:meth:`assert_aggregates_free` etc.).
        """
        if force:
            return
        path = self.run_dir(run_id)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Run {run_id!r} already exists at {path}. "
                "Pass --force to overwrite, or pick a new run_id."
            )

    def assert_aggregates_free(self, run_id: str, *, force: bool) -> None:
        if force:
            return
        path = self.aggregates_dir(run_id)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Aggregates for run {run_id!r} already exist at {path}. Pass --force to overwrite."
            )


DEFAULT_PATHS = EvalPaths()
