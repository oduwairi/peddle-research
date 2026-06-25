"""Industry benchmark calibration for proxy score tiers.

Compares Draper.ai's tier distributions against published industry benchmarks
(WordStream/LocaliQ 2025) as a sanity check. This is Stream C (supporting)
of the RQ3 validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from draper.scoring.schemas import ScoredAd

logger = logging.getLogger("draper")

# WordStream/LocaliQ 2025 industry benchmarks — Facebook Ads
# Source: https://www.wordstream.com/blog/ws/2017/02/28/facebook-advertising-benchmarks
# Updated with 2025 data where available.
# CTR values are percentages.
INDUSTRY_BENCHMARKS: dict[str, dict[str, float]] = {
    "ecommerce": {"avg_ctr": 1.24, "avg_cpc_usd": 0.70, "competitiveness": 0.6},
    "technology": {"avg_ctr": 1.04, "avg_cpc_usd": 1.27, "competitiveness": 0.7},
    "finance": {"avg_ctr": 0.56, "avg_cpc_usd": 3.77, "competitiveness": 0.9},
    "health": {"avg_ctr": 0.83, "avg_cpc_usd": 1.32, "competitiveness": 0.7},
    "education": {"avg_ctr": 0.73, "avg_cpc_usd": 1.06, "competitiveness": 0.5},
    "real_estate": {"avg_ctr": 0.99, "avg_cpc_usd": 1.81, "competitiveness": 0.7},
    "retail": {"avg_ctr": 1.59, "avg_cpc_usd": 0.70, "competitiveness": 0.6},
    "travel": {"avg_ctr": 0.90, "avg_cpc_usd": 0.63, "competitiveness": 0.5},
    "food": {"avg_ctr": 1.20, "avg_cpc_usd": 0.42, "competitiveness": 0.4},
    "beauty": {"avg_ctr": 1.16, "avg_cpc_usd": 1.00, "competitiveness": 0.5},
    "automotive": {"avg_ctr": 0.80, "avg_cpc_usd": 2.24, "competitiveness": 0.8},
}


@dataclass
class VerticalCalibration:
    """Calibration result for a single vertical."""

    vertical: str
    n_ads: int
    tier_counts: dict[str, int]
    high_pct: float
    benchmark_ctr: float | None
    benchmark_competitiveness: float | None
    notes: str = ""


@dataclass
class CalibrationReport:
    """Full benchmark calibration report."""

    verticals: list[VerticalCalibration]
    overall_notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "verticals": [
                {
                    "vertical": v.vertical,
                    "n_ads": v.n_ads,
                    "tier_counts": v.tier_counts,
                    "high_pct": round(v.high_pct, 2),
                    "benchmark_ctr": v.benchmark_ctr,
                    "benchmark_competitiveness": v.benchmark_competitiveness,
                    "notes": v.notes,
                }
                for v in self.verticals
            ],
            "overall_notes": self.overall_notes,
        }


class BenchmarkCalibrator:
    """Compare tier distributions against industry benchmarks."""

    def calibrate(self, scored_ads: list[ScoredAd]) -> CalibrationReport:
        """Run benchmark calibration across all verticals.

        Checks whether tier distributions align with industry expectations:
        - High-CTR verticals (retail, ecommerce, food) should have more
          engagement signal contribution
        - High-competition verticals (finance, automotive) may have lower
          tier assignment due to shorter ad lifespans
        """
        # Group by vertical
        verticals: dict[str, list[ScoredAd]] = {}
        for ad in scored_ads:
            v = ad.ad.vertical or "unknown"
            verticals.setdefault(v, []).append(ad)

        results: list[VerticalCalibration] = []
        for vertical, ads in sorted(verticals.items()):
            results.append(self._calibrate_vertical(vertical, ads))

        notes = self._overall_notes(results)
        return CalibrationReport(verticals=results, overall_notes=notes)

    def _calibrate_vertical(self, vertical: str, ads: list[ScoredAd]) -> VerticalCalibration:
        """Calibrate a single vertical against benchmarks."""
        tier_counts = {"high": 0, "medium": 0, "low": 0}
        for ad in ads:
            tier_counts[ad.tier] = tier_counts.get(ad.tier, 0) + 1

        n = len(ads)
        high_pct = (tier_counts["high"] / n * 100) if n > 0 else 0.0

        bench = INDUSTRY_BENCHMARKS.get(vertical)
        bench_ctr = bench["avg_ctr"] if bench else None
        bench_comp = bench["competitiveness"] if bench else None

        notes = ""
        if bench:
            # Sanity checks
            if bench_comp is not None and bench_comp >= 0.8 and high_pct > 30:
                notes = (
                    f"Warning: {vertical} is highly competitive "
                    f"(comp={bench_comp}) but has {high_pct:.0f}% high-tier ads"
                )
            elif bench_ctr is not None and bench_ctr >= 1.2 and high_pct < 10:
                notes = (
                    f"Warning: {vertical} has high industry CTR "
                    f"({bench_ctr}%) but only {high_pct:.0f}% high-tier ads"
                )

        return VerticalCalibration(
            vertical=vertical,
            n_ads=n,
            tier_counts=tier_counts,
            high_pct=high_pct,
            benchmark_ctr=bench_ctr,
            benchmark_competitiveness=bench_comp,
            notes=notes,
        )

    @staticmethod
    def _overall_notes(results: list[VerticalCalibration]) -> list[str]:
        """Generate overall calibration notes."""
        notes: list[str] = []
        warnings = [r for r in results if r.notes]
        if warnings:
            notes.append(
                f"{len(warnings)} verticals have benchmark mismatches — "
                "review scoring weights or data quality"
            )

        # Check for verticals with very few ads
        sparse = [r for r in results if r.n_ads < 50 and r.vertical != "unknown"]
        if sparse:
            names = ", ".join(r.vertical for r in sparse)
            notes.append(
                f"Sparse verticals (<50 ads): {names} — tier assignments may be unreliable"
            )

        return notes
