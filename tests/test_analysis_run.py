"""Test that every analysis run() produces valid output."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from src.analysis.polymarket.polymarket_trader_performance import PolymarketTraderPerformanceAnalysis
from src.common.analysis import Analysis, AnalysisOutput

_ALL_ANALYSES = Analysis.load()
_STATIC_ANALYSES = [c for c in _ALL_ANALYSES if c.__name__ != "WinRateByPriceAnimatedAnalysis"]
_ANIMATED_ANALYSES = [c for c in _ALL_ANALYSES if c.__name__ == "WinRateByPriceAnimatedAnalysis"]


def _build_kwargs(cls: type[Analysis], fixture_dirs: dict[str, Path]) -> dict[str, Path]:
    """Map constructor params to fixture paths based on platform module."""
    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters if p != "self"]

    module = cls.__module__
    is_kalshi = ".kalshi." in module
    is_polymarket = ".polymarket." in module

    kwargs: dict[str, Path] = {}
    for param in params:
        # Direct match — comparison module params use explicit platform prefixes
        if param in fixture_dirs:
            kwargs[param] = fixture_dirs[param]
        elif is_kalshi and param == "trades_dir":
            kwargs[param] = fixture_dirs["kalshi_trades_dir"]
        elif is_kalshi and param == "markets_dir":
            kwargs[param] = fixture_dirs["kalshi_markets_dir"]
        elif is_polymarket and param == "trades_dir":
            kwargs[param] = fixture_dirs["polymarket_trades_dir"]
        elif is_polymarket and param == "legacy_trades_dir":
            kwargs[param] = fixture_dirs["polymarket_legacy_trades_dir"]
        elif is_polymarket and param == "markets_dir":
            kwargs[param] = fixture_dirs["polymarket_markets_dir"]
        elif is_polymarket and param == "blocks_dir":
            kwargs[param] = fixture_dirs["polymarket_blocks_dir"]

    return kwargs


@pytest.mark.parametrize("cls", _STATIC_ANALYSES, ids=lambda c: c.__name__)
def test_analysis_run(cls: type[Analysis], all_fixture_dirs: dict[str, Path]):
    """Every non-animated analysis run() returns valid AnalysisOutput."""
    kwargs = _build_kwargs(cls, all_fixture_dirs)
    instance = cls(**kwargs)
    output = instance.run()

    assert isinstance(output, AnalysisOutput)

    if output.data is not None:
        assert isinstance(output.data, pd.DataFrame)

    if output.figure is not None:
        assert isinstance(output.figure, Figure)

    if output.chart is not None:
        json_str = output.chart.to_json()
        parsed = json.loads(json_str)
        assert "type" in parsed
        assert "data" in parsed

    # Close figure to prevent memory leaks
    if isinstance(output.figure, Figure):
        plt.close(output.figure)


@pytest.mark.slow
@pytest.mark.parametrize("cls", _ANIMATED_ANALYSES, ids=lambda c: c.__name__)
def test_animated_analysis_run(cls: type[Analysis], all_fixture_dirs: dict[str, Path]):
    """Animated analysis run() returns valid AnalysisOutput with FuncAnimation."""
    kwargs = _build_kwargs(cls, all_fixture_dirs)
    instance = cls(**kwargs)
    output = instance.run()

    assert isinstance(output, AnalysisOutput)
    assert isinstance(output.figure, FuncAnimation)

    if output.data is not None:
        assert isinstance(output.data, pd.DataFrame)


# ── Checkpoint tests for PolymarketTraderPerformanceAnalysis ─────────────────


def _make_trader_instance(fixture_dirs: dict[str, Path], checkpoint_dir: Path, fmt: str = "parquet"):
    return PolymarketTraderPerformanceAnalysis(
        trades_dir=fixture_dirs["polymarket_trades_dir"],
        markets_dir=fixture_dirs["polymarket_markets_dir"],
        blocks_dir=fixture_dirs["polymarket_blocks_dir"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_format=fmt,
    )


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_checkpoint_files_created(fmt: str, all_fixture_dirs: dict[str, Path], tmp_path: Path):
    """First run writes all three checkpoint files in the requested format."""
    ckpt = tmp_path / "ckpt"
    output = _make_trader_instance(all_fixture_dirs, ckpt, fmt).run()
    plt.close("all")

    assert isinstance(output, AnalysisOutput)
    assert (ckpt / f"token_resolution.{fmt}").exists()
    assert (ckpt / f"trade_pnl.{fmt}").exists()
    assert (ckpt / f"trader_metrics.{fmt}").exists()


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_resume_from_trader_metrics_checkpoint(fmt: str, all_fixture_dirs: dict[str, Path], tmp_path: Path):
    """Second run loads trader_metrics checkpoint and produces identical results."""
    ckpt = tmp_path / "ckpt"
    out1 = _make_trader_instance(all_fixture_dirs, ckpt, fmt).run()
    plt.close("all")

    # Remove upstream checkpoints to confirm only trader_metrics is needed
    (ckpt / f"token_resolution.{fmt}").unlink()
    (ckpt / f"trade_pnl.{fmt}").unlink()

    out2 = _make_trader_instance(all_fixture_dirs, ckpt, fmt).run()
    plt.close("all")

    assert out1.data is not None and out2.data is not None
    pd.testing.assert_frame_equal(out1.data.reset_index(drop=True), out2.data.reset_index(drop=True))


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_resume_from_trade_pnl_checkpoint(fmt: str, all_fixture_dirs: dict[str, Path], tmp_path: Path):
    """Second run skips SQL query when trade_pnl checkpoint exists."""
    ckpt = tmp_path / "ckpt"
    out1 = _make_trader_instance(all_fixture_dirs, ckpt, fmt).run()
    plt.close("all")

    # Remove only the trader_metrics checkpoint to force re-aggregation from trade_pnl
    (ckpt / f"trader_metrics.{fmt}").unlink()

    out2 = _make_trader_instance(all_fixture_dirs, ckpt, fmt).run()
    plt.close("all")

    assert out1.data is not None and out2.data is not None
    pd.testing.assert_frame_equal(out1.data.reset_index(drop=True), out2.data.reset_index(drop=True))


def test_no_checkpoint_dir_runs_clean(all_fixture_dirs: dict[str, Path]):
    """Without checkpoint_dir the analysis runs normally without writing any files."""
    instance = PolymarketTraderPerformanceAnalysis(
        trades_dir=all_fixture_dirs["polymarket_trades_dir"],
        markets_dir=all_fixture_dirs["polymarket_markets_dir"],
        blocks_dir=all_fixture_dirs["polymarket_blocks_dir"],
    )
    output = instance.run()
    plt.close("all")
    assert isinstance(output, AnalysisOutput)


def test_invalid_checkpoint_format_raises():
    """Unsupported format string raises ValueError at construction time."""
    with pytest.raises(ValueError, match="checkpoint_format"):
        PolymarketTraderPerformanceAnalysis(checkpoint_format="json")
