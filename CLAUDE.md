# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync              # Install dependencies
make test            # Run all tests
make lint            # Check style (ruff check + format --check)
make format          # Auto-fix style issues
make analyze         # Interactive menu to run analysis scripts
make index           # Interactive menu to run data indexers
make setup           # Download and extract the pre-collected dataset (~36 GiB)
```

Run a specific analysis directly:
```bash
uv run main.py analyze <analysis_name>   # e.g., win_rate_by_price
uv run main.py analyze all               # Run all analyses
```

Run a single test file:
```bash
uv run pytest tests/test_analysis_run.py -v
uv run pytest tests/test_analysis_run.py::TestKalshiAnalyses -v
```

Skip slow tests:
```bash
uv run pytest tests/ -v -m "not slow"
```

## Architecture

The project is a data collection + analysis framework for Polymarket and Kalshi prediction markets.

### Plugin Discovery Pattern

Both `Analysis` and `Indexer` base classes use the same auto-discovery mechanism: `Analysis.load()` and `Indexer.load()` scan their respective directories (`src/analysis/` and `src/indexers/`) for non-underscore `.py` files, dynamically import them, and collect all non-abstract subclasses. **Any new analysis or indexer is automatically available without registration.**

### Analysis Framework (`src/common/analysis.py`)

- Subclass `Analysis`, set `name` and `description` as instance attributes, implement `run() -> AnalysisOutput`
- `AnalysisOutput` holds: `figure` (matplotlib `Figure` or `FuncAnimation`), `data` (pandas DataFrame), `chart` (`ChartConfig` for JSON export), and optional `metadata`
- `save()` handles multi-format export: PNG/PDF/SVG at 300 DPI, GIF for animations, CSV, and JSON chart configs
- Use `self.progress("description")` context manager for long-running steps (shows a tqdm spinner)
- Outputs go to `output/<analysis_name>.<ext>`

### Chart Config (`src/common/interfaces/chart.py`)

`ChartConfig` / `ChartType` / `UnitType` generate typed JSON configurations for a `ResearchChart` frontend component. Helper factories: `line_chart()`, `bar_chart()`, `area_chart()`, `pie_chart()`, `scatter_chart()`, `heatmap()`, `treemap()`.

### Data Storage (`src/common/storage.py`)

`ParquetStorage` writes market data into chunked Parquet files (`markets_0_10000.parquet`, `markets_10000_20000.parquet`, …). It deduplicates by ticker using an in-memory set populated on first write.

### Indexers (`src/indexers/`)

- `src/indexers/kalshi/` — Kalshi REST API client + markets/trades indexers
- `src/indexers/polymarket/` — Polymarket REST API + Polygon blockchain indexers (CTF Exchange `OrderFilled` events, legacy FPMM events, block timestamp mapping)
- `src/common/client.py` — shared `retry_request()` decorator (tenacity, exponential backoff, retries on 429/5xx)

### Analysis Scripts (`src/analysis/`)

- `src/analysis/kalshi/` — ~18 analyses operating on `data/kalshi/trades/*.parquet` and `data/kalshi/markets/*.parquet`
- `src/analysis/polymarket/` — analyses operating on Polymarket trades/markets/blocks
- `src/analysis/comparison/` — cross-platform comparisons
- `src/analysis/kalshi/util/categories.py` — `get_group()`, `get_hierarchy()`, `GROUP_COLORS` for mapping Kalshi `event_ticker` prefixes to high-level categories (Sports, Politics, Crypto, etc.)

### Data Layer

Analyses query Parquet files directly using **DuckDB** with glob patterns:
```python
con = duckdb.connect()
df = con.execute(f"SELECT ... FROM '{data_dir}/*.parquet' WHERE ...").df()
```

Data directories (populated via `make setup` or `make index`):
- `data/kalshi/markets/` and `data/kalshi/trades/`
- `data/polymarket/markets/`, `data/polymarket/trades/`, `data/polymarket/blocks/`
- `data/polymarket/fpmm_collateral_lookup.json` — maps FPMM addresses to collateral token info

### Tests

Tests use session-scoped fixtures in `tests/conftest.py` that build in-memory DataFrames and write them to `tmp_path` Parquet files. Each analysis class is instantiated with fixture paths injected instead of the real `data/` directory. See `tests/test_analysis_run.py` for the pattern of how analysis constructors accept path overrides.

## Key Conventions

- Line length: 120 characters (ruff)
- Python 3.9+ compatibility required (no 3.10+ union syntax in runtime code; use `from __future__ import annotations`)
- Data paths are resolved relative to the analysis file using `Path(__file__).parent` chains up to the repo root, then into `data/`
- Kalshi prices are in **cents** (1–99); Polymarket prices are **decimals** (0–1)
- `_fetched_at` column is added by storage layer, not source APIs
