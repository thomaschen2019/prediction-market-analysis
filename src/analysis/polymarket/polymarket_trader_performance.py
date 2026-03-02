"""Analyze Polymarket trader performance by grouping CTF trades by wallet address.

For each resolved market, the PnL of every maker and taker is computed from first
principles using the CTF exchange settlement logic:

  Case A – maker_asset_id = '0'  (maker pays USDC, receives outcome tokens)
    token_id  = taker_asset_id
    maker PnL = (won ? taker_amount : 0) – maker_amount
    taker PnL = maker_amount – (won ? taker_amount : 0)
    USDC leg  = maker_amount

  Case B – taker_asset_id = '0'  (taker pays USDC, receives outcome tokens)
    token_id  = maker_asset_id
    taker PnL = (won ? maker_amount : 0) – taker_amount
    maker PnL = taker_amount – (won ? maker_amount : 0)
    USDC leg  = taker_amount

All amounts in the Parquet files use 6-decimal USDC units; outputs are in USD.

Checkpointing
-------------
Pass ``checkpoint_dir`` to persist intermediate results and enable resume-on-restart.
Three checkpoints are written in order:

  1. token_resolution  – market token→outcome mapping (dict saved as two-column table)
  2. trade_pnl         – per-trade PnL rows after the UNION ALL SQL query
  3. trader_metrics    – fully aggregated per-trader statistics

On the next run the script loads the deepest available checkpoint and skips all
preceding steps.  Pass ``checkpoint_format="csv"`` for human-readable output
(default is ``"parquet"``).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

# Traders with fewer resolved trades than this are excluded from the output.
MIN_RESOLVED_TRADES = 5

# Block bucket size for timestamp approximation (~6 hours at 2 sec/block on Polygon).
BLOCK_BUCKET_SIZE = 10800

_SUPPORTED_FORMATS = ("parquet", "csv")


class PolymarketTraderPerformanceAnalysis(Analysis):
    """Rank Polymarket traders by PnL, Sharpe ratio, win rate, and consistency."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
        checkpoint_format: str = "parquet",
    ):
        """
        Args:
            trades_dir: Directory containing CTF trade Parquet files.
            markets_dir: Directory containing market Parquet files.
            blocks_dir: Optional directory containing block Parquet files (for timestamps).
            checkpoint_dir: Directory for intermediate and final checkpoints.  When
                ``None`` (default) no checkpoints are written or read.
            checkpoint_format: Storage format for checkpoints – ``"parquet"`` (default,
                fast and type-safe) or ``"csv"`` (human-readable).
        """
        super().__init__(
            name="polymarket_trader_performance",
            description="Trader performance analysis: PnL, Sharpe, win rate, and profit factor by wallet",
        )
        if checkpoint_format not in _SUPPORTED_FORMATS:
            raise ValueError(f"checkpoint_format must be one of {_SUPPORTED_FORMATS}, got {checkpoint_format!r}")

        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "polymarket" / "markets")
        self.blocks_dir = Path(blocks_dir or base_dir / "data" / "polymarket" / "blocks")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_format = checkpoint_format

    def run(self) -> AnalysisOutput:
        # ── Try deepest checkpoint first to allow resuming a stopped run ────────
        trader_df = self._load_checkpoint("trader_metrics")
        if trader_df is None:
            trade_df = self._load_checkpoint("trade_pnl")

            if trade_df is None:
                con = duckdb.connect()

                # ── 1. Build token_id → won map from resolved markets ──────────
                token_won = self._load_token_resolution()
                if token_won is None:
                    with self.progress("Loading market resolutions"):
                        markets_df = con.execute(
                            f"""
                            SELECT clob_token_ids, outcome_prices
                            FROM '{self.markets_dir}/*.parquet'
                            WHERE closed = true
                            """
                        ).df()

                    token_won = {}
                    for _, row in markets_df.iterrows():
                        try:
                            prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                            if not prices or len(prices) != 2:
                                continue
                            p0, p1 = float(prices[0]), float(prices[1])
                            if p0 > 0.99 and p1 < 0.01:
                                winning_outcome = 0
                            elif p0 < 0.01 and p1 > 0.99:
                                winning_outcome = 1
                            else:
                                continue
                            token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                            if token_ids and len(token_ids) == 2:
                                token_won[token_ids[0]] = winning_outcome == 0
                                token_won[token_ids[1]] = winning_outcome == 1
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue

                    self._save_token_resolution(token_won)

                if not token_won:
                    return AnalysisOutput(
                        figure=self._empty_figure("No resolved markets found"),
                        data=pd.DataFrame(),
                        chart=self._empty_chart(),
                    )

                con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
                con.executemany("INSERT INTO token_resolution VALUES (?, ?)", list(token_won.items()))

                # ── 2. Optional: build block → timestamp lookup ────────────────
                has_blocks = self.blocks_dir.exists() and any(self.blocks_dir.glob("*.parquet"))
                if has_blocks:
                    con.execute(
                        f"""
                        CREATE TABLE blocks AS
                        SELECT
                            block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                            FIRST(timestamp) AS ts
                        FROM '{self.blocks_dir}/*.parquet'
                        GROUP BY block_number // {BLOCK_BUCKET_SIZE}
                        """
                    )
                    ts_select = f"DATE_TRUNC('month', b.ts::TIMESTAMP) AS trade_month"
                    ts_join = f"LEFT JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket"
                else:
                    ts_select = "NULL AS trade_month"
                    ts_join = ""

                # ── 3. Compute per-trade PnL for every maker and taker ─────────
                #
                # Each resolved trade generates exactly two rows:
                #   • one for the buyer  (USDC → tokens)
                #   • one for the seller (tokens → USDC)
                #
                with self.progress("Computing per-trade PnL"):
                    trade_df = con.execute(
                        f"""
                        SELECT user_address, pnl_usd, cost_usd, won, trade_month
                        FROM (
                            -- Case A: maker is the buyer ---------------------------------
                            SELECT
                                t.maker                                              AS user_address,
                                (CASE WHEN tr.won
                                      THEN  CAST(t.taker_amount AS DOUBLE) - CAST(t.maker_amount AS DOUBLE)
                                      ELSE -CAST(t.maker_amount AS DOUBLE)
                                 END) / 1e6                                          AS pnl_usd,
                                CAST(t.maker_amount AS DOUBLE) / 1e6                AS cost_usd,
                                tr.won,
                                {ts_select}
                            FROM '{self.trades_dir}/*.parquet' t
                            INNER JOIN token_resolution tr ON t.taker_asset_id = tr.token_id
                            {ts_join}
                            WHERE t.maker_asset_id = '0'
                              AND t.maker_amount > 0 AND t.taker_amount > 0
                              AND t.maker IS NOT NULL

                            UNION ALL

                            -- Case A: taker is the seller --------------------------------
                            SELECT
                                t.taker                                              AS user_address,
                                (CAST(t.maker_amount AS DOUBLE)
                                    - CASE WHEN tr.won
                                           THEN CAST(t.taker_amount AS DOUBLE)
                                           ELSE 0 END
                                ) / 1e6                                              AS pnl_usd,
                                CAST(t.maker_amount AS DOUBLE) / 1e6                AS cost_usd,
                                NOT tr.won                                           AS won,
                                {ts_select}
                            FROM '{self.trades_dir}/*.parquet' t
                            INNER JOIN token_resolution tr ON t.taker_asset_id = tr.token_id
                            {ts_join}
                            WHERE t.maker_asset_id = '0'
                              AND t.maker_amount > 0 AND t.taker_amount > 0
                              AND t.taker IS NOT NULL

                            UNION ALL

                            -- Case B: taker is the buyer ---------------------------------
                            SELECT
                                t.taker                                              AS user_address,
                                (CASE WHEN tr.won
                                      THEN  CAST(t.maker_amount AS DOUBLE) - CAST(t.taker_amount AS DOUBLE)
                                      ELSE -CAST(t.taker_amount AS DOUBLE)
                                 END) / 1e6                                          AS pnl_usd,
                                CAST(t.taker_amount AS DOUBLE) / 1e6                AS cost_usd,
                                tr.won,
                                {ts_select}
                            FROM '{self.trades_dir}/*.parquet' t
                            INNER JOIN token_resolution tr ON t.maker_asset_id = tr.token_id
                            {ts_join}
                            WHERE t.taker_asset_id = '0'
                              AND t.maker_amount > 0 AND t.taker_amount > 0
                              AND t.taker IS NOT NULL

                            UNION ALL

                            -- Case B: maker is the seller --------------------------------
                            SELECT
                                t.maker                                              AS user_address,
                                (CAST(t.taker_amount AS DOUBLE)
                                    - CASE WHEN tr.won
                                           THEN CAST(t.maker_amount AS DOUBLE)
                                           ELSE 0 END
                                ) / 1e6                                              AS pnl_usd,
                                CAST(t.taker_amount AS DOUBLE) / 1e6                AS cost_usd,
                                NOT tr.won                                           AS won,
                                {ts_select}
                            FROM '{self.trades_dir}/*.parquet' t
                            INNER JOIN token_resolution tr ON t.maker_asset_id = tr.token_id
                            {ts_join}
                            WHERE t.taker_asset_id = '0'
                              AND t.maker_amount > 0 AND t.taker_amount > 0
                              AND t.maker IS NOT NULL
                        )
                        WHERE user_address IS NOT NULL
                        """
                    ).df()

                self._save_checkpoint("trade_pnl", trade_df)

            if trade_df.empty:
                return AnalysisOutput(
                    figure=self._empty_figure("No resolved trades found"),
                    data=pd.DataFrame(),
                    chart=self._empty_chart(),
                )

            # ── 4. Aggregate per-trader metrics ──────────────────────────────────
            with self.progress("Aggregating trader metrics"):
                trader_df = self._compute_trader_metrics(trade_df)

            self._save_checkpoint("trader_metrics", trader_df)

        # ── 5. Filter, sort, and visualise (always re-run from checkpoint) ───────
        trader_df = trader_df[trader_df["n_trades"] >= MIN_RESOLVED_TRADES].reset_index(drop=True)

        if trader_df.empty:
            return AnalysisOutput(
                figure=self._empty_figure(f"No traders with ≥{MIN_RESOLVED_TRADES} resolved trades"),
                data=pd.DataFrame(),
                chart=self._empty_chart(),
            )

        trader_df = trader_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

        fig = self._create_figure(trader_df)
        chart = self._create_chart(trader_df)
        return AnalysisOutput(figure=fig, data=trader_df, chart=chart)

    # ── Checkpoint helpers ────────────────────────────────────────────────────────

    def _checkpoint_path(self, name: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"{name}.{self.checkpoint_format}"

    def _save_checkpoint(self, name: str, df: pd.DataFrame) -> None:
        """Write *df* to the checkpoint directory (no-op when checkpoint_dir is None)."""
        path = self._checkpoint_path(name)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

    def _load_checkpoint(self, name: str) -> pd.DataFrame | None:
        """Return the saved DataFrame for *name*, or ``None`` if it doesn't exist."""
        path = self._checkpoint_path(name)
        if path is None or not path.exists():
            return None
        if self.checkpoint_format == "parquet":
            return pd.read_parquet(path)
        # CSV: restore column types that don't survive text serialisation
        df = pd.read_csv(path)
        if "trade_month" in df.columns:
            df["trade_month"] = pd.to_datetime(df["trade_month"], errors="coerce")
        if "won" in df.columns:
            df["won"] = df["won"].astype(bool)
        return df

    def _save_token_resolution(self, token_won: dict[str, bool]) -> None:
        df = pd.DataFrame(list(token_won.items()), columns=["token_id", "won"])
        self._save_checkpoint("token_resolution", df)

    def _load_token_resolution(self) -> dict[str, bool] | None:
        df = self._load_checkpoint("token_resolution")
        if df is None:
            return None
        return dict(zip(df["token_id"], df["won"].astype(bool)))

    # ── Metric computation ────────────────────────────────────────────────────────

    def _compute_trader_metrics(self, trade_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-trader performance metrics from trade-level PnL data."""
        df = trade_df.copy()
        df["_ret"] = np.where(df["cost_usd"] > 0, df["pnl_usd"] / df["cost_usd"], 0.0)
        df["_gross_profit"] = df["pnl_usd"].clip(lower=0)
        df["_gross_loss"] = (-df["pnl_usd"]).clip(lower=0)
        df["_win"] = (df["pnl_usd"] > 0).astype(float)

        g = df.groupby("user_address", sort=False)
        agg = g.agg(
            n_trades=("pnl_usd", "count"),
            total_pnl_usd=("pnl_usd", "sum"),
            total_volume_usd=("cost_usd", "sum"),
            mean_return=("_ret", "mean"),
            std_return=("_ret", "std"),
            wins=("_win", "sum"),
            gross_profit_usd=("_gross_profit", "sum"),
            gross_loss_usd=("_gross_loss", "sum"),
            avg_trade_size_usd=("cost_usd", "mean"),
        ).reset_index()

        agg["std_return"] = agg["std_return"].fillna(0.0)
        agg["wins"] = agg["wins"].astype(int)
        agg["win_rate"] = agg["wins"] / agg["n_trades"]
        agg["roi"] = np.where(agg["total_volume_usd"] > 0, agg["total_pnl_usd"] / agg["total_volume_usd"], 0.0)

        # Sharpe: vectorised edge-case handling (zero std → capped sentinel)
        sharpe = np.where(
            agg["std_return"] > 0,
            agg["mean_return"] / agg["std_return"],
            np.where(agg["mean_return"] > 0, 10.0, np.where(agg["mean_return"] < 0, -10.0, 0.0)),
        )
        agg["sharpe"] = np.clip(sharpe.astype(float), -10.0, 10.0)

        # Profit factor
        pf = np.where(
            agg["gross_loss_usd"] > 0,
            agg["gross_profit_usd"] / agg["gross_loss_usd"],
            np.where(agg["gross_profit_usd"] > 0, 10.0, 0.0),
        )
        agg["profit_factor"] = np.clip(pf.astype(float), 0.0, 10.0)

        # Monthly consistency: % of active months with net positive PnL
        if "trade_month" in df.columns and df["trade_month"].notna().any():
            monthly = (
                df.dropna(subset=["trade_month"])
                .groupby(["user_address", "trade_month"])["pnl_usd"]
                .sum()
                .gt(0)
                .groupby(level="user_address")
                .mean()
                .rename("monthly_consistency")
                .reset_index()
            )
            agg = agg.merge(monthly, on="user_address", how="left")
        else:
            agg["monthly_consistency"] = None

        # Round numeric columns
        round_cols = [
            "total_pnl_usd", "total_volume_usd", "roi", "mean_return", "std_return",
            "sharpe", "win_rate", "profit_factor", "gross_profit_usd", "gross_loss_usd",
            "avg_trade_size_usd",
        ]
        for col in round_cols:
            agg[col] = agg[col].round(4)
        mc = agg["monthly_consistency"]
        if pd.api.types.is_numeric_dtype(mc):
            agg["monthly_consistency"] = mc.round(4)

        return agg[[
            "user_address", "n_trades", "total_pnl_usd", "total_volume_usd", "roi",
            "mean_return", "std_return", "sharpe", "win_rate", "profit_factor",
            "gross_profit_usd", "gross_loss_usd", "avg_trade_size_usd", "monthly_consistency",
        ]]

    # ── Visualisation ─────────────────────────────────────────────────────────────

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Polymarket Trader Performance Analysis", fontsize=14, fontweight="bold")

        # ── Panel 1: Total PnL distribution ──────────────────────────────────
        ax1 = axes[0, 0]
        lo, hi = df["total_pnl_usd"].quantile(0.01), df["total_pnl_usd"].quantile(0.99)
        ax1.hist(df["total_pnl_usd"].clip(lo, hi), bins=60, color="#4C72B0", edgecolor="none", alpha=0.85)
        ax1.axvline(0, color="#D65F5F", linestyle="--", linewidth=1.5, label="Break-even")
        n_profit = int((df["total_pnl_usd"] > 0).sum())
        n_loss = int((df["total_pnl_usd"] <= 0).sum())
        ax1.set_xlabel("Total PnL (USD)")
        ax1.set_ylabel("Number of Traders")
        ax1.set_title(f"Total PnL Distribution\n{n_profit:,} profitable · {n_loss:,} unprofitable")
        ax1.legend(fontsize=9)

        # ── Panel 2: Top traders by Sharpe ────────────────────────────────────
        ax2 = axes[0, 1]
        top_n = min(20, len(df))
        top = df.nlargest(top_n, "sharpe")
        bar_colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in top["total_pnl_usd"].values]
        ax2.barh(range(top_n), top["sharpe"].values, color=bar_colors)
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(
            [f"{a[:8]}…  ({p:+.0f} USD)" for a, p in zip(top["user_address"], top["total_pnl_usd"])],
            fontsize=7,
        )
        ax2.set_xlabel("Sharpe Ratio (per-trade return / std)")
        ax2.set_title(f"Top {top_n} Traders by Sharpe Ratio\n(green = profitable, red = unprofitable)")
        ax2.invert_yaxis()
        ax2.axvline(0, color="grey", linewidth=0.8)

        # ── Panel 3: Win rate vs Sharpe scatter (colour = PnL) ────────────────
        ax3 = axes[1, 0]
        lo_pnl = df["total_pnl_usd"].quantile(0.05)
        hi_pnl = df["total_pnl_usd"].quantile(0.95)
        sc = ax3.scatter(
            df["win_rate"] * 100,
            df["sharpe"],
            c=df["total_pnl_usd"].clip(lo_pnl, hi_pnl),
            cmap="RdYlGn",
            s=np.clip(df["n_trades"].values, 5, 200),
            alpha=0.6,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax3, label="Total PnL (USD)")
        ax3.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax3.axvline(50, color="grey", linestyle="--", linewidth=0.8)
        ax3.set_xlabel("Win Rate (%)")
        ax3.set_ylabel("Sharpe Ratio")
        ax3.set_title("Win Rate vs Sharpe Ratio\n(point size ∝ # trades, colour = total PnL)")

        # ── Panel 4: Profit factor distribution ──────────────────────────────
        ax4 = axes[1, 1]
        finite_pf = df["profit_factor"].replace(10.0, np.nan).dropna()
        finite_pf = finite_pf[finite_pf < 8]
        if len(finite_pf) > 0:
            ax4.hist(finite_pf, bins=40, color="#4C72B0", edgecolor="none", alpha=0.85)
            ax4.axvline(1.0, color="#D65F5F", linestyle="--", linewidth=1.5, label="Break-even (PF = 1)")
            ax4.set_xlabel("Profit Factor (gross profit / gross loss)")
            ax4.set_ylabel("Number of Traders")
            ax4.set_title("Profit Factor Distribution\n(> 1 means wins exceed losses)")
            ax4.legend(fontsize=9)
        else:
            ax4.text(0.5, 0.5, "Insufficient data", transform=ax4.transAxes, ha="center", va="center")
            ax4.set_title("Profit Factor Distribution")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        top_n = min(50, len(df))
        top = df.nlargest(top_n, "sharpe")
        chart_data = [
            {
                "address": row["user_address"][:10] + "…",
                "sharpe": round(row["sharpe"], 3),
                "pnl": round(row["total_pnl_usd"], 2),
                "win_rate": round(row["win_rate"] * 100, 1),
                "n_trades": int(row["n_trades"]),
                "roi": round(row["roi"] * 100, 2),
            }
            for _, row in top.iterrows()
        ]
        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="address",
            yKeys=["sharpe"],
            title="Top Polymarket Traders by Sharpe Ratio",
            xLabel="Trader Address",
            yLabel="Sharpe Ratio",
            yUnit=UnitType.NUMBER,
            caption=(
                "Sharpe = mean per-trade return / std per-trade return. "
                "Return = PnL / USDC invested per trade. "
                f"Minimum {MIN_RESOLVED_TRADES} resolved trades required."
            ),
        )

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _empty_figure(self, message: str = "No data available") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title("Polymarket Trader Performance Analysis")
        ax.axis("off")
        return fig

    def _empty_chart(self) -> ChartConfig:
        return ChartConfig(
            type=ChartType.BAR,
            data=[],
            xKey="address",
            yKeys=["sharpe"],
            title="Polymarket Trader Performance — No Data",
        )
