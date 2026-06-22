#!/usr/bin/env python3
"""Generate equity curve CSV/PNG files from portfolio trade CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=5_000_000.0)
    p.add_argument("--inputs", nargs="+", required=True, help="label=trades_csv")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def build_curve(label: str, path: Path, capital: float, out_dir: Path) -> dict[str, object]:
    trades = pd.read_csv(path, parse_dates=["exit_time"]).sort_values("exit_time")
    trades["cum_pnl"] = trades["pnl"].cumsum()
    trades["equity"] = capital + trades["cum_pnl"]
    trades["peak_equity"] = trades["equity"].cummax()
    trades["drawdown"] = trades["peak_equity"] - trades["equity"]
    trades["drawdown_pct"] = trades["drawdown"] / capital * 100.0
    curve = trades[["exit_time", "equity", "cum_pnl", "drawdown", "drawdown_pct"]].copy()
    csv_path = out_dir / f"{label}_equity_curve.csv"
    curve.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(curve["exit_time"], curve["equity"], linewidth=1.4)
    plt.title(f"{label} equity curve")
    plt.xlabel("Exit time")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    png_path = out_dir / f"{label}_equity_curve.png"
    plt.savefig(png_path, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(curve["exit_time"], -curve["drawdown_pct"], linewidth=1.2, color="firebrick")
    plt.title(f"{label} drawdown")
    plt.xlabel("Exit time")
    plt.ylabel("Drawdown (% of initial capital)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    dd_path = out_dir / f"{label}_drawdown.png"
    plt.savefig(dd_path, dpi=150)
    plt.close()

    pnl = trades["pnl"].values
    return {
        "label": label,
        "trades": int(len(trades)),
        "pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / capital * 100.0),
        "max_drawdown": float(trades["drawdown"].max()),
        "max_drawdown_pct": float(trades["drawdown_pct"].max()),
        "csv": str(csv_path),
        "equity_png": str(png_path),
        "drawdown_png": str(dd_path),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.inputs:
        label, raw_path = item.split("=", 1)
        rows.append(build_curve(label, Path(raw_path), args.capital, out_dir))
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "equity_curve_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
