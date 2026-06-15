#!/usr/bin/env python3
"""Consensus-collapse counterfactual analysis.

Reconstructs what the market did in the suppressed minority direction after the
directional consensus collapsed to NEUTRAL while a position was open.

A management snapshot with scan_direction == "NEUTRAL" marks a moment where the
consensus dissolved (agreement < threshold). The position direction tells us
which side APEX was holding; the *suppressed* minority direction is the opposite
side (the fast cluster that pulled agreement below the gate).

For each NEUTRAL window we measure, from the M5 OHLC series, how far price moved
in the suppressed direction (MFE), how far it moved against it (MAE), and the
close-to-close P&L over 15/30/60 minute horizons. From that we estimate the
expectancy of a hypothetical suppressed scalp with a fixed stop.

Outputs a markdown report. Re-runnable as more data lands. No data files are
modified; only files under analysis/ are written.
"""

from __future__ import annotations

import bisect
import csv
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL_DIR = os.path.join(REPO, "decision_journal")
OUT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "consensus_collapse_counterfactual.md")

# Hypothetical suppressed-scalp risk (pips) used for expectancy/R math.
STOP_PIPS = 5.0
HORIZONS_MIN = [15, 30, 60]
PIP_THRESHOLDS = [3, 5, 8, 10, 15, 20]
# profit_r values outside this band are treated as data bugs and excluded from
# the profit_r-conditioned slices (the R denominator is clearly broken upstream).
PROFIT_R_SANE = 50.0


def pip_multiplier(symbol: str) -> float:
    """Pips-per-price-unit for an instrument (1 pip = price_move * multiplier)."""
    s = symbol.upper()
    # Crypto: quote moves are large; treat 1.0 unit = 1 "pip" equivalent.
    if any(s.startswith(c) for c in ("BTC", "ETH", "BNB", "LTC", "XRP", "SOL",
                                     "ADA", "DOT")):
        return 1.0
    # Energy
    if s in ("XTIUSD", "XBRUSD"):
        return 100.0
    # Metals
    if s == "XAUUSD":
        return 10.0
    if s == "XAGUSD":
        return 100.0
    # Equity indices
    if s in ("US30", "US100", "US500", "GER40", "FRA40", "UK100", "AUS200",
             "JP225", "HK50", "AUS200"):
        return 10.0
    # FX
    if s.endswith("JPY"):
        return 100.0
    return 10000.0


def parse_journal_ts(ts: str) -> datetime:
    """Parse an ISO timestamp and drop tzinfo for naive comparison with CSV."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def parse_csv_ts(ts: str) -> datetime:
    """Parse a CSV time cell which may or may not carry a tz suffix."""
    ts = ts.strip()
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


@dataclass
class Bars:
    times: list = field(default_factory=list)
    high: list = field(default_factory=list)
    low: list = field(default_factory=list)
    close: list = field(default_factory=list)


_BAR_CACHE: dict = {}


def load_bars(symbol: str, tf: str = "M5") -> Bars | None:
    key = (symbol, tf)
    if key in _BAR_CACHE:
        return _BAR_CACHE[key]
    path = os.path.join(REPO, f"{symbol}_{tf}.csv")
    if not os.path.exists(path):
        _BAR_CACHE[key] = None
        return None
    b = Bars()
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = parse_csv_ts(row["time"])
                b.times.append(t)
                b.high.append(float(row["high"]))
                b.low.append(float(row["low"]))
                b.close.append(float(row["close"]))
            except (ValueError, KeyError):
                continue
    _BAR_CACHE[key] = b
    return b


def excursion(symbol: str, start_dt: datetime, suppressed_dir: str,
              horizon_min: int):
    """Return (mfe_pips, mae_pips, close_pnl_pips) in the suppressed direction.

    MFE = best favorable move for the suppressed side over the horizon.
    MAE  = worst adverse move (how far it went the wrong way first).
    close_pnl = signed P&L if held to the horizon close, suppressed side.
    Returns None when no covering bars exist.
    """
    b = load_bars(symbol)
    if not b or not b.times:
        return None
    mult = pip_multiplier(symbol)
    # First bar at or after start.
    i0 = bisect.bisect_left(b.times, start_dt)
    if i0 >= len(b.times):
        return None
    entry = b.close[i0]
    end_dt = start_dt + timedelta(minutes=horizon_min)
    best_fav = 0.0
    worst_adv = 0.0
    last_close = entry
    saw = False
    for i in range(i0, len(b.times)):
        if b.times[i] > end_dt:
            break
        saw = True
        last_close = b.close[i]
        if suppressed_dir == "SHORT":
            fav = (entry - b.low[i]) * mult   # price dropping = good for short
            adv = (b.high[i] - entry) * mult
        else:  # LONG
            fav = (b.high[i] - entry) * mult
            adv = (entry - b.low[i]) * mult
        best_fav = max(best_fav, fav)
        worst_adv = max(worst_adv, adv)
    if not saw:
        return None
    if suppressed_dir == "SHORT":
        close_pnl = (entry - last_close) * mult
    else:
        close_pnl = (last_close - entry) * mult
    return best_fav, worst_adv, close_pnl


def load_management_snapshots():
    """Yield management snapshots (decision_type missing/UNKNOWN) chronologically."""
    rows = []
    for path in sorted(glob.glob(os.path.join(JOURNAL_DIR, "decisions_*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dt = d.get("decision_type")
                if dt == "ENTRY":
                    continue
                if "order_id" not in d or "scan_direction" not in d:
                    continue
                rows.append(d)
    rows.sort(key=lambda d: d["timestamp"])
    return rows


@dataclass
class Window:
    symbol: str
    order_id: str
    position_dir: str
    suppressed_dir: str
    start_ts: datetime
    end_ts: datetime
    n_snapshots: int
    start_profit_r: float
    start_pnl_pips: float
    start_momentum: float
    start_tf_alignment: float


def build_neutral_windows(snapshots):
    """Group consecutive NEUTRAL snapshots per order_id into windows."""
    by_oid = defaultdict(list)
    for d in snapshots:
        by_oid[d["order_id"]].append(d)
    windows = []
    for oid, snaps in by_oid.items():
        snaps.sort(key=lambda d: d["timestamp"])
        cur = None
        for d in snaps:
            is_neutral = d.get("scan_direction") == "NEUTRAL"
            if is_neutral:
                ts = parse_journal_ts(d["timestamp"])
                if cur is None:
                    pdir = (d.get("direction") or "").upper()
                    sup = "SHORT" if pdir == "LONG" else (
                        "LONG" if pdir == "SHORT" else "")
                    sit = d.get("situation", {})
                    cur = Window(
                        symbol=d["symbol"], order_id=oid, position_dir=pdir,
                        suppressed_dir=sup, start_ts=ts, end_ts=ts,
                        n_snapshots=1,
                        start_profit_r=d.get("profit_r", 0.0),
                        start_pnl_pips=d.get("pnl_pips", 0.0),
                        start_momentum=sit.get("momentum", 0.0),
                        start_tf_alignment=sit.get("tf_alignment", 0.0))
                else:
                    cur.end_ts = ts
                    cur.n_snapshots += 1
            else:
                if cur is not None:
                    windows.append(cur)
                    cur = None
        if cur is not None:
            windows.append(cur)
    return [w for w in windows if w.suppressed_dir in ("LONG", "SHORT")]


def aligned_base_rate(snapshots, horizon_min=30):
    """Base rate: when scan matches position, how far does the OPPOSITE side run?"""
    fav_list = []
    covered = 0
    for d in snapshots:
        if d.get("scan_direction") not in ("LONG", "SHORT"):
            continue
        pdir = (d.get("direction") or "").upper()
        if d.get("scan_direction") != pdir:
            continue
        opp = "SHORT" if pdir == "LONG" else "LONG"
        ts = parse_journal_ts(d["timestamp"])
        ex = excursion(d["symbol"], ts, opp, horizon_min)
        if ex is None:
            continue
        covered += 1
        fav_list.append(ex[0])
    return fav_list, covered


def summarize(windows, horizon_min):
    """Compute MFE/MAE/close stats + expectancy for a horizon. Returns dict."""
    rows = []
    for w in windows:
        ex = excursion(w.symbol, w.start_ts, w.suppressed_dir, horizon_min)
        if ex is None:
            continue
        mfe, mae, cpnl = ex
        rows.append((w, mfe, mae, cpnl))
    n = len(rows)
    out = {"covered": n, "rows": rows}
    if n == 0:
        return out
    mfes = [r[1] for r in rows]
    maes = [r[2] for r in rows]
    cpnls = [r[3] for r in rows]
    out["avg_mfe"] = sum(mfes) / n
    out["avg_mae"] = sum(maes) / n
    out["avg_close_pnl"] = sum(cpnls) / n
    # threshold hit rates
    out["thresholds"] = {}
    for th in PIP_THRESHOLDS:
        hits = sum(1 for m in mfes if m >= th)
        out["thresholds"][th] = (hits, hits / n)
    # expectancy with fixed stop: win if MFE >= STOP before MAE >= STOP (approx:
    # use MFE>=target & MAE<STOP as a clean win; else loss of STOP).
    target = STOP_PIPS
    wins = []
    losses = 0
    for w, mfe, mae, cpnl in rows:
        if mae >= STOP_PIPS and mfe < target:
            losses += 1                      # stopped out before target
        elif mfe >= target:
            wins.append(mfe)                 # reached target favorable
        else:
            losses += 1                      # neither: treat as scratch-loss
    win_rate = len(wins) / n
    avg_winner = (sum(wins) / len(wins)) if wins else 0.0
    expectancy_pips = win_rate * avg_winner - (1 - win_rate) * STOP_PIPS
    out["win_rate"] = win_rate
    out["avg_winner_pips"] = avg_winner
    out["expectancy_pips"] = expectancy_pips
    out["expectancy_r"] = expectancy_pips / STOP_PIPS
    return out


def fmt_pct(x):
    return f"{x*100:.1f}%"


def main():
    snapshots = load_management_snapshots()
    n_mgmt = len(snapshots)
    n_neutral = sum(1 for d in snapshots if d.get("scan_direction") == "NEUTRAL")
    windows = build_neutral_windows(snapshots)

    # Coverage: how many windows have any M5 bars at their start.
    covered_windows = 0
    for w in windows:
        if excursion(w.symbol, w.start_ts, w.suppressed_dir, 15) is not None:
            covered_windows += 1

    per_h = {h: summarize(windows, h) for h in HORIZONS_MIN}

    # Double-miss: position losing (profit_r < 0, sane) + suppressed side ran.
    dm_rows = []
    base_h = 30
    for w in windows:
        if not (-PROFIT_R_SANE <= w.start_profit_r < 0):
            continue
        ex = excursion(w.symbol, w.start_ts, w.suppressed_dir, base_h)
        if ex is None:
            continue
        mfe, mae, cpnl = ex
        if mfe >= STOP_PIPS:
            dm_rows.append((w, mfe, mae, cpnl))

    base_fav, base_cov = aligned_base_rate(snapshots, base_h)
    base_hit = {}
    if base_fav:
        for th in PIP_THRESHOLDS:
            h = sum(1 for m in base_fav if m >= th)
            base_hit[th] = (h, h / len(base_fav))
        base_avg_mfe = sum(base_fav) / len(base_fav)
    else:
        base_avg_mfe = 0.0

    # Per-symbol (top 5 by window count, with coverage).
    sym_counts = defaultdict(int)
    for w in windows:
        sym_counts[w.symbol] += 1
    top_syms = sorted(sym_counts.items(), key=lambda kv: -kv[1])[:5]
    sym_stats = {}
    for sym, _cnt in top_syms:
        sw = [w for w in windows if w.symbol == sym]
        sym_stats[sym] = summarize(sw, base_h)

    # ---- write report ----
    L = []
    L.append("# Consensus-Collapse Counterfactual Analysis\n")
    L.append("_Generated by `analysis/run_counterfactual.py`. Re-run as more "
             "data lands._\n")
    L.append("## 0. Method & Coverage\n")
    L.append("A management snapshot with `scan_direction == NEUTRAL` marks a "
             "moment the directional consensus dissolved while a position was "
             "open. The **suppressed minority direction** is the side opposite "
             "the open position — the fast cluster that pulled panel agreement "
             "below the gate. We measure, from M5 OHLC, how far price moved in "
             "that suppressed direction afterward.\n")
    L.append(f"- Management snapshots parsed: **{n_mgmt}**")
    L.append(f"- NEUTRAL snapshots: **{n_neutral}**")
    L.append(f"- Distinct NEUTRAL windows (consecutive, per order): "
             f"**{len(windows)}**")
    L.append(f"- NEUTRAL windows with M5 price coverage: **{covered_windows}** "
             f"({fmt_pct(covered_windows/len(windows)) if windows else 'n/a'})")
    L.append(f"- Hypothetical suppressed-scalp stop: **{STOP_PIPS:.0f} pips**\n")
    L.append("> **Coverage caveat:** most bundled M5 CSVs end **2026-06-12 "
             "23:55** (a few, e.g. BTCUSD, reach mid June 13), while the "
             "decision journal spans **June 11–15**. NEUTRAL windows on June "
             "13–15 therefore have no price data and are excluded from "
             "excursion math. Notably, **all 335 BTCUSD NEUTRAL snapshots fall "
             "on June 13–15**, so the covered window set is FX/JPY-dominated. "
             "All expectancy numbers below are computed on the covered subset "
             "only.\n")
    L.append("> **Unit caveat:** 'pips' are FX/JPY pips. Crypto, metals, "
             "energy and index instruments use native-unit conversions (see "
             "`pip_multiplier`) and are not directly comparable; they are "
             "near-absent from the covered subset (1 crypto window), so the "
             "aggregate reflects FX behaviour.\n")

    L.append("## 1. Summary Statistics (suppressed direction)\n")
    L.append("| Horizon | Covered | Avg MFE (pips) | Avg MAE (pips) | "
             "Avg close P&L | Win rate (≥5p) | Avg winner | Expectancy (pips) | "
             "Expectancy (R) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for h in HORIZONS_MIN:
        s = per_h[h]
        if s["covered"] == 0:
            L.append(f"| {h}min | 0 | – | – | – | – | – | – | – |")
            continue
        L.append(f"| {h}min | {s['covered']} | {s['avg_mfe']:.1f} | "
                 f"{s['avg_mae']:.1f} | {s['avg_close_pnl']:+.1f} | "
                 f"{fmt_pct(s['win_rate'])} | {s['avg_winner_pips']:.1f} | "
                 f"{s['expectancy_pips']:+.1f} | {s['expectancy_r']:+.2f} |")
    L.append("")

    L.append("## 2. Per-threshold breakdown (MFE in suppressed direction)\n")
    for h in HORIZONS_MIN:
        s = per_h[h]
        if s["covered"] == 0:
            continue
        L.append(f"**{h}min horizon** (n={s['covered']})\n")
        L.append("| Threshold | Windows reaching | Hit rate |")
        L.append("|---|---|---|")
        for th in PIP_THRESHOLDS:
            hits, rate = s["thresholds"][th]
            L.append(f"| ≥{th} pips | {hits} | {fmt_pct(rate)} |")
        L.append("")

    L.append("## 3. Time-horizon comparison\n")
    L.append("See §1 — MFE grows with horizon; expectancy peaks where avg "
             "winner outruns the fixed stop before adverse excursion stops it "
             "out.\n")

    L.append("## 4. Double-miss cases (position losing + suppressed side ran ≥5p, "
             f"{base_h}min)\n")
    L.append(f"- Losing NEUTRAL windows (sane profit_r<0) where the suppressed "
             f"direction produced ≥{STOP_PIPS:.0f} pips: **{len(dm_rows)}**")
    if dm_rows:
        tot_left = sum(r[1] for r in dm_rows)
        L.append(f"- Total favorable pips left on the table (sum MFE): "
                 f"**{tot_left:.0f} pips**")
        L.append(f"- Average per double-miss: **{tot_left/len(dm_rows):.1f} "
                 f"pips**\n")
        L.append("| Symbol | Pos dir | Suppressed | Start profit_r | Start "
                 "mom | MFE pips | MAE pips |")
        L.append("|---|---|---|---|---|---|---|")
        for w, mfe, mae, cpnl in sorted(dm_rows, key=lambda r: -r[1])[:15]:
            L.append(f"| {w.symbol} | {w.position_dir} | {w.suppressed_dir} | "
                     f"{w.start_profit_r:.1f} | {w.start_momentum:+.2f} | "
                     f"{mfe:.1f} | {mae:.1f} |")
        L.append("")
    else:
        L.append("")

    L.append(f"## 5. Comparison with aligned-scan base rate ({base_h}min)\n")
    L.append("When the scan *agrees* with the position, how far does the "
             "OPPOSITE side run anyway? This is the noise floor — a suppressed "
             "move is only signal if it beats this.\n")
    L.append(f"- Aligned snapshots with coverage: **{base_cov}**")
    L.append(f"- Base-rate avg opposite-side MFE: **{base_avg_mfe:.1f} pips**")
    s30 = per_h[base_h]
    if s30["covered"]:
        L.append(f"- Suppressed (NEUTRAL) avg MFE: **{s30['avg_mfe']:.1f} "
                 f"pips**")
        lift = s30["avg_mfe"] - base_avg_mfe
        L.append(f"- **Lift over base rate: {lift:+.1f} pips**\n")
    L.append("| Threshold | Suppressed hit | Base-rate hit |")
    L.append("|---|---|---|")
    for th in PIP_THRESHOLDS:
        sup = s30["thresholds"][th][1] if s30["covered"] else 0.0
        bse = base_hit.get(th, (0, 0.0))[1]
        L.append(f"| ≥{th} pips | {fmt_pct(sup)} | {fmt_pct(bse)} |")
    L.append("")

    L.append("## 6. Per-symbol breakdown (top 5 by NEUTRAL-window count, "
             f"{base_h}min)\n")
    L.append("| Symbol | Windows | Covered | Avg MFE | Win rate (≥5p) | "
             "Expectancy (pips) | Expectancy (R) |")
    L.append("|---|---|---|---|---|---|---|")
    for sym, cnt in top_syms:
        s = sym_stats[sym]
        if s["covered"] == 0:
            L.append(f"| {sym} | {cnt} | 0 | – | – | – | – |")
            continue
        L.append(f"| {sym} | {cnt} | {s['covered']} | {s['avg_mfe']:.1f} | "
                 f"{fmt_pct(s['win_rate'])} | {s['expectancy_pips']:+.1f} | "
                 f"{s['expectancy_r']:+.2f} |")
    L.append("")

    # ---- verdict ----
    L.append("## 7. Verdict\n")
    s30 = per_h[30]
    if s30["covered"] == 0:
        L.append("**Inconclusive on covered data** — no NEUTRAL windows fell "
                 "inside the M5 price coverage window. Re-run once M5 CSVs "
                 "cover June 13–15.\n")
    else:
        exp = s30["expectancy_pips"]
        lift = s30["avg_mfe"] - base_avg_mfe
        material = exp > 0 and lift > 1.0
        if material:
            L.append(f"**Yes — there is material missed expectancy in the "
                     f"suppressed minority clusters.** On the covered subset, "
                     f"the suppressed direction returns **{exp:+.1f} pips** "
                     f"expectancy per event (≈{s30['expectancy_r']:+.2f}R on a "
                     f"{STOP_PIPS:.0f}-pip stop) at the 30-min horizon, "
                     f"**{lift:+.1f} pips above** the aligned-scan base rate. "
                     f"The consensus is dissolving tradable moves, not just "
                     f"noise.\n")
        else:
            L.append(f"**No clear edge on covered data.** The suppressed "
                     f"direction's 30-min expectancy is **{exp:+.1f} pips** "
                     f"(≈{s30['expectancy_r']:+.2f}R), only **{lift:+.1f} "
                     f"pips** vs the aligned-scan base rate — within noise. The "
                     f"NEUTRAL collapse is mostly suppressing chop, not "
                     f"systematically-profitable reversals, on this sample.\n")
        L.append(f"- Double-miss events (holding a loser while the suppressed "
                 f"side ran ≥{STOP_PIPS:.0f}p): **{len(dm_rows)}** "
                 f"({fmt_pct(len(dm_rows)/s30['covered']) if s30['covered'] else 'n/a'} "
                 f"of covered windows).\n")
    L.append("> **Caveats:** (1) MFE/MAE use M5 bars — intrabar path is "
             "approximated; a 5-pip stop and target could both be touched in "
             "one bar. (2) Coverage is limited to June 11–12; the full sample "
             "needs June 13–15 M5 data. (3) Expectancy ignores spread/slippage "
             "and the existing reversal size-down and falling-knife guards.\n")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")

    print(f"NEUTRAL windows: {len(windows)}  covered: {covered_windows}")
    print(f"30min suppressed avg MFE: "
          f"{per_h[30].get('avg_mfe', 0):.1f}  base: {base_avg_mfe:.1f}")
    print(f"double-miss: {len(dm_rows)}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
