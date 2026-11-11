"""Downloaders for the free public data sources used by the project.

* Historical S&P 500 membership: github.com/fja05680/sp500 (MIT license)
* Daily prices and volumes: Yahoo Finance via ``yfinance`` (split/dividend adjusted)
* Daily risk-free rate: Kenneth R. French data library (1-month T-bill)
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
FAMA_FRENCH_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def _get(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def download_membership(raw_dir: Path) -> Path:
    path = raw_dir / "sp500_components.csv"
    path.write_bytes(_get(MEMBERSHIP_URL))
    return path


def parse_fama_french_rf(zip_bytes: bytes) -> pd.Series:
    """Extract the daily risk-free rate (decimal, not percent) from the Fama-French zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append((parts[0], float(parts[4])))
    dates = pd.to_datetime([d for d, _ in rows], format="%Y%m%d")
    return pd.Series([v / 100.0 for _, v in rows], index=dates, name="rf")


def download_risk_free(raw_dir: Path) -> pd.Series:
    rf = parse_fama_french_rf(_get(FAMA_FRENCH_DAILY_URL))
    rf.to_frame().to_csv(raw_dir / "risk_free_daily.csv")
    return rf


# Companies that changed ticker but kept their price history on Yahoo under the new symbol.
# Without this, e.g. Meta would be missing for 2013-2022 (it was "FB", a symbol Yahoo now
# uses for a different security). Only same-company renames are listed, never mergers into
# a different company.
RENAMES: dict[str, str] = {
    "FB": "META",
    "ABC": "COR",  # AmerisourceBergen -> Cencora
    "ANTM": "ELV",  # Anthem -> Elevance Health
    "BK": "BNY",  # Bank of New York Mellon
    "BLL": "BALL",
    "CTL": "LUMN",  # CenturyLink -> Lumen
    "GPS": "GAP",
    "HCP": "DOC",  # HCP -> Healthpeak
    "PEAK": "DOC",
    "JEC": "J",  # Jacobs
    "PKI": "RVTY",  # PerkinElmer -> Revvity
    "UTX": "RTX",  # United Technologies -> RTX
    "FII": "FHI",  # Federated Hermes
    "HRS": "LHX",  # Harris -> L3Harris
    "SYMC": "GEN",  # Symantec -> NortonLifeLock -> Gen Digital
    "NLOK": "GEN",
    "BHGE": "BKR",  # Baker Hughes
    "MYL": "VTRS",  # Mylan -> Viatris
    "FI": "FISV",  # Fiserv
    "FBHS": "FBIN",  # Fortune Brands
    "HFC": "DINO",  # HollyFrontier -> HF Sinclair
    "WLTW": "WTW",  # Willis Towers Watson
    "RE": "EG",  # Everest
    "DWDP": "DD",  # DowDuPont -> DuPont
    "FLT": "CPAY",  # FleetCor -> Corpay
    "TMK": "GL",  # Torchmark -> Globe Life
    "ADS": "BFH",  # Alliance Data -> Bread Financial
    "WYND": "TNL",  # Wyndham Worldwide -> Travel + Leisure
    "KORS": "CPRI",  # Michael Kors -> Capri
    "MMC": "MRSH",  # Marsh McLennan
}


def to_yahoo(ticker: str) -> str:
    """Map an index-membership ticker to the Yahoo symbol holding that company's history."""
    if ticker in RENAMES:
        return RENAMES[ticker]
    return ticker.replace(".", "-")  # class shares: BRK.B -> BRK-B


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    batch_size: int = 100,
    pause: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices and volumes. Columns use the original ticker spelling.

    Tickers Yahoo no longer knows (mostly delisted companies) come back empty and are dropped.
    """
    orig_to_yahoo = {t: to_yahoo(t) for t in tickers}
    yahoo = sorted(set(orig_to_yahoo.values()))
    closes, volumes = [], []
    for i in range(0, len(yahoo), batch_size):
        batch = yahoo[i : i + batch_size]
        log.info("Downloading prices %d-%d of %d", i + 1, i + len(batch), len(yahoo))
        df = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
        if df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):  # single-ticker edge case
            df.columns = pd.MultiIndex.from_product([df.columns, batch])
        closes.append(df["Close"])
        volumes.append(df["Volume"])
        time.sleep(pause)
    close_y = pd.concat(closes, axis=1)
    volume_y = pd.concat(volumes, axis=1)
    # Several membership tickers can share one Yahoo symbol (e.g. HCP and PEAK -> DOC);
    # each keeps its own column so the point-in-time membership lookup still works.
    have = [t for t in tickers if orig_to_yahoo[t] in close_y.columns]
    close = pd.DataFrame({t: close_y[orig_to_yahoo[t]] for t in have})
    volume = pd.DataFrame({t: volume_y[orig_to_yahoo[t]] for t in have})
    close = close.loc[:, close.notna().any()].sort_index(axis=1)
    volume = volume.reindex(columns=close.columns)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = close.index
    return close, volume
