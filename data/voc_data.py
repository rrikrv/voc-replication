from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd

GOYAL_SHEET_ID = "1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ"
GOYAL_URL = f"https://docs.google.com/spreadsheets/d/{GOYAL_SHEET_ID}/export?format=xlsx"

PREDICTORS = [
    "dfy", "infl", "svar", "de", "lty", "tms", "tbl", "dfr",
    "dp", "dy", "ltr", "ep", "bm", "ntis", "mkt_lag",
]

_ALIASES = {
    "yyyymm": ["yyyymm", "date"],
    "Index": ["Index", "index", "price", "sp_index"],
    "D12": ["D12", "d12"],
    "E12": ["E12", "e12"],
    "bm": ["b/m", "bm", "b_m"],
    "tbl": ["tbl"],
    "AAA": ["AAA", "aaa"],
    "BAA": ["BAA", "baa"],
    "lty": ["lty"],
    "ntis": ["ntis"],
    "Rfree": ["Rfree", "rfree", "rf"],
    "infl": ["infl"],
    "ltr": ["ltr"],
    "corpr": ["corpr"],
    "svar": ["svar"],
    "CRSP_SPvw": ["CRSP_SPvw", "crsp_spvw", "vwret"],
}


def read_goyal_raw(source: str | None = None) -> pd.DataFrame:
    if source is None:
        source = GOYAL_URL
    if isinstance(source, str) and source.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(source) as r:
            content = r.read()
        buf = io.BytesIO(content)
        if content[:2] != b"PK":
            raise RuntimeError(
                "Не удалось скачать xlsx с Google Drive. Скачайте файл вручную со "
                "страницы Amit Goyal и передайте путь: load_voc_data('PredictorDataXXXX.xlsx')"
            )
        source = buf
    if isinstance(source, str) and source.lower().endswith(".csv"):
        raw = pd.read_csv(source)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xls = pd.ExcelFile(source)
            sheet = next((s for s in xls.sheet_names if s.lower().startswith("month")),
                         xls.sheet_names[0])
            raw = pd.read_excel(xls, sheet)
    return _rename_columns(raw)


def _rename_columns(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c.strip(): c for c in raw.columns.astype(str)}
    lower = {c.lower(): c for c in cols}
    out = {}
    for std_name, aliases in _ALIASES.items():
        found = None
        for a in aliases:
            if a in cols:
                found = cols[a]; break
            if a.lower() in lower:
                found = cols[lower[a.lower()]]; break
        if found is None:
            raise KeyError(f"Не найдена колонка '{std_name}'. Есть колонки: {list(raw.columns)}")
        out[std_name] = pd.to_numeric(raw[found], errors="coerce")
    df = pd.DataFrame(out)
    df = df.dropna(subset=["yyyymm"])
    df["yyyymm"] = df["yyyymm"].astype(int)
    df.index = pd.PeriodIndex(
        [pd.Period(year=y // 100, month=y % 100, freq="M") for y in df["yyyymm"]], name="date"
    )
    return df.drop(columns="yyyymm").sort_index()


def build_predictors(raw: pd.DataFrame, lag_infl: bool = False) -> pd.DataFrame:
    g = pd.DataFrame(index=raw.index)
    log_p, log_d, log_e = np.log(raw["Index"]), np.log(raw["D12"]), np.log(raw["E12"])

    g["dfy"] = raw["BAA"] - raw["AAA"]
    g["infl"] = raw["infl"].shift(1) if lag_infl else raw["infl"]
    g["svar"] = raw["svar"]
    g["de"] = log_d - log_e
    g["lty"] = raw["lty"]
    g["tms"] = raw["lty"] - raw["tbl"]
    g["tbl"] = raw["tbl"]
    g["dfr"] = raw["corpr"] - raw["ltr"]
    g["dp"] = log_d - log_p
    g["dy"] = log_d - log_p.shift(1)
    g["ltr"] = raw["ltr"]
    g["ep"] = log_e - log_p
    g["bm"] = raw["bm"]
    g["ntis"] = raw["ntis"]
    g["mkt_lag"] = raw["CRSP_SPvw"] - raw["Rfree"]
    return g[PREDICTORS]


def excess_return(raw: pd.DataFrame) -> pd.Series:
    return (raw["CRSP_SPvw"] - raw["Rfree"]).rename("R_raw")


def standardize_returns(r: pd.Series, window: int = 12) -> pd.DataFrame:
    vol = np.sqrt((r ** 2).shift(1).rolling(window, min_periods=window).mean())
    return pd.DataFrame({"R_raw": r, "R_vol": vol, "R": r / vol})


def standardize_predictors(g: pd.DataFrame, min_periods: int = 36) -> pd.DataFrame:
    sd = g.expanding(min_periods=min_periods).std()
    return g / sd


def load_voc_data(
    source: str | None = None,
    start: str | None = None,
    end: str | None = "2020-12",
    min_periods: int = 36,
    ret_window: int = 12,
    lag_infl: bool = False,
) -> pd.DataFrame:
    raw = read_goyal_raw(source)
    g = standardize_predictors(build_predictors(raw, lag_infl), min_periods)
    r = standardize_returns(excess_return(raw), ret_window)

    df = g.join(r)
    df["R_next"] = df["R"].shift(-1)
    df["R_next_raw"] = df["R_raw"].shift(-1)

    df = df.dropna(subset=PREDICTORS + ["R", "R_next"])
    if start is not None:
        df = df[df.index >= pd.Period(start, "M")]
    if end is not None:
        df = df[df.index <= pd.Period(end, "M")]
    return df


def check_no_lookahead(source: str | None = None, cut: str = "1990-12") -> bool:
    raw = read_goyal_raw(source)
    full = standardize_predictors(build_predictors(raw)).join(standardize_returns(excess_return(raw)))
    part_raw = raw[raw.index <= pd.Period(cut, "M")]
    part = standardize_predictors(build_predictors(part_raw)).join(standardize_returns(excess_return(part_raw)))
    a, b = full.loc[part.index], part
    return bool(np.allclose(a.values, b.values, equal_nan=True))


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_voc_data(src)
    print(df.shape, df.index[0], "->", df.index[-1])
    print(df[PREDICTORS + ["R_next"]].describe().T.round(3))
    print("no look-ahead:", check_no_lookahead(src))
