from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "voc_data.csv"
DATE_COL = "date"
RET_COL = "R_next"

TARGET_IS_NEXT_MONTH = True

PREDICTORS = [
    "dfy", "infl", "svar", "de", "lty", "tms", "tbl", "dfr",
    "dp", "dy", "ltr", "ep", "bm", "ntis", "mkt_lag",
]
ADD_LAG_MKT = False

STANDARDIZE_RETURNS = False
STANDARDIZE_PREDICTORS = False
PREDICTOR_WARMUP = 36

START_DATE = "1930-01"
END_DATE = "2020-12"

T = 12
P_MAX = 12_000
GAMMA = 2.0
LOG10_Z = [-3, -2, -1, 0, 1, 2, 3]
P_GRID = None
STANDARDIZE_IN_WINDOW = True

FEATURES = "core"
ENGINE = "core"

N_SEEDS = 20
SEED0 = 0
N_JOBS = 1
OUT_DIR = Path("results")
