import os, numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "voc_data.csv")
START, END = "1930-01", "2020-12"
PREDICTORS = ['dfy','infl','svar','de','lty','tms','tbl','dfr','dp','dy','ltr','ep','bm','ntis','mkt_lag']

def load():
    df = pd.read_csv(DATA, index_col="date")
    df = df[(df.index >= START) & (df.index <= END)]
    return df, df[PREDICTORS].values, df["R_next"].values, pd.PeriodIndex(df.index, freq="M")

def fit_beta(X, y, z):
    T = len(y); A = X.T @ X / T; b = X.T @ y / T
    Ainv = np.linalg.pinv(A) if z == 0 else np.linalg.inv(z*np.eye(A.shape[0]) + A)
    return Ainv @ b

def run_rolling(G, R, T, z, cols=None, std_in_window=True):
    cols = slice(None) if cols is None else cols
    X = G[:, cols]
    pis = []
    for t in range(T, len(R)):
        Xtr, g = X[t-T:t], X[t]
        if std_in_window:
            sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
            Xtr, g = Xtr / sd, g / sd
        pis.append(g @ fit_beta(Xtr, R[t-T:t], z))
    return np.array(pis), R[T:]

def metrics(pi, y):
    s = pi * y; n = len(s); sd = s.std(ddof=1)
    X = np.c_[np.ones(n), y]; c = np.linalg.lstsq(X, s, rcond=None)[0]; res = s - X @ c
    se = np.sqrt((res**2).sum()/(n-2) * np.linalg.inv(X.T@X)[0,0])
    return dict(R2=100*(1-np.var(y-pi)/np.var(y)),
                SR=s.mean()/sd*np.sqrt(12), t_SR=s.mean()/(sd/np.sqrt(n)),
                IR_mkt=c[0]/res.std(ddof=1)*np.sqrt(12), t_IR=c[0]/se,
                max_loss=-s.min(), skew=stats.skew(s), n=n)

NBER = [("1929-08","1933-03"),("1937-05","1938-06"),("1945-02","1945-10"),("1948-11","1949-10"),
        ("1953-07","1954-05"),("1957-08","1958-04"),("1960-04","1961-02"),("1969-12","1970-11"),
        ("1973-11","1975-03"),("1980-01","1980-07"),("1981-07","1982-11"),("1990-07","1991-03"),
        ("2001-03","2001-11"),("2007-12","2009-06"),("2020-02","2020-04")]
