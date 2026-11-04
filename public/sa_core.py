"""
sa_core.py : スポーツデータ分析システムのコアロジック（UI非依存）

- ファイル読み込み（CSVは自動で分割読み込み／xlsx対応）
- 列名の自動推定
- GPS・加速度・心拍の1秒集約と異常値除外
- 運動量指標（距離・HSR・スプリント・加減速・PlayerLoad・心拍）
- 15分／前後半の区間集計
- 気象データの付随情報化
- 運動量サマリーのクレンジング・負荷（ACWR）計算
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_FILES = 2
MAX_FILES = 10
CHUNK_ROWS = 500_000
ENCODINGS = ("utf-8-sig", "cp932", "utf-16")

WX_TEMP = "気温(℃)"
WX_HUM = "湿度(%)"
WX_WIND = "風速(m/s)"
WX_SKY = "天気"

# ── 列名の別名辞書 ───────────────────────────────────
RAW_ALIASES = {
    "time":  ["日時", "時刻", "タイムスタンプ", "時間", "timestamp", "datetime",
              "time", "elapsed", "経過時間", "経過秒"],
    "lat":   ["緯度", "latitude", "lat"],
    "lon":   ["経度", "longitude", "lon", "lng"],
    "speed": ["速度kmh", "速度ms", "speedkmh", "speedms", "速度", "スピード",
              "speed", "velocity"],
    "ax":    ["加速度x", "accx", "accelx", "accelerationx", "ax"],
    "ay":    ["加速度y", "accy", "accely", "accelerationy", "ay"],
    "az":    ["加速度z", "accz", "accelz", "accelerationz", "az"],
    "hr":    ["心拍数", "心拍", "heartrate", "hr", "bpm"],
}
RAW_EXCLUDE = {"speed": ["加速", "acc"]}

SUMMARY_ALIASES = {
    "player":   ["選手名", "選手", "氏名", "名前", "playername", "player", "name", "選手id"],
    "team":     ["チーム", "チーム名", "team", "クラブ"],
    "date":     ["日付", "試合日", "実施日", "date", "day"],
    "game":     ["試合", "ゲーム", "game", "match", "節", "対戦", "試合番号", "セッション"],
    "position": ["ポジション", "position", "pos", "役割"],
    "injury":   ["けが", "ケガ", "怪我", "負傷", "傷害", "injury"],
}

WEATHER_ALIASES = {
    "time":     ["日時", "観測日時", "時刻", "datetime", "timestamp", "日付", "date", "time"],
    "temp":     ["気温", "温度", "temperature", "temp"],
    "humidity": ["湿度", "humidity"],
    "wind":     ["風速", "windspeed", "wind"],
    "sky":      ["天気", "天候", "weather", "condition"],
}


# ════════════════════════════════════════════════════
# 列名の推定
# ════════════════════════════════════════════════════
def _norm(s) -> str:
    return re.sub(r"[\s\u3000_\-\(\)（）\[\]\{\}/%.,:：・]", "", str(s)).lower()


def guess_mapping(columns, aliases: dict, exclude: dict | None = None) -> dict:
    """列名から役割（time/lat/...）を推定する。見つからなければ None。"""
    exclude = exclude or {}
    columns = list(columns)
    norm = {c: _norm(c) for c in columns}
    res = {k: None for k in aliases}
    used: set = set()

    def ok(k, c):
        return c not in used and not any(_norm(x) in norm[c] for x in exclude.get(k, []))

    for k, al in aliases.items():                     # 完全一致
        for a in map(_norm, al):
            hit = next((c for c in columns if norm[c] == a and ok(k, c)), None)
            if hit is not None:
                res[k] = hit
                used.add(hit)
                break
    for k, al in aliases.items():                     # 部分一致
        if res[k] is not None:
            continue
        for a in map(_norm, al):
            if len(a) < 3 and a.isascii():
                continue
            hit = next((c for c in columns if a in norm[c] and ok(k, c)), None)
            if hit is not None:
                res[k] = hit
                used.add(hit)
                break
    return res


# ════════════════════════════════════════════════════
# ファイル読み込み
# ════════════════════════════════════════════════════
def read_head(file, nrows: int = 300):
    """先頭だけ読み、(DataFrame, 文字コード) を返す。"""
    name = file.name.lower()
    file.seek(0)
    if name.endswith(".csv"):
        err = None
        for enc in ENCODINGS:
            try:
                file.seek(0)
                head = pd.read_csv(file, nrows=nrows, encoding=enc)
                file.seek(0)
                return head, enc
            except (UnicodeError, pd.errors.ParserError) as e:
                err = e
        raise ValueError(f"文字コードを判定できません：{err}")
    head = pd.read_excel(file, nrows=nrows)
    file.seek(0)
    return head, None


def iter_chunks(file, enc, chunksize: int = CHUNK_ROWS):
    """CSVは分割して、xlsxは丸ごと読み込む。"""
    file.seek(0)
    if enc is not None:
        yield from pd.read_csv(file, encoding=enc, chunksize=chunksize)
    else:
        yield pd.read_excel(file)


def read_table(file, enc) -> pd.DataFrame:
    parts = list(iter_chunks(file, enc))
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def detect_kind(head: pd.DataFrame) -> str:
    """'raw'（GPS・加速度の生データ）か 'summary'（運動量サマリー）かを推定"""
    m = guess_mapping(head.columns, RAW_ALIASES, RAW_EXCLUDE)
    if m["lat"] and m["lon"]:
        return "raw"
    if m["time"] and m["speed"] and len(head) >= 100:
        t = pd.to_numeric(head[m["time"]], errors="coerce")
        if t.notna().mean() < 0.9:
            t = pd.to_datetime(head[m["time"]], errors="coerce")
        if t.notna().all() and t.is_monotonic_increasing and t.nunique() > 50:
            return "raw"
    return "summary"


# ════════════════════════════════════════════════════
# GPS・加速度・心拍（生データ）
# ════════════════════════════════════════════════════
@dataclass(frozen=True)
class RawParams:
    speed_unit: str = "km/h"        # 'km/h' or 'm/s'
    acc_unit: str = "g"             # 'g' or 'm/s²'
    max_speed_kmh: float = 45.0     # これ以上は異常値
    spike_ms: float = 6.0           # 1点だけ飛ぶ速度スパイク判定 (m/s)
    hr_min: int = 30
    hr_max: int = 230
    start_override: str = ""        # 経過秒データ用の開始日時


def _to_datetime(s: pd.Series, start_override: str = "") -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        med = num.median()
        if med > 1e11:
            return pd.to_datetime(num, unit="ms", errors="coerce")
        if med > 1e8:
            return pd.to_datetime(num, unit="s", errors="coerce")
        base = pd.Timestamp(start_override) if start_override else pd.Timestamp("2000-01-01")
        return base + pd.to_timedelta(num, unit="s")
    return pd.to_datetime(s, errors="coerce")


def _haversine(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _chunk_to_seconds(c: pd.DataFrame, m: dict, p: RawParams, rep: dict):
    t = _to_datetime(c[m["time"]], p.start_override)
    ok = t.notna().to_numpy()
    rep["time_invalid"] += int((~ok).sum())
    c = c.loc[ok].reset_index(drop=True)
    t = t[ok].reset_index(drop=True)
    if c.empty:
        return None
    d: dict[str, pd.Series] = {}

    if m.get("lat") and m.get("lon"):
        lat = pd.to_numeric(c[m["lat"]], errors="coerce")
        lon = pd.to_numeric(c[m["lon"]], errors="coerce")
        bad = ((lat.abs() > 90) | (lon.abs() > 180) | ((lat == 0) & (lon == 0))) & lat.notna()
        rep["gps_invalid"] += int(bad.sum())
        d["lat"], d["lon"] = lat.mask(bad), lon.mask(bad)

    if m.get("speed"):
        v = pd.to_numeric(c[m["speed"]], errors="coerce")
        if p.speed_unit == "m/s":
            v = v * 3.6
        bad = (v < -1) | (v > p.max_speed_kmh)      # 小さな負値はノイズとして0に丸める
        rep["speed_outlier"] += int(bad.sum())
        d["speed"] = v.mask(bad).clip(lower=0)

    if m.get("hr"):
        h = pd.to_numeric(c[m["hr"]], errors="coerce")
        bad = (h < p.hr_min) | (h > p.hr_max)
        rep["hr_outlier"] += int(bad.sum())
        d["hr"] = h.mask(bad)

    axes = [m.get(k) for k in ("ax", "ay", "az")]
    if all(axes):
        a = np.column_stack([pd.to_numeric(c[x], errors="coerce").to_numpy(dtype=float) for x in axes])
        if p.acc_unit == "m/s²":
            a = a / 9.80665
        diff = np.diff(a, axis=0, prepend=np.nan)
        d["pl"] = pd.Series(np.sqrt((diff ** 2).sum(axis=1)))

    g = pd.DataFrame(d)
    for k in ("lat", "lon", "speed", "hr"):
        if k in g:
            g[k + "_n"] = g[k].notna().astype("int32")
    g["_s"] = t.dt.floor("s")
    g["_rows"] = 1
    return g.groupby("_s").sum()


def load_raw(file, enc, mapping: dict, p: RawParams):
    """生データを1秒値に集約して返す。(DataFrame[index=time], 品質レポート)"""
    rep = dict(rows_in=0, time_invalid=0, gps_invalid=0, speed_outlier=0,
               hr_outlier=0, gps_jump=0, speed_spike=0, gap_seconds=0)
    parts = []
    for chunk in iter_chunks(file, enc):
        rep["rows_in"] += len(chunk)
        r = _chunk_to_seconds(chunk, mapping, p, rep)
        if r is not None:
            parts.append(r)
    if not parts:
        raise ValueError("有効な時刻データがありません（時刻列の指定を確認してください）")

    agg = pd.concat(parts).groupby(level=0).sum()
    out = pd.DataFrame(index=agg.index)
    for k in ("lat", "lon", "speed", "hr"):
        if k in agg:
            out[k] = agg[k] / agg[k + "_n"].where(agg[k + "_n"] > 0)
    if "pl" in agg:
        out["pl"] = agg["pl"]
    out.index.name = "time"

    if 1 < len(out) and (out.index[-1] - out.index[0]) < pd.Timedelta("36h"):
        out = out.reindex(pd.date_range(out.index[0], out.index[-1], freq="s"))
        out.index.name = "time"
        if "pl" in out:
            out["pl"] = out["pl"].fillna(0.0)

    # ── 位置のジャンプ除去（飛んで戻る1点）
    if "lat" in out:
        step = pd.Series(_haversine(out["lat"].shift(), out["lon"].shift(),
                                    out["lat"], out["lon"]), index=out.index)
        vmax = p.max_speed_kmh / 3.6
        jump = (step > vmax) & (step.shift(-1) > vmax)
        rep["gps_jump"] = int(jump.sum())
        out.loc[jump, ["lat", "lon"]] = np.nan
        out[["lat", "lon"]] = out[["lat", "lon"]].interpolate(limit=3, limit_area="inside")

    # ── 速度が無ければ位置から算出
    if "speed" not in out and "lat" in out:
        d = _haversine(out["lat"].shift(), out["lon"].shift(), out["lat"], out["lon"])
        sp = pd.Series(d, index=out.index) * 3.6
        bad = sp > p.max_speed_kmh
        rep["speed_outlier"] += int(bad.sum())
        out["speed"] = sp.mask(bad)
    if "speed" not in out:
        raise ValueError("速度列も緯度経度もありません。列の割り当てを確認してください")

    # ── 速度スパイク（1点だけ急上昇して戻る）
    v = out["speed"] / 3.6
    up, dn = v - v.shift(), v.shift(-1) - v
    spike = ((up > p.spike_ms) & (dn < -p.spike_ms)) | ((up < -p.spike_ms) & (dn > p.spike_ms))
    rep["speed_spike"] = int(spike.sum())
    out.loc[spike, "speed"] = np.nan
    out["speed"] = out["speed"].interpolate(limit=3, limit_area="inside")

    out["speed_ms"] = out["speed"] / 3.6
    out["dist_m"] = out["speed_ms"].fillna(0.0)
    out["acc"] = out["speed_ms"].diff()
    rep["gap_seconds"] = int(out["speed"].isna().sum())
    rep["seconds"] = int(len(out))
    return out, rep


# ════════════════════════════════════════════════════
# セッション整形・指標
# ════════════════════════════════════════════════════
def prepare_session(df: pd.DataFrame, start=None, end=None,
                    half_min: float = 45.0, break_min: float = 0.0,
                    seg_min: int = 15) -> pd.DataFrame:
    """開始・終了で切り出し、経過分・前後半・区間（15分）を付与"""
    d = df
    if start is not None and not pd.isna(start):
        d = d[d.index >= start]
    if end is not None and not pd.isna(end):
        d = d[d.index <= end]
    d = d.copy()
    if d.empty:
        return d
    t0 = start if (start is not None and not pd.isna(start)) else d.index[0]
    e = np.asarray((d.index - t0).total_seconds(), dtype=float) / 60.0
    in_break = (e >= half_min) & (e < half_min + break_min)
    d, e = d[~in_break].copy(), e[~in_break]
    e = np.where(e >= half_min + break_min, e - break_min, e)
    d["elapsed_min"] = e
    d["half"] = np.where(e < half_min, "前半", "後半")
    d["seg"] = (np.floor(e / seg_min) * seg_min).astype(int)
    d = d[e >= 0]
    return d


def _events(mask: pd.Series) -> int:
    m = mask.fillna(False).astype(bool)
    return int((m & ~m.shift(1, fill_value=False)).sum())


def compute_metrics(d: pd.DataFrame, hsr_kmh=19.8, sprint_kmh=25.2,
                    acc_th=2.0, hr_ref=None) -> dict:
    if d.empty:
        return {}
    sp = d["speed"]
    sec = int(sp.notna().sum())
    mins = sec / 60.0
    dist = float(d["dist_m"].sum())
    hsr = float(d.loc[sp >= hsr_kmh, "dist_m"].sum())
    spr = float(d.loc[sp >= sprint_kmh, "dist_m"].sum())
    res = {
        "時間(分)": round(mins, 1),
        "総走行距離(m)": round(dist, 1),
        "距離/分(m/min)": round(dist / mins, 1) if mins > 0 else np.nan,
        "HSR距離(m)": round(hsr, 1),
        "HSR率(%)": round(hsr / dist * 100, 1) if dist > 0 else np.nan,
        "スプリント距離(m)": round(spr, 1),
        "スプリント回数": _events(sp >= sprint_kmh),
        "最高速度(km/h)": round(float(sp.max()), 1) if sec else np.nan,
        "加速回数": _events(d["acc"] >= acc_th),
        "減速回数": _events(d["acc"] <= -acc_th),
    }
    if "pl" in d and d["pl"].notna().any():
        pl = float(d["pl"].sum())
        res["PlayerLoad"] = round(pl, 1)
        res["PlayerLoad/分"] = round(pl / mins, 2) if mins > 0 else np.nan
    if "hr" in d and d["hr"].notna().any():
        res["平均心拍(bpm)"] = round(float(d["hr"].mean()), 1)
        res["最大心拍(bpm)"] = round(float(d["hr"].max()), 0)
        if hr_ref:
            res["高強度心拍時間(分)"] = round(float((d["hr"] >= 0.85 * hr_ref).sum()) / 60, 1)
    if WX_TEMP in d and d[WX_TEMP].notna().any():
        res[WX_TEMP] = round(float(d[WX_TEMP].mean()), 1)
    return res


def segment_table(d: pd.DataFrame, key: str, **kw) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(key, sort=True):
        r = compute_metrics(g, **kw)
        if r:
            rows.append({key: k, **r})
    return pd.DataFrame(rows)


def seg_label(s: int, seg_min: int = 15) -> str:
    return f"{int(s)}-{int(s) + seg_min}分"


# ════════════════════════════════════════════════════
# ピッチ座標・スプリント
# ════════════════════════════════════════════════════
def _local_xy(lat, lon, lat0, lon0):
    x = (np.asarray(lon, float) - lon0) * np.cos(np.radians(lat0)) * 111_320.0
    y = (np.asarray(lat, float) - lat0) * 110_540.0
    return x, y


def parse_latlon(text):
    """「34.650123, 135.601234」のような文字列から (緯度, 経度) を取り出す。Googleマップからのコピーをそのまま貼れる。
    読み取れないときは None。経度・緯度の順に貼られた場合は自動で入れ替える。"""
    t = str(text or "")
    if "°" in t or "′" in t or "″" in t:
        return None                              # 度分秒表記は未対応（10進数で入力してもらう）
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    if len(nums) < 2:
        return None
    a, b = float(nums[0]), float(nums[1])
    if abs(a) > 90 and abs(b) <= 90:
        a, b = b, a
    if abs(a) > 90 or abs(b) > 180:
        return None
    return a, b


def fit_pitch_corners(corners, L: float = 105.0, W: float = 68.0) -> dict | None:
    """ピッチ4隅の緯度経度から変換を作る。corners は [(lat, lon)] ×4 を
    ①左下 → ②右下 → ③右上 → ④左上 の順（①→②が長辺＝ピッチ長）で渡す。
    4点への最小二乗あてはめなので、ピッチが北から回転していても、多少の測定誤差があっても使える。"""
    c = np.asarray(corners, float)
    if c.shape != (4, 2) or np.isnan(c).any():
        return None
    lat0, lon0 = float(c[:, 0].mean()), float(c[:, 1].mean())
    X, Y = _local_xy(c[:, 0], c[:, 1], lat0, lon0)
    src = np.column_stack([X, Y, np.ones(4)])
    dst = np.array([[0, 0], [L, 0], [L, W], [0, W]], float)
    M, *_ = np.linalg.lstsq(src, dst, rcond=None)
    if abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) < 1e-9:
        return None
    rms = float(np.sqrt(((src @ M - dst) ** 2).sum(axis=1).mean()))
    return dict(lat0=lat0, lon0=lon0, swap=False, affine=M.tolist(), rms=rms,
                xmin=0.0, xmax=float(L), ymin=0.0, ymax=float(W))


def fit_pitch(frames: list, swap: bool = False, manual: dict | None = None) -> dict | None:
    """複数セッションのGPS範囲から、ピッチ(0..L,0..W)への変換パラメータを作る"""
    pos = pd.concat([f[["lat", "lon"]] for f in frames if "lat" in f], ignore_index=True).dropna()
    if pos.empty:
        return None
    lat0, lon0 = float(pos["lat"].median()), float(pos["lon"].median())
    X, Y = _local_xy(pos["lat"], pos["lon"], lat0, lon0)
    if swap:
        X, Y = Y, X
    if manual:
        cx, cy = _local_xy([manual["lat_min"], manual["lat_max"]],
                           [manual["lon_min"], manual["lon_max"]], lat0, lon0)
        if swap:
            cx, cy = cy, cx
        xmin, xmax, ymin, ymax = cx.min(), cx.max(), cy.min(), cy.max()
    else:
        xmin, xmax = np.quantile(X, [0.005, 0.995])
        ymin, ymax = np.quantile(Y, [0.005, 0.995])
    if xmax - xmin < 1e-6 or ymax - ymin < 1e-6:
        return None
    return dict(lat0=lat0, lon0=lon0, swap=swap, xmin=float(xmin), xmax=float(xmax),
                ymin=float(ymin), ymax=float(ymax))


def apply_pitch(d: pd.DataFrame, fp: dict, L=105.0, W=68.0,
                flip_x=False, flip_y=False,
                attack_right_first=True, flip_second=True,
                clip: bool = True) -> pd.DataFrame:
    """x,y（攻撃方向が常に右になるよう揃えた座標）を付与。
    clip=False のときはピッチの外に出た点もそのまま残す（4隅の設定が合っているかの確認用）"""
    d = d.copy()
    X, Y = _local_xy(d["lat"], d["lon"], fp["lat0"], fp["lon0"])
    if "affine" in fp:                      # ピッチ4隅から作った変換（回転したピッチにも対応）
        M = np.asarray(fp["affine"], float)
        P = np.column_stack([X, Y, np.ones(len(X))]) @ M
        x, y = P[:, 0], P[:, 1]
    else:
        if fp["swap"]:
            X, Y = Y, X
        x = (X - fp["xmin"]) / (fp["xmax"] - fp["xmin"]) * L
        y = (Y - fp["ymin"]) / (fp["ymax"] - fp["ymin"]) * W
    if clip:
        x, y = np.clip(x, 0, L), np.clip(y, 0, W)
    if flip_x:
        x = L - x
    if flip_y:
        y = W - y
    first_sign = 1 if attack_right_first else -1
    sign = np.where(d["half"].to_numpy() == "前半", first_sign,
                    -first_sign if flip_second else first_sign)
    d["x"] = np.where(sign > 0, x, L - x)
    d["y"] = np.where(sign > 0, y, W - y)
    return d


def sprint_events(d: pd.DataFrame, sprint_kmh: float, L: float = 105.0,
                  min_dx: float = 2.0) -> pd.DataFrame:
    """スプリント区間ごとの開始/終了位置・向き（x は攻撃方向が右）"""
    cols = ["t0", "x0", "y0", "x1", "y1", "half", "dist", "vmax", "向き", "攻撃サード"]
    if "x" not in d or d.empty:
        return pd.DataFrame(columns=cols)
    dd = d.reset_index()
    tcol = dd.columns[0]
    m = ((dd["speed"] >= sprint_kmh) & dd["x"].notna()).to_numpy()
    if not m.any():
        return pd.DataFrame(columns=cols)
    chg = np.empty(len(m), dtype=bool)
    chg[0] = m[0]
    chg[1:] = m[1:] != m[:-1]
    grp = np.cumsum(chg)
    d2 = dd[m].assign(_g=grp[m])
    rows = []
    for _, g in d2.groupby("_g"):
        dx = g["x"].iloc[-1] - g["x"].iloc[0]
        way = "攻撃方向" if dx >= min_dx else ("守備方向" if dx <= -min_dx else "横・その他")
        rows.append(dict(t0=g[tcol].iloc[0], x0=g["x"].iloc[0], y0=g["y"].iloc[0],
                         x1=g["x"].iloc[-1], y1=g["y"].iloc[-1], half=g["half"].iloc[0],
                         dist=g["dist_m"].sum(), vmax=g["speed"].max(), 向き=way,
                         攻撃サード=bool(g["x"].iloc[0] >= L * 2 / 3)))
    return pd.DataFrame(rows, columns=cols)


# ════════════════════════════════════════════════════
# 気象データ
# ════════════════════════════════════════════════════
def prepare_weather(df: pd.DataFrame, m: dict):
    if not m.get("time"):
        raise ValueError("気象データの日時列を指定してください")
    w = pd.DataFrame({"time": pd.to_datetime(df[m["time"]], errors="coerce")})
    rep = {}

    def put(key, name, lo, hi):
        if m.get(key):
            v = pd.to_numeric(df[m[key]], errors="coerce")
            bad = (v < lo) | (v > hi)
            rep[name] = int(bad.sum())
            w[name] = v.mask(bad)
        else:
            w[name] = np.nan

    put("temp", WX_TEMP, -30, 50)
    put("humidity", WX_HUM, 0, 100)
    put("wind", WX_WIND, 0, 60)
    w[WX_SKY] = df[m["sky"]].astype(str) if m.get("sky") else np.nan
    rep["日時が不正な行"] = int(w["time"].isna().sum())
    w = w.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return w, rep


def attach_weather(d: pd.DataFrame, w: pd.DataFrame | None, tol_min: int = 90) -> pd.DataFrame:
    """秒データ d に、最も近い時刻の気象を付与"""
    if w is None or w.empty or d.empty:
        return d
    d = d.copy()
    wcols = [WX_TEMP, WX_HUM, WX_WIND, WX_SKY]
    wt = w["time"]
    date_only = bool((wt.dt.hour.eq(0) & wt.dt.minute.eq(0)).all())
    left = pd.DataFrame({"time": d.index.astype("datetime64[ns]")})
    if date_only:
        wd = weather_daily(w).reset_index().rename(columns={"_date": "_k"})
        left["_k"] = left["time"].dt.normalize()
        merged = left.merge(wd, on="_k", how="left")
    else:
        ww = w.copy()
        ww["time"] = ww["time"].astype("datetime64[ns]")
        merged = pd.merge_asof(left, ww, on="time", direction="nearest",
                               tolerance=pd.Timedelta(minutes=tol_min))
    for c in wcols:
        d[c] = merged[c].to_numpy() if c in merged else np.nan
    return d


def weather_daily(w: pd.DataFrame) -> pd.DataFrame:
    """日別の気象（9〜18時の平均を優先、無ければ1日平均）。index='_date'"""
    x = w.copy()
    x["_date"] = x["time"].dt.normalize()

    def mode(s):
        s = s.dropna()
        s = s[s != "nan"]
        return s.mode().iloc[0] if len(s) else np.nan

    agg = {WX_TEMP: "mean", WX_HUM: "mean", WX_WIND: "mean", WX_SKY: mode}
    g_all = x.groupby("_date").agg(agg)
    day = x[x["time"].dt.hour.between(9, 18)]
    if day.empty:
        return g_all
    return day.groupby("_date").agg(agg).combine_first(g_all)


# ════════════════════════════════════════════════════
# 運動量サマリー
# ════════════════════════════════════════════════════
def _clean_str(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), None)
    out = out.map(lambda v: None if v is None else re.sub(r"[\s\u3000]+", " ", str(v)).strip())
    return out.map(lambda v: None if v in (None, "", "nan", "None", "NaN") else v)


def injury_flag(s: pd.Series) -> pd.Series:
    def f(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return False
        if isinstance(v, (int, float, np.number)):
            return bool(v > 0)
        return str(v).strip().lower() not in (
            "", "0", "なし", "無", "無し", "no", "n", "false", "-", "×", "x", "nan", "none")
    return s.map(f).astype(bool)


STD_SUMMARY = {"player": "選手", "team": "チーム", "date": "日付",
               "game": "試合", "position": "ポジション", "injury": "ケガ"}


def prepare_summary(df: pd.DataFrame, m: dict, sigma: float = 3.0):
    """標準列（選手/チーム/日付/試合/ポジション/ケガ）を作り、数値列を指標として抽出し外れ値(±σ)を除外"""
    d = df.copy()
    used = [m[k] for k in STD_SUMMARY if m.get(k)]
    for k, new in STD_SUMMARY.items():
        col = m.get(k)
        if col:
            d[new] = d[col] if col != new else d[new]
        else:
            if new in d.columns:
                d = d.rename(columns={new: new + "_元"})
            d[new] = np.nan
    d["選手"] = _clean_str(d["選手"])
    d = d[d["選手"].notna()].reset_index(drop=True)
    for c in ("チーム", "試合", "ポジション"):
        d[c] = _clean_str(d[c])
    d["日付"] = pd.to_datetime(d["日付"], errors="coerce") if m.get("date") else pd.NaT

    std_names = set(STD_SUMMARY.values())
    metrics = []
    for c in d.columns:
        if c in std_names or c in used or str(c).endswith("_元") or c == "ソースファイル":
            continue
        conv = pd.to_numeric(d[c], errors="coerce")
        nn = d[c].notna().sum()
        if nn > 0 and conv.notna().sum() >= 0.8 * nn and conv.nunique() > 1:
            d[c] = conv
            metrics.append(c)

    rep = []
    if sigma:
        for c in metrics:
            x = d[c]
            sd = x.std()
            if not sd or np.isnan(sd) or sd == 0:
                continue
            z = (x - x.mean()) / sd
            bad = z.abs() > sigma
            for i in d.index[bad]:
                rep.append({"指標": c, "選手": d.at[i, "選手"],
                            "日付": d.at[i, "日付"], "値": x[i], "Zスコア": round(float(z[i]), 2)})
            d.loc[bad, c] = np.nan
    d["ケガ有無"] = injury_flag(d["ケガ"]) if m.get("injury") else False
    return d, metrics, pd.DataFrame(rep)


def attach_weather_daily(d: pd.DataFrame, w: pd.DataFrame | None) -> pd.DataFrame:
    if w is None or w.empty or d["日付"].isna().all():
        return d
    wd = weather_daily(w)
    key = d["日付"].dt.normalize()
    d = d.copy()
    for c in (WX_TEMP, WX_HUM, WX_WIND, WX_SKY):
        d[c] = key.map(wd[c]) if c in wd else np.nan
    return d


def calc_acwr(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    """選手ごとの急性:慢性負荷比（直近7日 ÷ 直近28日の週平均）"""
    out = []
    x = d.dropna(subset=["日付", metric])
    for p, g in x.groupby("選手"):
        s = g.groupby("日付")[metric].sum().sort_index()
        if len(s) < 2:
            continue
        s = s.resample("D").sum()
        acute = s.rolling(7, min_periods=1).sum()
        chronic = s.rolling(28, min_periods=28).sum() / 4
        acwr = acute / chronic.where(chronic > 0)
        out.append(pd.DataFrame({"選手": p, "日付": s.index, "日負荷": s.values,
                                 "急性(7日)": acute.values, "慢性(週平均)": chronic.values,
                                 "ACWR": acwr.values}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def acwr_flag(v) -> str:
    if v is None or pd.isna(v):
        return "判定不可"
    if v > 1.5:
        return "過負荷リスク"
    if v > 1.3:
        return "注意"
    if v < 0.8:
        return "負荷不足"
    return "適正"


# ════════════════════════════════════════════════════
# 軌跡アニメーション用：複数セッションを共通の時間軸（経過秒）にそろえる
# ════════════════════════════════════════════════════
def anim_tracks(frames: list, start_s: float, end_s: float, step_s: int = 5,
                max_frames: int = 600) -> dict:
    """apply_pitch 済みの各セッションから、一定間隔の位置(x,y)・速度を取り出す。
    経過分（elapsed_min）で揃えるので、同じ試合の複数選手を同時に再生できる。"""
    dur = max(float(end_s) - float(start_s), 1.0)
    step = max(int(step_s), int(np.ceil(dur / max_frames)))
    grid = np.arange(int(start_s), int(end_s) + 1, step)
    tol = max(step, 2)
    X, Y, V = [], [], []
    for f in frames:
        t = np.round(f["elapsed_min"].to_numpy(dtype=float) * 60.0)
        v = f["speed"].to_numpy(dtype=float) if "speed" in f else np.full(len(f), np.nan)
        g = pd.DataFrame({"x": f["x"].to_numpy(dtype=float),
                          "y": f["y"].to_numpy(dtype=float), "v": v}, index=t)
        g = g[~g.index.duplicated(keep="first")].sort_index()
        r = g.reindex(grid, method="nearest", tolerance=tol)
        X.append(r["x"].to_numpy())
        Y.append(r["y"].to_numpy())
        V.append(r["v"].to_numpy())
    return dict(grid=grid, step=step, X=X, Y=Y, V=V)
