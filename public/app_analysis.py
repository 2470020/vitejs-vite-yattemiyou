"""
app_analysis.py : スポーツデータ分析システム（データ分析専用版）

元の app.py は変更せず、新規に作成した別システムです。
  ・GPS／3軸加速度／心拍の生データ ＋ 運動量サマリー を 2〜10 ファイルまでアップロード
  ・気象データは別アップロードし、付随情報（気温・湿度・天気）としてデータに追加
  ・コーチ向けレポート／AIチャット／トレーニング提案／カレンダー連携は削除済み
起動： streamlit run app_analysis.py
"""
import json
from pathlib import Path
from dataclasses import astuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import sa_core as sc
from sa_core import WX_TEMP, WX_HUM, WX_WIND, WX_SKY

# ── ページ設定 ──────────────────────────────────────
st.set_page_config(page_title="SPORTS DATA ANALYSIS SYSTEM", layout="wide",
                   initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def _toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


DARK = {"bg": "#1a2028", "sidebar_bg": "#20262f", "card_bg": "#232b38", "card_bg_alt": "#2a3341",
        "border": "#2e3847", "border_strong": "#3d4d63", "text_primary": "#e8edf4",
        "text_secondary": "#8ea3bd", "text_muted": "#5f7692", "text_faint": "#465b73",
        "accent": "#5eb0ff", "accent_border": "#2a7fd4", "accent_soft": "rgba(94,176,255,0.13)",
        "white": "#ffffff", "gold": "#ffd700", "warn": "#ff9a4d", "danger": "#ff6b6b",
        "ok": "#7bd88f", "shadow": "0 2px 12px rgba(0,0,0,0.35)"}
LIGHT = {"bg": "#f2f4f8", "sidebar_bg": "#ffffff", "card_bg": "#ffffff", "card_bg_alt": "#eef1f6",
         "border": "#dce3ec", "border_strong": "#b9c6d6", "text_primary": "#1c2733",
         "text_secondary": "#51637a", "text_muted": "#8296ac", "text_faint": "#a8b8c9",
         "accent": "#1a6fc4", "accent_border": "#1a6fc4", "accent_soft": "rgba(26,111,196,0.13)",
         "white": "#0c2a4a", "gold": "#8a6d00", "warn": "#a85a10", "danger": "#b23a3a",
         "ok": "#2e8b57", "shadow": "0 2px 10px rgba(20,30,50,0.12)"}
T = DARK if st.session_state.theme == "dark" else LIGHT
PALETTE = [T["accent"], T["warn"], T["ok"], T["gold"], "#c792ea", T["danger"], "#4dd0e1", "#f78fb3",
           "#a3be8c", "#d08770"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Noto+Sans+JP:wght@400;700&display=swap');
.stApp {{ background-color: {T['bg']}; color: {T['text_primary']}; }}
[data-testid="stSidebar"] {{ background-color: {T['sidebar_bg']}; border-right: 1px solid {T['border']}; }}
.stButton > button, .stDownloadButton > button {{
    background: transparent; border: 1px solid {T['accent_border']}; color: {T['accent']};
    font-family: 'Rajdhani', sans-serif; letter-spacing: 0.08em; transition: all 0.2s; }}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: {T['accent_soft']}; border-color: {T['accent']}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{
    background-color: {T['card_bg']}; border: 1px solid {T['border_strong']}; border-radius: 0;
    font-family: 'Rajdhani','Noto Sans JP',sans-serif; letter-spacing: 0.08em;
    color: {T['text_secondary']}; padding: 8px 18px; }}
.stTabs [aria-selected="true"] {{
    background-color: {T['accent_soft']} !important; border-color: {T['accent']} !important;
    color: {T['accent']} !important; }}
hr {{ border-color: {T['border']}; }}
.section-header {{
    font-family: 'Rajdhani','Noto Sans JP',sans-serif; font-size: 13px; font-weight: 700;
    color: {T['accent']}; letter-spacing: 0.2em; text-transform: uppercase;
    border-bottom: 1px solid {T['border_strong']}; padding-bottom: 4px; margin: 20px 0 12px; }}
.hero-title {{ font-family: 'Rajdhani', sans-serif; font-size: 42px; font-weight: 700;
    letter-spacing: 0.15em; color: {T['white']}; line-height: 1.1; }}
.hero-sub {{ font-family: 'Rajdhani', sans-serif; font-size: 13px; letter-spacing: 0.3em;
    color: {T['accent']}; margin-bottom: 32px; }}
.stat-box {{ border: 1px solid {T['border_strong']}; background: {T['card_bg']};
    padding: 14px 18px; margin-bottom: 8px; }}
.stat-box-label {{ font-size: 10px; letter-spacing: 0.2em; color: {T['accent']};
    font-family: 'Rajdhani','Noto Sans JP',sans-serif; text-transform: uppercase; }}
.stat-box-value {{ font-size: 26px; font-weight: 700; font-family: 'Rajdhani', sans-serif;
    color: {T['white']}; }}
.null-warning {{ background: {T['card_bg_alt']}; border-left: 3px solid {T['accent_border']};
    padding: 8px 14px; font-family: 'Noto Sans JP', sans-serif; font-size: 12px;
    color: {T['text_secondary']}; margin-bottom: 12px; line-height: 1.8; }}
.progress-info {{ font-family: 'Rajdhani', sans-serif; font-size: 12px; color: {T['accent']};
    letter-spacing: 0.1em; margin-bottom: 8px; }}
.badge-ok, .badge-warn, .badge-ng {{ display:inline-block; font-family:'Rajdhani',sans-serif;
    font-size: 13px; font-weight: 700; padding: 2px 10px; border: 1px solid; letter-spacing: 0.1em; }}
.badge-ok {{ color: {T['ok']}; border-color: {T['ok']}; }}
.badge-warn {{ color: {T['warn']}; border-color: {T['warn']}; }}
.badge-ng {{ color: {T['danger']}; border-color: {T['danger']}; }}
#theme-toggle-anchor + div[data-testid="stButton"] {{ position: fixed; bottom: 24px; right: 24px; z-index: 9999; }}
#theme-toggle-anchor + div[data-testid="stButton"] button {{
    border-radius: 20px; padding: 8px 20px; box-shadow: {T['shadow']}; background: {T['card_bg']}; }}
[data-baseweb="select"] > div, [data-baseweb="input"] {{
    background-color: {T['card_bg']} !important; border-color: {T['border_strong']} !important; }}
[data-baseweb="select"] > div *, [data-baseweb="input"] input {{ color: {T['text_primary']} !important; }}
[data-baseweb="tag"] {{ background-color: {T['accent_border']} !important; }}
[data-baseweb="tag"] span {{ color: {T['white']} !important; }}
div[role="listbox"], ul[data-testid="stSelectboxVirtualDropdown"] {{ background-color: {T['card_bg']} !important; }}
li[role="option"] {{ background-color: {T['card_bg']} !important; color: {T['text_primary']} !important; }}
li[role="option"]:hover, li[aria-selected="true"] {{ background-color: {T['accent_soft']} !important; }}
.stSelectbox label, .stMultiSelect label, .stTextInput label, .stNumberInput label,
.stSlider label, .stRadio label, .stFileUploader label, .stCheckbox label {{ color: {T['text_secondary']} !important; }}
.stRadio div[role="radiogroup"] label span {{ color: {T['text_primary']} !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div id="theme-toggle-anchor"></div>', unsafe_allow_html=True)
st.button("LIGHT MODE" if st.session_state.theme == "dark" else "DARK MODE",
          key="theme_toggle_btn", on_click=_toggle_theme)

# ── 共通ヘルパー ────────────────────────────────────
try:
    _NEW_ST = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)
except Exception:
    _NEW_ST = False
_chart_n = [0]


def show(fig, h=380, margin=None):
    """Plotlyをテーマに合わせて表示"""
    fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg"], height=h,
                      font=dict(color=T["text_primary"], family="Noto Sans JP, Rajdhani, sans-serif"),
                      margin=margin or dict(t=44, b=34, l=50, r=24),
                      legend=dict(bgcolor=T["card_bg"], bordercolor=T["border"], borderwidth=1))
    fig.update_xaxes(gridcolor=T["border"], zerolinecolor=T["border"])
    fig.update_yaxes(gridcolor=T["border"], zerolinecolor=T["border"])
    _chart_n[0] += 1
    if _NEW_ST:
        st.plotly_chart(fig, width="stretch", key=f"pc{_chart_n[0]}")
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"pc{_chart_n[0]}")


def table(df: pd.DataFrame, **kw):
    if _NEW_ST:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def editor(df: pd.DataFrame, **kw):
    if _NEW_ST:
        return st.data_editor(df, width="stretch", hide_index=True, **kw)
    return st.data_editor(df, use_container_width=True, hide_index=True, **kw)


def sec(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="null-warning">{text}</div>', unsafe_allow_html=True)


def kpi(col, label, value):
    col.markdown(f'<div class="stat-box"><div class="stat-box-label">{label}</div>'
                 f'<div class="stat-box-value">{value}</div></div>', unsafe_allow_html=True)


def colsel(label, cols, default, key, optional=True):
    opts = (["(なし)"] if optional else []) + list(cols)
    idx = opts.index(default) if default in opts else 0
    v = st.selectbox(label, opts, index=idx, key=key)
    return None if v == "(なし)" else v


def fmt_size(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def pick(cols, keywords, fallback=0):
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return cols[min(fallback, len(cols) - 1)] if cols else None


def pitch_shapes(L, W, color):
    ln = dict(color=color, width=1.4)
    s = [dict(type="rect", x0=0, y0=0, x1=L, y1=W, line=ln),
         dict(type="line", x0=L / 2, y0=0, x1=L / 2, y1=W, line=ln),
         dict(type="circle", x0=L / 2 - 9.15, y0=W / 2 - 9.15, x1=L / 2 + 9.15, y1=W / 2 + 9.15, line=ln)]
    for x0, x1 in ((0, 16.5), (L - 16.5, L)):
        s.append(dict(type="rect", x0=x0, y0=(W - 40.32) / 2, x1=x1, y1=(W + 40.32) / 2, line=ln))
    for x0, x1 in ((0, 5.5), (L - 5.5, L)):
        s.append(dict(type="rect", x0=x0, y0=(W - 18.32) / 2, x1=x1, y1=(W + 18.32) / 2, line=ln))
    return s


def build_anim_fig(trk, names, colors, L, W, theme, sprint_kmh, trail_s, fps):
    """ピッチ上の選手位置を時系列で再生するPlotlyアニメーション（再生/一時停止＋スライダー）"""
    grid, step = trk["grid"], int(trk["step"])
    n = len(grid)
    n_trail = int(trail_s // step)

    def mmss(sec):
        sec = int(sec)
        return f"{sec // 60:02d}:{sec % 60:02d}"

    def traces(k):
        out = []
        lo = max(0, k - n_trail)
        for i, nm in enumerate(names):
            out.append(go.Scatter(x=trk["X"][i][lo:k + 1], y=trk["Y"][i][lo:k + 1], mode="lines",
                                  line=dict(color=colors[i], width=2), opacity=0.55,
                                  hoverinfo="skip", showlegend=False))
            v = trk["V"][i][k]
            sp = bool(np.isfinite(v) and v >= sprint_kmh)
            out.append(go.Scatter(
                x=[trk["X"][i][k]], y=[trk["Y"][i][k]], mode="markers+text",
                text=[nm], textposition="top center",
                textfont=dict(color=theme["text_primary"], size=11),
                marker=dict(size=17 if sp else 12, color=colors[i],
                            line=dict(color="#ff3b3b" if sp else theme["text_primary"], width=3 if sp else 1)),
                hovertemplate=nm + "<br>%{x:.0f} m, %{y:.0f} m<extra></extra>", showlegend=False))
        return out

    frames = [go.Frame(data=traces(k), name=str(k),
                       layout=go.Layout(title_text=f"経過 {mmss(grid[k])}")) for k in range(n)]
    fig = go.Figure(data=traces(0), frames=frames)
    tick = max(1, n // 10)
    ani = dict(mode="immediate", transition=dict(duration=0))
    fig.update_layout(
        title=f"経過 {mmss(grid[0])}",
        shapes=pitch_shapes(L, W, theme["text_muted"]),
        xaxis=dict(range=[-3, L + 3], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-3, W + 3], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top",
            showactive=False, bgcolor=theme["card_bg"], bordercolor=theme["border"],
            font=dict(color=theme["text_primary"]),
            buttons=[
                dict(label="▶ 再生", method="animate",
                     args=[None, dict(frame=dict(duration=int(1000 / fps), redraw=True),
                                      fromcurrent=True, **ani)]),
                dict(label="⏸ 一時停止", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), **ani)])])],
        sliders=[dict(
            active=0, x=0.18, len=0.80, y=-0.02, yanchor="top", pad=dict(t=8),
            currentvalue=dict(visible=False), font=dict(color=theme["text_muted"]),
            steps=[dict(method="animate", label=(mmss(grid[k]) if k % tick == 0 else ""),
                        args=[[str(k)], dict(frame=dict(duration=0, redraw=True), **ani)])
                   for k in range(n)])])
    return fig


# ════════════════════════════════════════════════════
# サイドバー
# ════════════════════════════════════════════════════
with st.sidebar:
    sec("FIELD SETUP — STEP 1")
    with st.expander("フィールド（ピッチ4隅のGPS）", expanded=True):
        st.caption("データを入れる前に、ピッチの大きさと4隅のGPSを設定します。"
                   "Googleマップの地点を右クリックして出る「34.650123, 135.601234」を、そのまま貼り付けられます。"
                   "①左下 → ②右下 → ③右上 → ④左上の順で、①→②が長辺（ピッチ長）になるようにしてください。")
        for _k, _v in dict(field_L=105.0, field_W=68.0, fc_0="", fc_1="", fc_2="", fc_3="").items():
            st.session_state.setdefault(_k, _v)
        _fj = st.file_uploader("保存したフィールド設定（JSON）を読み込む", type=["json"], key="field_json")
        if _fj is not None and st.session_state.get("_field_sig") != (_fj.name, _fj.size):
            try:
                _cfg = json.loads(_fj.getvalue().decode("utf-8"))
                st.session_state["field_L"] = float(min(max(float(_cfg["length"]), 50.0), 130.0))
                st.session_state["field_W"] = float(min(max(float(_cfg["width"]), 30.0), 100.0))
                for _i, _c in enumerate(list(_cfg["corners"])[:4]):
                    st.session_state[f"fc_{_i}"] = f"{float(_c[0])}, {float(_c[1])}"
                st.session_state["_field_sig"] = (_fj.name, _fj.size)
            except Exception:
                st.error("フィールド設定のファイルを読み込めませんでした。")
        _f1, _f2 = st.columns(2)
        field_L = float(_f1.number_input("ピッチ長 (m)", min_value=50.0, max_value=130.0, step=1.0, key="field_L"))
        field_W = float(_f2.number_input("ピッチ幅 (m)", min_value=30.0, max_value=100.0, step=1.0, key="field_W"))
        _txt = [st.text_input(nm, key=f"fc_{i}", placeholder="34.650123, 135.601234")
                for i, nm in enumerate(["① 左下の角", "② 右下の角", "③ 右上の角", "④ 左上の角"])]
        _pts = [sc.parse_latlon(t) for t in _txt]
        _bad = [i + 1 for i, (t, p_) in enumerate(zip(_txt, _pts)) if t.strip() and p_ is None]
        if _bad:
            st.warning("読み取れない入力があります（" + "・".join("①②③④"[i - 1] for i in _bad) +
                       "）。「緯度, 経度」を10進数で入力してください。")
        _have = [p_ for p_ in _pts if p_ is not None]
        field_fp = None
        if len(_have) == 4:
            field_fp = sc.fit_pitch_corners(_pts, field_L, field_W)
        if field_fp is not None:
            _e12 = float(sc._haversine(*_pts[0], *_pts[1]))
            _e23 = float(sc._haversine(*_pts[1], *_pts[2]))
            st.markdown('<span class="badge-ok">FIELD OK</span>', unsafe_allow_html=True)
            st.caption(f"測った長さ：①→② 約 {_e12:.1f} m（ピッチ長 {field_L:.0f} m）／"
                       f"②→③ 約 {_e23:.1f} m（ピッチ幅 {field_W:.0f} m）／あてはめ誤差 約 {field_fp['rms']:.2f} m")
            if abs(_e12 - field_L) > 0.15 * field_L or abs(_e23 - field_W) > 0.15 * field_W:
                st.warning("測った長さがピッチの大きさと大きく違います。入力の順番（①→②が長辺）か、ピッチ長・幅を確認してください。")
        elif _have:
            st.markdown('<span class="badge-ng">FIELD: {}/4</span>'.format(len(_have)), unsafe_allow_html=True)
        if _have:
            _la0 = float(np.mean([p_[0] for p_ in _have]))
            _lo0 = float(np.mean([p_[1] for p_ in _have]))
            _px, _py = sc._local_xy([p_[0] for p_ in _have], [p_[1] for p_ in _have], _la0, _lo0)
            _xs, _ys = list(_px), list(_py)
            _lab = list("①②③④"[:len(_have)])
            if len(_have) == 4:
                _xs, _ys, _lab = _xs + [_xs[0]], _ys + [_ys[0]], _lab + [""]
            _pf = go.Figure(go.Scatter(x=_xs, y=_ys, mode="lines+markers+text", text=_lab, textposition="top center",
                                       line=dict(color=T["accent"], width=2), fill="toself" if len(_have) == 4 else None,
                                       fillcolor=T["accent_soft"], showlegend=False,
                                       hovertemplate="東へ %{x:.1f} m, 北へ %{y:.1f} m<extra></extra>"))
            _pf.update_layout(title="入力した4隅（上が北・単位 m）",
                              xaxis=dict(visible=False),
                              yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
            show(_pf, 260, margin=dict(t=40, b=10, l=10, r=10))
        if field_fp is not None:
            st.download_button("このフィールド設定をJSONで保存",
                               data=json.dumps(dict(length=field_L, width=field_W, corners=[list(p_) for p_ in _pts]),
                                               ensure_ascii=False, indent=1).encode("utf-8"),
                               file_name="field_setup.json", mime="application/json", key="field_dl")

    sec("DATA INPUT")
    st.markdown(f"""<div style="font-family:'Noto Sans JP',sans-serif; font-size:11px;
        color:{T['text_muted']}; margin:0 0 6px; line-height:1.8;">
        ① 試合・練習データを <b>2〜{sc.MAX_FILES}ファイル</b> までアップロード<br>
        　（GPS・加速度の生データ／運動量サマリー）<br>
        大きいファイルは自動で分割して読み込みます（CSV推奨）</div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("① 運動量・GPS・加速度データ（xlsx / csv）", type=["csv", "xlsx"],
                                accept_multiple_files=True, key="main_files")
    n_up = len(uploaded) if uploaded else 0
    badge = "badge-ok" if sc.MIN_FILES <= n_up <= sc.MAX_FILES else "badge-ng"
    st.markdown(f'<span class="{badge}">{n_up} / {sc.MAX_FILES} FILES</span>', unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    weather_file = st.file_uploader("② 気象データ（別アップロード・任意）", type=["csv", "xlsx"],
                                    key="weather_file")

    sec("ANALYSIS PARAMETERS")
    with st.expander("速度・加減速のしきい値", expanded=False):
        hsr_kmh = st.number_input("HSR（高強度走）下限 km/h", 10.0, 40.0, 19.8, 0.1)
        sprint_kmh = st.number_input("スプリント下限 km/h", 15.0, 45.0, 25.2, 0.1)
        acc_th = st.number_input("加減速イベント判定 m/s²", 0.5, 6.0, 2.0, 0.1)
    with st.expander("前後半・区間", expanded=False):
        half_min = st.number_input("前半の長さ（分）", 1.0, 120.0, 45.0, 1.0)
        break_min = st.number_input("ハーフタイム（分）※除外", 0.0, 60.0, 0.0, 1.0)
        seg_min = st.selectbox("区間の長さ（分）", [5, 10, 15, 20, 30], index=2)
    with st.expander("異常値の除外", expanded=False):
        max_speed = st.number_input("速度の上限 km/h（超過は異常値）", 20.0, 80.0, 45.0, 1.0)
        spike_ms = st.number_input("速度スパイク判定 m/s", 2.0, 15.0, 6.0, 0.5)
        hr_lo, hr_hi = st.slider("心拍の有効範囲 bpm", 20, 260, (30, 230))
        sigma = st.slider("サマリーの外れ値 ±σ（0=除外しない）", 0.0, 6.0, 3.0, 0.5)
    with st.expander("心拍", expanded=False):
        hr_ref_in = st.number_input("最大心拍（基準）0=実測の最大値を使用", 0, 230, 0, 1)

# ════════════════════════════════════════════════════
# タイトル画面
# ════════════════════════════════════════════════════
if not uploaded:
    st.markdown(f"""
    <div style="padding: 50px 0 30px;">
        <div class="hero-sub">SPORTS PERFORMANCE ANALYTICS</div>
        <div class="hero-title">SPORTS DATA<br>ANALYSIS<br>SYSTEM</div>
        <div style="width:60px; height:3px; background:{T['accent_border']}; margin:24px 0;"></div>
        <div style="font-family:'Noto Sans JP',sans-serif; font-size:13px;
                    color:{T['text_muted']}; line-height:2;">
            サッカーの試合・練習で計測したGPS／3軸加速度／心拍データと運動量サマリーを解析し、<br>
            量（距離）と強度（速度）を別々に可視化します。</div>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for c, lab, val in ((c1, "STEP 01", "UPLOAD<br>2〜10 FILES"), (c2, "STEP 02", "ADD WEATHER<br>（別アップロード）"),
                        (c3, "STEP 03", "ANALYZE<br>& EXPORT")):
        kpi(c, lab, val)
    st.markdown(f"""<div style="margin-top:36px; font-family:'Rajdhani',sans-serif; font-size:12px;
        color:{T['text_faint']}; letter-spacing:0.2em;">LOAD FILES FROM THE SIDEBAR TO BEGIN</div>""",
                unsafe_allow_html=True)
    st.stop()

files = list(uploaded)
if len(files) > sc.MAX_FILES:
    st.error(f"アップロードできるのは最大 {sc.MAX_FILES} ファイルまでです（現在 {len(files)} ファイル）。"
             "サイドバーで不要なファイルを × で外してください。")
    st.stop()
if len(files) < sc.MIN_FILES:
    st.warning(f"データは {sc.MIN_FILES} ファイル以上アップロードしてください"
               f"（現在 {len(files)} ファイル）。あと {sc.MIN_FILES - len(files)} ファイル必要です。")
    st.stop()

# ════════════════════════════════════════════════════
# STEP1: ファイルの役割と列の割り当て
# ════════════════════════════════════════════════════
ROLE_RAW, ROLE_SUM = "GPS・加速度（生データ）", "運動量サマリー"
heads = st.session_state.setdefault("_heads", {})
cache = st.session_state.setdefault("_cache", {})
if len(cache) > 24:
    for k in list(cache)[:8]:
        cache.pop(k, None)

sec("FILE SETUP — ファイルの種類と列の割り当て（自動推定済み・必要なら変更）")
raw_items, sum_items = [], []

for i, f in enumerate(files):
    hk = f"{f.name}|{f.size}"
    if hk not in heads:
        try:
            heads[hk] = sc.read_head(f)
        except Exception as e:  # noqa: BLE001
            heads[hk] = e
    h = heads[hk]
    if isinstance(h, Exception):
        st.error(f"{f.name}：読み込めませんでした — {h}")
        continue
    head, enc = h
    guess = sc.detect_kind(head)
    with st.expander(f"{i + 1}. {f.name}　（{fmt_size(f.size)}）", expanded=False):
        role = st.radio("このファイルの種類", [ROLE_RAW, ROLE_SUM], horizontal=True,
                        index=0 if guess == "raw" else 1, key=f"role_{i}_{hk}")
        if role == ROLE_SUM:
            sum_items.append(dict(file=f, enc=enc, head=head))
            st.caption("運動量サマリーの列設定は、下の「運動量サマリーの列設定」で共通設定します。")
            continue
        g = sc.guess_mapping(head.columns, sc.RAW_ALIASES, sc.RAW_EXCLUDE)
        cols = list(head.columns)
        r1 = st.columns(4)
        r2 = st.columns(4)
        mp = {}
        with r1[0]:
            mp["time"] = colsel("時刻（必須）", cols, g["time"], f"c_time_{i}_{hk}", optional=False)
        with r1[1]:
            mp["lat"] = colsel("緯度", cols, g["lat"], f"c_lat_{i}_{hk}")
        with r1[2]:
            mp["lon"] = colsel("経度", cols, g["lon"], f"c_lon_{i}_{hk}")
        with r1[3]:
            mp["speed"] = colsel("速度", cols, g["speed"], f"c_spd_{i}_{hk}")
        with r2[0]:
            mp["ax"] = colsel("加速度X", cols, g["ax"], f"c_ax_{i}_{hk}")
        with r2[1]:
            mp["ay"] = colsel("加速度Y", cols, g["ay"], f"c_ay_{i}_{hk}")
        with r2[2]:
            mp["az"] = colsel("加速度Z", cols, g["az"], f"c_az_{i}_{hk}")
        with r2[3]:
            mp["hr"] = colsel("心拍（ある選手のみ）", cols, g["hr"], f"c_hr_{i}_{hk}")
        u1, u2, u3 = st.columns(3)
        s_default = 0
        if mp["speed"]:
            nm = str(mp["speed"]).lower()
            vmax = pd.to_numeric(head[mp["speed"]], errors="coerce").max()
            if ("m/s" in nm or "ms" in nm.replace("kmh", "")) and "km" not in nm:
                s_default = 1
            elif "km" not in nm and pd.notna(vmax) and vmax <= 12:
                s_default = 1
        with u1:
            s_unit = st.radio("速度の単位", ["km/h", "m/s"], index=s_default, horizontal=True,
                              key=f"su_{i}_{hk}")
        a_default = 0
        if mp["ax"]:
            amax = pd.to_numeric(head[mp["ax"]], errors="coerce").abs().max()
            a_default = 1 if (pd.notna(amax) and amax > 4) else 0
        with u2:
            a_unit = st.radio("加速度の単位", ["g", "m/s²"], index=a_default, horizontal=True,
                              key=f"au_{i}_{hk}")
        with u3:
            player = st.text_input("選手名（このファイルの選手）", value=Path(f.name).stem, key=f"pl_{i}_{hk}")
        start_ov = ""
        if mp["time"]:
            tnum = pd.to_numeric(head[mp["time"]], errors="coerce")
            if tnum.notna().mean() > 0.9 and tnum.median() < 1e8:
                start_ov = st.text_input("時刻が経過秒のため、開始日時を入力（例 2025-06-14 14:00:00）",
                                         key=f"so_{i}_{hk}")
        raw_items.append(dict(
            file=f, enc=enc, mapping=mp, player=player.strip() or Path(f.name).stem,
            params=sc.RawParams(speed_unit=s_unit, acc_unit=a_unit, max_speed_kmh=float(max_speed),
                                spike_ms=float(spike_ms), hr_min=int(hr_lo), hr_max=int(hr_hi),
                                start_override=start_ov.strip())))

sum_map = None
if sum_items:
    with st.expander("運動量サマリーの列設定（全サマリーファイル共通）", expanded=False):
        h0 = sum_items[0]["head"]
        cols = list(h0.columns)
        g = sc.guess_mapping(cols, sc.SUMMARY_ALIASES)
        sc1 = st.columns(3)
        sum_map = {}
        labels = {"player": "選手名（必須）", "team": "チーム", "date": "日付", "game": "試合",
                  "position": "ポジション", "injury": "ケガ（有無／件数）"}
        for j, (k, lab) in enumerate(labels.items()):
            with sc1[j % 3]:
                sum_map[k] = colsel(lab, cols, g[k], f"sm_{k}", optional=(k != "player"))
        st.caption("上記以外の数値列は、すべて「運動量の指標」として自動で分析対象になります。")

wx_map, wx_head = None, None
if weather_file is not None:
    wk = f"{weather_file.name}|{weather_file.size}"
    if wk not in heads:
        try:
            heads[wk] = sc.read_head(weather_file)
        except Exception as e:  # noqa: BLE001
            heads[wk] = e
    if isinstance(heads[wk], Exception):
        st.error(f"気象データを読み込めません：{heads[wk]}")
    else:
        wx_head = heads[wk]
        with st.expander(f"気象データの列設定：{weather_file.name}", expanded=False):
            cols = list(wx_head[0].columns)
            g = sc.guess_mapping(cols, sc.WEATHER_ALIASES)
            wc = st.columns(5)
            wx_map = {}
            for (k, lab), c in zip({"time": "日時（必須）", "temp": "気温", "humidity": "湿度",
                                    "wind": "風速", "sky": "天気"}.items(), wc):
                with c:
                    wx_map[k] = colsel(lab, cols, g[k], f"wx_{k}", optional=(k != "time"))

if not raw_items and not sum_items:
    st.error("読み込める有効なファイルがありません。")
    st.stop()
if sum_items and not sum_map.get("player"):
    st.error("運動量サマリーの「選手名」列を指定してください。")
    st.stop()

sig = tuple((f.name, f.size) for f in files) + ((weather_file.name, weather_file.size) if weather_file else ())
if st.session_state.get("_sig") != sig:
    st.session_state["_sig"] = sig
    st.session_state["_started"] = False
if st.button("▶  この設定で読み込み・分析を開始"):
    st.session_state["_started"] = True
if not st.session_state.get("_started"):
    note("列の割り当てを確認したら「読み込み・分析を開始」を押してください。"
         "サイドバーの設定を変えたあとも、自動で再計算されます。")
    st.stop()

# ════════════════════════════════════════════════════
# データ読み込み（進捗表示・キャッシュ）
# ════════════════════════════════════════════════════
total_jobs = len(raw_items) + len(sum_items) + (1 if wx_map else 0)
prog = st.progress(0.0, text="ファイルを読み込んでいます...")
done, load_errors = 0, []

for it in raw_items:
    f = it["file"]
    ck = ("raw", f.name, f.size, repr(sorted(it["mapping"].items(), key=lambda kv: kv[0])),
          astuple(it["params"]))
    try:
        if ck not in cache:
            cache[ck] = sc.load_raw(f, it["enc"], it["mapping"], it["params"])
        it["df"], it["rep"] = cache[ck]
    except Exception as e:  # noqa: BLE001
        load_errors.append(f"{f.name}：{e}")
        it["df"] = None
    done += 1
    prog.progress(done / total_jobs, text=f"読み込み中... {done}/{total_jobs}　{f.name}")

sum_frames = []
for it in sum_items:
    f = it["file"]
    ck = ("sum", f.name, f.size)
    try:
        if ck not in cache:
            cache[ck] = sc.read_table(f, it["enc"])
        x = cache[ck].copy()
        x["ソースファイル"] = f.name
        sum_frames.append(x)
    except Exception as e:  # noqa: BLE001
        load_errors.append(f"{f.name}：{e}")
    done += 1
    prog.progress(done / total_jobs, text=f"読み込み中... {done}/{total_jobs}　{f.name}")

wx, wx_rep = None, {}
if wx_map and wx_head:
    try:
        ck = ("wx", weather_file.name, weather_file.size, repr(sorted(wx_map.items(), key=lambda kv: kv[0])))
        if ck not in cache:
            wdf = sc.read_table(weather_file, wx_head[1])
            cache[ck] = sc.prepare_weather(wdf, wx_map)
        wx, wx_rep = cache[ck]
    except Exception as e:  # noqa: BLE001
        load_errors.append(f"{weather_file.name}（気象）：{e}")
    done += 1
    prog.progress(1.0, text="読み込み完了")
prog.empty()

for err in load_errors:
    st.error(f"読み込み失敗 — {err}")
raw_items = [it for it in raw_items if it.get("df") is not None]

sdf, s_metrics, s_outliers = None, [], pd.DataFrame()
if sum_frames:
    try:
        sdf, s_metrics, s_outliers = sc.prepare_summary(pd.concat(sum_frames, ignore_index=True),
                                                        sum_map, float(sigma))
        sdf = sc.attach_weather_daily(sdf, wx)
        if sdf.empty or not s_metrics:
            st.error("サマリーに分析できる数値列、または選手データがありません。列設定を確認してください。")
            sdf = None
    except Exception as e:  # noqa: BLE001
        st.error(f"サマリーの処理に失敗しました：{e}")
        sdf = None

if not raw_items and sdf is None:
    st.stop()

st.markdown(f"""<div class="progress-info">
    FILES : {len(files)} &nbsp;|&nbsp; RAW SESSIONS : {len(raw_items)} &nbsp;|&nbsp;
    SUMMARY ROWS : {0 if sdf is None else len(sdf)} &nbsp;|&nbsp;
    WEATHER : {'ON' if wx is not None else 'OFF'}</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# STEP2: セッション情報（開始・終了・種別・目的）＋気象の付与
# ════════════════════════════════════════════════════
sessions = []
if raw_items:
    sec("SESSION INFO — 試合・練習の開始／終了時刻と付随情報（編集できます）")
    note("開始・終了時刻はデータから自動で入力されます。修正するとその範囲だけが分析対象になります"
         "（形式：2025-06-14 14:00:00）。気象データは開始時刻に最も近い観測値が自動で付与されます。")
    info0 = pd.DataFrame([{
        "ファイル": it["file"].name, "選手": it["player"], "種別": "試合", "練習目的": "",
        "開始": it["df"].index[0].strftime("%Y-%m-%d %H:%M:%S"),
        "終了": it["df"].index[-1].strftime("%Y-%m-%d %H:%M:%S"), "メモ": ""} for it in raw_items])
    ek = "session_info_" + "_".join(f"{it['file'].name}{it['file'].size}" for it in raw_items)
    info = editor(info0, key=ek, disabled=["ファイル"], column_config={
        "種別": st.column_config.SelectboxColumn("種別", options=["試合", "練習", "その他"], required=True),
        "練習目的": st.column_config.TextColumn("練習目的"),
        "メモ": st.column_config.TextColumn("メモ")})

    used_labels = set()
    for (_, r), it in zip(info.iterrows(), raw_items):
        st_t = pd.to_datetime(r["開始"], errors="coerce")
        en_t = pd.to_datetime(r["終了"], errors="coerce")
        if pd.isna(st_t) or pd.isna(en_t):
            st.warning(f"{r['ファイル']}：開始/終了の形式が不正なため、自動値を使用します。")
            st_t, en_t = it["df"].index[0], it["df"].index[-1]
        d = sc.prepare_session(it["df"], st_t, en_t, float(half_min), float(break_min), int(seg_min))
        if d.empty:
            st.warning(f"{r['ファイル']}：指定した時間範囲にデータがありません。")
            continue
        d = sc.attach_weather(d, wx)
        hr_ref = None
        if "hr" in d and d["hr"].notna().any():
            hr_ref = float(hr_ref_in) if hr_ref_in else float(d["hr"].max())
        kw = dict(hsr_kmh=float(hsr_kmh), sprint_kmh=float(sprint_kmh), acc_th=float(acc_th), hr_ref=hr_ref)
        label = f"{r['選手']}｜{Path(r['ファイル']).stem}"
        if label in used_labels:
            label += f"#{len(used_labels)}"
        used_labels.add(label)
        sky = d[WX_SKY].dropna().mode() if WX_SKY in d else pd.Series(dtype=object)
        sessions.append(dict(
            label=label, player=r["選手"], kind=r["種別"], purpose=r["練習目的"], memo=r["メモ"],
            start=st_t, end=en_t, df=d, hr_ref=hr_ref, rep=it["rep"], file=r["ファイル"],
            sky=(sky.iloc[0] if len(sky) else ""),
            whole=sc.compute_metrics(d, **kw), seg=sc.segment_table(d, "seg", **kw),
            half=sc.segment_table(d, "half", **kw)))

# ════════════════════════════════════════════════════
# メインタブ
# ════════════════════════════════════════════════════
tab_gps, tab_sum, tab_dq = st.tabs(["GPS・加速度（試合/練習）", "運動量サマリー", "データ品質・出力"])
color_of = {s["label"]: PALETTE[i % len(PALETTE)] for i, s in enumerate(sessions)}

# ════════════════════════════════════════════════════
# タブ1：GPS・加速度
# ════════════════════════════════════════════════════
with tab_gps:
    if not sessions:
        st.info("GPS・加速度の生データファイルがありません。")
    else:
        sel = st.multiselect("対象セッション", [s["label"] for s in sessions],
                             default=[s["label"] for s in sessions])
        S = [s for s in sessions if s["label"] in sel]
        if not S:
            st.info("セッションを1つ以上選択してください。")
        else:
            g_over, g_seg, g_half, g_area, g_hr = st.tabs(
                ["セッション概要（量と強度）", "区間ごとの推移", "前半・後半の比較", "行動エリア", "心拍"])

            # ── セッション概要 ───────────────────────
            with g_over:
                ov = pd.DataFrame([{
                    "セッション": s["label"], "選手": s["player"], "種別": s["kind"], "練習目的": s["purpose"],
                    "開始": s["start"].strftime("%Y-%m-%d %H:%M"), "終了": s["end"].strftime("%H:%M"),
                    WX_SKY: s["sky"], **s["whole"]} for s in S])
                sec("SESSION SUMMARY")
                table(ov)
                vol_opts = [c for c in ["総走行距離(m)", "HSR距離(m)", "スプリント距離(m)", "時間(分)"] if c in ov]
                int_opts = [c for c in ["距離/分(m/min)", "HSR率(%)", "最高速度(km/h)", "PlayerLoad/分",
                                        "スプリント回数", "加速回数", "減速回数", "平均心拍(bpm)"] if c in ov]
                c1, c2 = st.columns(2)
                with c1:
                    sec("量（どれだけ動いたか）")
                    vm = st.selectbox("量の指標", vol_opts, key="ov_vol")
                    f1 = px.bar(ov, x="セッション", y=vm, color="セッション", color_discrete_map=color_of)
                    f1.update_layout(showlegend=False)
                    show(f1, 330)
                with c2:
                    sec("強度（どれだけ激しく動いたか）")
                    im = st.selectbox("強度の指標", int_opts, key="ov_int")
                    f2 = px.bar(ov, x="セッション", y=im, color="セッション", color_discrete_map=color_of)
                    f2.update_layout(showlegend=False)
                    show(f2, 330)
                note("量＝総走行距離などの『合計』、強度＝距離/分・HSR率・最高速度などの『密度・速さ』。"
                     "両者を分けて見ることで「走った量は多いが強度は低い」といった練習内容のズレを把握できます。")

            # ── 区間ごと ────────────────────────────
            with g_seg:
                seg_all = pd.concat([s["seg"].assign(セッション=s["label"]) for s in S], ignore_index=True)
                metric_opts = [c for c in seg_all.columns if c not in ("seg", "セッション", "時間(分)", WX_TEMP)]
                sec(f"{seg_min}分ごとの推移")
                mname = st.selectbox("指標", metric_opts,
                                     index=metric_opts.index("総走行距離(m)") if "総走行距離(m)" in metric_opts else 0,
                                     key="seg_metric")
                has_t = WX_TEMP in seg_all and seg_all[WX_TEMP].notna().any()
                fig = make_subplots(specs=[[{"secondary_y": has_t}]])
                for s in S:
                    t = s["seg"]
                    if mname not in t:
                        continue
                    xs = [sc.seg_label(v, seg_min) for v in t["seg"]]
                    fig.add_trace(go.Bar(x=xs, y=t[mname], name=s["label"],
                                         marker_color=color_of[s["label"]]), secondary_y=False)
                    if has_t and WX_TEMP in t:
                        fig.add_trace(go.Scatter(x=xs, y=t[WX_TEMP], mode="lines+markers",
                                                 name=f"気温｜{s['label']}", line=dict(dash="dot", width=2),
                                                 marker_color=color_of[s["label"]]), secondary_y=True)
                fig.update_layout(barmode="group", title=f"{mname}（{seg_min}分ごと）")
                fig.update_yaxes(title_text=mname, secondary_y=False)
                if has_t:
                    fig.update_yaxes(title_text="気温(℃)", secondary_y=True, showgrid=False)
                show(fig, 420)
                piv = seg_all.assign(区間=seg_all["seg"].map(lambda v: sc.seg_label(v, seg_min))).pivot_table(
                    index="seg", columns="セッション", values=mname)
                piv.index = [sc.seg_label(v, seg_min) for v in piv.index]
                st.dataframe(piv.round(1))
                note("後半に向けた低下や、特定の区間の落ち込み・急上昇を見つけて試合の振り返りに使えます。"
                     "気象データがある場合は気温を重ねて表示します。")

            # ── 前半・後半 ──────────────────────────
            with g_half:
                rows = []
                for s in S:
                    h = s["half"].set_index("half")
                    if not {"前半", "後半"} <= set(h.index):
                        continue
                    for c in h.columns:
                        a, b = h.at["前半", c], h.at["後半", c]
                        if isinstance(a, (int, float, np.number)) and c not in (WX_TEMP,):
                            ch = (b - a) / a * 100 if a not in (0, np.nan) and pd.notna(a) and a != 0 else np.nan
                            rows.append({"セッション": s["label"], "指標": c, "前半": a, "後半": b,
                                         "変化率(%)": round(ch, 1) if pd.notna(ch) else np.nan})
                hd = pd.DataFrame(rows)
                if hd.empty:
                    st.info("前半・後半の両方にデータがあるセッションがありません（前半の長さ設定を確認）。")
                else:
                    sec("FIRST HALF vs SECOND HALF")
                    want = [m for m in ["総走行距離(m)", "距離/分(m/min)", "HSR距離(m)", "スプリント回数",
                                        "最高速度(km/h)", "PlayerLoad/分", "平均心拍(bpm)"] if m in set(hd["指標"])]
                    pick_m = st.multiselect("比較する指標", sorted(set(hd["指標"])), default=want[:5])
                    hs = hd[hd["指標"].isin(pick_m)]
                    if not hs.empty:
                        fig = px.bar(hs, x="指標", y="変化率(%)", color="セッション", barmode="group",
                                     color_discrete_map=color_of,
                                     title="後半の変化率（前半を基準・％）")
                        fig.add_hline(y=0, line_color=T["text_muted"])
                        show(fig, 380)
                    table(hs)
                    note("距離/分やHSR・スプリントの後半低下は疲労の目安になります。"
                         "『量』は同じでも『強度』が落ちていないかを確認してください。")

            # ── 行動エリア ──────────────────────────
            with g_area:
                sec("PLAYING AREA — 走行の軌跡・スプリントの向き")
                MIN_GPS_PTS = 30     # これ未満のセッションは位置が描けないので除外
                gps_n = {s["label"]: (int(s["df"]["lat"].notna().sum()) if "lat" in s["df"] else 0) for s in S}
                gps_ok = [s for s in S if gps_n[s["label"]] >= MIN_GPS_PTS]
                gps_ng = [f'{k}（有効な位置 {v} 点）' for k, v in gps_n.items() if v < MIN_GPS_PTS]
                if gps_ng and gps_ok:
                    st.caption("位置データが少ないため、行動エリアから除外したセッション： " + " ／ ".join(gps_ng))
                if not gps_ok:
                    st.info("行動エリアを表示できるGPS（緯度・経度）のデータがありません。"
                            "位置データがあるセッションを「対象セッション」に含めてください。")
                    if gps_ng:
                        st.caption("除外されたセッション： " + " ／ ".join(gps_ng))
                else:
                    L, Wd = float(field_L), float(field_W)
                    if field_fp is None:
                        note("STEP 1 が未設定です。サイドバー上部の「FIELD SETUP」でピッチ4隅のGPSを入力すると、"
                             "実際のピッチ上の位置に変換できます。いまは、GPSの分布範囲をピッチ全体にあてはめた近似で表示しています。")
                    else:
                        st.caption(f"STEP 1 のフィールド設定を使用中：ピッチ {L:.0f}×{Wd:.0f} m"
                                   f"（4隅へのあてはめ誤差 約 {field_fp['rms']:.2f} m）")
                    p3, p4 = st.columns(2)
                    swap = p3.checkbox("緯度/経度の軸を入れ替える（STEP 1 未設定のときのみ有効）", value=False)
                    atk = p4.radio("前半の攻撃方向（図上）", ["右 →", "← 左"], horizontal=True)
                    q1, q2, q3 = st.columns(3)
                    fx = q1.checkbox("左右を反転", value=False)
                    fy = q2.checkbox("上下を反転", value=False)
                    flip2 = q3.checkbox("後半は攻撃方向が逆になる", value=True)
                    fp = field_fp if field_fp is not None else sc.fit_pitch([s["df"] for s in gps_ok], swap, None)
                    if fp is None:
                        st.warning("位置データの範囲が狭すぎて座標変換できません。")
                    else:
                        def pitch_layout(fig, title):
                            fig.update_layout(title=title, shapes=pitch_shapes(L, Wd, T["text_muted"]),
                                              xaxis=dict(range=[-3, L + 3], showgrid=False, zeroline=False,
                                                         visible=False),
                                              yaxis=dict(range=[-3, Wd + 3], showgrid=False, zeroline=False,
                                                         visible=False, scaleanchor="x", scaleratio=1))
                            return fig

                        sec("STEP 2 — フィールド上でGPSを確認")
                        MARGIN = st.number_input("ピッチ外とみなす余白（m）", 0.0, 30.0, 3.0, 0.5, key="fld_margin")

                        def field_xy(s):
                            d = sc.apply_pitch(s["df"], fp, L, Wd, fx, fy, atk.startswith("右"), flip2, clip=False)
                            ok = d["x"].between(-MARGIN, L + MARGIN) & d["y"].between(-MARGIN, Wd + MARGIN)
                            return d, ok

                        def to_field(s):
                            d, ok = field_xy(s)
                            d.loc[~ok, ["x", "y"]] = np.nan      # ピッチ外の点は図・集計から外す
                            return d

                        stat_rows = []
                        for s in gps_ok:
                            d0, ok0 = field_xy(s)
                            v0 = d0["x"].notna()
                            n_v, n_o = int(v0.sum()), int((v0 & ~ok0).sum())
                            stat_rows.append({"セッション": s["label"], "位置データ(点)": n_v, "ピッチ外(点)": n_o,
                                              "ピッチ内の割合(%)": round((1 - n_o / n_v) * 100, 1) if n_v else np.nan})
                        stat = pd.DataFrame(stat_rows)
                        table(stat)
                        tot_v, tot_o = int(stat["位置データ(点)"].sum()), int(stat["ピッチ外(点)"].sum())
                        in_pct = (1 - tot_o / tot_v) * 100 if tot_v else 0.0
                        if field_fp is not None and in_pct < 90:
                            st.warning(f"ピッチ内に入っている点は {in_pct:.0f}% です。4隅の座標・順番・ピッチの大きさを見直してください"
                                       "（ピッチ外の点は、下の図と集計から除外されます）。")
                        elif field_fp is not None:
                            st.caption(f"ピッチ内に入っている点：{in_pct:.1f}%")
                        chk = st.selectbox("確認するセッション", [s["label"] for s in gps_ok], key="fld_chk")
                        d_chk, ok_chk = field_xy(next(s for s in gps_ok if s["label"] == chk))
                        v_chk = d_chk["x"].notna().to_numpy()
                        dd_chk, ok_arr = d_chk[v_chk], ok_chk.to_numpy()[v_chk]
                        if len(dd_chk):
                            ins, outs = dd_chk[ok_arr], dd_chk[~ok_arr]
                            f_chk = go.Figure()
                            f_chk.add_trace(go.Scattergl(x=ins["x"].iloc[::max(1, len(ins) // 4000)],
                                                         y=ins["y"].iloc[::max(1, len(ins) // 4000)], mode="markers",
                                                         name="ピッチ内", marker=dict(size=4, color=T["accent"])))
                            if len(outs):
                                f_chk.add_trace(go.Scattergl(x=outs["x"].iloc[::max(1, len(outs) // 2000)],
                                                             y=outs["y"].iloc[::max(1, len(outs) // 2000)],
                                                             mode="markers", name="ピッチ外",
                                                             marker=dict(size=5, color=T["danger"])))
                            pitch_layout(f_chk, "GPSの点をフィールドに重ねた確認図（赤＝ピッチ外）")
                            xlo, xhi = np.quantile(dd_chk["x"], [0.002, 0.998])
                            ylo, yhi = np.quantile(dd_chk["y"], [0.002, 0.998])
                            f_chk.update_layout(xaxis=dict(range=[min(-3, xlo - 2), max(L + 3, xhi + 2)]),
                                                yaxis=dict(range=[min(-3, ylo - 2), max(Wd + 3, yhi + 2)]))
                            show(f_chk, 460)
                        else:
                            st.info("このセッションには位置データがありません。")

                        view_opts = ["選択セッションを合算"] + [s["label"] for s in gps_ok]
                        view = st.selectbox("表示するセッション", view_opts, index=1 if len(gps_ok) == 1 else 0)
                        frames = []
                        for s in gps_ok:
                            if view != view_opts[0] and s["label"] != view:
                                continue
                            frames.append(to_field(s))
                        dv = pd.concat(frames)
                        dv_xy = dv.dropna(subset=["x", "y"])

                        c1, c2 = st.columns(2)
                        with c1:
                            fh = go.Figure(go.Histogram2d(
                                x=dv_xy["x"], y=dv_xy["y"], colorscale="Blues" if st.session_state.theme == "light" else "Viridis",
                                xbins=dict(start=0, end=L, size=L / 21), ybins=dict(start=0, end=Wd, size=Wd / 14),
                                colorbar=dict(title="秒")))
                            show(pitch_layout(fh, "滞在ヒートマップ（攻撃方向→）"), 430)
                        with c2:
                            ft = go.Figure()
                            tr = dv_xy.iloc[::3]
                            ft.add_trace(go.Scattergl(x=tr["x"], y=tr["y"], mode="lines", name="軌跡",
                                                      line=dict(color=T["text_faint"], width=1), opacity=0.7))
                            ev = pd.concat([sc.sprint_events(fr, float(sprint_kmh), L) for fr in frames], ignore_index=True)
                            col_way = {"攻撃方向": T["accent"], "守備方向": T["warn"], "横・その他": T["text_secondary"]}
                            evs = ev.sort_values("dist", ascending=False).head(250) if len(ev) else ev
                            for _, r in evs.iterrows():
                                if abs(r["x1"] - r["x0"]) + abs(r["y1"] - r["y0"]) < 1:
                                    continue
                                ft.add_annotation(x=r["x1"], y=r["y1"], ax=r["x0"], ay=r["y0"], xref="x", yref="y",
                                                  axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1,
                                                  arrowwidth=2, arrowcolor=col_way[r["向き"]])
                            for way, c in col_way.items():
                                ft.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"スプリント：{way}",
                                                        line=dict(color=c, width=3)))
                            show(pitch_layout(ft, f"走行軌跡とスプリントの向き（{sprint_kmh} km/h以上）"), 430)

                        sec("SPRINT ANALYSIS")
                        if ev.empty:
                            st.info("この条件ではスプリントがありません（スプリント下限を下げて確認できます）。")
                        else:
                            n = len(ev)
                            n_att = int((ev["向き"] == "攻撃方向").sum())
                            n_def = int((ev["向き"] == "守備方向").sum())
                            n_third = int(ev["攻撃サード"].sum())
                            k1, k2, k3, k4 = st.columns(4)
                            kpi(k1, "SPRINTS", n)
                            kpi(k2, "攻撃方向", f"{n_att}（{n_att / n * 100:.0f}%）")
                            kpi(k3, "守備方向", f"{n_def}（{n_def / n * 100:.0f}%）")
                            kpi(k4, "相手ゴール前1/3で開始", f"{n_third}（{n_third / n * 100:.0f}%）")
                            bh = ev.groupby(["half", "向き"]).size().reset_index(name="回数")
                            fb = px.bar(bh, x="half", y="回数", color="向き", barmode="group",
                                        color_discrete_map=col_way, title="前後半×向き別のスプリント回数")
                            show(fb, 300)
                            note("得点につなげるには、相手ゴールに近いエリアでのスプリントを増やすことが重要です。"
                                 "攻撃方向・相手ゴール前1/3の割合を、試合ごと・選手ごとに比べてみてください。")
                        thirds = pd.cut(dv_xy["x"].clip(0, L), [-0.01, L / 3, 2 * L / 3, L + 0.01],
                                        labels=["自陣1/3", "中盤1/3", "敵陣1/3"]).value_counts(normalize=True)
                        st.caption("滞在割合（攻撃方向を右に揃えて集計）： " +
                                   " ／ ".join(f"{k} {v * 100:.0f}%" for k, v in thirds.reindex(
                                       ["自陣1/3", "中盤1/3", "敵陣1/3"]).items()))
                        st.caption("※ 座標はGPS範囲をピッチへあてはめた近似です。走行距離そのものは速度から算出しています。")

                        # ── 時系列の軌跡 ──────────────────────
                        sec("STEP 3 — 時系列の軌跡（時間の流れで見る）")
                        ts_label = st.selectbox("セッション", [s["label"] for s in gps_ok], key="ts_sess")
                        ts_s = next(s for s in gps_ok if s["label"] == ts_label)
                        tsd = to_field(ts_s).dropna(subset=["x", "y"])
                        if len(tsd) < 2:
                            st.info("このセッションには、ピッチ内の位置データがほとんどありません。")
                        else:
                            sub = tsd.iloc[::max(1, len(tsd) // 3000)]
                            f1 = go.Figure()
                            f1.add_trace(go.Scattergl(x=sub["x"], y=sub["y"], mode="lines", hoverinfo="skip",
                                                      line=dict(color=T["text_faint"], width=1), opacity=0.5,
                                                      showlegend=False))
                            f1.add_trace(go.Scattergl(
                                x=sub["x"], y=sub["y"], mode="markers", showlegend=False,
                                customdata=sub["elapsed_min"],
                                marker=dict(size=5, color=sub["elapsed_min"], colorscale="Turbo", showscale=True,
                                            colorbar=dict(title="経過(分)")),
                                hovertemplate="経過 %{customdata:.1f} 分<br>%{x:.0f} m, %{y:.0f} m<extra></extra>"))
                            show(pitch_layout(f1, "時間で色分けした軌跡（青→赤の順に時間が進みます）"), 480)

                            segs = sorted(tsd["seg"].unique())
                            if len(segs) > 12:
                                st.caption("区間が多いため、最初の12区間だけ表示します（サイドバーの「区間の長さ」を長くすると減ります）。")
                                segs = segs[:12]
                            ncol = min(3, len(segs))
                            nrow = int(np.ceil(len(segs) / ncol))
                            f2 = make_subplots(rows=nrow, cols=ncol, horizontal_spacing=0.03, vertical_spacing=0.12,
                                               subplot_titles=[sc.seg_label(int(g), int(seg_min)) for g in segs])
                            shp = []
                            for n_, g in enumerate(segs, start=1):
                                r_, c_ = (n_ - 1) // ncol + 1, (n_ - 1) % ncol + 1
                                gd = tsd[tsd["seg"] == g]
                                gd = gd.iloc[::max(1, len(gd) // 800)]
                                f2.add_trace(go.Scattergl(x=gd["x"], y=gd["y"], mode="lines", hoverinfo="skip",
                                                          line=dict(color=PALETTE[(n_ - 1) % len(PALETTE)], width=1.6),
                                                          showlegend=False), row=r_, col=c_)
                                ax = "" if n_ == 1 else str(n_)
                                for sh in pitch_shapes(L, Wd, T["text_muted"]):
                                    shp.append({**sh, "xref": f"x{ax}", "yref": f"y{ax}"})
                                f2.update_xaxes(range=[-3, L + 3], visible=False, row=r_, col=c_)
                                f2.update_yaxes(range=[-3, Wd + 3], visible=False, scaleanchor=f"x{ax}",
                                                scaleratio=1, row=r_, col=c_)
                            f2.update_layout(shapes=shp)
                            show(f2, 250 * nrow + 60)

                            ds = tsd.iloc[::max(1, len(tsd) // 1500)]
                            f3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                               subplot_titles=("攻撃方向の位置 x（m）… 右ほど相手ゴール側",
                                                               "左右の位置 y（m）", "速度（km/h）"))
                            f3.add_trace(go.Scatter(x=ds["elapsed_min"], y=ds["x"], mode="lines", showlegend=False,
                                                    line=dict(color=T["accent"], width=1.2)), row=1, col=1)
                            f3.add_trace(go.Scatter(x=ds["elapsed_min"], y=ds["y"], mode="lines", showlegend=False,
                                                    line=dict(color=T["warn"], width=1.2)), row=2, col=1)
                            if "speed" in ds:
                                f3.add_trace(go.Scatter(x=ds["elapsed_min"], y=ds["speed"], mode="lines",
                                                        showlegend=False, line=dict(color=T["ok"], width=1)), row=3, col=1)
                                f3.add_hline(y=float(sprint_kmh), line_dash="dot", line_color=T["danger"], row=3, col=1)
                            f3.update_yaxes(range=[0, L], row=1, col=1)
                            f3.update_yaxes(range=[0, Wd], row=2, col=1)
                            f3.update_xaxes(title_text="経過（分）", row=3, col=1)
                            show(f3, 560)

                            cols_ = [c for c in ("elapsed_min", "half", "x", "y", "speed") if c in tsd]
                            ex = tsd[cols_].rename(columns={"elapsed_min": "経過(分)", "half": "前後半", "x": "x(m)",
                                                            "y": "y(m)", "speed": "速度(km/h)"}).round(2)
                            ex.insert(0, "時刻", tsd.index.astype(str))
                            st.download_button("この軌跡の時系列データをCSVで保存（時刻・位置・速度）",
                                               data=ex.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                               file_name="trajectory_timeseries_session.csv", mime="text/csv",
                                               key="ts_csv")

                        # ── 軌跡アニメーション（時系列の動き）──────────
                        sec("TRAJECTORY ANIMATION — 時系列の動きを再生")
                        st.caption("試合の進行にあわせて、選手の位置の変化を再生します。"
                                   "複数のセッションを選ぶと、同じ経過時間で重ねて動きます（同じ試合の選手どうしなら同時再生、"
                                   "別の日の試合どうしなら試合の比較になります。赤い縁取り＝スプリント中）。")
                        if st.checkbox("軌跡アニメーションを作成する", value=False, key="anim_on"):
                            anim_opts = [s["label"] for s in gps_ok]
                            anim_sel = st.multiselect("再生するセッション", anim_opts,
                                                      default=anim_opts[:1], key="anim_sel")
                            max_min = max(float(s["df"]["elapsed_min"].max()) for s in gps_ok)
                            a1, a2, a3, a4 = st.columns(4)
                            a_start = a1.number_input("開始（試合の何分から）", 0.0, float(max(max_min, 1.0)),
                                                      0.0, 1.0, key="anim_start")
                            a_dur = a2.number_input("再生する長さ（分）", 1.0, 120.0, 10.0, 1.0, key="anim_dur")
                            a_step = a3.number_input("1コマの間隔（秒）", 1, 60, 5, 1, key="anim_step")
                            a_trail = a4.number_input("軌跡の長さ（秒）", 0, 600, 60, 10, key="anim_trail")
                            a_fps = st.slider("再生速度（コマ/秒）", 1, 30, 10, key="anim_fps")
                            a_end = min(a_start + a_dur, max_min) * 60.0
                            if not anim_sel:
                                st.info("再生するセッションを1つ以上選んでください。")
                            elif a_end <= a_start * 60.0:
                                st.warning(f"開始位置がデータの終わり（約 {max_min:.0f} 分）を超えています。開始（分）を小さくしてください。")
                            else:
                                a_sess = [s for s in gps_ok if s["label"] in anim_sel]
                                a_frames = [to_field(s) for s in a_sess]
                                trk = sc.anim_tracks(a_frames, a_start * 60.0, a_end, int(a_step))
                                players = [s["player"] for s in a_sess]
                                names = players if len(set(players)) == len(players) else [s["label"] for s in a_sess]
                                fig_a = build_anim_fig(trk, names, [color_of[s["label"]] for s in a_sess],
                                                       L, Wd, T, float(sprint_kmh), int(a_trail), int(a_fps))
                                show(fig_a, 620, margin=dict(t=60, b=120, l=20, r=20))
                                if trk["step"] > int(a_step):
                                    st.caption(f"コマ数が多すぎるため、1コマの間隔を {trk['step']} 秒に自動調整しました。")
                                ts = pd.DataFrame({"経過秒": trk["grid"]})
                                for i, nm in enumerate(names):
                                    ts[f"{nm}_x(m)"] = trk["X"][i].round(2)
                                    ts[f"{nm}_y(m)"] = trk["Y"][i].round(2)
                                    ts[f"{nm}_速度(km/h)"] = trk["V"][i].round(1)
                                st.download_button("時系列データをCSVで保存（位置・速度）",
                                                   data=ts.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                                   file_name="trajectory_timeseries.csv", mime="text/csv",
                                                   key="anim_csv")
                                st.download_button("アニメーションをHTMLで保存（ブラウザで再生できます）",
                                                   data=fig_a.to_html(include_plotlyjs="cdn").encode("utf-8"),
                                                   file_name="trajectory_animation.html", mime="text/html",
                                                   key="anim_dl")

            # ── 心拍 ────────────────────────────────
            with g_hr:
                hs_list = [s for s in S if s["hr_ref"]]
                if not hs_list:
                    st.info("心拍データのあるセッションがありません（心拍は選手Bのみの想定です）。")
                else:
                    for s in hs_list:
                        d = s["df"]
                        sec(f"HEART RATE — {s['label']}（基準 最大心拍 {s['hr_ref']:.0f} bpm）")
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(go.Scatter(x=d.index, y=d["hr"].rolling(30, min_periods=1).mean(),
                                                 name="心拍(30秒平均)", line=dict(color=T["danger"])), secondary_y=False)
                        fig.add_trace(go.Scatter(x=d.index, y=d["speed"].rolling(30, min_periods=1).mean(),
                                                 name="速度(30秒平均)", line=dict(color=T["accent"], width=1),
                                                 opacity=0.8), secondary_y=True)
                        fig.update_yaxes(title_text="bpm", secondary_y=False)
                        fig.update_yaxes(title_text="km/h", secondary_y=True, showgrid=False)
                        show(fig, 340)
                        pct = d["hr"] / s["hr_ref"] * 100
                        zone = pd.cut(pct, [0, 60, 70, 80, 90, 200], labels=["<60%", "60-70%", "70-80%", "80-90%", "≥90%"],
                                      right=False)
                        zt = (zone.value_counts().reindex(["<60%", "60-70%", "70-80%", "80-90%", "≥90%"]) / 60).round(1)
                        fz = px.bar(x=zt.index.astype(str), y=zt.values, labels={"x": "%HRmax ゾーン", "y": "時間（分）"},
                                    title="心拍ゾーン別の滞在時間（分）")
                        fz.update_traces(marker_color=T["danger"])
                        show(fz, 300)

# ════════════════════════════════════════════════════
# タブ2：運動量サマリー
# ════════════════════════════════════════════════════
def render_summary():
    if sdf is None:
        st.info("運動量サマリーのファイルがありません。")
        return
    has_date = sdf["日付"].notna().any()
    # ── フィルター
    sec("FILTER")
    f1, f2, f3 = st.columns(3)
    teams = sorted(sdf["チーム"].dropna().unique())
    poss = sorted(sdf["ポジション"].dropna().unique())
    sel_t = f1.multiselect("チーム", teams, default=teams) if teams else []
    sel_p = f2.multiselect("ポジション", poss, default=poss) if poss else []
    sf = sdf.copy()
    if teams:
        sf = sf[sf["チーム"].isin(sel_t)]
    if poss:
        sf = sf[sf["ポジション"].isin(sel_p)]
    if has_date:
        dmin, dmax = sdf["日付"].min().date(), sdf["日付"].max().date()
        if dmin < dmax:
            rng = f3.slider("期間", dmin, dmax, (dmin, dmax))
            sf = sf[(sf["日付"].dt.date >= rng[0]) & (sf["日付"].dt.date <= rng[1])]
    if sf.empty:
        st.warning("条件に合うデータがありません。")
        return
    sf = sf.copy()
    sf["ポジション"] = sf["ポジション"].fillna("不明")
    sf["チーム"] = sf["チーム"].fillna("不明")
    sf["月"] = sf["日付"].dt.strftime("%Y-%m") if has_date else sf["試合"].fillna("不明")

    vol_def = pick(s_metrics, ["総走行距離", "走行距離", "distance", "距離"])
    int_def = pick(s_metrics, ["距離/分", "m/min", "最高速度", "speed", "速度", "スプリント"], 1)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "ROWS", f"{len(sf):,}")
    kpi(k2, "PLAYERS", sf["選手"].nunique())
    kpi(k3, "TEAMS", sf["チーム"].nunique())
    kpi(k4, "METRICS", len(s_metrics))

    a_tab, b_tab, c_tab, d_tab, e_tab, f_tab, g_tab = st.tabs(
        ["量と強度", "ポジション・時期", "選手別の推移", "負荷チェック", "ケガ", "気温との関係", "ランキング"])

    # ── 量と強度 ────────────────────────────────
    with a_tab:
        sec("VOLUME × INTENSITY — 量と強度を別々に評価")
        c1, c2, c3 = st.columns(3)
        vm = c1.selectbox("量の指標（x軸）", s_metrics, index=s_metrics.index(vol_def), key="qa_vol")
        im = c2.selectbox("強度の指標（y軸）", s_metrics, index=s_metrics.index(int_def), key="qa_int")
        cb = c3.selectbox("色分け", ["ポジション", "チーム", "選手"], key="qa_color")
        if vm == im:
            st.warning("量と強度には別の指標を選んでください。")
        else:
            t1, t2 = st.columns(2)
            vt = t1.number_input(f"{vm} の目標値（0=未設定）", 0.0, value=0.0, key="qa_vt")
            it_ = t2.number_input(f"{im} の目標値（0=未設定）", 0.0, value=0.0, key="qa_it")
            pdta = sf.dropna(subset=[vm, im])
            fig = px.scatter(pdta, x=vm, y=im, color=cb, hover_data=["選手", "チーム", "試合", "日付"],
                             color_discrete_sequence=PALETTE, opacity=0.8)
            fig.add_vline(x=pdta[vm].mean(), line_dash="dash", line_color=T["text_muted"],
                          annotation_text="平均")
            fig.add_hline(y=pdta[im].mean(), line_dash="dash", line_color=T["text_muted"])
            if vt:
                fig.add_vline(x=vt, line_color=T["gold"], annotation_text="量の目標")
            if it_:
                fig.add_hline(y=it_, line_color=T["gold"], annotation_text="強度の目標")
            show(fig, 460)
            pm = pdta.groupby("選手").agg(試合数=(vm, "size"), 量=(vm, "mean"), 強度=(im, "mean")).reset_index()
            pm["量の判定"] = np.where(pm["量"] >= pm["量"].mean(), "多い", "少ない")
            pm["強度の判定"] = np.where(pm["強度"] >= pm["強度"].mean(), "高い", "低い")
            if vt:
                pm["量の達成率(%)"] = (pm["量"] / vt * 100).round(0)
            if it_:
                pm["強度の達成率(%)"] = (pm["強度"] / it_ * 100).round(0)
            pm = pm.rename(columns={"量": f"量:{vm}", "強度": f"強度:{im}"}).round(1)
            table(pm)
            note("右上＝量も強度も高い／右下＝量は多いが強度は低い／左上＝量は少ないが強度は高い。"
                 "目標値を入れると、練習が目的どおり（量・強度とも）だったかを達成率で確認できます。")

    # ── ポジション・時期 ─────────────────────────
    with b_tab:
        sec("POSITION × PERIOD — ポジション・時期ごとの違い")
        mm = st.selectbox("指標", s_metrics, index=s_metrics.index(vol_def), key="pp_metric")
        c1, c2 = st.columns(2)
        with c1:
            fb = px.box(sf, x="ポジション", y=mm, color="ポジション", color_discrete_sequence=PALETTE, points="outliers")
            fb.update_layout(showlegend=False, title=f"ポジション別の分布：{mm}")
            show(fb, 380)
        with c2:
            pv = sf.pivot_table(index="ポジション", columns="月", values=mm, aggfunc="mean")
            fhm = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index),
                                       colorscale="Blues" if st.session_state.theme == "light" else "Viridis",
                                       text=np.round(pv.values, 0), texttemplate="%{text}",
                                       colorbar=dict(title=mm[:8])))
            fhm.update_layout(title="ポジション × 時期（平均）")
            show(fhm, 380)
        lp = sf.groupby(["月", "ポジション"])[mm].mean().reset_index()
        fl = px.line(lp, x="月", y=mm, color="ポジション", markers=True, color_discrete_sequence=PALETTE,
                     title="時期ごとの推移（ポジション別平均）")
        show(fl, 360)

    # ── 選手別の推移 ─────────────────────────────
    with c_tab:
        sec("PLAYER SEASON TREND — いい時と悪い時の比較")
        c1, c2 = st.columns(2)
        pl = c1.selectbox("選手", sorted(sf["選手"].unique()), key="tr_player")
        mt = c2.selectbox("指標", s_metrics, index=s_metrics.index(vol_def), key="tr_metric")
        pdx = sf[sf["選手"] == pl].dropna(subset=[mt])
        xk = "日付" if has_date else "試合"
        if pdx.empty:
            st.info("この選手の該当データがありません。")
        else:
            pdx = pdx.sort_values(xk)
            mean, sd = pdx[mt].mean(), pdx[mt].std()
            team_avg = sf.dropna(subset=[mt]).groupby(xk)[mt].mean()
            fig = go.Figure()
            if pd.notna(sd) and sd > 0:
                fig.add_hrect(y0=mean - sd, y1=mean + sd, fillcolor=T["accent_soft"], line_width=0,
                              annotation_text="自己平均±1SD", annotation_position="top left")
            fig.add_trace(go.Scatter(x=team_avg.index, y=team_avg.values, name="全体平均", mode="lines",
                                     line=dict(color=T["text_muted"], dash="dot")))
            fig.add_trace(go.Scatter(x=pdx[xk], y=pdx[mt], name=pl, mode="lines+markers",
                                     line=dict(color=T["accent"], width=2)))
            best, worst = pdx.loc[pdx[mt].idxmax()], pdx.loc[pdx[mt].idxmin()]
            for r, lab, c in ((best, "BEST", T["ok"]), (worst, "WORST", T["danger"])):
                fig.add_trace(go.Scatter(x=[r[xk]], y=[r[mt]], mode="markers+text", text=[lab],
                                         textposition="top center", showlegend=False,
                                         marker=dict(size=14, color=c, symbol="star")))
            fig.add_hline(y=mean, line_color=T["accent"], line_dash="dash")
            fig.update_layout(title=f"{pl}：{mt}")
            show(fig, 420)
            show_cols = [c for c in [xk, "試合", "チーム", "ポジション", mt, WX_TEMP, WX_SKY] if c in pdx.columns]
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**いい時（上位3）**")
                table(pdx.nlargest(3, mt)[list(dict.fromkeys(show_cols))])
            with cc2:
                st.markdown("**悪い時（下位3）**")
                table(pdx.nsmallest(3, mt)[list(dict.fromkeys(show_cols))])
            note("良い時・悪い時の試合条件（気温・天気・相手など）を見比べて、コンディション調整に活かせます。")

    # ── 負荷チェック ─────────────────────────────
    with d_tab:
        sec("LOAD CHECK — 負荷のかけすぎ・かけなさ過ぎ（ACWR）")
        if not has_date:
            st.info("負荷チェックには「日付」列が必要です。")
        else:
            lm = st.selectbox("負荷の指標", s_metrics, index=s_metrics.index(vol_def), key="ac_metric")
            ac = sc.calc_acwr(sf, lm)
            if ac.empty or ac["ACWR"].notna().sum() == 0:
                st.info("ACWRの計算には、1選手あたり28日以上の期間のデータが必要です。")
            else:
                last = ac.dropna(subset=["ACWR"]).sort_values("日付").groupby("選手").tail(1).copy()
                last["判定"] = last["ACWR"].map(sc.acwr_flag)
                last["ACWR"] = last["ACWR"].round(2)
                cnt = last["判定"].value_counts()
                k1, k2, k3, k4 = st.columns(4)
                kpi(k1, "過負荷リスク", cnt.get("過負荷リスク", 0))
                kpi(k2, "注意", cnt.get("注意", 0))
                kpi(k3, "適正", cnt.get("適正", 0))
                kpi(k4, "負荷不足", cnt.get("負荷不足", 0))
                order = {"過負荷リスク": 0, "注意": 1, "負荷不足": 2, "適正": 3}
                last = last.sort_values("判定", key=lambda s: s.map(order))
                lt = last[["選手", "日付", "急性(7日)", "慢性(週平均)", "ACWR", "判定"]].copy()
                lt["日付"] = lt["日付"].dt.strftime("%Y-%m-%d")
                table(lt.round(1))
                pl = st.selectbox("推移を見る選手", sorted(ac["選手"].unique()), key="ac_player")
                ap = ac[(ac["選手"] == pl)].dropna(subset=["ACWR"])
                fig = go.Figure()
                fig.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(123,216,143,0.15)", line_width=0)
                fig.add_hrect(y0=1.5, y1=max(2.5, ap["ACWR"].max() + 0.2), fillcolor="rgba(255,107,107,0.15)",
                              line_width=0)
                fig.add_trace(go.Scatter(x=ap["日付"], y=ap["ACWR"], mode="lines+markers", name="ACWR",
                                         line=dict(color=T["accent"])))
                fig.update_layout(title=f"{pl}：ACWRの推移（緑=適正 0.8〜1.3／赤=1.5超）")
                show(fig, 360)
                note("ACWR＝直近7日の負荷 ÷ 直近28日の週平均負荷。0.8〜1.3を適正、1.5超を過負荷の目安とするのが"
                     "一般的ですが、最適な範囲はチーム・指標により異なります。しきい値は参考値としてご利用ください。")

    # ── ケガ ────────────────────────────────────
    with e_tab:
        sec("INJURY — ケガの割合")
        if not sf["ケガ有無"].any():
            st.info("「ケガ」列が未指定、または発生がありません（サマリー設定でケガ列を指定してください）。")
        else:
            rate = sf["ケガ有無"].mean() * 100
            k1, k2 = st.columns(2)
            kpi(k1, "ケガ発生件数", int(sf["ケガ有無"].sum()))
            kpi(k2, "ケガの割合（全データ行あたり）", f"{rate:.1f}%")
            c1, c2 = st.columns(2)
            with c1:
                ip = sf.groupby("ポジション")["ケガ有無"].mean().mul(100).reset_index()
                fi = px.bar(ip, x="ポジション", y="ケガ有無", labels={"ケガ有無": "ケガの割合(%)"},
                            title="ポジション別のケガの割合")
                fi.update_traces(marker_color=T["danger"])
                show(fi, 340)
            with c2:
                im_ = sf.groupby("月")["ケガ有無"].mean().mul(100).reset_index()
                fi2 = px.bar(im_, x="月", y="ケガ有無", labels={"ケガ有無": "ケガの割合(%)"},
                             title="時期別のケガの割合")
                fi2.update_traces(marker_color=T["warn"])
                show(fi2, 340)
            sel_m = st.multiselect("ケガあり／なしで比較する指標", s_metrics, default=[vol_def, int_def]
                                   if vol_def != int_def else [vol_def], key="inj_metrics")
            if sel_m:
                cmp_ = sf.groupby("ケガ有無")[sel_m].mean().round(1).T
                cmp_.columns = ["ケガなし" if not c else "ケガあり" for c in cmp_.columns]
                st.dataframe(cmp_)
            note("ケガが出た時の負荷（量・強度）が普段と違うかを確認し、ケガ予防・コンディション調整の目安に結び付けます。"
                 "件数が少ない場合は偶然の影響が大きい点にご注意ください。")

    # ── 気温との関係 ─────────────────────────────
    with f_tab:
        sec("TEMPERATURE × INTENSITY — 気温と強度")
        if WX_TEMP not in sf or sf[WX_TEMP].notna().sum() < 3:
            st.info("気温と結び付けられるデータがありません。気象データをアップロードし、"
                    "サマリーに「日付」列があることを確認してください。")
        else:
            tm = st.selectbox("強度の指標", s_metrics, index=s_metrics.index(int_def), key="tp_metric")
            td = sf.dropna(subset=[WX_TEMP, tm])
            fig = px.scatter(td, x=WX_TEMP, y=tm, color="ポジション", color_discrete_sequence=PALETTE,
                             hover_data=["選手", "試合", "日付"], opacity=0.75)
            if len(td) >= 3 and td[WX_TEMP].nunique() > 1:
                co = np.polyfit(td[WX_TEMP], td[tm], 1)
                xs = np.linspace(td[WX_TEMP].min(), td[WX_TEMP].max(), 20)
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(co, xs), mode="lines", name="傾向線",
                                         line=dict(color=T["gold"], width=3)))
                r = np.corrcoef(td[WX_TEMP], td[tm])[0, 1]
                st.caption(f"相関係数 r = {r:.2f}（気温が1℃上がると {tm} は約 {co[0]:.1f} 変化）")
            show(fig, 430)
            bins = pd.cut(td[WX_TEMP], [-50, 15, 20, 25, 30, 60], labels=["〜15℃", "15〜20℃", "20〜25℃", "25〜30℃", "30℃〜"])
            bt = td.groupby(bins, observed=True)[tm].agg(["mean", "count"]).reset_index()
            bt.columns = ["気温帯", f"{tm}（平均）", "件数"]
            table(bt.round(1))
            note("気温帯ごとに強度がどう変わるかを見て、暑熱下での負荷調整や水分補給計画の検討に使えます。")

    # ── ランキング ──────────────────────────────
    with g_tab:
        sec("RANKING")
        rm = st.selectbox("指標", s_metrics, index=s_metrics.index(vol_def), key="rk_metric")
        rk = (sf.groupby("選手")[rm].agg(["mean", "max", "count"]).reset_index()
              .rename(columns={"mean": "平均", "max": "最大", "count": "試合数"}).sort_values("平均", ascending=False))
        rk.insert(0, "順位", range(1, len(rk) + 1))
        c1, c2 = st.columns([1, 1])
        with c1:
            table(rk.round(1))
        with c2:
            top = rk.head(15).iloc[::-1]
            fr = px.bar(top, x="平均", y="選手", orientation="h", title=f"{rm}（平均・上位15）")
            fr.update_traces(marker_color=T["accent"])
            show(fr, 440)
        sec("RADAR — 選手とチーム平均の比較（パーセンタイル）")
        rp = st.selectbox("選手", sorted(sf["選手"].unique()), key="rd_player")
        rmetrics = st.multiselect("指標（3つ以上）", s_metrics, default=s_metrics[:min(6, len(s_metrics))], key="rd_m")
        if len(rmetrics) >= 3:
            pm_ = sf.groupby("選手")[rmetrics].mean()
            pct = pm_.rank(pct=True) * 100
            if rp in pct.index:
                cats = rmetrics + [rmetrics[0]]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=[50] * len(cats), theta=cats, name="チーム中央値",
                                              line=dict(color=T["border_strong"], dash="dot")))
                vals = pct.loc[rp, rmetrics].tolist()
                fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats, fill="toself", name=rp,
                                              line=dict(color=T["accent"]), fillcolor=T["accent_soft"]))
                fig.update_layout(polar=dict(bgcolor=T["bg"], radialaxis=dict(range=[0, 100], visible=True,
                                                                                gridcolor=T["border"]),
                                             angularaxis=dict(gridcolor=T["border"])))
                show(fig, 440)
                st.caption("各軸はチーム内でのパーセンタイル順位（0〜100）です。")


with tab_sum:
    render_summary()

# ════════════════════════════════════════════════════
# タブ3：データ品質・出力
# ════════════════════════════════════════════════════
with tab_dq:
    sec("DATA QUALITY — 異常値の除外結果")
    if sessions:
        q = pd.DataFrame([{
            "セッション": s["label"], "入力行数": s["rep"]["rows_in"], "1秒データ数": s["rep"]["seconds"],
            "時刻不正": s["rep"]["time_invalid"], "GPS座標不正": s["rep"]["gps_invalid"],
            "速度の異常値": s["rep"]["speed_outlier"], "速度スパイク": s["rep"]["speed_spike"],
            "位置ジャンプ": s["rep"]["gps_jump"], "心拍の異常値": s["rep"]["hr_outlier"],
            "欠測秒数": s["rep"]["gap_seconds"]} for s in sessions])
        table(q)
        note("異常値は削除し、短い欠測（3秒以内）は前後から補間しています。"
             "しきい値はサイドバーの「異常値の除外」で変更できます。")
    if sdf is not None:
        st.markdown(f"**運動量サマリー：外れ値（±{sigma}σ）として除外した値 — {len(s_outliers)} 件**")
        if len(s_outliers):
            table(s_outliers)
        else:
            st.caption("該当なし")
    if wx is not None:
        st.markdown(f"**気象データ：{len(wx):,} 行（{wx['time'].min()} 〜 {wx['time'].max()}）**　不正値の除外："
                    + "、".join(f"{k} {v}件" for k, v in wx_rep.items()))

    sec("EXPORT — 気象などの付随情報を付与した結果をCSVで保存")
    cols = st.columns(4)

    def csv_bytes(df):
        return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    if sessions:
        ov_all = pd.DataFrame([{"セッション": s["label"], "選手": s["player"], "種別": s["kind"],
                                "練習目的": s["purpose"], "メモ": s["memo"], "開始": s["start"], "終了": s["end"],
                                WX_SKY: s["sky"], **s["whole"]} for s in sessions])
        cols[0].download_button("セッション概要 CSV", csv_bytes(ov_all), "session_summary.csv", "text/csv")
        seg_x = pd.concat([s["seg"].assign(セッション=s["label"]) for s in sessions], ignore_index=True)
        cols[1].download_button(f"{seg_min}分ごと CSV", csv_bytes(seg_x), "segment_metrics.csv", "text/csv")
        half_x = pd.concat([s["half"].assign(セッション=s["label"]) for s in sessions], ignore_index=True)
        cols[2].download_button("前後半 CSV", csv_bytes(half_x), "half_metrics.csv", "text/csv")
    if sdf is not None:
        out = sdf.drop(columns=[c for c in ["ケガ有無"] if c in sdf and not sum_map.get("injury")])
        cols[3].download_button("サマリー（気象付き）CSV", csv_bytes(out), "summary_with_weather.csv", "text/csv")
