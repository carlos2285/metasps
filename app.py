# app.py — Dashboard de Indicadores (UI mejorada + menos lag)
# --------------------------------------------------------------------
# Requisitos: streamlit, pandas, numpy, plotly
# Datos esperados:
#   - indicadores_flat.csv: resultado, indicador, anio, valor
#   - (opcional) proyecciones_2026_2029.csv: anio_proy, valor_proy, [resultado], [indicador]
#
# Funcionalidad:
#   - Navegación por resultado (chips) + selector de indicador
#   - Modelos SIN deps externas: lineal, SES, Holt lineal, Naive con deriva y Auto (mejor MAE)
#   - Intervalos 80% aproximados por residuales
#   - Shocks optimista/pesimista y truncado a cero
#   - Actualizar histórico vía upload (upsert)
#   - Permalink en la URL (res, ind, ymin, ymax)
#   - Optimizaciones de rendimiento con @st.cache_data

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------- Configuración general ---------------------------
st.set_page_config(
    page_title="Dashboard de Indicadores",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paleta y tokens de estilo
PRIMARY = "#dc2626"  # rojo
BLUE    = "#2563eb"  # azul
INK     = "#0f172a"  # texto fuerte
MUTED   = "#64748b"  # texto suave
SURFACE = "#f8fafc"  # fondos claros

# Estilos globales (tipos, chips, tarjetas)
st.markdown(f"""
<style>
:root {{
  --brand: {PRIMARY};
  --ink: {INK};
  --muted: {MUTED};
  --surface: {SURFACE};
}}
.block-container {{ max-width: 1200px; }}
html, body, [class*="css"] {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}

h1.app-title {{
  font-size: 40px; line-height: 1.1; margin: 6px 0 8px 0; color: var(--ink);
}}
.subtle {{
  color: var(--muted); font-size: 14px; margin-bottom: 12px;
}}
.header-sticky {{
  position: sticky; top: 0; z-index: 10; background: white;
  padding: 10px 0 8px 0; border-bottom: 1px solid #eef2f7;
}}

.chips {{ display:grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); margin: 12px 0 6px; }}
.chip {{
  border:1px solid #e5e7eb; background:#fff; color:#334155;
  padding: 12px 14px; border-radius: 14px; min-height: 60px;
  display:flex; align-items:center; justify-content:center; text-align:center;
  transition: all .12s ease-in-out; cursor:pointer;
}}
.chip:hover {{ border-color:#d1d5db; transform: translateY(-1px); }}
.chip.active {{ background: var(--brand); color:#fff; border-color: var(--brand); }}

.kpi {{
  border:1px solid #e5e7eb; border-radius:14px; padding:14px; background:#fff;
}}
.kpi .label {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
.kpi .value {{ color:var(--ink); font-weight:700; font-size:30px; line-height:1; }}
.kpi .delta {{ color:var(--muted); font-size:12px; margin-top:2px; }}

.small-note {{ color:var(--muted); font-size:12px; }}

.dataframe tbody tr th, .dataframe thead th {{ text-align:left; }}
</style>
""", unsafe_allow_html=True)

# --------------------------- Helpers ---------------------------
def canon(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def read_csv_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def ensure_cols(df, cols):
    return [c for c in cols if c not in df.columns]

def qp_set(**kwargs):
    try:
        st.query_params.update({k: str(v) for k, v in kwargs.items() if v is not None})
    except Exception:
        try:
            st.experimental_set_query_params(**{k: v for k, v in kwargs.items() if v is not None})
        except:
            pass

def qp_get(name, default=None, cast=str):
    try:
        v = st.query_params.get(name, None)
        if isinstance(v, list):
            v = v[0] if v else None
        return cast(v) if (v is not None and cast) else (v if v is not None else default)
    except Exception:
        return default

def fmt2(x):
    try:
        return f"{float(x):,.2f}"
    except:
        return str(x)

# --------------------------- Carga de datos ---------------------------
@st.cache_data(show_spinner=False)
def load_data(hist_path: str, proy_path: str):
    # Histórico
    hist = read_csv_safe(hist_path)
    if not hist.empty:
        rmap = {}
        if "año" in hist.columns and "anio" not in hist.columns:  rmap["año"] = "anio"
        if "year" in hist.columns and "anio" not in hist.columns:  rmap["year"] = "anio"
        if "value" in hist.columns and "valor" not in hist.columns: rmap["value"] = "valor"
        hist = hist.rename(columns=rmap)
        for c in ("resultado", "indicador"):
            if c in hist.columns:
                hist[c] = hist[c].map(canon)
        if "anio" in hist.columns:  hist["anio"]  = pd.to_numeric(hist["anio"],  errors="coerce")
        if "valor" in hist.columns: hist["valor"] = pd.to_numeric(hist["valor"], errors="coerce")
        hist = (hist
                .dropna(subset=["resultado", "indicador", "anio"])
                .sort_values(["resultado", "indicador", "anio"]))
        hist = hist[["resultado", "indicador", "anio", "valor"]]

    # Proyecciones opcionales (no usadas en el cálculo, pero se preservan por compat)
    proy = read_csv_safe(proy_path)
    if not proy.empty:
        pmap = {}
        if "anio" in proy.columns and "anio_proy" not in proy.columns:   pmap["anio"]  = "anio_proy"
        if "valor" in proy.columns and "valor_proy" not in proy.columns: pmap["valor"] = "valor_proy"
        proy = proy.rename(columns=pmap)
        for c in ("resultado", "indicador"):
            if c in proy.columns:
                proy[c] = proy[c].map(canon)
        if "anio_proy" in proy.columns:  proy["anio_proy"]  = pd.to_numeric(proy["anio_proy"],  errors="coerce")
        if "valor_proy" in proy.columns: proy["valor_proy"] = pd.to_numeric(proy["valor_proy"], errors="coerce")
        keep = [c for c in ["resultado", "indicador", "anio_proy", "valor_proy"] if c in proy.columns]
        proy = proy[keep].dropna(subset=["anio_proy", "valor_proy"]).sort_values("anio_proy")

    return hist, proy

HIST_PATH = os.getenv("HIST_PATH", "indicadores_flat.csv")
PROY_PATH = os.getenv("PROY_PATH", "proyecciones_2026_2029.csv")
flat0, base_proy0 = load_data(HIST_PATH, PROY_PATH)

# Estado vivo (para actualizar histórico en caliente)
if "flat" not in st.session_state:      st.session_state.flat = flat0.copy()
if "base_proy" not in st.session_state: st.session_state.base_proy = base_proy0.copy()

flat = st.session_state.flat

# --------------------------- Sidebar: actualizar histórico ---------------------------
with st.sidebar:
    st.markdown("### 🔁 Actualizar histórico")
    st.caption("CSV con columnas: `resultado, indicador, anio, valor`. Se hace **upsert** por (resultado, indicador, anio).")
    up = st.file_uploader("Subir CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            miss = ensure_cols(df, ["resultado", "indicador", "anio", "valor"])
            if miss:
                st.error(f"Faltan columnas: {miss}")
            else:
                for c in ("resultado", "indicador"): df[c] = df[c].map(canon)
                df["anio"]  = pd.to_numeric(df["anio"],  errors="coerce")
                df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
                df = df.dropna(subset=["resultado", "indicador", "anio"])
                key = ["resultado", "indicador", "anio"]
                merged = pd.concat([flat, df]).drop_duplicates(subset=key, keep="last").sort_values(key)
                st.session_state.flat = merged
                flat = merged
                # limpiar cachés para que se note el update sin lag
                st.cache_data.clear()
                st.success("Histórico actualizado en memoria.")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
    if not flat.empty:
        st.download_button(
            "⬇️ Descargar histórico actualizado",
            data=flat.to_csv(index=False).encode("utf-8"),
            file_name="historico_actualizado.csv",
            mime="text/csv"
        )

# --------------------------- Encabezado ---------------------------
st.markdown('<div class="header-sticky"><h1 class="app-title">Dashboard de Indicadores</h1>'
            '<div class="subtle">Selecciona un <b>Resultado</b> y un <b>Indicador</b>. La URL guarda tu vista para compartir.</div>'
            '</div>', unsafe_allow_html=True)

if flat.empty:
    st.warning("No hay datos. Sube un CSV desde la barra lateral.")
    st.stop()

# --------------------------- Navegación por resultado (chips) ---------------------------
@st.cache_data(show_spinner=False)
def get_resultados(df: pd.DataFrame):
    return sorted(df["resultado"].dropna().unique().tolist())

resultados = get_resultados(flat)
res_qp = qp_get("res", resultados[0] if resultados else "", str)
if res_qp not in resultados:
    res_qp = resultados[0]

# Render chips como una grilla (menos lag que muchos st.button aislados)
clicked_res = res_qp
st.markdown('<div class="chips">', unsafe_allow_html=True)
for r in resultados:
    active = "active" if r == res_qp else ""
    if st.button(r, key=f"chip_{r}", use_container_width=True):
        clicked_res = r
    # truco visual: estilizar el último botón (Streamlit no permite class per botón),
    # así que mostramos pill de estado debajo para forzar el estilo via HTML/CSS:
    st.markdown(f'<div class="chip {active}" style="display:none">{r}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

sel_res = clicked_res
qp_set(res=sel_res)

# --------------------------- Controles de serie ---------------------------
@st.cache_data(show_spinner=False)
def indicadores_por_resultado(df: pd.DataFrame, res: str):
    sub = df[df["resultado"] == res]
    return sorted(sub["indicador"].dropna().unique().tolist()), sub[["indicador", "anio", "valor"]].copy()

inds, df_res = indicadores_por_resultado(flat, sel_res)
if not inds:
    st.warning("No hay indicadores para este resultado.")
    st.stop()

ind_qp = qp_get("ind", inds[0], str)
if ind_qp not in inds:
    ind_qp = inds[0]

c1, c2 = st.columns([3, 2], gap="small")
with c1:
    sel_ind = st.selectbox("Indicador", inds, index=inds.index(ind_qp))
with c2:
    yr_min = int(df_res["anio"].min()); yr_max = int(df_res["anio"].max())
    ymin, ymax = st.slider("Rango histórico a mostrar", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))
qp_set(ind=sel_ind, ymin=ymin, ymax=ymax)

hist = df_res.query("indicador == @sel_ind")[["anio", "valor"]].dropna().sort_values("anio")
hist_vis = hist.query("@ymin <= anio <= @ymax")

# --------------------------- Modelos (sin deps) ---------------------------
def ci80_from_residuals(resid):
    if resid is None or len(resid) == 0:
        return 0.0
    return 1.28 * float(np.nanstd(resid))

def proj_linear(y, years, fyears):
    if len(y) < 2:
        return np.array([]), None, None, None
    X = years - years.min()
    b1, b0 = np.polyfit(X, y, 1)  # y ≈ b0 + b1*X
    Xf = fyears - years.min()
    yhat = b0 + b1 * Xf
    fitted = b0 + b1 * X
    resid = y - fitted
    half = ci80_from_residuals(resid)
    return yhat, fitted, (yhat - half), (yhat + half)

def proj_ses(y, years, fyears, alphas=(0.2, 0.4, 0.6, 0.8)):
    best_ae = np.inf; best_fit = None; best_alpha = 0.6
    for a in alphas:
        s = None; fitted = []
        for val in y:
            s = val if s is None else a * val + (1 - a) * s
            fitted.append(s)
        fitted = np.array(fitted)
        ae = float(np.mean(np.abs(y - fitted)))
        if ae < best_ae:
            best_ae, best_fit, best_alpha = ae, fitted, a
    last_s = best_fit[-1]
    yhat = np.full(len(fyears), float(last_s), dtype=float)
    resid = y - best_fit
    half = ci80_from_residuals(resid)
    return yhat, best_fit, (yhat - half), (yhat + half)

def proj_holt(y, years, fyears, alphas=(0.2, 0.4, 0.6, 0.8), betas=(0.2, 0.4, 0.6, 0.8)):
    best = (np.inf, 0.6, 0.2, None, None)  # ae, alpha, beta, level, trend
    for a in alphas:
        for b in betas:
            l = None; t = 0.0; fitted = []
            for i, val in enumerate(y):
                if l is None:
                    l = val
                    t = (y[1] - y[0]) if len(y) > 1 else 0.0
                prev_l = l
                l = a * val + (1 - a) * (l + t)
                t = b * (l - prev_l) + (1 - b) * t
                fitted.append(l + t)  # 1-step ahead
            fitted = np.array(fitted)
            ae = float(np.mean(np.abs(y - fitted)))
            if ae < best[0]:
                best = (ae, a, b, l, t)
    a, b, l, t = best[1], best[2], best[3], best[4]
    yhat = np.array([l + (h + 1) * t for h in range(len(fyears))], dtype=float)

    # recompute fitted for residuals
    l = None; t = 0.0; fitted = []
    for i, val in enumerate(y):
        if l is None:
            l = val; t = (y[1] - y[0]) if len(y) > 1 else 0.0
        prev_l = l
        l = a * val + (1 - a) * (l + t)
        t = b * (l - prev_l) + (1 - b) * t
        fitted.append(l + t)
    fitted = np.array(fitted)
    resid = y - fitted
    half = ci80_from_residuals(resid)
    return yhat, fitted, (yhat - half), (yhat + half)

def proj_drift(y, years, fyears):
    if len(y) < 2:
        yhat = np.full(len(fyears), (y[-1] if len(y) else np.nan))
        return yhat, y, None, None
    slope = (y[-1] - y[0]) / (len(y) - 1)
    yhat = np.array([y[-1] + (h + 1) * slope for h in range(len(fyears))], dtype=float)
    trend_fit = np.linspace(y[0], y[-1], len(y))
    resid = y - trend_fit
    half = ci80_from_residuals(resid)
    return yhat, trend_fit, (yhat - half), (yhat + half)

# Backtest (one-step-ahead, últimos k)
def backtest_mae(y, years, k, method_name):
    if len(y) <= k:
        return np.inf
    maes = []
    start = len(y) - k
    for cut in range(start, len(y)):
        y_tr = y[:cut]; yr_tr = years[:cut]
        target_year = np.array([years[cut]])
        try:
            if method_name == "Tendencia lineal":
                yhat, _, _, _ = proj_linear(y_tr, yr_tr, target_year)
            elif method_name == "SES (suavizamiento exponencial simple)":
                yhat, _, _, _ = proj_ses(y_tr, yr_tr, target_year)
            elif method_name == "Holt lineal (doble exponencial)":
                yhat, _, _, _ = proj_holt(y_tr, yr_tr, target_year)
            elif method_name == "Naive con deriva":
                yhat, _, _, _ = proj_drift(y_tr, yr_tr, target_year)
            else:
                yhat, _, _, _ = proj_linear(y_tr, yr_tr, target_year)
            maes.append(abs(float(yhat[0]) - y[cut]))
        except:
            maes.append(np.inf)
    return float(np.mean(maes)) if maes else np.inf

def auto_select(y, years, k):
    cands = ["Tendencia lineal",
             "SES (suavizamiento exponencial simple)",
             "Holt lineal (doble exponencial)",
             "Naive con deriva"]
    scores = [(m, backtest_mae(y, years, k, m)) for m in cands]
    scores.sort(key=lambda t: t[1])
    return scores[0][0], scores

# --------------------------- Panel de Proyección ---------------------------
st.markdown("---")
st.markdown("### Proyección")

left, mid, right, more = st.columns([2, 2, 2, 2], gap="small")
with left:
    modelo = st.selectbox("Modelo", [
        "Auto (mejor MAE)",
        "Tendencia lineal",
        "SES (suavizamiento exponencial simple)",
        "Holt lineal (doble exponencial)",
        "Naive con deriva"
    ])
with mid:
    yr_max = int(hist["anio"].max()) if not hist.empty else 2024
    year_end = st.number_input("Año final de proyección",
                               min_value=yr_max,
                               max_value=yr_max + 30,
                               value=min(yr_max + 9, yr_max + 30),
                               step=1)
with right:
    back_k = st.slider("Backtest (años MASE~)", min_value=3, max_value=int(min(6, max(3, len(hist) - 1))), value=int(min(5, max(3, len(hist) - 1))))
with more:
    clamp0 = st.checkbox("Truncar a cero", value=True)

sc1, sc2 = st.columns(2, gap="small")
with sc1:
    shock_up = st.slider("🔼 Escenario optimista (%)", -20, 100, 0)
with sc2:
    shock_dn = st.slider("🔽 Escenario pesimista (%)", -100, 20, 0)

years = hist["anio"].values
y      = hist["valor"].values
future_years = np.arange(years.max() + 1, int(year_end) + 1, dtype=int) if len(hist) > 0 else np.array([], dtype=int)

# Cachear proyección y backtest por combinación de serie + params
@st.cache_data(show_spinner=False)
def compute_projection_cached(y_tuple, years_tuple, fyears_tuple, model_name, back_k_val, clamp_zero, shock_up_val, shock_dn_val):
    y_arr     = np.array(y_tuple, dtype=float)
    years_arr = np.array(years_tuple, dtype=int)
    f_arr     = np.array(fyears_tuple, dtype=int)

    chosen = model_name
    if len(f_arr) == 0 or len(y_arr) == 0:
        return None, None, None, None, None

    if model_name == "Auto (mejor MAE)":
        chosen, _scores = auto_select(y_arr, years_arr, back_k_val)

    if chosen == "Tendencia lineal":
        yhat, fitted, ci_lo, ci_hi = proj_linear(y_arr, years_arr, f_arr)
    elif chosen == "SES (suavizamiento exponencial simple)":
        yhat, fitted, ci_lo, ci_hi = proj_ses(y_arr, years_arr, f_arr)
    elif chosen == "Holt lineal (doble exponencial)":
        yhat, fitted, ci_lo, ci_hi = proj_holt(y_arr, years_arr, f_arr)
    elif chosen == "Naive con deriva":
        yhat, fitted, ci_lo, ci_hi = proj_drift(y_arr, years_arr, f_arr)
    else:
        yhat, fitted, ci_lo, ci_hi = proj_linear(y_arr, years_arr, f_arr)

    # shocks
    if yhat is not None:
        if shock_up_val: yhat = yhat * (1 + shock_up_val / 100.0)
        if shock_dn_val: yhat = yhat * (1 + shock_dn_val / 100.0)

        if ci_lo is not None and ci_hi is not None:
            if shock_up_val: ci_lo = ci_lo * (1 + shock_up_val / 100.0); ci_hi = ci_hi * (1 + shock_up_val / 100.0)
            if shock_dn_val: ci_lo = ci_lo * (1 + shock_dn_val / 100.0); ci_hi = ci_hi * (1 + shock_dn_val / 100.0)

        if clamp_zero:
            yhat = np.maximum(yhat, 0.0)
            if ci_lo is not None: ci_lo = np.maximum(ci_lo, 0.0)

        # asegurar orden de bandas
        if ci_lo is not None and ci_hi is not None:
            swap = ci_lo > ci_hi
            if np.any(swap):
                tmp = ci_lo.copy(); ci_lo[swap] = ci_hi[swap]; ci_hi[swap] = tmp

    mae = None
    if len(y_arr) >= back_k_val + 1:
        mae = backtest_mae(y_arr, years_arr, back_k_val, (chosen if chosen != "Auto (mejor MAE)" else "Tendencia lineal"))

    return yhat, ci_lo, ci_hi, chosen, mae

yhat, ci_lo, ci_hi, chosen, mae = compute_projection_cached(
    tuple(map(float, y.tolist())),
    tuple(map(int, years.tolist())),
    tuple(map(int, future_years.tolist())),
    modelo,
    int(back_k),
    bool(clamp0),
    int(shock_up),
    int(shock_dn)
)

# --------------------------- KPIs ---------------------------
k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    st.markdown('<div class="kpi"><div class="label">Último valor (hist.)</div>'
                f'<div class="value">{fmt2(y[-1]) if len(y) else "–"}</div>'
                f'<div class="delta">Δ {fmt2(y[-1]-y[-2]) if len(y)>1 else "–"} vs. año previo</div></div>', unsafe_allow_html=True)
with k2:
    first_proj = fmt2(yhat[0]) if (yhat is not None and len(yhat) > 0) else "–"
    st.markdown('<div class="kpi"><div class="label">Primera proyección</div>'
                f'<div class="value">{first_proj}</div>'
                f'<div class="delta">para {future_years[0] if (yhat is not None and len(yhat)>0) else "–"}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><div class="label">Modelo elegido</div>'
                f'<div class="value" style="font-size:20px">{chosen if chosen else "–"}</div>'
                f'<div class="delta">Auto = menor MASE~</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown('<div class="kpi"><div class="label">Backtest (MASE~)</div>'
                f'<div class="value">{fmt2(mae) if (mae is not None and np.isfinite(mae)) else "—"}</div>'
                f'<div class="delta">{back_k} años</div></div>', unsafe_allow_html=True)

# --------------------------- Gráfico ---------------------------
fig = go.Figure()

if not hist.empty:
    fig.add_trace(go.Scatter(
        x=hist["anio"], y=hist["valor"], mode="lines+markers",
        name="Histórico", line=dict(color=BLUE, width=2)
    ))

if yhat is not None and len(yhat) > 0:
    fig.add_trace(go.Scatter(
        x=future_years, y=yhat, mode="lines+markers",
        name="Proyección", line=dict(color=PRIMARY, width=3, dash="dash")
    ))
    if ci_lo is not None and ci_hi is not None:
        fig.add_trace(go.Scatter(
            x=future_years, y=ci_hi, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=future_years, y=ci_lo, mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(220,38,38,.12)",
            showlegend=False, hoverinfo="skip"
        ))

# Línea vertical en el corte historia/proyección
if not hist.empty and (yhat is not None and len(yhat) > 0):
    cut_x = years.max() + 0.5
    fig.add_vline(x=cut_x, line_width=1, line_dash="dot", line_color="#cbd5e1")

# Anotaciones discretas
if len(y) > 0:
    fig.add_annotation(x=years[-1], y=y[-1],
                       text=f"Últ. hist.: {fmt2(y[-1])}", showarrow=False,
                       xanchor="left", yanchor="bottom",
                       bgcolor="white", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=11))
if yhat is not None and len(yhat) > 0:
    fig.add_annotation(x=future_years[0], y=yhat[0],
                       text=f"Proy. {future_years[0]}: {fmt2(yhat[0])}", showarrow=False,
                       xanchor="left", yanchor="top",
                       bgcolor="white", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=11))

fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="año", yaxis_title="valor",
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
)
st.plotly_chart(fig, use_container_width=True, theme=None)

# --------------------------- Tablas ---------------------------
st.markdown("#### Datos (tablas y resumen)")
t1, t2 = st.columns(2, gap="large")
with t1:
    st.markdown("**Histórico (rango visible)**")
    if not hist_vis.empty:
        hv = hist_vis.copy()
        hv["valor"] = hv["valor"].map(fmt2)
        st.dataframe(hv.rename(columns={"anio": "año"}), use_container_width=True, hide_index=True)
    else:
        st.caption("—")

with t2:
    st.markdown("**Proyección**")
    if yhat is not None and len(yhat) > 0:
        dfp = pd.DataFrame({"año": future_years, "valor": [fmt2(v) for v in yhat]})
        st.dataframe(dfp, use_container_width=True, hide_index=True)
    else:
        st.caption("—")

# Resumen
if not hist_vis.empty:
    n = len(hist_vis)
    vmin = float(np.nanmin(hist_vis["valor"]))
    vmax = float(np.nanmax(hist_vis["valor"]))
    vprom = float(np.nanmean(hist_vis["valor"]))
    st.markdown(f'<div class="small-note">n={n} · min={fmt2(vmin)} · prom={fmt2(vprom)} · max={fmt2(vmax)}</div>', unsafe_allow_html=True)

# --------------------------- Descarga serie (hist + proy) ---------------------------
if not hist.empty or (yhat is not None and len(yhat) > 0):
    full = []
    if not hist.empty:
        full.append(hist.assign(tipo="Histórico"))
    if yhat is not None and len(yhat) > 0:
        full.append(pd.DataFrame({"anio": future_years, "valor": yhat, "tipo": "Proyección"}))
    out = pd.concat(full, ignore_index=True)
    fname = f"serie_{canon(sel_res).replace(' ', '_')}_{canon(sel_ind).replace(' ', '_')}.csv"
    st.download_button("⬇️ Descargar serie (hist+proy)",
                       data=out.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv")

# --------------------------- Ayuda ---------------------------
with st.expander("Ayuda"):
    st.markdown("""
- **Modelos**: Tendencia lineal, SES, Holt lineal y Naive con deriva (sin librerías externas).
- **Auto** selecciona el de menor MASE~ en backtest rolling 1 paso.
- **Shocks** aplican % sobre la proyección; puedes truncar a 0 si procede.
- **Actualizar**: sube un CSV en la barra lateral (se hace *upsert* y se limpian cachés).
- **Compartir**: la URL guarda `res`, `ind`, `ymin`, `ymax`.
""")
