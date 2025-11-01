# app.py — Dashboard de Indicadores (versión ejecutiva)
# Visual refinado + modelos series cortas + Auto condicional + export limpio
# ------------------------------------------------
# Datos esperados:
# - indicadores_flat.csv:  resultado, indicador, anio, valor  (normaliza año/year/value si vienen así)
# - (opcional) proyecciones_2026_2029.csv: anio_proy, valor_proy, [resultado], [indicador]

import os, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------- opcionales (no rompen si faltan) ----------
HAS_PM = False
try:
    import pmdarima as pm  # Auto-ARIMA opcional
    HAS_PM = True
except Exception:
    HAS_PM = False

HAS_SM = False
try:
    import statsmodels.tsa.holtwinters as sm_hw  # ETS opcional
    HAS_SM = True
except Exception:
    HAS_SM = False

HAS_KALEIDO = False
try:
    import plotly.io as pio  # PNG con kaleido
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

HAS_PDF = False
try:
    from fpdf import FPDF  # PDF simple
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# --------------------------- Config & Estilos ---------------------------
st.set_page_config(page_title="Dashboard de Indicadores", layout="wide")

PRIMARY = "#dc2626"  # rojo proyección
BLUE    = "#2563eb"  # azul histórico
MUTED   = "#64748b"  # gris texto suave
BAND    = "rgba(220,38,38,.12)"  # banda incertidumbre

st.markdown(f"""
<style>
.block-container {{ max-width: 1200px; }}
html, body, [class*="css"] {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}}
.header-sticky {{
  position: sticky; top: 0; z-index: 10; background: white;
  padding: 8px 0 10px 0; border-bottom: 1px solid #eee;
}}
h1.title {{ margin:0 0 2px 0; font-weight: 700; }}
.subtle {{ color:{MUTED}; font-size: 0.95rem; }}
.btnbar {{ display:flex; flex-wrap:wrap; gap:.5rem; margin:.75rem 0 0; }}
.btn {{
  border:1px solid #e5e7eb; background:#f8fafc; color:#334155; padding:.45rem .9rem; border-radius:999px;
}}
.btn.active {{ background:{PRIMARY}; color:white; border-color:{PRIMARY}; text-decoration: underline; }}
.kpi {{
  border:1px solid #e5e7eb; border-radius:14px; padding:14px; background: #fff;
}}
.kpi .label {{ color:{MUTED}; font-size: .9rem; margin-bottom: 2px; }}
.kpi .value {{ font-weight:700; font-size: 1.35rem; }}
.kpi .delta {{ color:{MUTED}; font-size: .85rem; }}
hr.divider {{ border:none; border-top:1px solid #eee; margin: 8px 0 12px 0; }}
.small-note {{ color:{MUTED}; }}
</style>
""", unsafe_allow_html=True)

# --------------------------- Helpers ---------------------------
def canon(s:str)->str:
    return re.sub(r"\s+"," ", str(s)).strip()

def read_csv_safe(path):
    if not os.path.exists(path): return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def ensure_cols(df, cols):
    return [c for c in cols if c not in df.columns]

def qp_set(**kwargs):
    try:
        st.query_params.update({k:str(v) for k,v in kwargs.items() if v is not None})
    except Exception:
        try: st.experimental_set_query_params(**{k:v for k,v in kwargs.items() if v is not None})
        except: pass

def qp_get(name, default=None, cast=str):
    try:
        v = st.query_params.get(name, None)
        if isinstance(v, list): v = v[0] if v else None
        return cast(v) if (v is not None and cast) else (v if v is not None else default)
    except Exception:
        return default

def _as_array(x, n: int):
    if x is None: return None
    x = np.asarray(x)
    if x.ndim == 0: return np.full(n, float(x))
    return x

# --------------------------- Métricas ---------------------------
def mase(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n < 2: return np.inf
    denom = np.mean(np.abs(np.diff(y_true)))
    if denom == 0: denom = 1e-9
    return np.mean(np.abs(y_true - y_pred)) / denom

def smape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom==0] = 1e-9
    return 100 * np.mean(2*np.abs(y_pred - y_true)/denom)

# --------------------------- Carga de datos (cache lectura) ---------------------------
@st.cache_data(show_spinner=False)
def load_data(hist_path:str, proy_path:str):
    hist = read_csv_safe(hist_path)
    if not hist.empty:
        rmap={}
        if "año" in hist.columns and "anio" not in hist.columns: rmap["año"]="anio"
        if "year" in hist.columns and "anio" not in hist.columns: rmap["year"]="anio"
        if "value" in hist.columns and "valor" not in hist.columns: rmap["value"]="valor"
        hist = hist.rename(columns=rmap)
        for c in ("resultado","indicador"):
            if c in hist.columns: hist[c]=hist[c].map(canon)
        if "anio" in hist.columns:  hist["anio"]=pd.to_numeric(hist["anio"], errors="coerce")
        if "valor" in hist.columns: hist["valor"]=pd.to_numeric(hist["valor"], errors="coerce")
        hist = hist.dropna(subset=["resultado","indicador","anio"]).sort_values(["resultado","indicador","anio"])
        hist = hist[["resultado","indicador","anio","valor"]]

    proy = read_csv_safe(proy_path)
    if not proy.empty:
        pmap={}
        if "anio" in proy.columns and "anio_proy" not in proy.columns: pmap["anio"]="anio_proy"
        if "valor" in proy.columns and "valor_proy" not in proy.columns: pmap["valor"]="valor_proy"
        proy = proy.rename(columns=pmap)
        for c in ("resultado","indicador"):
            if c in proy.columns: proy[c]=proy[c].map(canon)
        if "anio_proy" in proy.columns:  proy["anio_proy"]=pd.to_numeric(proy["anio_proy"], errors="coerce")
        if "valor_proy" in proy.columns: proy["valor_proy"]=pd.to_numeric(proy["valor_proy"], errors="coerce")
        keep=[c for c in ["resultado","indicador","anio_proy","valor_proy"] if c in proy.columns]
        proy = proy[keep].dropna(subset=["anio_proy","valor_proy"]).sort_values("anio_proy")

    return hist, proy

HIST_PATH = os.getenv("HIST_PATH", "indicadores_flat.csv")
PROY_PATH = os.getenv("PROY_PATH", "proyecciones_2026_2029.csv")
flat0, base_proy0 = load_data(HIST_PATH, PROY_PATH)

if "flat" not in st.session_state: st.session_state.flat = flat0.copy()
if "base_proy" not in st.session_state: st.session_state.base_proy = base_proy0.copy()
flat = st.session_state.flat
base_proy = st.session_state.base_proy

# --------------------------- Sidebar: actualizar histórico ---------------------------
with st.sidebar:
    st.markdown("### 🔁 Actualizar histórico")
    st.caption("CSV: `resultado, indicador, anio, valor`. Upsert por clave (resultado, indicador, anio).")
    up = st.file_uploader("Subir CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            miss = ensure_cols(df, ["resultado","indicador","anio","valor"])
            if miss:
                st.error(f"Faltan columnas: {miss}")
            else:
                for c in ("resultado","indicador"): df[c]=df[c].map(canon)
                df["anio"]=pd.to_numeric(df["anio"], errors="coerce")
                df["valor"]=pd.to_numeric(df["valor"], errors="coerce")
                df = df.dropna(subset=["resultado","indicador","anio"])
                key=["resultado","indicador","anio"]
                merged = pd.concat([flat, df]).drop_duplicates(subset=key, keep="last").sort_values(key)
                st.session_state.flat = merged
                flat = merged
                st.success("Histórico actualizado en memoria.")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
    if not flat.empty:
        st.download_button("⬇️ Descargar histórico actualizado", data=flat.to_csv(index=False).encode("utf-8"),
                           file_name="historico_actualizado.csv", mime="text/csv")

# --------------------------- Encabezado ---------------------------
st.markdown("""
<div class="header-sticky">
  <h1 class="title">Dashboard de Indicadores</h1>
  <div class="subtle">Selecciona un <b>Resultado</b> y un <b>Indicador</b>. La URL guarda tu vista para compartir.</div>
</div>
""", unsafe_allow_html=True)

if flat.empty:
    st.warning("No hay datos. Sube un CSV desde la barra lateral.")
    st.stop()

# --------------------------- Navegación por resultado (una sola) ---------------------------
resultados = sorted(flat["resultado"].unique())
res_qp = qp_get("res", resultados[0] if resultados else "", str)
if res_qp not in resultados: res_qp = resultados[0]

# barra de botones (píldoras funcionales)
st.write("")  # respiro
btn_cols = st.columns(min(6, max(1, len(resultados))))
sel_res = res_qp
for i, r in enumerate(resultados):
    with btn_cols[i % len(btn_cols)]:
        label = f"{'✅ ' if r==res_qp else ''}{r}"
        if st.button(label, use_container_width=True, key=f"resbtn_{i}"):
            sel_res = r
qp_set(res=sel_res)

# --------------------------- Controles de serie ---------------------------
df_res = flat[flat["resultado"]==sel_res].copy()
inds = sorted(df_res["indicador"].unique())
ind_qp = qp_get("ind", inds[0], str)
if ind_qp not in inds: ind_qp = inds[0]

c1, c2 = st.columns([3,2])
with c1:
    sel_ind = st.selectbox("Indicador", inds, index=inds.index(ind_qp))
with c2:
    yr_min = int(df_res["anio"].min()); yr_max = int(df_res["anio"].max())
    ymin, ymax = st.slider("Rango histórico a mostrar", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))
qp_set(ind=sel_ind, ymin=ymin, ymax=ymax)

# serie
hist = df_res.query("indicador == @sel_ind")[["anio","valor"]].dropna().sort_values("anio")
hist_vis = hist.query("@ymin <= anio <= @ymax")
years = hist["anio"].values
y     = hist["valor"].values

# --------------------------- Modelos (series cortas) ---------------------------
def ci_bootstrap(y, fitted, h, reps=400):
    resid = np.asarray(y) - np.asarray(fitted)
    resid = resid[~np.isnan(resid)]
    if resid.size == 0: return None, None
    draws = np.random.choice(resid, size=(reps, h), replace=True)
    p10 = np.percentile(draws, 10, axis=0)
    p90 = np.percentile(draws, 90, axis=0)
    return p10, p90

def proj_linear(y, years, fyears):
    if len(y) < 2:
        yhat = np.full(len(fyears), y[-1] if len(y) else np.nan); return yhat, y, None, None
    X  = years - years.min()
    b1, b0 = np.polyfit(X, y, 1)
    Xf = fyears - years.min()
    yhat = b0 + b1*Xf
    fitted = b0 + b1*X
    p10, p90 = ci_bootstrap(y, fitted, len(Xf))
    lo = yhat + p10 if p10 is not None else None
    hi = yhat + p90 if p90 is not None else None
    return yhat, fitted, lo, hi

def proj_linear_lastk(y, years, fyears, k=5):
    n = len(y); k = max(2, min(k, n))
    return proj_linear(y[-k:], years[-k:], fyears)

def proj_theilsen_np(y, years, fyears):
    x = years.astype(float); y = y.astype(float)
    n = len(y)
    if n < 2:
        yhat = np.full(len(fyears), y[-1] if n else np.nan); return yhat, y, None, None
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x[j]-x[i]
            if dx != 0:
                slopes.append((y[j]-y[i])/dx)
    m = np.median(slopes); b = np.median(y - m*x)
    yhat = m*(fyears) + b
    fitted = m*(x) + b
    p10, p90 = ci_bootstrap(y, fitted, len(fyears))
    lo = yhat + p10 if p10 is not None else None
    hi = yhat + p90 if p90 is not None else None
    return yhat, fitted, lo, hi

def proj_ses(y, years, fyears, alphas=(0.2,0.4,0.6,0.8)):
    best_ae = np.inf; best_fit=None
    for a in alphas:
        s=None; fitted=[]
        for val in y:
            s = val if s is None else a*val + (1-a)*s
            fitted.append(s)
        ae = np.mean(np.abs(y - np.array(fitted)))
        if ae < best_ae: best_ae, best_fit = ae, np.array(fitted)
    last_s = best_fit[-1]
    yhat = np.full(len(fyears), last_s, dtype=float)
    p10, p90 = ci_bootstrap(y, best_fit, len(fyears))
    lo = yhat + p10 if p10 is not None else None
    hi = yhat + p90 if p90 is not None else None
    return yhat, best_fit, lo, hi

def proj_holt(y, years, fyears, alphas=(0.2,0.4,0.6,0.8), betas=(0.2,0.4,0.6,0.8)):
    best = (np.inf, 0.6, 0.2, None, None)
    for a in alphas:
        for b in betas:
            l=None; t=0.0; fitted=[]
            for i, val in enumerate(y):
                if l is None:
                    l = val; t = (y[1]-y[0]) if len(y)>1 else 0.0
                prev_l = l
                l = a*val + (1-a)*(l + t)
                t = b*(l - prev_l) + (1-b)*t
                fitted.append(l + t)
            ae = np.mean(np.abs(y - np.array(fitted)))
            if ae < best[0]: best = (ae, a, b, l, t)
    a, b, l, t = best[1], best[2], best[3], best[4]
    yhat = np.array([l + (h+1)*t for h in range(len(fyears))], dtype=float)
    # recompute fitted
    l=None; t=0.0; fitted=[]
    for i, val in enumerate(y):
        if l is None:
            l = val; t = (y[1]-y[0]) if len(y)>1 else 0.0
        prev_l = l
        l = a*val + (1-a)*(l + t)
        t = b*(l - prev_l) + (1-b)*t
        fitted.append(l + t)
    fitted = np.array(fitted)
    p10, p90 = ci_bootstrap(y, fitted, len(fyears))
    lo = yhat + p10 if p10 is not None else None
    hi = yhat + p90 if p90 is not None else None
    return yhat, fitted, lo, hi

def proj_drift(y, years, fyears):
    if len(y) < 2:
        yhat = np.full(len(fyears), y[-1] if len(y) else np.nan); return yhat, y, None, None
    slope = (y[-1]-y[0])/(len(y)-1)
    yhat = np.array([y[-1] + (h+1)*slope for h in range(len(fyears))], dtype=float)
    trend_fit = np.linspace(y[0], y[-1], len(y))
    p10, p90 = ci_bootstrap(y, trend_fit, len(fyears))
    lo = yhat + p10 if p10 is not None else None
    hi = yhat + p90 if p90 is not None else None
    return yhat, trend_fit, lo, hi

def proj_arima_pm(y, years, fyears):
    if not HAS_PM or len(y) < 6: return None, None, None, None
    try:
        m = pm.auto_arima(y, start_p=0, start_q=0, max_p=2, max_q=2,
                          start_d=0, max_d=2, seasonal=False, stepwise=True,
                          information_criterion='aicc', suppress_warnings=True,
                          error_action='ignore', maxiter=50)
        h = len(fyears)
        if h <= 0: return None, None, None, None
        fc, confint = m.predict(n_periods=h, return_conf_int=True, alpha=0.20)
        fitted = m.predict_in_sample()
        lo = confint[:,0]; hi = confint[:,1]
        return np.array(fc), np.array(fitted), lo, hi
    except Exception:
        return None, None, None, None

def proj_ets_hw(y, years, fyears):
    if not HAS_SM or len(y) < 6: return None, None, None, None
    try:
        model = sm_hw.ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        h = len(fyears)
        yhat = fit.forecast(h)
        fitted = fit.fittedvalues
        p10, p90 = ci_bootstrap(y, fitted, h)
        lo = yhat + p10 if p10 is not None else None
        hi = yhat + p90 if p90 is not None else None
        return np.array(yhat), np.array(fitted), lo, hi
    except Exception:
        return None, None, None, None

def backtest_metric(y, years, k, project_once):
    if len(y) <= k: return np.inf
    preds, reals = [], []
    for cut in range(len(y)-k, len(y)):
        y_tr, yr_tr = y[:cut], years[:cut]
        target_x = np.array([years[cut]])
        try:
            pred, *_ = project_once(y_tr, yr_tr, target_x)
            pv = float(pred[0]) if pred is not None and len(pred)>0 else np.nan
        except Exception:
            pv = np.nan
        preds.append(pv); reals.append(float(y[cut]))
    preds = np.array(preds); reals = np.array(reals)
    mask = ~np.isnan(preds)
    if mask.sum()==0: return np.inf
    preds = preds[mask]; reals = reals[mask]
    return 0.7*mase(reals, preds) + 0.3*(smape(reals, preds)/100.0)

def project_with_transform(y, years, fyears, base_project_fn, use_log=False, **kwargs):
    use_log = (use_log and np.all(y > 0))
    if use_log:
        yt = np.log(y)
        yhat_t, fitted_t, lo_t, hi_t = base_project_fn(yt, years, fyears, **kwargs)
        nF = len(fyears)
        if yhat_t is None: return None, None, None, None
        yhat = np.exp(np.asarray(yhat_t))
        fitted = np.exp(np.asarray(fitted_t)) if fitted_t is not None else None
        lo = np.exp(_as_array(lo_t, nF)) if lo_t is not None else None
        hi = np.exp(_as_array(hi_t, nF)) if hi_t is not None else None
        return yhat, fitted, lo, hi
    else:
        return base_project_fn(y, years, fyears, **kwargs)

def model_runner(name, y, years, fyears, transform, k_lin):
    use_log = (transform.startswith("Log"))
    if name == "Tendencia lineal":
        return project_with_transform(y, years, fyears, proj_linear, use_log=use_log)
    if name == "Tendencia lineal (últimos k años)":
        return project_with_transform(y, years, fyears, proj_linear_lastk, use_log=use_log, k=k_lin)
    if name == "Mediana de pendientes (Theil–Sen)":
        return project_with_transform(y, years, fyears, proj_theilsen_np, use_log=use_log)
    if name == "SES (suavizamiento exponencial simple)":
        return project_with_transform(y, years, fyears, proj_ses, use_log=use_log)
    if name == "Holt lineal (doble exponencial)":
        return project_with_transform(y, years, fyears, proj_holt, use_log=use_log)
    if name == "Naive con deriva":
        return project_with_transform(y, years, fyears, proj_drift, use_log=use_log)
    if name == "Auto-ARIMA (si disponible)":
        return project_with_transform(y, years, fyears, proj_arima_pm, use_log=False)
    if name == "ETS/Holt-Winters (si disponible)":
        return project_with_transform(y, years, fyears, proj_ets_hw, use_log=use_log)
    return project_with_transform(y, years, fyears, proj_linear, use_log=use_log)

def auto_select(y, years, k, transform, k_lin):
    cands = [
        "Tendencia lineal",
        "Tendencia lineal (últimos k años)",
        "Mediana de pendientes (Theil–Sen)",
        "SES (suavizamiento exponencial simple)",
        "Holt lineal (doble exponencial)",
        "Naive con deriva",
    ]
    if HAS_PM and len(y) >= 8:
        cands.append("Auto-ARIMA (si disponible)")
    if HAS_SM and len(y) >= 8:
        cands.append("ETS/Holt-Winters (si disponible)")

    scores = []
    for m in cands:
        def once(tr_y, tr_x, target_x, mm=m):
            return model_runner(mm, tr_y, tr_x, target_x, transform, k_lin)
        score = backtest_metric(y, years, k, once)
        scores.append((m, score))
    scores.sort(key=lambda t: t[1])
    return scores[0][0], scores

# --------------------------- UI Proyección (controles plegados) ---------------------------
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
st.subheader("Proyección", divider="gray")

yr_max = int(hist["anio"].max()) if not hist.empty else 2030
yr_end_default = min(yr_max+9, yr_max+30)

left, right = st.columns([2,2])
with left:
    modelos_lista = [
        "Auto (mejor MASE)",
        "Tendencia lineal",
        "Tendencia lineal (últimos k años)",
        "Mediana de pendientes (Theil–Sen)",
        "SES (suavizamiento exponencial simple)",
        "Holt lineal (doble exponencial)",
        "Naive con deriva",
    ]
    if HAS_PM: modelos_lista += ["Auto-ARIMA (si disponible)"]
    if HAS_SM: modelos_lista += ["ETS/Holt-Winters (si disponible)"]
    modelo = st.selectbox("Modelo", modelos_lista)

with right:
    transform = st.selectbox("Transformación", ["Ninguna", "Log (positiva)"])

with st.expander("Opciones avanzadas", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        year_end = st.number_input("Año final de proyección", min_value=yr_max, max_value=yr_max+30, value=yr_end_default, step=1)
    with c2:
        back_k = st.slider("Backtest (años para Auto)", min_value=3, max_value=min(8, max(3, len(hist))), value=min(5, max(3, len(hist))))
    with c3:
        k_lin = st.slider("k para Lineal últimos k", min_value=3, max_value=max(3, len(hist)), value=min(5, max(3, len(hist))))
    c4, c5 = st.columns(2)
    with c4:
        clamp0 = st.checkbox("No-negativos", value=True)
    with c5:
        show_ci = st.checkbox("Mostrar banda de incertidumbre", value=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        shock_up = st.slider("🔼 Escenario optimista (%)", -20, 100, 0)
    with sc2:
        shock_dn = st.slider("🔽 Escenario pesimista (%)", -100, 20, 0)

# horizonte futuro robusto
if len(hist)>0:
    steps = max(int(year_end - years.max()), 0)
    future_years = np.arange(years.max()+1, years.max()+1+steps) if steps>0 else np.array([], dtype=int)
else:
    future_years = np.array([], dtype=int)

# --------------------------- Ejecutar modelado ---------------------------
yhat = None; ci_lo=None; ci_hi=None; chosen = modelo
if len(future_years)>0 and len(y)>=1:
    if modelo == "Auto (mejor MASE)":
        chosen, _ = auto_select(y, years, back_k, transform, k_lin)
    yhat, fitted, ci_lo, ci_hi = model_runner(chosen, y, years, future_years, transform, k_lin)

    # shocks y clamps
    nF = len(future_years)
    if yhat is not None:
        yhat = np.asarray(yhat, dtype=float)
        ci_lo = _as_array(ci_lo, nF)
        ci_hi = _as_array(ci_hi, nF)
        if shock_up:
            yhat *= (1 + shock_up/100.0)
            if ci_lo is not None: ci_lo *= (1 + shock_up/100.0)
            if ci_hi is not None: ci_hi *= (1 + shock_up/100.0)
        if shock_dn:
            yhat *= (1 + shock_dn/100.0)
            if ci_lo is not None: ci_lo *= (1 + shock_dn/100.0)
            if ci_hi is not None: ci_hi *= (1 + shock_dn/100.0)
        if clamp0:
            yhat = np.maximum(yhat, 0.0)
            if ci_lo is not None: ci_lo = np.maximum(ci_lo, 0.0)
        if (ci_lo is not None) and (ci_hi is not None):
            # asegurar orden y longitudes
            ci_lo = ci_lo[:nF]; ci_hi = ci_hi[:nF]
            swap = ci_lo > ci_hi
            if np.any(swap):
                lo_new = np.where(swap, ci_hi, ci_lo)
                hi_new = np.where(swap, ci_lo, ci_hi)
                ci_lo, ci_hi = lo_new, hi_new
else:
    chosen = "—"

# --------------------------- Subtítulo dinámico ---------------------------
periodo_txt = "—"
if not hist.empty:
    periodo_txt = f"{int(hist['anio'].min())}–{int(hist['anio'].max())}"
st.caption(f"**Vista:** {sel_res} › {sel_ind} · **Período:** {periodo_txt}")

# --------------------------- KPIs (tarjetas) ---------------------------
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown("<div class='kpi'><div class='label'>Último valor (hist.)</div>", unsafe_allow_html=True)
    if len(y)>0:
        last = y[-1]; prev = y[-2] if len(y)>1 else np.nan
        st.markdown(f"<div class='value'>{last:.2f}</div><div class='delta'>{'' if np.isnan(prev) else f'Δ {last-prev:+.2f} vs. año previo'}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='value'>–</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='kpi'><div class='label'>Primera proyección</div>", unsafe_allow_html=True)
    if yhat is not None and len(yhat)>0:
        st.markdown(f"<div class='value'>{yhat[0]:.2f}</div><div class='delta'>para {future_years[0] if len(future_years)>0 else '—'}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='value'>–</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='kpi'><div class='label'>Modelo elegido</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{chosen}</div><div class='delta'></div></div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='kpi'><div class='label'>Backtest (MASE~)</div>", unsafe_allow_html=True)
    if len(y) >= max(4, back_k + 1) and chosen not in ("—", None):
        def chosen_once(tr_y, tr_x, target_x):
            return model_runner(chosen, tr_y, tr_x, target_x, transform, k_lin)
        val = backtest_metric(y, years, back_k, chosen_once)
        st.markdown(f"<div class='value'>{val:.3f}</div><div class='delta'>últ. {back_k} años</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='value'>—</div></div>", unsafe_allow_html=True)

# --------------------------- Gráfico (con anotaciones) ---------------------------
fig = go.Figure()
if not hist.empty:
    fig.add_trace(go.Scatter(
        x=hist["anio"], y=hist["valor"],
        mode="lines+markers", name="Histórico",
        line=dict(color=BLUE, width=2)
    ))

if yhat is not None and len(yhat)>0:
    fig.add_trace(go.Scatter(
        x=future_years, y=yhat, mode="lines+markers",
        name="Proyección", line=dict(color=PRIMARY, width=3, dash="dash")
    ))
    if show_ci and ci_lo is not None and ci_hi is not None:
        fig.add_trace(go.Scatter(x=future_years, y=ci_hi, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future_years, y=ci_lo, mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor=BAND,
                                 showlegend=False, hoverinfo="skip"))

# línea vertical de corte y anotaciones
if len(y)>0:
    x_cut = years.max() + 0.02
    fig.add_vline(x=x_cut, line_width=1, line_dash="dot", line_color="#94a3b8")
    # anotación último valor histórico
    fig.add_annotation(
        x=years.max(), y=y[-1],
        text=f"Último hist.: {y[-1]:.2f}",
        showarrow=True, arrowhead=2, arrowcolor="#475569", ax=30, ay=-30,
        font=dict(size=11, color="#111827"), bgcolor="rgba(255,255,255,.8)"
    )
    # anotación primera proyección
    if yhat is not None and len(yhat)>0:
        fig.add_annotation(
            x=future_years[0], y=yhat[0],
            text=f"Proy. {future_years[0]}: {yhat[0]:.2f}",
            showarrow=True, arrowhead=2, arrowcolor=PRIMARY, ax=-30, ay=-30,
            font=dict(size=11, color="#111827"), bgcolor="rgba(255,255,255,.85)"
        )

fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="año", yaxis_title="valor",
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------- Exportaciones (fila compacta) ---------------------------
ex1, ex2, ex3, ex4 = st.columns([1,1,1,2])
with ex1:
    if st.button("🔗 Copiar enlace actual"):
        st.toast("Copia la URL del navegador: ya incluye tus filtros (res/ind/ymin/ymax).")
with ex2:
    if HAS_KALEIDO:
        png_name = f"grafico_{canon(sel_res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.png"
        if st.button("🖼️ Exportar PNG"):
            try:
                pio.write_image(fig, png_name, scale=2, engine="kaleido")
                with open(png_name, "rb") as f:
                    st.download_button("Descargar PNG", data=f.read(), file_name=png_name, mime="image/png")
            except Exception as e:
                st.toast(f"No se pudo exportar PNG: {e}")
    else:
        st.caption("PNG: instala `kaleido`", help="pip install -U kaleido")
with ex3:
    if HAS_PDF:
        if st.button("📄 Exportar PDF (lámina)"):
            try:
                img_path = None
                if HAS_KALEIDO:
                    img_path = "tmp_plot.png"
                    pio.write_image(fig, img_path, scale=2, engine="kaleido")
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, f"Resultado: {sel_res} — Indicador: {sel_ind}", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 6, f"Período: {periodo_txt}", ln=True)
                pdf.cell(0, 6, f"Modelo: {chosen}", ln=True)
                if img_path and os.path.exists(img_path):
                    pdf.ln(4); pdf.image(img_path, x=10, y=40, w=190); pdf.set_y(140)
                pdf.ln(60)
                pdf.set_font("Arial", "B", 11); pdf.cell(0, 8, "Tabla resumida", ln=True)
                pdf.set_font("Arial", "", 9)
                tbl = []
                htail = hist.tail(6).copy(); htail["tipo"]="Histórico"; tbl.append(htail)
                if (yhat is not None) and len(yhat)>0:
                    pf = pd.DataFrame({"anio": future_years, "valor": yhat}).head(4)
                    pf["tipo"]="Proyección"; tbl.append(pf)
                if tbl:
                    t = pd.concat(tbl, ignore_index=True)
                    for _,r in t.iterrows():
                        pdf.cell(0, 6, f"{int(r['anio'])}: {r['valor']:.2f} ({r['tipo']})", ln=True)
                pdf.ln(4); pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 5, "Banda ≈80% (bootstrap). Métrica: MASE (backtest 1-paso).", ln=True)
                outpdf = f"reporte_{canon(sel_res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.pdf"
                pdf.output(outpdf)
                with open(outpdf, "rb") as f:
                    st.download_button("Descargar PDF", data=f.read(), file_name=outpdf, mime="application/pdf")
            except Exception as e:
                st.toast(f"No se pudo exportar PDF: {e}")
    else:
        st.caption("PDF: instala `fpdf2`", help="pip install -U fpdf2")
with ex4:
    if not hist.empty:
        full = []
        full.append(hist.assign(tipo="Histórico"))
        if yhat is not None and len(yhat)>0:
            full.append(pd.DataFrame({"anio": future_years, "valor": yhat, "tipo":"Proyección"}))
        out = pd.concat(full, ignore_index=True)
        fname = f"serie_{canon(sel_res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.csv"
        st.download_button("⬇️ Descargar serie (hist+proy)", data=out.to_csv(index=False).encode("utf-8"),
                           file_name=fname, mime="text/csv")

# --------------------------- Datos (plegado, limpio) ---------------------------
with st.expander("Datos (tablas y resumen)"):
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Histórico (rango visible)**")
        show = hist_vis.rename(columns={"anio":"año","valor":"valor"})
        st.dataframe(show.style.format({"valor":"{:,.2f}"}), use_container_width=True, hide_index=True)
        if not hist_vis.empty:
            stats = hist_vis["valor"].agg(["count","min","mean","max"]).rename({"count":"n","mean":"prom"})
            st.caption(f"n={int(stats['n'])} · min={stats['min']:.2f} · prom={stats['prom']:.2f} · max={stats['max']:.2f}")
    with t2:
        st.markdown("**Proyección**")
        if yhat is not None and len(yhat)>0:
            dfp = pd.DataFrame({"año": future_years, "valor": yhat})
            st.dataframe(dfp.style.format({"valor":"{:,.2f}"}), use_container_width=True, hide_index=True)
        else:
            st.caption("Sin proyección para los parámetros actuales.")

# --------------------------- Notas finales (no intrusivas) ---------------------------
with st.expander("Ayuda"):
    st.markdown("""
- **Auto (mejor MASE)** compara varios modelos ligeros y, si están disponibles, ARIMA/ETS. Selección por backtest rolling 1-paso (MASE).
- **Transformación Log** solo si todos los valores son positivos; estabiliza varianza en algunas series.
- **Banda de incertidumbre**: bootstrap de residuales (≈ IC 80%).
- **No-negativos**: trunca proyecciones por debajo de 0 si aplica.
- **Compartir**: la URL almacena `res`, `ind`, `ymin`, `ymax`.
""")
