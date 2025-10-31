# app.py — Dashboard v4 (series cortas + Auto condicional + export)
# ------------------------------------------------
# Requisitos base: streamlit, pandas, numpy, plotly
# Opcionales: kaleido (PNG), fpdf2 (PDF), pmdarima y/or statsmodels (ARIMA/ETS)
#
# Datos:
#   - indicadores_flat.csv: resultado, indicador, anio, valor  (normaliza año/year/value si vienen así)
#   - (opcional) proyecciones_2026_2029.csv: anio_proy, valor_proy, [resultado], [indicador]

import os, re, io, math, random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---- imports opcionales ----
HAS_PM = False
try:
    import pmdarima as pm  # AutoARIMA opcional
    HAS_PM = True
except Exception:
    HAS_PM = False

HAS_SM = False
try:
    import statsmodels.tsa.holtwinters as sm_hw  # ETS/Holt-Winters opcional
    HAS_SM = True
except Exception:
    HAS_SM = False

HAS_KALEIDO = False
try:
    import plotly.io as pio  # write_image necesita kaleido
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

HAS_PDF = False
try:
    from fpdf import FPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# --------------------------- Config & Estilos ---------------------------
st.set_page_config(page_title="Dashboard de Indicadores", layout="wide")

PRIMARY = "#dc2626"  # rojo
BLUE    = "#2563eb"  # azul

st.markdown(f"""
<style>
.block-container {{ max-width: 1200px; }}
html, body, [class*="css"]  {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
.header-sticky {{
  position: sticky; top: 0; z-index: 10; background: white; padding: 8px 0 6px 0; border-bottom: 1px solid #eee;
}}
.pills {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0 1rem; }}
.pill {{
  border:1px solid #e5e7eb; background:#f8fafc; color:#334155; padding:.45rem .9rem; border-radius:999px;
}}
.pill.active {{ background:{PRIMARY}; color:white; border-color:{PRIMARY}; text-decoration: underline; }}
.small-note {{ color:#64748b; }}
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
    try: st.query_params.update({k:str(v) for k,v in kwargs.items() if v is not None})
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

# --------------------------- Carga de datos (cache) ---------------------------
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
  <h1 style="margin:0 0 4px 0;">Dashboard de Indicadores</h1>
  <div class="small-note">Selecciona un <b>Resultado</b> y luego un <b>Indicador</b>. La URL guarda tu selección para compartir.</div>
</div>
""", unsafe_allow_html=True)

if flat.empty:
    st.warning("No hay datos. Sube un CSV desde la barra lateral.")
    st.stop()

# --------------------------- Navegación por resultado ---------------------------
resultados = sorted(flat["resultado"].unique())
res_qp = qp_get("res", resultados[0] if resultados else "", str)
if res_qp not in resultados: res_qp = resultados[0]

st.write("")  # espaciado
pill_html = []
for r in resultados:
    active = "active" if r==res_qp else ""
    pill_html.append(f'<span class="pill {active}">{r}</span>')
st.markdown(f'<div class="pills">{"".join(pill_html)}</div>', unsafe_allow_html=True)

# botones reales (para cambiar qp)
cols = st.columns(min(6,len(resultados)))
sel_res = res_qp
for i, r in enumerate(resultados):
    with cols[i % len(cols)]:
        if st.button(("✅ " if r==res_qp else "  ") + r, use_container_width=True, key=f"resbtn_{i}"):
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
    ymin, ymax = st.slider("Rango histórico a mostrar", min_value=yr_min, max_value=yr_max,
                           value=(yr_min, yr_max))
qp_set(ind=sel_ind, ymin=ymin, ymax=ymax)

hist = df_res.query("indicador == @sel_ind")[["anio","valor"]].dropna().sort_values("anio")
hist_vis = hist.query("@ymin <= anio <= @ymax")
years = hist["anio"].values
y     = hist["valor"].values

# --------------------------- Modelos ---------------------------
def ci_bootstrap(y, fitted, fyears, h, reps=400):
    # residual bootstrap (simple, no-block) -> IC p10/p90 ≈ 80%
    resid = np.asarray(y) - np.asarray(fitted)
    if resid.size == 0 or np.all(np.isnan(resid)): return None, None
    draws_hi = []
    draws_lo = []
    for _ in range(reps):
        noise = np.random.choice(resid[~np.isnan(resid)], size=h, replace=True)
        draws_hi.append(noise)
        draws_lo.append(noise)
    # se devolverán bandas alrededor de yhat fuera (se sumarán afuera si aplica)
    # aquí devolvemos solo el semiancho robusto por paso (DEPRECATED approach). Mejor devolvemos percentiles 10 y 90 de ruido 0-centrado
    noise_mat = np.stack(draws_hi, axis=0)  # (reps, h)
    p90 = np.percentile(noise_mat, 90, axis=0)
    p10 = np.percentile(noise_mat, 10, axis=0)
    return p10, p90

def proj_linear(y, years, fyears):
    if len(y) < 2: 
        yhat = np.full(len(fyears), y[-1] if len(y) else np.nan)
        return yhat, y, None, None
    X = years - years.min()
    b1, b0 = np.polyfit(X, y, 1)
    Xf = fyears - years.min()
    yhat = b0 + b1*Xf
    fitted = b0 + b1*X
    p10, p90 = ci_bootstrap(y, fitted, fyears, len(Xf))
    if p10 is not None:
        lo = yhat + p10; hi = yhat + p90
    else:
        lo = hi = None
    return yhat, fitted, lo, hi

def proj_linear_lastk(y, years, fyears, k=5):
    n = len(y)
    k = max(2, min(k, n))
    return proj_linear(y[-k:], years[-k:], fyears)

def proj_theilsen_np(y, years, fyears):
    x = years.astype(float); y = y.astype(float)
    n = len(y)
    if n < 2:
        yhat = np.full(len(fyears), y[-1] if n else np.nan)
        return yhat, y, None, None
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x[j]-x[i]
            if dx != 0:
                slopes.append((y[j]-y[i])/dx)
    m = np.median(slopes)
    b = np.median(y - m*x)
    yhat = m*(fyears) + b
    fitted = m*(x) + b
    p10, p90 = ci_bootstrap(y, fitted, fyears, len(fyears))
    if p10 is not None:
        lo = yhat + p10; hi = yhat + p90
    else:
        lo = hi = None
    return yhat, fitted, lo, hi

def proj_ses(y, years, fyears, alphas=(0.2,0.4,0.6,0.8)):
    best_ae = np.inf; best_fit=None
    for a in alphas:
        s = None; fitted=[]
        for val in y:
            s = val if s is None else a*val + (1-a)*s
            fitted.append(s)
        ae = np.mean(np.abs(y - np.array(fitted)))
        if ae < best_ae: best_ae, best_fit = ae, np.array(fitted)
    last_s = best_fit[-1]
    yhat = np.full(len(fyears), last_s, dtype=float)
    p10, p90 = ci_bootstrap(y, best_fit, fyears, len(fyears))
    if p10 is not None:
        lo = yhat + p10; hi = yhat + p90
    else:
        lo = hi = None
    return yhat, best_fit, lo, hi

def proj_holt(y, years, fyears, alphas=(0.2,0.4,0.6,0.8), betas=(0.2,0.4,0.6,0.8)):
    best = (np.inf, 0.6, 0.2, None, None)
    for a in alphas:
        for b in betas:
            l=None; t=0.0; fitted=[]
            for i, val in enumerate(y):
                if l is None:
                    l = val
                    t = (y[1]-y[0]) if len(y)>1 else 0.0
                prev_l = l
                l = a*val + (1-a)*(l + t)
                t = b*(l - prev_l) + (1-b)*t
                fitted.append(l + t)
            ae = np.mean(np.abs(y - np.array(fitted)))
            if ae < best[0]:
                best = (ae, a, b, l, t)
    a, b, l, t = best[1], best[2], best[3], best[4]
    yhat = np.array([l + (h+1)*t for h in range(len(fyears))], dtype=float)
    # recompute fitted for resid
    l=None; t=0.0; fitted=[]
    for i, val in enumerate(y):
        if l is None:
            l = val; t = (y[1]-y[0]) if len(y)>1 else 0.0
        prev_l = l
        l = a*val + (1-a)*(l + t)
        t = b*(l - prev_l) + (1-b)*t
        fitted.append(l + t)
    fitted = np.array(fitted)
    p10, p90 = ci_bootstrap(y, fitted, fyears, len(fyears))
    lo = yhat + p10 if p10 is not None else None
    hi = yhat + p90 if p90 is not None else None
    return yhat, fitted, lo, hi

def proj_drift(y, years, fyears):
    if len(y) < 2:
        yhat = np.full(len(fyears), y[-1] if len(y) else np.nan)
        return yhat, y, None, None
    slope = (y[-1]-y[0])/(len(y)-1)
    yhat = np.array([y[-1] + (h+1)*slope for h in range(len(fyears))], dtype=float)
    trend_fit = np.linspace(y[0], y[-1], len(y))
    p10, p90 = ci_bootstrap(y, trend_fit, fyears, len(fyears))
    lo = yhat + p10 if p10 is not None else None
    hi = yhat + p90 if p90 is not None else None
    return yhat, trend_fit, lo, hi

def proj_arima_pm(y, years, fyears):
    # Requiere pmdarima
    if not HAS_PM or len(y) < 6:
        return None, None, None, None
    try:
        # sin estacionalidad por defecto, grilla acotada
        m = pm.auto_arima(y, start_p=0, start_q=0, max_p=2, max_q=2,
                          start_d=0, max_d=2, seasonal=False, stepwise=True,
                          information_criterion='aicc', suppress_warnings=True,
                          error_action='ignore', maxiter=50)
        h = len(fyears)
        if h <= 0: return None, None, None, None
        fc, confint = m.predict(n_periods=h, return_conf_int=True, alpha=0.20)  # ~80%
        fitted = m.predict_in_sample()
        lo = confint[:,0]; hi = confint[:,1]
        return np.array(fc), np.array(fitted), lo, hi
    except Exception:
        return None, None, None, None

def proj_ets_hw(y, years, fyears):
    # Requiere statsmodels
    if not HAS_SM or len(y) < 6:
        return None, None, None, None
    try:
        # Holt-Winters sin estacionalidad explícita (trend additive)
        model = sm_hw.ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        h = len(fyears)
        yhat = fit.forecast(h)
        # statsmodels no da CI por defecto aquí -> usamos bootstrap alrededor de fitted
        fitted = fit.fittedvalues
        p10, p90 = ci_bootstrap(y, fitted, fyears, h)
        lo = yhat + p10 if p10 is not None else None
        hi = yhat + p90 if p90 is not None else None
        return np.array(yhat), np.array(fitted), lo, hi
    except Exception:
        return None, None, None, None

# backtest 1-step con callback de proyección
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
        preds.append(pv)
        reals.append(float(y[cut]))
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

# --------------------------- UI Proyección ---------------------------
st.markdown("---")
st.markdown("#### Proyección")

left, mid, right = st.columns([2,2,2])
with left:
    modelo = st.selectbox("Modelo", [
        "Auto (mejor MASE)",
        "Tendencia lineal",
        "Tendencia lineal (últimos k años)",
        "Mediana de pendientes (Theil–Sen)",
        "SES (suavizamiento exponencial simple)",
        "Holt lineal (doble exponencial)",
        "Naive con deriva",
        *(["Auto-ARIMA (si disponible)"] if HAS_PM else []),
        *(["ETS/Holt-Winters (si disponible)"] if HAS_SM else []),
    ])
with mid:
    yr_max = int(hist["anio"].max()) if not hist.empty else 2030
    year_end = st.number_input("Año final de proyección", min_value=yr_max, max_value=yr_max+30, value=min(yr_max+9, yr_max+30), step=1)
with right:
    transform = st.selectbox("Transformación", ["Ninguna", "Log (positiva)"])

with st.expander("Opciones avanzadas"):
    c1,c2,c3 = st.columns(3)
    with c1:
        back_k = st.slider("Backtest (años para Auto)", min_value=3, max_value=min(8, len(hist)), value=min(5, len(hist)))
    with c2:
        k_lin = st.slider("k para Lineal últimos k", min_value=3, max_value=max(3, len(hist)), value=min(5, max(3, len(hist))))
    with c3:
        clamp0 = st.checkbox("No-negativos", value=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        shock_up = st.slider("🔼 Escenario optimista (%)", -20, 100, 0)
    with sc2:
        shock_dn = st.slider("🔽 Escenario pesimista (%)", -100, 20, 0)

# Horizonte futuro robusto
if len(hist)>0:
    steps = max(int(year_end - years.max()), 0)
    future_years = np.arange(years.max()+1, years.max()+1+steps) if steps>0 else np.array([], dtype=int)
else:
    future_years = np.array([], dtype=int)

# --------------------------- Selección/Ejecución ---------------------------
def model_runner(name, y, years, fyears):
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

def auto_select(y, years, k):
    # define candidatos según disponibilidad y tamaño de serie
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
            return model_runner(mm, tr_y, tr_x, target_x)
        score = backtest_metric(y, years, k, once)
        scores.append((m, score))
    scores.sort(key=lambda t: t[1])
    return scores[0][0], scores

yhat = None; ci_lo=None; ci_hi=None; chosen = modelo
if len(future_years)>0 and len(y)>=1:
    if modelo == "Auto (mejor MASE)":
        chosen, _scores = auto_select(y, years, back_k)
    yhat, fitted, ci_lo, ci_hi = model_runner(chosen, y, years, future_years)

    # shocks, clamp y normalización
    nF = len(future_years)
    if yhat is not None:
        yhat = np.asarray(yhat, dtype=float)
        ci_lo = _as_array(ci_lo, nF)
        ci_hi = _as_array(ci_hi, nF)

        if shock_up:
            yhat = yhat * (1 + shock_up/100.0)
            if ci_lo is not None: ci_lo = ci_lo * (1 + shock_up/100.0)
            if ci_hi is not None: ci_hi = ci_hi * (1 + shock_up/100.0)
        if shock_dn:
            yhat = yhat * (1 + shock_dn/100.0)
            if ci_lo is not None: ci_lo = ci_lo * (1 + shock_dn/100.0)
            if ci_hi is not None: ci_hi = ci_hi * (1 + shock_dn/100.0)
        if clamp0:
            yhat = np.maximum(yhat, 0.0)
            if ci_lo is not None: ci_lo = np.maximum(ci_lo, 0.0)

        if (ci_lo is not None) and (ci_hi is not None):
            ci_lo = ci_lo[:nF]; ci_hi = ci_hi[:nF]
            swap = ci_lo > ci_hi
            if np.any(swap):
                lo_new = np.where(swap, ci_hi, ci_lo)
                hi_new = np.where(swap, ci_lo, ci_hi)
                ci_lo, ci_hi = lo_new, hi_new
else:
    chosen = "—"

# --------------------------- KPIs ---------------------------
k1,k2,k3,k4 = st.columns(4)
if len(y)>0:
    last = y[-1]
    prev = y[-2] if len(y)>1 else np.nan
    k1.metric("Último valor (hist.)", f"{last:.2f}", None if np.isnan(prev) else f"{(last-prev):+.2f}")
else:
    k1.metric("Último valor (hist.)", "–")

if yhat is not None and len(yhat)>0:
    k2.metric("Primera proyección", f"{yhat[0]:.2f}")
else:
    k2.metric("Primera proyección", "–")

k3.metric("Modelo elegido", chosen if chosen else "–")

# MASE backtest del modelo elegido (si aplica)
if len(y) >= back_k + 1 and chosen not in ("—", None):
    def chosen_once(tr_y, tr_x, target_x):
        return model_runner(chosen, tr_y, tr_x, target_x)
    val = backtest_metric(y, years, back_k, chosen_once)
    k4.metric("Backtest (MASE~)", f"{val:.3f}")
else:
    k4.metric("Backtest (MASE~)", "—")

# --------------------------- Gráfico ---------------------------
fig = go.Figure()
if not hist.empty:
    fig.add_trace(go.Scatter(x=hist["anio"], y=hist["valor"], mode="lines+markers",
                             name="Histórico", line=dict(color=BLUE)))
if yhat is not None and len(yhat)>0:
    fig.add_trace(go.Scatter(x=future_years, y=yhat, mode="lines+markers",
                             name="Proyección", line=dict(color=PRIMARY, dash="dash")))
    if ci_lo is not None and ci_hi is not None:
        fig.add_trace(go.Scatter(x=future_years, y=ci_hi, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future_years, y=ci_lo, mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor="rgba(220,38,38,.12)",
                                 showlegend=False, hoverinfo="skip"))
# línea vertical de corte
if len(y)>0:
    fig.add_vline(x=years.max()+0.02, line_width=1, line_dash="dot", line_color="#94a3b8")

fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  xaxis_title="año", yaxis_title="valor")
st.plotly_chart(fig, use_container_width=True)

# --------------------------- Exportaciones ---------------------------
colx, coly, colz = st.columns([1,1,2])
with colx:
    if HAS_KALEIDO and (not hist.empty):
        # export PNG del gráfico
        png_name = f"grafico_{canon(sel_res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.png"
        if st.button("🖼️ Exportar PNG"):
            try:
                pio.write_image(fig, png_name, scale=2, engine="kaleido")
                with open(png_name, "rb") as f:
                    st.download_button("Descargar PNG", data=f.read(), file_name=png_name, mime="image/png")
            except Exception as e:
                st.error(f"No se pudo exportar PNG: {e}")
    else:
        st.caption("Instala `kaleido` para exportar PNG.")

with coly:
    if HAS_PDF and (not hist.empty):
        if st.button("📄 Exportar PDF"):
            try:
                # Renderizamos imagen temporal del gráfico si está kaleido
                img_path = None
                if HAS_KALEIDO:
                    img_path = "tmp_plot.png"
                    pio.write_image(fig, img_path, scale=2, engine="kaleido")
                # PDF simple
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, f"Resultado: {sel_res} — Indicador: {sel_ind}", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 6, f"Periodo: {int(hist['anio'].min()) if not hist.empty else '—'}–{int(hist['anio'].max()) if not hist.empty else '—'}", ln=True)
                pdf.cell(0, 6, f"Modelo: {chosen}", ln=True)
                if img_path and os.path.exists(img_path):
                    pdf.ln(4)
                    pdf.image(img_path, x=10, y=40, w=190)
                    pdf.set_y(140)
                pdf.ln(60)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 8, "Tabla resumida", ln=True)
                pdf.set_font("Arial", "", 9)
                # últimas 6 filas de hist y primeras 4 de proy
                tbl = []
                htail = hist.tail(6).copy()
                htail["tipo"]="Histórico"
                tbl.append(htail)
                if yhat is not None and len(yhat)>0:
                    pf = pd.DataFrame({"anio": future_years, "valor": yhat}).head(4)
                    pf["tipo"]="Proyección"
                    tbl.append(pf)
                if tbl:
                    t = pd.concat(tbl, ignore_index=True)
                    for _,r in t.iterrows():
                        pdf.cell(0, 6, f"{int(r['anio'])}: {r['valor']:.2f} ({r['tipo']})", ln=True)
                pdf.ln(4)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 5, "Banda ≈80%. Métrica: MASE (backtest 1-paso).", ln=True)
                outpdf = f"reporte_{canon(sel_res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.pdf"
                pdf.output(outpdf)
                with open(outpdf, "rb") as f:
                    st.download_button("Descargar PDF", data=f.read(), file_name=outpdf, mime="application/pdf")
            except Exception as e:
                st.error(f"No se pudo exportar PDF: {e}")
    else:
        st.caption("Instala `fpdf2` (y opcional `kaleido`) para exportar PDF.")

# --------------------------- Tablas ---------------------------
with st.expander("Datos"):
    t1,t2 = st.columns(2)
    with t1:
        st.markdown("**Histórico (rango visible)**")
        st.dataframe(hist_vis.rename(columns={"anio":"año","valor":"valor"}), use_container_width=True, hide_index=True)
        if not hist_vis.empty:
            stats = hist_vis["valor"].agg(["count","min","mean","max"]).rename({"count":"n","mean":"prom"})
            st.caption(f"n={int(stats['n'])} · min={stats['min']:.2f} · prom={stats['prom']:.2f} · max={stats['max']:.2f}")
    with t2:
        st.markdown("**Proyección**")
        if yhat is not None and len(yhat)>0:
            dfp = pd.DataFrame({"año": future_years, "valor": yhat})
            st.dataframe(dfp, use_container_width=True, hide_index=True)
        else:
            st.caption("Sin proyección para los parámetros actuales.")

# --------------------------- Descarga serie CSV ---------------------------
if not hist.empty or (yhat is not None and len(yhat)>0):
    full = []
    if not hist.empty:
        full.append(hist.assign(tipo="Histórico"))
    if yhat is not None and len(yhat)>0:
        full.append(pd.DataFrame({"anio": future_years, "valor": yhat, "tipo":"Proyección"}))
    out = pd.concat(full, ignore_index=True)
    fname = f"serie_{canon(sel_res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.csv"
    st.download_button("⬇️ Descargar serie (hist+proy)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv")
