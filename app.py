# app.py — Dashboard v3 (sin sklearn/statsmodels)
# ------------------------------------------------
# Requisitos: streamlit, pandas, numpy, plotly (normalmente ya están en Streamlit Cloud)
#
# Datos esperados:
#   - indicadores_flat.csv  -> columnas: resultado, indicador, anio, valor
#   - (opcional) proyecciones_2026_2029.csv -> anio_proy, valor_proy, [resultado], [indicador]
#
# Funciones clave:
#   - Navegación visible por "resultado" (botones) + selectbox de indicador
#   - Modelos de proyección SIN dependencias externas:
#       1) Tendencia lineal (np.polyfit)
#       2) Suavizamiento exponencial simple (SES) con búsqueda pequeña de alpha
#       3) Holt lineal (doble exponencial) implementado a mano con búsqueda pequeña de (alpha, beta)
#       4) Naive con deriva (drift)
#       5) Auto (escoge el mejor MAE en backtest rolling)
#   - Intervalo de confianza aproximado (80%) con residuales
#   - Escenarios de shocks (%), y opción de truncar en cero
#   - Actualizar histórico via sidebar (subes CSV y hace upsert por llave)
#   - Permalink en la URL (res, ind, ymin, ymax)

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------- Config & Estilos ---------------------------
st.set_page_config(page_title="Dashboard de Indicadores", layout="wide", initial_sidebar_state="collapsed")

PRIMARY = "#dc2626"  # rojo
BLUE    = "#2563eb"  # azul

st.markdown(f"""
<style>
.block-container {{ max-width: 1200px; }}
html, body, [class*="css"]  {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
.header-sticky {{
  position: sticky; top: 0; z-index: 10; background: white; padding: 8px 0 6px 0; border-bottom: 1px solid #eee;
}}
.btn-row {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0 1rem; }}
.btn-row button {{
  border:1px solid #e5e7eb; background:#f8fafc; color:#334155; padding:.45rem .9rem; border-radius:999px;
}}
.btn-row .active {{ background:{PRIMARY}; color:white; border-color:{PRIMARY}; }}
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

# --------------------------- Carga de datos (cache) ---------------------------
@st.cache_data(show_spinner=False)
def load_data(hist_path:str, proy_path:str):
    # histórico
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

    # proyecciones base opcionales
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

# Estado vivo (permite actualizar por upload)
if "flat" not in st.session_state: st.session_state.flat = flat0.copy()
if "base_proy" not in st.session_state: st.session_state.base_proy = base_proy0.copy()
flat = st.session_state.flat
base_proy = st.session_state.base_proy

# --------------------------- Sidebar: actualizar histórico ---------------------------
with st.sidebar:
    st.markdown("### 🔁 Actualizar histórico")
    st.caption("CSV con columnas: `resultado, indicador, anio, valor`. Se hace **upsert** por clave (resultado, indicador, anio).")
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
  <small style="color:#64748b">Elige un <b>resultado</b> y luego un indicador. La URL guarda tu selección para compartir.</small>
</div>
""", unsafe_allow_html=True)

if flat.empty:
    st.warning("No hay datos. Sube un CSV desde la barra lateral.")
    st.stop()

# --------------------------- Navegación por resultado (botones) ---------------------------
resultados = sorted(flat["resultado"].unique())
res_qp = qp_get("res", resultados[0] if resultados else "", str)
if res_qp not in resultados: res_qp = resultados[0]

cols = st.columns(len(resultados)) if len(resultados)<=6 else st.columns(6)
sel_res = res_qp
for i, r in enumerate(resultados):
    tgt_col = cols[i % len(cols)]
    with tgt_col:
        if st.button(r, type=("primary" if r==res_qp else "secondary"), use_container_width=True, key=f"resbtn_{i}"):
            sel_res = r
st.markdown(
    "".join([f'<span class="{"btn-row"}"></span>']), unsafe_allow_html=True
)
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

# serie
hist = df_res.query("indicador == @sel_ind")[["anio","valor"]].dropna().sort_values("anio")
hist_vis = hist.query("@ymin <= anio <= @ymax")

# --------------------------- Modelos de proyección (sin deps) ---------------------------
st.markdown("---")
st.markdown("#### Proyección")

left, mid, right, more = st.columns([2,2,2,2])
with left:
    modelo = st.selectbox("Modelo", [
        "Auto (mejor MAE)",
        "Tendencia lineal",
        "SES (suavizamiento exponencial simple)",
        "Holt lineal (doble exponencial)",
        "Naive con deriva"
    ])
with mid:
    year_end = st.number_input("Año final de proyección", min_value=yr_max, max_value=yr_max+30, value=min(yr_max+9, yr_max+30), step=1)
with right:
    back_k = st.slider("Backtest (años MAE)", min_value=3, max_value=min(8, len(hist)), value=min(5, len(hist)))
with more:
    clamp0 = st.checkbox("No-negativos", value=True)

sc1, sc2 = st.columns(2)
with sc1:
    shock_up = st.slider("🔼 Optimista (%)", -20, 100, 0)
with sc2:
    shock_dn = st.slider("🔽 Pesimista (%)", -100, 20, 0)

years = hist["anio"].values
y     = hist["valor"].values
future_years = np.arange(years.max()+1, year_end+1) if len(hist)>0 else np.array([], dtype=int)

def ci80_from_residuals(resid, n_future):
    # +/- 1.28 * std(resid) ~ 80%
    if resid is None or len(resid)==0: return 0.0
    return 1.28 * np.nanstd(resid)

# 1) Tendencia lineal (np.polyfit)
def proj_linear(y, years, fyears):
    if len(y) < 2: return np.array([]), None, None, None
    X = years - years.min()
    b1, b0 = np.polyfit(X, y, 1)  # y ≈ b0 + b1*X
    Xf = fyears - years.min()
    yhat = b0 + b1*Xf
    fitted = b0 + b1*X
    resid = y - fitted
    half = ci80_from_residuals(resid, len(Xf))
    return yhat, fitted, (yhat-half), (yhat+half)

# 2) SES
def proj_ses(y, years, fyears, alphas=(0.2,0.4,0.6,0.8)):
    best_ae = np.inf; best_alpha=0.6; best_fit=None
    for a in alphas:
        # fit
        s = None; fitted=[]
        for val in y:
            s = val if s is None else a*val + (1-a)*s
            fitted.append(s)
        ae = np.mean(np.abs(y - np.array(fitted)))
        if ae < best_ae: best_ae, best_alpha, best_fit = ae, a, np.array(fitted)
    # forecast: repetir último estado
    last_s = best_fit[-1]
    yhat = np.full(len(fyears), last_s, dtype=float)
    resid = y - best_fit
    half = ci80_from_residuals(resid, len(fyears))
    return yhat, best_fit, (yhat-half), (yhat+half)

# 3) Holt lineal (doble exponencial, aditivo)
def proj_holt(y, years, fyears, alphas=(0.2,0.4,0.6,0.8), betas=(0.2,0.4,0.6,0.8)):
    best = (np.inf, 0.6, 0.2, None, None)  # ae, alpha, beta, level, trend
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
                fitted.append(l + t)  # 1-step ahead
            ae = np.mean(np.abs(y - np.array(fitted)))
            if ae < best[0]:
                best = (ae, a, b, l, t)
    # forecast h pasos
    a, b, l, t = best[1], best[2], best[3], best[4]
    yhat = np.array([l + (h+1)*t for h in range(len(fyears))], dtype=float)
    # recompute fitted for residuals with best params
    l=None; t=0.0; fitted=[]
    for i, val in enumerate(y):
        if l is None:
            l = val; t = (y[1]-y[0]) if len(y)>1 else 0.0
        prev_l = l
        l = a*val + (1-a)*(l + t)
        t = b*(l - prev_l) + (1-b)*t
        fitted.append(l + t)
    resid = y - np.array(fitted)
    half = ci80_from_residuals(resid, len(fyears))
    return yhat, np.array(fitted), (yhat-half), (yhat+half)

# 4) Naive con deriva (drift)
def proj_drift(y, years, fyears):
    if len(y) < 2:  # fallback a naive
        yhat = np.full(len(fyears), y[-1] if len(y) else np.nan)
        return yhat, y, None, None
    slope = (y[-1]-y[0])/(len(y)-1)
    yhat = np.array([y[-1] + (h+1)*slope for h in range(len(fyears))], dtype=float)
    # residuos ~ diferencia vs. línea entre extremos (aprox)
    trend_fit = np.linspace(y[0], y[-1], len(y))
    resid = y - trend_fit
    half = ci80_from_residuals(resid, len(fyears))
    return yhat, trend_fit, (yhat-half), (yhat+half)

# Backtest (one-step-ahead, últimos k años)
def backtest_mae(y, years, k, method_name):
    if len(y) <= k: return np.inf
    maes=[]
    for cut in range(len(y)-k, len(y)):
        y_tr = y[:cut]; yr_tr = years[:cut]
        target_year = np.array([years[cut]])
        try:
            if method_name == "Tendencia lineal":
                yhat,_,_,_ = proj_linear(y_tr, yr_tr, target_year)
            elif method_name == "SES (suavizamiento exponencial simple)":
                yhat,_,_,_ = proj_ses(y_tr, yr_tr, target_year)
            elif method_name == "Holt lineal (doble exponencial)":
                yhat,_,_,_ = proj_holt(y_tr, yr_tr, target_year)
            elif method_name == "Naive con deriva":
                yhat,_,_,_ = proj_drift(y_tr, yr_tr, target_year)
            else:
                yhat,_,_,_ = proj_linear(y_tr, yr_tr, target_year)
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

# Ejecutar proyección
yhat = None; ci_lo=None; ci_hi=None; chosen = modelo
if len(future_years)>0 and len(y)>=1:
    if modelo == "Auto (mejor MAE)":
        chosen, _scores = auto_select(y, years, back_k)
    if chosen == "Tendencia lineal":
        yhat, fitted, ci_lo, ci_hi = proj_linear(y, years, future_years)
    elif chosen == "SES (suavizamiento exponencial simple)":
        yhat, fitted, ci_lo, ci_hi = proj_ses(y, years, future_years)
    elif chosen == "Holt lineal (doble exponencial)":
        yhat, fitted, ci_lo, ci_hi = proj_holt(y, years, future_years)
    elif chosen == "Naive con deriva":
        yhat, fitted, ci_lo, ci_hi = proj_drift(y, years, future_years)

    # shocks y truncado
    if yhat is not None:
        if shock_up: yhat = yhat * (1 + shock_up/100.0)
        if shock_dn: yhat = yhat * (1 + shock_dn/100.0)
        if clamp0:   yhat = np.maximum(yhat, 0.0)
        if ci_lo is not None and ci_hi is not None:
            if clamp0:
                ci_lo = np.maximum(ci_lo, 0.0)
            # asegurar orden
            swap = ci_lo > ci_hi
            if np.any(swap):
                tmp = ci_lo.copy(); ci_lo[swap]=ci_hi[swap]; ci_hi[swap]=tmp

# --------------------------- KPIs ---------------------------
k1,k2,k3 = st.columns(3)
if len(y)>0:
    last = y[-1]
    prev = y[-2] if len(y)>1 else np.nan
    k1.metric("Último valor (hist.)", f"{last:.2f}", None if np.isnan(prev) else f"{(last-prev):+.2f}")
else:
    k1.metric("Último valor (hist.)", "–")

if yhat is not None and len(yhat)>0:
    k2.metric("Primera proyección", f"{yhat[0]:.2f}")
    k3.metric("Modelo elegido", chosen)
else:
    k2.metric("Primera proyección", "–")
    k3.metric("Modelo elegido", "–")

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
fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  xaxis_title="año", yaxis_title="valor")
st.plotly_chart(fig, use_container_width=True)

# --------------------------- Tablas ---------------------------
t1,t2 = st.columns(2)
with t1:
    st.markdown("**Histórico (rango visible)**")
    st.dataframe(hist_vis.rename(columns={"anio":"año","valor":"valor"}), use_container_width=True, hide_index=True)
with t2:
    st.markdown("**Proyección**")
    if yhat is not None and len(yhat)>0:
        st.dataframe(pd.DataFrame({"año": future_years, "valor": yhat}), use_container_width=True, hide_index=True)
    else:
        st.caption("Sin proyección para los parámetros actuales.")

# --------------------------- MAE backtest ---------------------------
if len(y) >= back_k + 1:
    mae = backtest_mae(y, years, back_k, (chosen if chosen!="Auto (mejor MAE)" else "Tendencia lineal"))
    st.caption(f"**MAE** (backtest {back_k} años): **{mae:.2f}**")

# --------------------------- Descarga serie ---------------------------
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

# --------------------------- Ayuda ---------------------------
with st.expander("Ayuda"):
    st.markdown("""
- **Modelos sin dependencias externas**: lineal, SES, Holt lineal y Naive con deriva.
- **Auto** selecciona el que obtiene menor MAE en un backtest rolling de 1 paso.
- **Escenarios** aplican shocks porcentuales sobre la proyección base.
- **Actualizar**: usa la barra lateral para subir un CSV y actualizar el histórico (upsert).
- **Compartir**: la URL guarda `res`, `ind`, `ymin`, `ymax`.
""")

