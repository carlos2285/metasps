# app.py — Dashboard de Indicadores (v2 Pro: navegación visible + proyecciones robustas + updater)
# ================================================================================================
# Requiere: streamlit, plotly, pandas, numpy, statsmodels, scikit-learn (para TheilSen).
# Datos esperados:
#   Historico: indicadores_flat.csv  -> columnas mínimas: resultado, indicador, anio, valor
#   (Opc.)    Proyecciones base: proyecciones_2026_2029.csv -> anio_proy, valor_proy, [resultado], [indicador]
#
# Si subes nuevos CSV desde la UI, se validan/mergen por (resultado, indicador, anio)
# y puedes descargar el CSV combinado.

import os, re, math, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, TheilSenRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# ============================== CONFIG ==============================
st.set_page_config(page_title="Dashboard de Indicadores", layout="wide", initial_sidebar_state="collapsed")

PRIMARY = "#dc2626"   # rojo
BLUE    = "#2563eb"   # azul
SLATE   = "#0f172a"

# Global CSS (mejor lectura + píldoras)
st.markdown(f"""
<style>
/* ancho y tipografía */
.main > div {{ padding-top: 0.6rem; }}
section[data-testid="stSidebar"] .st-emotion-cache-1rtdyuf {{ padding: 1rem 0.6rem; }}
html, body, [class*="css"]  {{ font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; }}

/* contenedor centrado */
.block-container {{ max-width: 1200px; }}

/* encabezado sticky */
.sticky {{
  position: sticky; top: 0; z-index: 50; background: white; padding-top: 10px; padding-bottom: 6px;
  border-bottom: 1px solid #eee;
}}

/* píldoras navegación */
.pills {{
  display: flex; flex-wrap: wrap; gap: .5rem; margin: .4rem 0 1rem 0;
}}
.pill {{
  border: 1px solid #e5e7eb; color: #334155; padding: .4rem .75rem; border-radius: 999px; cursor: pointer;
  background: #f8fafc; transition: all .15s ease;
}}
.pill:hover {{ transform: translateY(-1px); border-color: #d1d5db; }}
.pill.active {{ background: {PRIMARY}; color: white; border-color: {PRIMARY}; }}

small.help {{ color:#64748b; }}

/* botones principales */
button[kind="primary"] {{ background:{PRIMARY}; }}
</style>
""", unsafe_allow_html=True)

# ============================== HELPERS ==============================
def canon(s:str)->str:
    return re.sub(r"\s+", " ", str(s)).strip()

def clamp_nonneg(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0.0)

def ensure_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing

def read_csv_safe(path, required=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception as e:
            st.error(f"No se pudo leer {path}: {e}")
            return pd.DataFrame()
    if required:
        m = ensure_cols(df, required)
        if m: st.warning(f"Faltan columnas en {path}: {m}")
    return df

def persist_qp(**kwargs):
    qp = {k:str(v) for k,v in kwargs.items() if v is not None}
    try: st.query_params.update(qp)         # streamlit >= 1.32
    except Exception:
        try: st.experimental_set_query_params(**qp)
        except: pass

def get_qp(name, default=None, cast=str):
    try:
        val = st.query_params.get(name, None)
        if isinstance(val, list): val = val[0] if val else None
        if val is None: return default
        return cast(val) if cast else val
    except Exception:
        return default

# ============================== DATA LOAD ==============================
@st.cache_data(show_spinner=False)
def load_data(hist_path:str, proy_path:str):
    hist = read_csv_safe(hist_path)
    if not hist.empty:
        # map names
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
        prmap={}
        if "anio" in proy.columns and "anio_proy" not in proy.columns: prmap["anio"]="anio_proy"
        if "valor" in proy.columns and "valor_proy" not in proy.columns: prmap["valor"]="valor_proy"
        proy = proy.rename(columns=prmap)
        for c in ("resultado","indicador"):
            if c in proy.columns: proy[c]=proy[c].map(canon)
        if "anio_proy" in proy.columns:   proy["anio_proy"]=pd.to_numeric(proy["anio_proy"], errors="coerce")
        if "valor_proy" in proy.columns:  proy["valor_proy"]=pd.to_numeric(proy["valor_proy"], errors="coerce")
        keep = [c for c in ["resultado","indicador","anio_proy","valor_proy"] if c in proy.columns]
        proy = proy[keep].dropna(subset=["anio_proy","valor_proy"]).sort_values("anio_proy")

    return hist, proy

HIST_PATH = os.getenv("HIST_PATH", "indicadores_flat.csv")
PROY_PATH = os.getenv("PROY_PATH", "proyecciones_2026_2029.csv")
flat, base_proy = load_data(HIST_PATH, PROY_PATH)

if "flat" not in st.session_state: st.session_state.flat = flat.copy()
if "base_proy" not in st.session_state: st.session_state.base_proy = base_proy.copy()
flat = st.session_state.flat
base_proy = st.session_state.base_proy

# ============================== SIDEBAR: DATA UPDATER ==============================
with st.sidebar:
    st.markdown("### 🔁 Actualizar datos")
    st.caption("Sube un CSV de histórico para **reemplazar/actualizar filas** por llave `(resultado, indicador, anio)`.\nColumnas mínimas: `resultado, indicador, anio, valor`.")
    up_hist = st.file_uploader("Histórico (CSV)", type=["csv"], key="up_hist")
    if up_hist is not None:
        try:
            new_hist = pd.read_csv(up_hist)
            miss = ensure_cols(new_hist, ["resultado","indicador","anio","valor"])
            if miss:
                st.error(f"Faltan columnas: {miss}")
            else:
                for c in ("resultado","indicador"): new_hist[c]=new_hist[c].map(canon)
                new_hist["anio"]=pd.to_numeric(new_hist["anio"], errors="coerce")
                new_hist["valor"]=pd.to_numeric(new_hist["valor"], errors="coerce")
                new_hist = new_hist.dropna(subset=["resultado","indicador","anio"])

                # merge-upsert
                key = ["resultado","indicador","anio"]
                merged = pd.concat([flat, new_hist]).drop_duplicates(subset=key, keep="last")
                merged = merged.sort_values(key)
                st.session_state.flat = merged
                flat = merged
                st.success("Histórico actualizado en sesión.")
        except Exception as e:
            st.error(f"No se pudo procesar el CSV de histórico: {e}")

    # descarga del dataset actual
    if not flat.empty:
        st.download_button("⬇️ Descargar histórico actualizado", data=flat.to_csv(index=False).encode("utf-8"),
                           file_name="historico_actualizado.csv", mime="text/csv")

# ============================== HEADER ==============================
st.markdown(f"""
<div class="sticky">
  <h1 style="margin:0 0 4px 0;">Dashboard de Indicadores</h1>
  <small class="help">Selecciona <b>un</b> indicador dentro de cada resultado. Usa el enlace del navegador para compartir esta vista.</small>
</div>
""", unsafe_allow_html=True)

if flat.empty:
    st.warning("No hay datos de histórico. Sube un CSV en la barra lateral.")
    st.stop()

# ============================== NAV (píldoras) ==============================
resultados = sorted(flat["resultado"].unique())
res_qp = get_qp("res", resultados[0] if resultados else "", str)
if res_qp not in resultados: res_qp = resultados[0]

# Render manual de píldoras como radio + CSS
res_idx = resultados.index(res_qp)
cols = st.columns(len(resultados))
sel_res = res_qp
for i, r in enumerate(resultados):
    label = r
    with cols[i]:
        if st.button(label, key=f"pill_{i}", type=("primary" if i==res_idx else "secondary"),
                     use_container_width=True):
            sel_res = r
st.markdown(
    "".join([f'<span class="pill {"active" if r==sel_res else ""}">{r}</span>' for r in resultados]),
    unsafe_allow_html=True
)

persist_qp(res=sel_res)

# ============================== CONTROLES SERIE ==============================
df_res = flat[flat["resultado"]==sel_res].copy()
inds = sorted(df_res["indicador"].unique())

# recuperar de URL
ind_qp = get_qp("ind", inds[0], str)
if ind_qp not in inds: ind_qp = inds[0]

c1, c2 = st.columns([3,2])
with c1:
    sel_ind = st.selectbox("Indicador", inds, index=inds.index(ind_qp))
with c2:
    yr_min = int(df_res["anio"].min()); yr_max = int(df_res["anio"].max())
    ymin, ymax = st.slider("Rango histórico a mostrar", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))

persist_qp(ind=sel_ind, ymin=ymin, ymax=ymax)

# DATA de la serie
hist = df_res.query("indicador == @sel_ind")[["anio","valor"]].dropna().sort_values("anio")
hist_vis = hist.query("@ymin <= anio <= @ymax")

# ============================== BLOQUE PROYECCIÓN ==============================
st.markdown("---")
st.markdown("#### Proyección")

# parámetros proyección
left, mid, right, rx = st.columns([2,2,2,2])
with left:
    modelo = st.selectbox("Modelo de proyección", [
        "Auto (mejor MAE)",
        "Lineal (OLS)",
        "Theil–Sen (robusto)",
        "Holt-Winters (ETS)",
        "Holt-Winters amortiguado",
        "ARIMA (búsqueda pequeña)"
    ])
with mid:
    horizon_last = max(yr_max+5, yr_max+10)  # por defecto 5–10 años
    year_end = st.number_input("Año final de proyección", min_value=yr_max, max_value=yr_max+30, value=yr_max+9, step=1)
with right:
    back_k = st.slider("Backtest (años para MAE)", min_value=3, max_value=min(8, len(hist)), value=min(5, len(hist)))
with rx:
    clamp0 = st.checkbox("Forzar no-negativos", value=True)

# escenarios (shocks %)
sc1, sc2 = st.columns(2)
with sc1:
    shock_up = st.slider("🔼 Escenario optimista (%)", -20, 100, 0, help="Se aplica sobre la proyección base")
with sc2:
    shock_dn = st.slider("🔽 Escenario pesimista (%)", -100, 20, 0)

# ---------- Modelos ----------
def proj_linear(y, years, future_years):
    # y ~ a + b*t
    X = (years - years.min()).values.reshape(-1,1)
    reg = LinearRegression().fit(X, y)
    Xf = (future_years - years.min()).values.reshape(-1,1)
    yhat = reg.predict(Xf)
    return yhat, reg

def proj_theilsen(y, years, future_years):
    X = (years - years.min()).values.reshape(-1,1)
    reg = TheilSenRegressor(random_state=0).fit(X, y)
    Xf = (future_years - years.min()).values.reshape(-1,1)
    yhat = reg.predict(Xf)
    return yhat, reg

def proj_holt(y, years, future_years, damped=False):
    # índices equidistantes
    fit = ExponentialSmoothing(y, trend="add", seasonal=None, damped_trend=damped).fit(optimized=True)
    steps = int(future_years.max() - years.max())
    yhat = fit.forecast(steps)
    # alineamos con años futuros
    return np.array(yhat), fit

def proj_arima(y, years, future_years):
    # pequeña búsqueda (p,d,q) en [0..2], d en [0..1] si la serie es corta
    best_aic = np.inf; best = None
    for p in range(0,3):
        for d in range(0,2):
            for q in range(0,3):
                try:
                    model = ARIMA(y, order=(p,d,q))
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic; best = res
                except:
                    continue
    if best is None:
        # fallback a ARIMA(1,0,0)
        best = ARIMA(y, order=(1,0,0)).fit()
    steps = int(future_years.max() - years.max())
    fc = best.get_forecast(steps=steps)
    mean = fc.predicted_mean.values
    lower, upper = fc.conf_int(alpha=0.2).T.values  # 80% CI
    return mean, best, lower, upper

def rolling_backtest(y, years, method_fn, window=3):
    # usa últimos "window" puntos para evaluar MAE con orígenes crecientes
    if len(y) <= window: return np.inf
    maes = []
    for cut in range(len(y)-window, len(y)):
        y_train = y[:cut]
        yr_train = years[:cut]
        y_true = y[cut]
        yr_pred = np.array([years[cut]])
        try:
            if method_fn.__name__ == "proj_holt":
                yhat,_ = method_fn(y_train, yr_train, yr_pred, damped=False)
            else:
                yhat,_ = method_fn(y_train, yr_train, yr_pred)
            maes.append(abs(y_true - float(yhat[0])))
        except:
            maes.append(np.inf)
    return float(np.mean(maes)) if len(maes)>0 else np.inf

# preparar insumos
years = hist["anio"].values
y     = hist["valor"].values
future_years = np.arange(years.max()+1, year_end+1)

yhat = None; ci_lower=None; ci_upper=None; chosen = None

def apply_shock(arr, pct):
    return arr * (1.0 + pct/100.0)

def make_ci_from_residuals(residuals, horizon_len):
    # CI simple: +/- 1.28*std (≈80%) suponiendo residuales ~N(0,sigma)
    s = np.nanstd(residuals) if residuals.size else 0.0
    half = 1.28*s
    return half

# AUTO selección
def auto_select():
    candidates = [
        ("Lineal (OLS)", proj_linear),
        ("Theil–Sen (robusto)", proj_theilsen),
        ("Holt-Winters (ETS)", lambda a,b,c: proj_holt(a,b,c,False)),
        ("Holt-Winters amortiguado", lambda a,b,c: proj_holt(a,b,c,True)),
        ("ARIMA (búsqueda pequeña)", proj_arima)
    ]
    scores=[]
    for name, fn in candidates:
        if name.startswith("ARIMA"):
            # backtest simple con 1-paso ARIMA
            try:
                mae = rolling_backtest(y, years, proj_linear, back_k)  # warm start
                # mejor: calc direct ARIMA rolling (más costoso). Para agilidad, comparamos por AIC base:
                res = ARIMA(y, order=(1,0,0)).fit()
                # pseudo score: MAE base + normalización por AIC
                scores.append((name, mae + (res.aic/1000.0)))
            except:
                scores.append((name, np.inf))
        else:
            mae = rolling_backtest(y, years, fn, back_k)
            scores.append((name, mae))
    scores = sorted(scores, key=lambda t: t[1])
    return scores[0][0]

if len(future_years)>0 and len(y)>=2:
    if modelo == "Auto (mejor MAE)":
        chosen = auto_select()
    else:
        chosen = modelo

    if chosen == "Lineal (OLS)":
        yhat, _ = proj_linear(y, years, future_years)
        resid = y - LinearRegression().fit((years-years.min()).reshape(-1,1), y).predict((years-years.min()).reshape(-1,1))
        half = make_ci_from_residuals(resid, len(future_years))
        ci_lower = yhat - half; ci_upper = yhat + half
    elif chosen == "Theil–Sen (robusto)":
        yhat, _ = proj_theilsen(y, years, future_years)
        ci_lower = yhat - np.std(yhat - np.mean(yhat)); ci_upper = yhat + np.std(yhat - np.mean(yhat))
    elif chosen == "Holt-Winters (ETS)":
        yhat, fit = proj_holt(y, years, future_years, False)
        resid = y - fit.fittedvalues
        half = make_ci_from_residuals(resid, len(future_years))
        ci_lower = yhat - half; ci_upper = yhat + half
    elif chosen == "Holt-Winters amortiguado":
        yhat, fit = proj_holt(y, years, future_years, True)
        resid = y - fit.fittedvalues
        half = make_ci_from_residuals(resid, len(future_years))
        ci_lower = yhat - half; ci_upper = yhat + half
    elif chosen == "ARIMA (búsqueda pequeña)":
        mean, best, lower, upper = proj_arima(y, years, future_years)
        yhat = mean; ci_lower = lower; ci_upper = upper

    # shocks
    if yhat is not None:
        if shock_up != 0:  yhat = apply_shock(yhat, shock_up)
        if shock_dn != 0:  yhat = apply_shock(yhat, shock_dn)
        if clamp0:         yhat = clamp_nonneg(yhat)
        if ci_lower is not None and ci_upper is not None:
            if clamp0:
                ci_lower = clamp_nonneg(ci_lower)
            # asegurar orden
            swap = ci_lower > ci_upper
            if np.any(swap):
                tmp = ci_lower.copy(); ci_lower[swap]=ci_upper[swap]; ci_upper[swap]=tmp

# ============================== KPIs ==============================
k1,k2,k3 = st.columns(3)
if not hist.empty:
    last = hist.iloc[-1]["valor"]
    prev = hist.iloc[-2]["valor"] if len(hist)>1 else np.nan
    k1.metric("Último valor (hist.)", f"{last:.2f}", None if np.isnan(prev) else f"{(last-prev):+.2f}")
else:
    k1.metric("Último valor (hist.)", "–")

if yhat is not None:
    k2.metric("Primera proyección", f"{yhat[0]:.2f}")
    k3.metric("Modelo elegido", chosen or modelo)
else:
    k2.metric("Primera proyección", "–")
    k3.metric("Modelo elegido", "–")

# ============================== GRÁFICO ==============================
g_parts = []
if not hist.empty: g_parts.append(hist.assign(tipo="Histórico"))
if yhat is not None:
    proy_df = pd.DataFrame({"anio": future_years, "valor": yhat}).assign(tipo="Proyección")
    g_parts.append(proy_df)

fig = go.Figure()
if not hist.empty:
    fig.add_trace(go.Scatter(x=hist["anio"], y=hist["valor"], mode="lines+markers",
                             name="Histórico", line=dict(color=BLUE)))
if yhat is not None:
    fig.add_trace(go.Scatter(x=future_years, y=yhat, mode="lines+markers",
                             name="Proyección", line=dict(color=PRIMARY, dash="dash")))
    if ci_lower is not None and ci_upper is not None:
        fig.add_traces([
            go.Scatter(x=future_years, y=ci_upper, mode="lines",
                       line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=future_years, y=ci_lower, mode="lines",
                       fill="tonexty", line=dict(width=0), showlegend=False,
                       hoverinfo="skip", fillcolor="rgba(220,38,38,0.12)")
        ])
fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  xaxis_title="año", yaxis_title="valor")
st.plotly_chart(fig, use_container_width=True)

# ============================== TABLAS ==============================
t1,t2 = st.columns(2)
with t1:
    st.markdown("**Histórico (rango visible)**")
    st.dataframe(hist_vis.rename(columns={"anio":"año","valor":"valor"}),
                 use_container_width=True, hide_index=True)
with t2:
    st.markdown("**Proyección**")
    if yhat is not None:
        tmp = pd.DataFrame({"anio": future_years, "valor": yhat})
        st.dataframe(tmp.rename(columns={"anio":"año","valor":"valor"}),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("Sin proyección (elige un modelo o reduce el año final).")

# MAE de backtest mostrado
if len(y)>=back_k+1:
    # usamos el modelo elegido si no es AUTO; si es AUTO, mostramos el score del elegido vs back_k
    def one_step_mae(method):
        maes=[]
        for cut in range(len(y)-back_k, len(y)):
            yt = y[:cut]; yr = years[:cut]; y_true=y[cut]
            target_year = np.array([years[cut]])
            try:
                if method=="Lineal (OLS)":
                    pr,_ = proj_linear(yt,yr,target_year)
                elif method=="Theil–Sen (robusto)":
                    pr,_ = proj_theilsen(yt,yr,target_year)
                elif method=="Holt-Winters (ETS)":
                    pr,_ = proj_holt(yt,yr,target_year,False)
                elif method=="Holt-Winters amortiguado":
                    pr,_ = proj_holt(yt,yr,target_year,True)
                elif method=="ARIMA (búsqueda pequeña)":
                    pr,_,_,_ = proj_arima(yt,yr,target_year)
                else:
                    pr,_ = proj_linear(yt,yr,target_year)
                maes.append(abs(float(pr[0]) - y_true))
            except:
                pass
        return np.mean(maes) if maes else np.nan
    mae_val = one_step_mae(chosen or modelo)
    st.caption(f"**MAE** (backtest {back_k} años): **{mae_val:.2f}**")

# ============================== DESCARGA SERIE ==============================
if g_parts:
    out = pd.concat(g_parts, ignore_index=True).sort_values("anio")
    st.download_button("⬇️ Descargar serie (hist+proy)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name=f"serie_{canon(sel_res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.csv", mime="text/csv")

# ============================== AYUDA ==============================
with st.expander("Ayuda"):
    st.markdown(f"""
- **Pestañas tipo píldora:** arriba puedes cambiar de *resultado* con botones visibles.  
- **Permalink:** la URL conserva `res`, `ind`, `ymin`, `ymax`.  
- **Proyecciones:** usa *Auto (mejor MAE)* para escoger el método con menor error en los últimos años.  
- **Escenarios:** aplica shocks % (optimista / pesimista). Marca **Forzar no-negativos** para truncar bajo 0.  
- **Actualizar datos:** usa la barra lateral para subir un CSV y luego descarga el histórico combinado.
""")
