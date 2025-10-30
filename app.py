
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Dashboard de Indicadores — Serie única con proyección",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== Estilos ====================
st.markdown(
    """
    <style>
    :root { --radius: 14px; }
    .block-container { padding-top: .6rem; padding-bottom: 1.2rem; }
    .card { border-radius: var(--radius); padding: 16px; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.03); }
    .kpi {display:flex; gap:12px; align-items:center}
    .kpi .v {font-size: 1.35rem; font-weight:700}
    .kpi .l {font-size:.85rem; opacity:.8}
    .stTabs [data-baseweb="tab-list"]{ gap:.25rem; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"]{ padding:8px 14px; border-radius: var(--radius); }
    .dataframe tbody tr:hover { background: rgba(180,180,180,0.06) }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== Helpers ====================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    # normalización frecuente
    ren = {
        "año": "anio",
        "year": "anio",
        "valor_n": "valor",
        "value": "valor",
        "nombre_indicador": "indicador",
    }
    for k,v in ren.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)
    return df

def cand(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def available_models():
    opts = ["Naive (último valor)", "Media móvil (3)", "Lineal (OLS)"]
    # Opcionales con statsmodels si disponible
    try:
        import statsmodels.api as sm  # noqa
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa
        opts += ["Holt-Winters (additiva)"]
    except Exception:
        pass
    return opts

def fit_forecast(y: pd.Series, x: pd.Series, horizon_years: List[int], model_name: str) -> pd.Series:
    # Asegura orden
    order = np.argsort(x.values)
    xv = x.values[order].astype(float)
    yv = y.values[order].astype(float)

    if len(yv) == 0:
        return pd.Series(dtype=float)

    if model_name.startswith("Naive"):
        last = yv[-1]
        return pd.Series([last]*len(horizon_years), index=horizon_years, dtype="float")

    if model_name.startswith("Media móvil"):
        if len(yv) < 3:
            mv = np.mean(yv)
        else:
            mv = np.mean(yv[-3:])
        return pd.Series([mv]*len(horizon_years), index=horizon_years, dtype="float")

    if model_name.startswith("Lineal"):
        # y = a + b*x (x=anio)
        try:
            b, a = np.polyfit(xv, yv, deg=1)
            preds = a + b*np.array(horizon_years, dtype=float)
            return pd.Series(preds, index=horizon_years, dtype="float")
        except Exception:
            return pd.Series([yv[-1]]*len(horizon_years), index=horizon_years, dtype="float")

    if model_name.startswith("Holt-Winters"):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            # Índice equiespaciado por año
            s = pd.Series(yv, index=pd.Index(xv.astype(int), name="anio")).sort_index()
            # Sin estacionalidad (anual) — tendencia aditiva
            model = ExponentialSmoothing(s, trend="add", seasonal=None, initialization_method="estimated")
            fitted = model.fit()
            # statsmodels pronostica por pasos, armamos steps como diferencia de años
            last_year = int(s.index.max())
            steps = [y - last_year for y in horizon_years if y > last_year]
            fc = fitted.forecast(steps=len(steps))
            # reindex a horizon_years (para años <= last_year, repetimos último)
            out = []
            for y in horizon_years:
                if y <= last_year:
                    out.append(float(s.loc[last_year]))
                else:
                    out.append(float(fc.loc[y]) if y in fc.index else float(fc.iloc[-1]))
            return pd.Series(out, index=horizon_years, dtype="float")
        except Exception:
            return pd.Series([yv[-1]]*len(horizon_years), index=horizon_years, dtype="float")

    # Fallback
    return pd.Series([yv[-1]]*len(horizon_years), index=horizon_years, dtype="float")

def backtest_mae(y: pd.Series, x: pd.Series, model_name: str, k:int=3) -> Optional[float]:
    if len(y) <= k+1:
        return None
    train_y = y.iloc[:-k]
    train_x = x.iloc[:-k]
    horizon = list(x.iloc[-k:].astype(int).values)
    preds = fit_forecast(train_y, train_x, horizon, model_name)
    mae = np.mean(np.abs(y.iloc[-k:].values - preds.values))
    return float(mae)

def human(n):
    try:
        return f"{n:,.0f}".replace(",", " ")
    except Exception:
        return str(n)

# ==================== Sidebar: archivo base ====================
st.sidebar.title("Datos")
base_path = st.sidebar.text_input("Ruta del CSV base (largo):", value="indicadores_flat.csv")
uploaded = st.sidebar.file_uploader("O arrastra y suelta el CSV base", type=["csv"])
if uploaded is not None:
    base = pd.read_csv(uploaded)
    tmp_name = "_uploaded_flat.csv"
    base.to_csv(tmp_name, index=False)
    base_path = tmp_name

with st.spinner("Cargando…"):
    base = load_csv(base_path)

# Columnas clave
resultado_col = cand(base, ["resultado", "tipo", "bloque", "categoria", "dimension"])
indicador_col = cand(base, ["indicador", "nombre_indicador"])
subind_col    = cand(base, ["subindicador", "sub_indicador", "desagregacion", "categoria_detalle", "tema"])
anio_col      = cand(base, ["anio", "año", "year"])
valor_col     = cand(base, ["valor", "value", "y"])

required = [resultado_col, indicador_col, anio_col, valor_col]
if any(c is None for c in required):
    st.error("Faltan columnas mínimas en el CSV: resultado, indicador, anio, valor. Revisa los nombres.")
    st.stop()

# Tipos
base[anio_col] = pd.to_numeric(base[anio_col], errors="coerce")
base[valor_col] = pd.to_numeric(base[valor_col], errors="coerce")

min_year = int(base[anio_col].min())
max_year = int(base[anio_col].max())

st.markdown("# Dashboard de Indicadores")
st.caption("Pestañas por **resultado**. Dentro de cada pestaña selecciona **una sola serie** (indicador + subindicador opcional) para visualizar y proyectar.")

# ==================== Tabs por resultado ====================
resultados = list(pd.Series(base[resultado_col].dropna().unique()).sort_values().values)
if not resultados:
    st.warning("No hay resultados en la columna de agrupación.")
    st.stop()

tabs = st.tabs(resultados)

for idx, res in enumerate(resultados):
    with tabs[idx]:
        tdf = base[base[resultado_col] == res].copy()
        if tdf.empty:
            st.info("Sin datos para esta categoría.")
            continue

        # Filtros específicos del resultado
        c1, c2, c3 = st.columns([3,3,4])
        with c1:
            inds = list(pd.Series(tdf[indicador_col].dropna().unique()).sort_values().values)
            ind_sel = st.selectbox("Indicador", options=inds, key=f"{res}-ind")
        sdf = tdf[tdf[indicador_col] == ind_sel].copy()

        with c2:
            if subind_col and sdf[subind_col].notna().any():
                subopts = list(pd.Series(sdf[subind_col].dropna().unique()).sort_values().values)
                sub_sel = st.selectbox("Sub-indicador (opcional)", options=["(todos)"]+subopts, key=f"{res}-sub")
            else:
                sub_sel = "(todos)"

        with c3:
            # serie 2014-2024 (histórica) + horizonte
            vis_min = max(min_year, 2014) if max_year >= 2014 else min_year
            vis_max = min(max_year, 2024) if max_year >= 2024 else max_year
            rango = st.slider("Rango histórico a mostrar", min_value=min_year, max_value=max_year, value=(vis_min, vis_max), key=f"{res}-rango")

        # Filtrado para la **serie única**
        s = sdf.copy()
        if sub_sel != "(todos)" and subind_col:
            s = s[s[subind_col] == sub_sel]

        # Asegura una sola serie (por indicador + sub)
        s = s[[anio_col, valor_col, indicador_col] + ([subind_col] if subind_col else [])].dropna(subset=[anio_col, valor_col])
        s = s.groupby(anio_col, as_index=False)[valor_col].mean()  # si vinieran múltiples filas por año, agregamos

        s_hist = s[(s[anio_col] >= rango[0]) & (s[anio_col] <= rango[1])].sort_values(anio_col)

        # KPIs
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="card kpi"><div class="v">{res}</div><div class="l">resultado</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="card kpi"><div class="v">{ind_sel}</div><div class="l">indicador</div></div>', unsafe_allow_html=True)
        with k3:
            last_val = s_hist[valor_col].iloc[-1] if not s_hist.empty else None
            st.markdown(f'<div class="card kpi"><div class="v">{human(last_val) if last_val is not None else "—"}</div><div class="l">último valor</div></div>', unsafe_allow_html=True)

        st.divider()

        # ==================== Modelado ====================
        m1, m2, m3 = st.columns([2,2,2])
        with m1:
            modelo = st.selectbox("Modelo de proyección", options=available_models(), key=f"{res}-modelo")
        with m2:
            h_fin = st.number_input("Año final de proyección", min_value=max(2025, rango[1]+1), max_value=2035, value=2029, step=1, key=f"{res}-h")
        with m3:
            back_k = st.slider("Backtest (años para MAE)", min_value=0, max_value=5, value=3, step=1, key=f"{res}-bt")

        # Entrenamiento con todo el histórico (no sólo rango) hasta 2024 (si existe)
        full_hist = s[s[anio_col] <= min(h_fin-1, max_year)].sort_values(anio_col).reset_index(drop=True)
        if full_hist.empty:
            st.warning("No hay datos históricos para esta serie.")
            continue

        # Backtest
        mae_val = None
        if back_k and len(full_hist) > back_k+1:
            mae_val = backtest_mae(full_hist[valor_col], full_hist[anio_col], modelo, k=back_k)

        # Forecast en horizonte seleccionado
        horizon_years = list(range(int(min(max_year, 2024))+1, int(h_fin)+1))
        fc = fit_forecast(full_hist[valor_col], full_hist[anio_col], horizon_years, modelo)

        # ==================== Plot ====================
        # Construimos df para figura (hist vs forecast). Siempre UNA serie.
        plot_df = pd.DataFrame({
            "anio": list(s_hist[anio_col].values) + list(fc.index.values if len(fc)>0 else []),
            "valor": list(s_hist[valor_col].values) + list(fc.values if len(fc)>0 else []),
            "tipo": ["Histórico"]*len(s_hist) + (["Proyección"]*len(fc) if len(fc)>0 else []),
        })
        plot_df = plot_df.sort_values("anio")

        fig = px.line(
            plot_df,
            x="anio", y="valor", color="tipo",
            markers=True,
            title=f"{ind_sel}" + (f" — {sub_sel}" if sub_sel != '(todos)' else ""),
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ==================== Tablas ====================
        tcol1, tcol2 = st.columns([1,1])
        with tcol1:
            st.markdown("**Histórico (rango visible)**")
            st.dataframe(s_hist.rename(columns={anio_col:"anio", valor_col:"valor"}), use_container_width=True, hide_index=True)
        with tcol2:
            st.markdown("**Proyección**")
            if len(fc) == 0:
                st.info("Sin años de proyección (ajusta el año final).")
            else:
                df_fc = pd.DataFrame({"anio": fc.index, "valor": fc.values})
                st.dataframe(df_fc, use_container_width=True, hide_index=True)

        # Métrica de error opcional
        if mae_val is not None:
            st.caption(f"MAE (backtest {back_k} años): **{mae_val:,.2f}**")
