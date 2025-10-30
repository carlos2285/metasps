
import os
import re
import io
import json
import pathlib
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Dashboard de Indicadores — Tabs por tipo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ Estilos ============
st.markdown("""
<style>
/* Tipografía y espaciados */
:root { --radius: 14px; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px }
.stTabs [data-baseweb="tab-list"]{ gap:.25rem; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"]{ padding:8px 14px; border-radius: var(--radius); }
.st-emotion-cache-13k62yr { padding-top: 0 !important } /* reduce header padding */

/* Card */
.card {
  border-radius: var(--radius);
  padding: 16px;
  border: 1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.03);
}

/* Métricas compactas */
.kpi {display:flex; gap:12px; align-items:center}
.kpi .v {font-size: 1.35rem; font-weight:700}
.kpi .l {font-size:.85rem; opacity:.8}

/* Tablas */
.dataframe tbody tr:hover { background: rgba(180,180,180,0.06) }
</style>
""", unsafe_allow_html=True)

# ============ Helpers ============

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalizamos nombres
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c.lower() in df.columns:
            return c.lower()
    return None

def humanize(n):
    try:
        return f"{n:,.0f}".replace(",", " ")
    except Exception:
        return str(n)

# ============ Sidebar: Archivos ============
st.sidebar.title("Archivos")

default_flat = "indicadores_flat.csv"
default_pivot_1423 = "pivot_2014_2023.csv"
default_pivot_1419 = "pivot_2014_2019.csv"
default_pivot_2024 = "pivot_2020_2024.csv"
default_proj_2629 = "proyecciones_2026_2029.csv"

flat_path = st.sidebar.text_input("Ruta: indicadores_flat.csv", value=default_flat)
p1423_path = st.sidebar.text_input("Ruta: pivot_2014_2023.csv", value=default_pivot_1423)
p1419_path = st.sidebar.text_input("Ruta: pivot_2014_2019.csv", value=default_pivot_1419)
p2024_path = st.sidebar.text_input("Ruta: pivot_2020_2024.csv", value=default_pivot_2024)
proj_path  = st.sidebar.text_input("Ruta (opcional): proyecciones_2026_2029.csv", value=default_proj_2629)

st.sidebar.caption("También puedes arrastrar y soltar **indicadores_flat.csv** abajo.")
uploaded = st.sidebar.file_uploader("Cargar indicadores_flat.csv", type=["csv"])
if uploaded is not None:
    flat_df = pd.read_csv(uploaded)
    flat_df.to_csv("_uploaded_flat.csv", index=False)
    flat_path = "_uploaded_flat.csv"

# ============ Carga de datos ============
with st.spinner("Cargando datos…"):
    base = load_csv(flat_path)

anio_col = find_col(base, ["anio","año","year"])
val_col  = find_col(base, ["valor","value","valor_n","y"])
ind_col  = find_col(base, ["indicador","nombre_indicador","indicator"])
tipo_col = find_col(base, ["resultado","tipo","dimension","eje","categoria"])
ct_col   = find_col(base, ["cumple_tendencia","cumple","tendencia_ok","meets_trend"])

if not anio_col or not val_col or not ind_col:
    st.error("No se encontraron columnas mínimas requeridas: año, valor, indicador. Revisa el CSV.")
    st.stop()

# Casts seguros
base[anio_col] = pd.to_numeric(base[anio_col], errors="coerce")
base[val_col]  = pd.to_numeric(base[val_col], errors="coerce")

# Rango de años dinámico
min_year, max_year = int(base[anio_col].min()), int(base[anio_col].max())

# ============ Encabezado ============
st.markdown("# Dashboard de Indicadores")
st.caption("Vista con pestañas por tipo de indicador (campo: **{}**).".format(tipo_col if tipo_col else "—"))

# ============ Filtros globales ============
c1, c2, c3, c4 = st.columns([2,2,2,4])
with c1:
    desarrollo = st.selectbox("Resultado de desarrollo", ["Todos"] + sorted(list(base.get(tipo_col, pd.Series(["—"])).dropna().unique()))) if tipo_col else "Todos"
with c2:
    ind_opt = sorted(list(base[ind_col].dropna().unique()))
    ind_sel = st.multiselect("Indicador", options=ind_opt, default=[])
with c3:
    trend_opt = ["Todos"]
    if ct_col:
        trend_opt += sorted(list(base[ct_col].dropna().astype(str).unique()))
    trend_sel = st.selectbox("¿Cumple tendencia?", options=trend_opt, index=0)
with c4:
    yr = st.slider("Rango de años", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Aplica filtros globales
df = base.copy()
if desarrollo != "Todos" and tipo_col:
    df = df[df[tipo_col] == desarrollo]
if ind_sel:
    df = df[df[ind_col].isin(ind_sel)]
if trend_sel != "Todos" and ct_col:
    df = df[df[ct_col].astype(str) == str(trend_sel)]
df = df[(df[anio_col] >= yr[0]) & (df[anio_col] <= yr[1])]

# ============ Pestañas por tipo (resultado) ============
if tipo_col:
    categorias = list(df[tipo_col].dropna().unique())
else:
    categorias = ["Todos"]

if not categorias:
    st.info("No hay datos para los filtros seleccionados.")
    st.stop()

tab_objs = st.tabs([f"{str(cat)}" for cat in categorias])

for i, cat in enumerate(categorias):
    with tab_objs[i]:
        if tipo_col:
            tdf = df[df[tipo_col] == cat].copy()
        else:
            tdf = df.copy()

        # KPIs
        k1, k2, k3 = st.columns(3)
        n_ind = tdf[ind_col].nunique()
        n_obs = len(tdf)
        last_year = int(tdf[anio_col].max()) if not tdf.empty else None
        with k1:
            st.markdown(f'<div class="card kpi"><div class="v">{n_ind}</div><div class="l">indicadores</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="card kpi"><div class="v">{humanize(n_obs)}</div><div class="l">observaciones</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="card kpi"><div class="v">{last_year if last_year else "—"}</div><div class="l">último año</div></div>', unsafe_allow_html=True)

        st.divider()

        # Selector interno por indicador (para gráfica)
        ind_local = sorted(list(tdf[ind_col].dropna().unique()))
        chosen = st.multiselect("Selecciona 1–6 indicadores para graficar", ind_local, default=ind_local[:min(3, len(ind_local))], max_selections=6)

        gdf = tdf[tdf[ind_col].isin(chosen)].copy()
        if gdf.empty:
            st.warning("Sin datos para graficar con la selección actual.")
        else:
            # línea temporal
            fig = px.line(
                gdf.sort_values([ind_col, anio_col]),
                x=anio_col, y=val_col, color=ind_col,
                markers=True,
                title=f"Evolución temporal — {cat if tipo_col else 'Todos'}"
            )
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Datos en formato largo")
        st.dataframe(
            tdf.sort_values([ind_col, anio_col]).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

# ============ Sección opcional: Otras tablas ============
with st.expander("Ver tablas opcionales (pivots y proyecciones)"):
    for label, path in [
        ("Serie completa 2014–2023", p1423_path),
        ("2014–2019 (pre-base)",   p1419_path),
        ("2020–2024 (post-base)",  p2024_path),
        ("Proyecciones 2026–2029", proj_path),
    ]:
        try:
            if path and os.path.exists(path):
                p = load_csv(path)
                st.markdown(f"**{label}**")
                st.dataframe(p, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"No se pudo cargar {label}: {e}")
