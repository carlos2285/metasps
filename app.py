# app.py — Dashboard de Indicadores (Streamlit)
# -------------------------------------------------------------
# Requisitos: streamlit, pandas, numpy, plotly-express
# Ejecución local:  streamlit run app.py
# Estructura de archivos esperada (puedes cambiar rutas en la sidebar):
#   - indicadores_flat.csv
#   - pivot_2014_2023.csv
#   - pivot_2014_2019.csv
#   - pivot_2020_2024.csv
#   - proyecciones_2026_2029.csv (opcional)
# -------------------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard de Indicadores", layout="wide")

# ---------- Utilidades ----------
def clean_str(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalización ligera
    for c in df.select_dtypes(include=[object]).columns:
        df[c] = df[c].map(clean_str)
    return df

@st.cache_data(show_spinner=False)
def melt_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte una pivot en formato largo (resultado, indicador, anio, valor)."""
    if df.empty:
        return df
    # Detectar columnas de años
    year_cols = [c for c in df.columns if str(c).isdigit() or (isinstance(c, (int, np.integer)))]
    base_cols = [c for c in df.columns if c not in year_cols]
    m = df.melt(id_vars=base_cols, value_vars=year_cols, var_name="anio", value_name="valor")
    m["anio"] = pd.to_numeric(m["anio"], errors="coerce").astype("Int64")
    m["valor"] = pd.to_numeric(m["valor"], errors="coerce")
    return m.dropna(subset=["anio", "valor"]) if not m.empty else m

# ---------- Sidebar (rutas) ----------
st.sidebar.header("Archivos")
flat_path = st.sidebar.text_input("Ruta: indicadores_flat.csv", value="indicadores_flat.csv")
piv_full_path = st.sidebar.text_input("Ruta: pivot_2014_2023.csv", value="pivot_2014_2023.csv")
piv_old_path  = st.sidebar.text_input("Ruta: pivot_2014_2019.csv", value="pivot_2014_2019.csv")
piv_new_path  = st.sidebar.text_input("Ruta: pivot_2020_2024.csv", value="pivot_2020_2024.csv")
proj_path     = st.sidebar.text_input("Ruta (opcional): proyecciones_2026_2029.csv", value="proyecciones_2026_2029.csv")

st.sidebar.markdown("---")
up = st.sidebar.file_uploader("O cargar indicadores_flat.csv manualmente", type=["csv"]) 

# ---------- Carga de datos ----------
flat = pd.read_csv(up) if up is not None else load_csv(flat_path)
piv_full = load_csv(piv_full_path)
piv_old  = load_csv(piv_old_path)
piv_new  = load_csv(piv_new_path)
proj     = load_csv(proj_path)

# Asegurar columnas claves si existen
for col in ["resultado","indicador","tendencia_esperada","anio","valor","cumple"]:
    if col in flat.columns:
        if col in ["anio"]:
            flat[col] = pd.to_numeric(flat[col], errors="coerce").astype("Int64")
        elif col in ["valor"]:
            flat[col] = pd.to_numeric(flat[col], errors="coerce")
        else:
            flat[col] = flat[col].astype(str)

# ---------- Filtros ----------
st.title("📊 Dashboard de Indicadores")

if flat.empty:
    st.info("Carga 'indicadores_flat.csv' en la barra lateral o usa el cargador para continuar.")
else:
    cols = st.columns([1,1,1,1])
    res_opts = sorted([x for x in flat.get("resultado", pd.Series(dtype=str)).dropna().unique()])
    ind_opts = sorted([x for x in flat.get("indicador", pd.Series(dtype=str)).dropna().unique()])
    cum_opts = sorted([x for x in flat.get("cumple", pd.Series(dtype=str)).dropna().unique()])

    with cols[0]:
        sel_res = st.multiselect("Resultado de desarrollo", res_opts, default=res_opts[:5] if res_opts else [])
    with cols[1]:
        sel_ind = st.multiselect("Indicador", ind_opts, default=ind_opts[:10] if ind_opts else [])
    with cols[2]:
        sel_cum = st.multiselect("¿Cumple tendencia?", cum_opts, default=cum_opts)
    with cols[3]:
        year_min, year_max = int(np.nanmin(flat["anio"])) if "anio" in flat.columns and not flat["anio"].isna().all() else 2014, int(np.nanmax(flat["anio"])) if "anio" in flat.columns and not flat["anio"].isna().all() else 2024
        sel_years = st.slider("Rango de años", min_value=min(2010, year_min), max_value=max(2029, year_max), value=(2014, year_max))

    q = flat.copy()
    if sel_res:
        q = q[q.get("resultado").isin(sel_res)] if "resultado" in q.columns else q
    if sel_ind:
        q = q[q.get("indicador").isin(sel_ind)] if "indicador" in q.columns else q
    if sel_cum:
        q = q[q.get("cumple").isin(sel_cum)] if "cumple" in q.columns else q
    if "anio" in q.columns:
        q = q[(q["anio"]>=sel_years[0]) & (q["anio"]<=sel_years[1])]

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resumen / Serie (largo)",
        "Serie completa 2014-2023",
        "2014-2019 (pre-base)",
        "2020-2024 (post-base)",
        "Proyecciones 2026–2029"
    ])

    with tab1:
        st.subheader("Datos en formato largo")
        st.dataframe(q, use_container_width=True)
        # Gráfico línea por indicador
        if not q.empty and {"anio","valor"}.issubset(q.columns):
            st.markdown("**Serie temporal**")
            color_col = "indicador" if "indicador" in q.columns else None
            fig = px.line(q.sort_values(["indicador","anio"]) if color_col else q.sort_values("anio"), 
                          x="anio", y="valor", color=color_col, markers=True,
                          hover_data=[c for c in ["resultado","tendencia_esperada","cumple"] if c in q.columns])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Tabla dinámica — Serie completa 2014–2023")
        st.dataframe(piv_full, use_container_width=True)
        mfull = melt_pivot(piv_full)
        if not mfull.empty:
            fig = px.line(mfull, x="anio", y="valor", color="indicador", markers=True,
                          hover_data=[c for c in ["resultado"] if c in mfull.columns])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Tabla dinámica — 2014–2019 (pre-base)")
        st.dataframe(piv_old, use_container_width=True)
        mold = melt_pivot(piv_old)
        if not mold.empty:
            fig = px.line(mold, x="anio", y="valor", color="indicador", markers=True,
                          hover_data=[c for c in ["resultado"] if c in mold.columns])
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Tabla dinámica — 2020–2024 (post-base)")
        st.dataframe(piv_new, use_container_width=True)
        mnew = melt_pivot(piv_new)
        if not mnew.empty:
            fig = px.line(mnew, x="anio", y="valor", color="indicador", markers=True,
                          hover_data=[c for c in ["resultado"] if c in mnew.columns])
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Proyecciones 2026–2029 (tendencia lineal)")
        if proj.empty:
            st.info("No se encontró 'proyecciones_2026_2029.csv'. Puedes generarlo o subirlo en la barra lateral.")
        else:
            st.dataframe(proj, use_container_width=True)
            if not q.empty and {"anio","valor"}.issubset(q.columns):
                # Combinar q (histórico filtrado) + proyecciones del/los indicadores visibles
                merge_cols = ["indicador"] + (["resultado"] if "resultado" in proj.columns and "resultado" in q.columns else [])
                psub = proj.copy()
                if sel_ind:
                    psub = psub[psub["indicador"].isin(sel_ind)]
                if sel_res and "resultado" in psub.columns:
                    psub = psub[psub["resultado"].isin(sel_res)]
                hist = q[[c for c in q.columns if c in merge_cols + ["anio","valor"]]].copy()
                hist = hist.rename(columns={"anio":"anio_proy","valor":"valor_hist"})
                # Ensamble para gráfico
                g_hist = hist.rename(columns={"anio_proy":"anio"})[[*merge_cols, "anio", "valor_hist"]]
                g_hist = g_hist.rename(columns={"valor_hist":"valor"})
                g_hist["tipo"] = "observado"
                g_proj = psub.rename(columns={"valor_proy":"valor"})[[*merge_cols, "anio_proy","valor"]]
                g_proj = g_proj.rename(columns={"anio_proy":"anio"})
                g_proj["tipo"] = "proyectado"
                gg = pd.concat([g_hist, g_proj], ignore_index=True)
                fig = px.line(gg.sort_values([*merge_cols, "anio"]), x="anio", y="valor", color="indicador", line_dash="tipo",
                              markers=True, hover_data=[c for c in ["resultado","tipo"] if c in gg.columns])
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Esta app lee archivos CSV generados del Excel fuente y permite filtros por resultado, indicador y cumplimiento de tendencia, con tablas dinámicas y proyecciones simples.")