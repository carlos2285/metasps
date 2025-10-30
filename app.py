# app.py — Dashboard de Indicadores (versión limpia y compartible)
# ---------------------------------------------------------------
# Características:
# - Pestañas por "resultado"
# - Un solo indicador a la vez (evita confusión visual)
# - Histórico en AZUL, Proyección en ROJO punteado
# - Slider de años controla TODO (gráfico + tabla histórico)
# - KPIs: último valor, variación vs. año previo, y primera proyección
# - Enlace compartible: parámetros en la URL (?res=...&ind=...&ymin=...&ymax=...)
# - Descarga CSV de la serie (histórico + proyección)
# - Manejo robusto de archivos (mensajes claros si falta algún CSV)

import os
import re
import sys
import pandas as pd
import streamlit as st

# ============ Config general ============
st.set_page_config(
    page_title="Dashboard de Indicadores",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ Helpers ============
def canon(s: str) -> str:
    """Normaliza texto: quita dobles espacios, leading/trailing, convierte a str."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

def load_csv_safe(path: str, required_cols=None) -> pd.DataFrame:
    """
    Lee CSV si existe. Si faltan columnas requeridas, devuelve df vacío con aviso.
    """
    if not os.path.exists(path):
        st.warning(f"⚠️ No se encontró el archivo: `{path}`")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error leyendo `{path}`: {e}")
        return pd.DataFrame()

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"El archivo `{path}` no tiene columnas requeridas: {missing}")
            return pd.DataFrame()
    return df

@st.cache_data(show_spinner=False)
def load_data(
    hist_path="indicadores_flat.csv",
    proy_path="proyecciones_2026_2029.csv"
):
    # Histórico
    hist = load_csv_safe(hist_path)
    if not hist.empty:
        # columnas mínimas esperadas
        # intentamos mapear nombres alternativos si existen
        # (por si 'anio' llega como 'año' o 'year')
        rename_map = {}
        if "año" in hist.columns and "anio" not in hist.columns:
            rename_map["año"] = "anio"
        if "year" in hist.columns and "anio" not in hist.columns:
            rename_map["year"] = "anio"
        if "value" in hist.columns and "valor" not in hist.columns:
            rename_map["value"] = "valor"
        hist = hist.rename(columns=rename_map)

        # normalización
        for c in ("resultado", "indicador"):
            if c in hist.columns:
                hist[c] = hist[c].map(canon)

        # tipos
        if "anio" in hist.columns:
            hist["anio"] = pd.to_numeric(hist["anio"], errors="coerce")
        if "valor" in hist.columns:
            hist["valor"] = pd.to_numeric(hist["valor"], errors="coerce")

        # limpieza básica
        keep_cols = [c for c in ["resultado", "indicador", "subindicador", "anio", "valor"] if c in hist.columns]
        hist = hist[keep_cols].dropna(subset=["resultado", "indicador", "anio"]).sort_values(["resultado","indicador","anio"])

    # Proyección
    proy = load_csv_safe(proy_path)
    if not proy.empty:
        # normalizar posibles nombres
        rmap = {}
        if "anio" in proy.columns and "anio_proy" not in proy.columns:
            rmap["anio"] = "anio_proy"
        if "valor" in proy.columns and "valor_proy" not in proy.columns:
            rmap["valor"] = "valor_proy"
        proy = proy.rename(columns=rmap)

        for c in ("resultado", "indicador"):
            if c in proy.columns:
                proy[c] = proy[c].map(canon)

        if "anio_proy" in proy.columns:
            proy["anio_proy"] = pd.to_numeric(proy["anio_proy"], errors="coerce")
        if "valor_proy" in proy.columns:
            proy["valor_proy"] = pd.to_numeric(proy["valor_proy"], errors="coerce")

        # columnas útiles
        keep_cols_p = [c for c in ["resultado","indicador","anio_proy","valor_proy"] if c in proy.columns]
        proy = proy[keep_cols_p].dropna(subset=["anio_proy","valor_proy"]).sort_values(["anio_proy"])

    return hist, proy

def persist_query_params(**kwargs):
    """
    Guarda parámetros en la URL para compartir la vista actual.
    Streamlit recientes: st.query_params; fallback: experimental_set_query_params.
    """
    qp = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            qp[k] = [str(x) for x in v]
        else:
            qp[k] = str(v)
    try:
        st.query_params.update(qp)  # type: ignore[attr-defined]
    except Exception:
        try:
            st.experimental_set_query_params(**qp)
        except Exception:
            pass

def get_query_param(name, default=None, cast=int):
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        val = qp.get(name, None)
        if val is None:
            return default
        if isinstance(val, list):  # algunos navegadores devuelven lista
            val = val[0] if val else None
        if cast is None:
            return val
        return cast(val) if val is not None else default
    except Exception:
        return default

# ============ Carga de datos ============
HIST_PATH = os.environ.get("HIST_PATH", "indicadores_flat.csv")
PROY_PATH = os.environ.get("PROY_PATH", "proyecciones_2026_2029.csv")

flat, proy_df = load_data(HIST_PATH, PROY_PATH)

# ============ UI: Cabecera ============
st.markdown(
    """
    <div style="padding:6px 0 0 0">
      <h1 style="margin-bottom:4px;">Dashboard de Indicadores</h1>
      <p style="color:#475569;margin-top:0">
        Pestañas por <b>resultado</b>. Dentro de cada pestaña selecciona <b>un</b> indicador para visualizar y proyectar.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

if flat.empty:
    st.stop()

# ============ Tabs por Resultado ============
resultados = sorted(flat["resultado"].dropna().unique())
tabs = st.tabs(resultados if resultados else ["(sin datos)"])

# ============ Contenido por pestaña ============
import plotly.express as px

for i, res in enumerate(resultados):
    with tabs[i]:
        df_res = flat[flat["resultado"] == res].copy()
        if df_res.empty:
            st.info("Sin datos para este resultado.")
            continue

        # lista de indicadores disponibles
        inds = sorted(df_res["indicador"].dropna().unique())

        # leer de URL si viene pre-cargado
        ind_from_url = get_query_param("ind", default=None, cast=str)
        res_from_url = get_query_param("res", default=None, cast=str)
        yrmin_from_url = get_query_param("ymin", default=None, cast=int)
        yrmax_from_url = get_query_param("ymax", default=None, cast=int)

        # selector de indicador (una sola serie)
        c1, c2 = st.columns([3,2], vertical_alignment="bottom")
        with c1:
            if ind_from_url in inds and res_from_url == res:
                default_idx = inds.index(ind_from_url)
            else:
                default_idx = 0
            sel_ind = st.selectbox("Indicador", inds, index=default_idx, key=f"ind_{i}")

        # yrs para slider
        yr_min_real = int(df_res["anio"].min())
        yr_max_real = int(df_res["anio"].max())
        default_range = (
            yrmin_from_url if (yrmin_from_url and res_from_url == res) else yr_min_real,
            yrmax_from_url if (yrmax_from_url and res_from_url == res) else yr_max_real
        )
        with c2:
            ymin, ymax = st.slider(
                "Rango histórico a mostrar",
                min_value=yr_min_real,
                max_value=max(yr_max_real, yr_min_real),
                value=(int(default_range[0]), int(default_range[1])),
                key=f"yrs_{i}"
            )

        # persistimos en URL para compartir
        persist_query_params(res=res, ind=sel_ind, ymin=ymin, ymax=ymax)

        # ---------- Datos histórico (serie única) ----------
        hist = (
            df_res.query("indicador == @sel_ind")[["anio","valor"]]
            .dropna()
            .sort_values("anio")
        )
        hist_vis = hist.query("@ymin <= anio <= @ymax") if not hist.empty else hist.copy()

        # ---------- Datos proyección ----------
        proy = None
        if not proy_df.empty and {"anio_proy","valor_proy"}.issubset(set(proy_df.columns)):
            p = proy_df.copy()
            # si proyección tiene columna indicador, filtramos por sel_ind
            if "indicador" in p.columns:
                p = p.query("indicador == @sel_ind")
            # si proyección tiene columna resultado, intentamos empatar
            if "resultado" in p.columns:
                p = p.query("resultado == @res")
            if not p.empty:
                proy = p.rename(columns={"anio_proy":"anio","valor_proy":"valor"})[["anio","valor"]].dropna().sort_values("anio")

        # ---------- KPIs ----------
        k1, k2, k3 = st.columns(3)
        if not hist.empty:
            last_row = hist.iloc[-1]
            prev_row = hist.iloc[-2] if len(hist) > 1 else None
            delta = (last_row["valor"] - prev_row["valor"]) if prev_row is not None else None
            k1.metric("Último valor (hist.)", f"{last_row['valor']:.2f}", f"{delta:+.2f}" if delta is not None else None)
        else:
            k1.metric("Último valor (hist.)", "–")

        if proy is not None and not proy.empty:
            k2.metric("Primera proyección", f"{proy.iloc[0]['valor']:.2f}")
            k3.metric("Horizonte de proyección", f"{int(proy['anio'].min())}–{int(proy['anio'].max())}")
        else:
            k2.metric("Primera proyección", "–")
            k3.metric("Horizonte de proyección", "–")

        # ---------- Gráfico ----------
        g_parts = []
        if not hist.empty:
            g_parts.append(hist.assign(tipo="Histórico"))
        if proy is not None and not proy.empty:
            g_parts.append(proy.assign(tipo="Proyección"))

        if g_parts:
            gg = pd.concat(g_parts, ignore_index=True).sort_values("anio")
            fig = px.line(
                gg, x="anio", y="valor", color="tipo",
                markers=True,
                color_discrete_map={"Histórico":"#2563eb","Proyección":"#dc2626"},
                line_dash="tipo",
            )
            # Asegurar punteado para Proyección
            for tr in fig.data:
                if tr.name == "Proyección":
                    tr.line.update(dash="dash")
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="año",
                yaxis_title="valor",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos para graficar este indicador.")

        # ---------- Tablas y descarga ----------
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Histórico (rango visible)**")
            if not hist_vis.empty:
                st.dataframe(
                    hist_vis.rename(columns={"anio":"año","valor":"valor"}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.caption("Sin datos históricos en el rango seleccionado.")

        with t2:
            st.markdown("**Proyección**")
            if proy is not None and not proy.empty:
                st.dataframe(
                    proy.rename(columns={"anio":"año","valor":"valor"}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.caption("Sin proyecciones disponibles para este indicador.")

        # CSV combinado (hist + proy)
        if g_parts:
            out = pd.concat(g_parts, ignore_index=True).sort_values("anio")
            st.download_button(
                "⬇️ Descargar serie (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"serie_{canon(res).replace(' ','_')}_{canon(sel_ind).replace(' ','_')}.csv",
                mime="text/csv",
                key=f"dl_{i}"
            )

# ============ Sección de ayuda opcional ============
with st.expander("Ayuda y notas"):
    st.markdown(
        """
        **Consejos de uso**
        - Cada pestaña corresponde a un *resultado*.
        - Selecciona **un** indicador por pestaña.
        - Usa el **slider** para ajustar el rango histórico visible.
        - El enlace de tu navegador incluye los parámetros actuales: compártelo para que otros vean **exactamente la misma vista**.
        - El botón **Descargar serie (CSV)** exporta solo lo que necesitas (histórico + proyección de la serie seleccionada).

        **Colores y estilos**
        - **Histórico**: azul sólido  
        - **Proyección**: rojo punteado
        """
    )
