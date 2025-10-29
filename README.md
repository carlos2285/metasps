# Dashboard de Indicadores — Streamlit

## Estructura
- `app.py`: aplicación Streamlit
- `indicadores_flat.csv`: tabla plana a partir del Excel fuente
- `pivot_2014_2023.csv`, `pivot_2014_2019.csv`, `pivot_2020_2024.csv`: tablas dinámicas
- `proyecciones_2026_2029.csv`: proyecciones lineales por indicador (si hay ≥3 años con datos)
- `mapeo_columnas_por_hoja.csv`: auditoría del mapeo de columnas por hoja
- `requirements.txt`: dependencias

## Cómo ejecutar
```bash
pip install -r requirements.txt
streamlit run app.py
```

> En la barra lateral puedes ajustar las rutas de los CSV o subir `indicadores_flat.csv` manualmente.