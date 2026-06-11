import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Alineación Estratégica", layout="wide")
st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("""
Ingresa el objetivo o descripción de tu proyecto.  
La herramienta buscará **objetivos**, **metas** y **conceptos estratégicos** usando inteligencia semántica.
""")

# -------------------------------------------------------------
# Carga de datos desde SQLite
# -------------------------------------------------------------
@st.cache_resource
def cargar_datos():
    db_path = "mi_base_de_datos.db"
    try:
        conn = sqlite3.connect(db_path)
        df_inst = pd.read_sql_query("SELECT * FROM INSTRUMENTOS", conn)
        df_objs = pd.read_sql_query("SELECT * FROM OBJETIVOS", conn)
        df_metas = pd.read_sql_query("SELECT * FROM METAS", conn)
        df_rel = pd.read_sql_query("SELECT * FROM REL_META_CONCEPTO", conn)
        df_conceptos = pd.read_sql_query("SELECT * FROM CONCEPTOS_ESTRATEGICOS", conn)
        try:
            df_amenazas = pd.read_sql_query("SELECT * FROM AMENAZAS", conn)
        except:
            df_amenazas = pd.DataFrame()
        conn.close()
        # Limpiar duplicados en objetivos por si acaso
        df_objs = df_objs.drop_duplicates(subset=['id_objetivo'])
        return df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        return None, None, None, None, None, None

@st.cache_resource
def cargar_modelo():
    """Carga el modelo multilingüe de Sentence Transformers."""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def preparar_embeddings(model, df_objs, df_metas, df_conceptos):
    """Genera embeddings para objetivos, metas y conceptos."""
    # Textos de objetivos (nombre + descripción)
    textos_objs = (df_objs['nombre'].fillna('') + " " + df_objs['descripción'].fillna('')).tolist()
    # Textos de metas (descripción)
    textos_metas = df_metas['descripcion'].fillna('').tolist()
    # Textos de conceptos (nombre + definición)
    textos_conceptos = (df_conceptos['Concepto'].fillna('') + " " + df_conceptos['Definición'].fillna('')).tolist()

    # Codificar en lotes (muestra progreso)
    with st.spinner("Generando embeddings (puede tomar un minuto la primera vez)..."):
        emb_objs = model.encode(textos_objs, show_progress_bar=False)
        emb_metas = model.encode(textos_metas, show_progress_bar=False)
        emb_conceptos = model.encode(textos_conceptos, show_progress_bar=False)
    return emb_objs, emb_metas, emb_conceptos

def buscar_semantica(query, model, emb_objs, emb_metas, emb_conceptos,
                     df_objs, df_metas, df_conceptos, df_rel, top_n=10):
    """Codifica la consulta y devuelve los más similares en cada categoría."""
    emb_query = model.encode([query])
    sim_objs = cosine_similarity(emb_query, emb_objs).flatten()
    sim_metas = cosine_similarity(emb_query, emb_metas).flatten()
    sim_conceptos = cosine_similarity(emb_query, emb_conceptos).flatten()

    idx_objs = np.argsort(sim_objs)[::-1][:top_n]
    idx_metas = np.argsort(sim_metas)[::-1][:top_n]
    idx_conceptos = np.argsort(sim_conceptos)[::-1][:top_n]

    resultados_objs = []
    for idx in idx_objs:
        score = sim_objs[idx]
        if score < 0.1: continue
        row = df_objs.iloc[idx]
        resultados_objs.append({
            'instrumento': row['instrumento'],
            'nombre': row['nombre'],
            'descripcion': row['descripción'],
            'nivel': row['nivel'],
            'similitud': round(score, 3)
        })

    # Para metas, necesitamos obtener conceptos asociados
    # Determinar columna de id_meta en df_rel
    col_meta = None
    posibles = ['id_meta', 'meta_id', 'ID_META', 'idMeta']
    for col in posibles:
        if col in df_rel.columns:
            col_meta = col
            break
    if col_meta is None:
        for col in df_rel.columns:
            if 'meta' in col.lower():
                col_meta = col
                break

    resultados_metas = []
    for idx in idx_metas:
        score = sim_metas[idx]
        if score < 0.1: continue
        row = df_metas.iloc[idx]
        obj_row = df_objs[df_objs['id_objetivo'] == row['id_objetivo']]
        instrumento = obj_row.iloc[0]['instrumento'] if not obj_row.empty else 'No encontrado'
        objetivo_nombre = obj_row.iloc[0]['nombre'] if not obj_row.empty else ''
        conceptos = []
        if col_meta is not None and 'Concepto' in df_rel.columns:
            conceptos_df = df_rel[df_rel[col_meta] == row['id_meta']]
            conceptos = conceptos_df['Concepto'].dropna().unique().tolist()
        resultados_metas.append({
            'id_meta': row['id_meta'],
            'instrumento': instrumento,
            'objetivo_nombre': objetivo_nombre,
            'descripcion': row['descripcion'],
            'horizonte': row['horizonte'],
            'sector': row['sector'],
            'similitud': round(score, 3),
            'conceptos': conceptos
        })

    resultados_conceptos = []
    for idx in idx_conceptos:
        score = sim_conceptos[idx]
        if score < 0.1: continue
        row = df_conceptos.iloc[idx]
        resultados_conceptos.append({
            'concepto': row['Concepto'],
            'definicion': row['Definición'],
            'fuente': row.get('Fuente / Marco', ''),
            'similitud': round(score, 3)
        })

    # Ordenar por similitud descendente (ya lo están por el orden de índices)
    return resultados_objs, resultados_metas, resultados_conceptos

def exportar_resultados_csv(res_objs, res_metas, res_conceptos):
    rows = []
    for r in res_objs:
        rows.append({'Tipo': 'Objetivo', 'Nombre': r['nombre'], 'Descripción': r['descripcion'],
                     'Instrumento': r['instrumento'], 'Similitud': r['similitud']})
    for r in res_metas:
        rows.append({'Tipo': 'Meta', 'Descripción': r['descripcion'], 'Instrumento': r['instrumento'],
                     'Objetivo asociado': r['objetivo_nombre'], 'Horizonte': r['horizonte'],
                     'Sector': r['sector'], 'Conceptos': ', '.join(r['conceptos']), 'Similitud': r['similitud']})
    for r in res_conceptos:
        rows.append({'Tipo': 'Concepto', 'Concepto': r['concepto'], 'Definición': r['definicion'],
                     'Fuente': r['fuente'], 'Similitud': r['similitud']})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, sep=';', encoding='utf-8-sig')

# -------------------------------------------------------------
# Dashboard de análisis estadístico
# -------------------------------------------------------------
def mostrar_dashboard(df_inst, df_objs, df_metas, df_conceptos):
    st.header("📊 Dashboard de la base de conocimiento")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Instrumentos", len(df_inst))
    col2.metric("Objetivos", len(df_objs))
    col3.metric("Metas", len(df_metas))
    col4.metric("Conceptos estratégicos", len(df_conceptos))

    # Distribución de metas por sector
    if not df_metas.empty and 'sector' in df_metas.columns:
        sector_counts = df_metas['sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Cantidad']
        fig_sector = px.bar(sector_counts, x='Sector', y='Cantidad', title='Metas por sector',
                            color='Cantidad', color_continuous_scale='Blues')
        st.plotly_chart(fig_sector, use_container_width=True)

    # Distribución por horizonte
    if not df_metas.empty and 'horizonte' in df_metas.columns:
        horizonte_counts = df_metas['horizonte'].value_counts().reset_index()
        horizonte_counts.columns = ['Horizonte', 'Cantidad']
        fig_horizonte = px.bar(horizonte_counts, x='Horizonte', y='Cantidad', title='Metas por horizonte temporal',
                               color='Cantidad', color_continuous_scale='Oranges')
        st.plotly_chart(fig_horizonte, use_container_width=True)

    # Top instrumentos con más objetivos
    if not df_objs.empty and 'instrumento' in df_objs.columns:
        top_inst = df_objs['instrumento'].value_counts().head(10).reset_index()
        top_inst.columns = ['Instrumento', 'Cantidad de objetivos']
        fig_top_inst = px.bar(top_inst, x='Instrumento', y='Cantidad de objetivos',
                              title='Top 10 instrumentos con más objetivos',
                              color='Cantidad de objetivos', color_continuous_scale='Greens')
        st.plotly_chart(fig_top_inst, use_container_width=True)

    # Nube de conceptos (si hay muchos, mejor tabla de frecuencias)
    if not df_conceptos.empty and 'Concepto' in df_conceptos.columns:
        st.subheader("Principales conceptos estratégicos")
        # Podríamos mostrar una tabla con los 20 primeros (o usar wordcloud, pero requiere PIL)
        # Mostramos una tabla simple
        conceptos_list = df_conceptos['Concepto'].dropna().tolist()
        freq = pd.Series(conceptos_list).value_counts().head(20).reset_index()
        freq.columns = ['Concepto', 'Frecuencia']
        st.dataframe(freq, use_container_width=True)

# -------------------------------------------------------------
# Exploración jerárquica y filtros
# -------------------------------------------------------------
def mostrar_instrumentos_jerarquico(df_inst, df_objs, df_metas, df_rel):
    instrumentos = df_inst['nombre'].unique()
    for inst in instrumentos:
        with st.expander(f"📜 {inst}"):
            objs = df_objs[df_objs['instrumento'] == inst]
            if objs.empty:
                st.write("No hay objetivos para este instrumento.")
                continue
            for _, obj in objs.iterrows():
                with st.expander(f"🎯 {obj['nombre']}"):
                    st.write(f"**Descripción:** {obj['descripción']}")
                    metas = df_metas[df_metas['id_objetivo'] == obj['id_objetivo']]
                    if metas.empty:
                        st.write("No hay metas para este objetivo.")
                    else:
                        for _, meta in metas.iterrows():
                            col_meta = None
                            posibles = ['id_meta', 'meta_id', 'ID_META', 'idMeta']
                            for col in posibles:
                                if col in df_rel.columns:
                                    col_meta = col
                                    break
                            conceptos = []
                            if col_meta is not None and 'Concepto' in df_rel.columns:
                                conceptos_df = df_rel[df_rel[col_meta] == meta['id_meta']]
                                conceptos = conceptos_df['Concepto'].dropna().unique().tolist()
                            conceptos_str = ", ".join(conceptos[:3]) + ("..." if len(conceptos) > 3 else "")
                            st.markdown(f"**Meta:** {meta['descripcion']}")
                            st.caption(f"Horizonte: {meta['horizonte']} | Sector: {meta['sector']} | Conceptos: {conceptos_str}")

def filtrar_tabla(df, texto_buscar):
    if df.empty or not texto_buscar:
        return df
    texto = texto_buscar.lower()
    mask = False
    for col in df.columns:
        if df[col].dtype == 'object':
            mask |= df[col].fillna('').astype(str).str.lower().str.contains(texto, na=False)
    return df[mask]

# -------------------------------------------------------------
# Interfaz principal
# -------------------------------------------------------------
def main():
    modo = st.sidebar.radio(
        "Modo de uso",
        ["🔍 Alineación de proyectos", "📚 Explorar catálogos", "📊 Dashboard de análisis"]
    )
    datos = cargar_datos()
    if datos[0] is None:
        return
    df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas = datos
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Objetivos: {len(df_objs)} | Metas: {len(df_metas)} | Conceptos: {len(df_conceptos)}")

    if modo == "🔍 Alineación de proyectos":
        # Cargar modelo y embeddings (cacheados)
        with st.spinner("Cargando modelo de lenguaje y preparando índices (primera vez puede tomar un minuto)..."):
            model = cargar_modelo()
            emb_objs, emb_metas, emb_conceptos = preparar_embeddings(model, df_objs, df_metas, df_conceptos)

        consulta = st.text_area("Describe tu proyecto:", height=100)
        top_n = st.slider("Número de resultados por categoría", 5, 20, 10)

        if st.button("Buscar", type="primary"):
            if not consulta.strip():
                st.warning("Ingresa una descripción.")
            else:
                with st.spinner("Buscando semánticamente..."):
                    res_objs, res_metas, res_conceptos = buscar_semantica(
                        consulta, model, emb_objs, emb_metas, emb_conceptos,
                        df_objs, df_metas, df_conceptos, df_rel, top_n)

                if not (res_objs or res_metas or res_conceptos):
                    st.info("No se encontraron resultados relevantes.")
                else:
                    st.success(f"Encontrados: {len(res_objs)} objetivos, {len(res_metas)} metas, {len(res_conceptos)} conceptos.")
                    csv_data = exportar_resultados_csv(res_objs, res_metas, res_conceptos)
                    st.download_button("📥 Exportar resultados a CSV", data=csv_data,
                                       file_name="resultados_alineacion.csv", mime="text/csv")

                    tabs = st.tabs(["🎯 Objetivos", "📋 Metas", "🧠 Conceptos"])
                    with tabs[0]:
                        if not res_objs:
                            st.info("No hay objetivos coincidentes.")
                        else:
                            for r in res_objs:
                                score = r['similitud']
                                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                                with st.expander(f"{color} {r['nombre']} (similitud: {score})"):
                                    st.write(f"**Instrumento:** {r['instrumento']}")
                                    st.write(f"**Descripción:** {r['descripcion']}")
                    with tabs[1]:
                        if not res_metas:
                            st.info("No hay metas coincidentes.")
                        else:
                            for r in res_metas:
                                score = r['similitud']
                                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                                with st.expander(f"{color} Meta (similitud: {score})"):
                                    st.write(f"**Instrumento:** {r['instrumento']}")
                                    st.write(f"**Objetivo asociado:** {r['objetivo_nombre']}")
                                    st.write(f"**Descripción:** {r['descripcion']}")
                                    st.write(f"**Horizonte:** {r['horizonte']} | **Sector:** {r['sector']}")
                                    if r['conceptos']:
                                        st.write("**Conceptos clave:**", ", ".join(r['conceptos'][:5]))
                    with tabs[2]:
                        if not res_conceptos:
                            st.info("No hay conceptos coincidentes.")
                        else:
                            for r in res_conceptos:
                                score = r['similitud']
                                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                                with st.expander(f"{color} {r['concepto']} (similitud: {score})"):
                                    st.write(f"**Definición:** {r['definicion']}")
                                    if r['fuente']:
                                        st.caption(f"Fuente: {r['fuente']}")

    elif modo == "📚 Explorar catálogos":
        st.header("📚 Explorar catálogos")
        cat_seleccion = st.selectbox(
            "Selecciona un catálogo",
            ["Instrumentos", "Objetivos", "Metas", "Conceptos Estratégicos", "Amenazas"]
        )
        texto_buscar = st.text_input("Filtrar por texto", key="filtro_catalogo")
        if cat_seleccion == "Instrumentos":
            df = df_inst
            columnas_mostrar = ['nombre', 'escala', 'año_inicio', 'año_fin', 'entidad_lider', 'tematica_principal']
        elif cat_seleccion == "Objetivos":
            df = df_objs
            columnas_mostrar = ['id_objetivo', 'instrumento', 'nombre', 'descripción', 'nivel']
        elif cat_seleccion == "Metas":
            df = df_metas
            columnas_mostrar = ['id_meta', 'id_objetivo', 'descripcion', 'horizonte', 'sector']
        elif cat_seleccion == "Conceptos Estratégicos":
            df = df_conceptos
            columnas_mostrar = ['id_concepto', 'Concepto', 'Definición', 'Fuente / Marco']
        else:  # Amenazas
            if df_amenazas.empty:
                st.warning("No hay datos de amenazas en la base de datos.")
                return
            df = df_amenazas
            columnas_mostrar = ['id_amenazas', 'categoria', 'subcategoria', 'amenaza', 'descripcion_amenazas']

        df_filtrado = filtrar_tabla(df, texto_buscar)
        st.write(f"Mostrando {len(df_filtrado)} de {len(df)} filas")
        st.dataframe(df_filtrado[columnas_mostrar], use_container_width=True)

    else:  # Dashboard de análisis
        mostrar_dashboard(df_inst, df_objs, df_metas, df_conceptos)

if __name__ == "__main__":
    main()
