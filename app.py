import streamlit as st
import pandas as pd
import numpy as np
import re
import sqlite3
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Alineación Estratégica", layout="wide")
st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("""
Ingresa el objetivo o descripción de tu proyecto.  
La herramienta buscará **objetivos**, **metas** y **conceptos estratégicos** relevantes usando **embeddings multilingües**.
""")

# -------------------------------------------------------------
# Funciones de utilidad
# -------------------------------------------------------------
def color_similitud(score):
    if score >= 0.7:
        return "🟢"
    elif score >= 0.4:
        return "🟡"
    else:
        return "🔴"

def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    return text.strip()

# -------------------------------------------------------------
# Carga de datos desde SQLite
# -------------------------------------------------------------
@st.cache_data
def cargar_datos():
    db_path = "mi_base_de_datos_v2.db"
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
        return df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        return None, None, None, None, None, None

def obtener_columna_meta(df_rel):
    posibles = ['id_meta', 'meta_id', 'ID_META', 'idMeta']
    for col in posibles:
        if col in df_rel.columns:
            return col
    for col in df_rel.columns:
        if 'meta' in col.lower():
            return col
    return None

# -------------------------------------------------------------
# Cargar modelo de embeddings (cacheado como recurso)
# -------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# -------------------------------------------------------------
# Generar embeddings y guardar en session_state
# -------------------------------------------------------------
def generar_embeddings(model, df_objs, df_metas, df_conceptos):
    # Textos
    textos_objs = (df_objs['nombre'].fillna('') + " " + df_objs['descripción'].fillna('')).apply(preprocess).tolist()
    textos_metas = df_metas['descripcion'].fillna('').apply(preprocess).tolist()
    textos_conceptos = (df_conceptos['Concepto'].fillna('') + " " + df_conceptos['Definición'].fillna('')).apply(preprocess).tolist()
    
    # Generar embeddings
    with st.spinner("Generando embeddings (puede tardar un minuto la primera vez)..."):
        emb_objs = model.encode(textos_objs, show_progress_bar=False)
        emb_metas = model.encode(textos_metas, show_progress_bar=False)
        emb_conceptos = model.encode(textos_conceptos, show_progress_bar=False)
    
    return emb_objs, emb_metas, emb_conceptos

def buscar_alineacion(query, model, emb_objs, emb_metas, emb_conceptos,
                      df_objs, df_metas, df_conceptos, df_rel, top_n=10):
    query_limpia = preprocess(query)
    if not query_limpia:
        return [], [], []
    emb_q = model.encode([query_limpia])
    sim_objs = cosine_similarity(emb_q, emb_objs).flatten()
    sim_metas = cosine_similarity(emb_q, emb_metas).flatten()
    sim_conceptos = cosine_similarity(emb_q, emb_conceptos).flatten()
    
    idx_objs = np.argsort(sim_objs)[::-1][:top_n]
    idx_metas = np.argsort(sim_metas)[::-1][:top_n]
    idx_conceptos = np.argsort(sim_conceptos)[::-1][:top_n]
    
    resultados_objs = []
    for idx in idx_objs:
        score = sim_objs[idx]
        if score < 0.01: continue
        row = df_objs.iloc[idx]
        resultados_objs.append({
            'instrumento': row['instrumento'],
            'nombre': row['nombre'],
            'descripcion': row['descripción'],
            'nivel': row['nivel'],
            'similitud': round(score, 3)
        })
    
    resultados_metas = []
    col_meta = obtener_columna_meta(df_rel)
    for idx in idx_metas:
        score = sim_metas[idx]
        if score < 0.01: continue
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
        if score < 0.01: continue
        row = df_conceptos.iloc[idx]
        resultados_conceptos.append({
            'concepto': row['Concepto'],
            'definicion': row['Definición'],
            'fuente': row.get('Fuente / Marco', ''),
            'similitud': round(score, 3)
        })
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
# Dashboard estadístico
# -------------------------------------------------------------
def mostrar_dashboard(df_inst, df_objs, df_metas, df_conceptos, df_amenazas):
    st.header("📊 Dashboard de la base de datos")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Instrumentos", len(df_inst))
    col2.metric("Objetivos", len(df_objs))
    col3.metric("Metas", len(df_metas))
    col4.metric("Conceptos", len(df_conceptos))
    
    if not df_inst.empty and 'año_inicio' in df_inst.columns:
        df_inst_year = df_inst[df_inst['año_inicio'].notna() & df_inst['año_inicio'].astype(str).str.isnumeric()]
        if not df_inst_year.empty:
            df_inst_year['año_inicio'] = df_inst_year['año_inicio'].astype(int)
            fig = px.bar(df_inst_year.groupby('año_inicio').size().reset_index(name='cantidad'),
                         x='año_inicio', y='cantidad', title="Instrumentos por año de inicio")
            st.plotly_chart(fig, use_container_width=True)
    
    if not df_metas.empty and 'sector' in df_metas.columns:
        sector_counts = df_metas['sector'].dropna().value_counts().reset_index()
        sector_counts.columns = ['sector', 'cantidad']
        fig = px.bar(sector_counts, x='sector', y='cantidad', title="Metas por sector")
        st.plotly_chart(fig, use_container_width=True)
    
    if not df_metas.empty and 'horizonte' in df_metas.columns:
        horizonte_counts = df_metas['horizonte'].dropna().value_counts().reset_index()
        horizonte_counts.columns = ['horizonte', 'cantidad']
        fig = px.pie(horizonte_counts, values='cantidad', names='horizonte', title="Distribución de horizontes")
        st.plotly_chart(fig, use_container_width=True)
    
    if not df_conceptos.empty and 'Concepto' in df_conceptos.columns:
        conceptos_counts = df_conceptos['Concepto'].value_counts().reset_index()
        conceptos_counts.columns = ['concepto', 'frecuencia']
        fig = px.bar(conceptos_counts.head(20), x='concepto', y='frecuencia', title="Conceptos más frecuentes")
        st.plotly_chart(fig, use_container_width=True)
    
    if not df_amenazas.empty and 'categoria' in df_amenazas.columns:
        cat_counts = df_amenazas['categoria'].dropna().value_counts().reset_index()
        cat_counts.columns = ['categoría', 'cantidad']
        fig = px.bar(cat_counts, x='categoría', y='cantidad', title="Amenazas por categoría")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Exploración jerárquica y catalogos
# -------------------------------------------------------------
def mostrar_instrumentos_jerarquico(df_inst, df_objs, df_metas, df_rel):
    for inst in df_inst['nombre'].unique():
        with st.expander(f"📜 {inst}"):
            objs = df_objs[df_objs['instrumento'] == inst]
            for _, obj in objs.iterrows():
                with st.expander(f"🎯 {obj['nombre']}"):
                    st.write(f"**Descripción:** {obj['descripción']}")
                    metas = df_metas[df_metas['id_objetivo'] == obj['id_objetivo']]
                    for _, meta in metas.iterrows():
                        col_meta = obtener_columna_meta(df_rel)
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
        ["🔍 Alineación de proyectos", "📚 Explorar catálogos", "📊 Dashboard estadístico"]
    )
    datos = cargar_datos()
    if datos[0] is None:
        return
    df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas = datos
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Objetivos: {len(df_objs)} | Metas: {len(df_metas)} | Conceptos: {len(df_conceptos)}")
    
    if modo == "📊 Dashboard estadístico":
        mostrar_dashboard(df_inst, df_objs, df_metas, df_conceptos, df_amenazas)
        return
    
    if modo == "🔍 Alineación de proyectos":
        model = cargar_modelo()
        # Inicializar embeddings en session_state si no existen
        if 'emb_objs' not in st.session_state:
            with st.spinner("Generando índices de búsqueda (puede tardar un minuto la primera vez)..."):
                emb_objs, emb_metas, emb_conceptos = generar_embeddings(model, df_objs, df_metas, df_conceptos)
                st.session_state.emb_objs = emb_objs
                st.session_state.emb_metas = emb_metas
                st.session_state.emb_conceptos = emb_conceptos
        else:
            emb_objs = st.session_state.emb_objs
            emb_metas = st.session_state.emb_metas
            emb_conceptos = st.session_state.emb_conceptos
        
        consulta = st.text_area("Describe tu proyecto:", height=100)
        top_n = st.slider("Número de resultados por categoría", 5, 20, 10)
        # Filtros laterales
        with st.sidebar:
            st.subheader("🔎 Filtros")
            mostrar = st.multiselect("Mostrar", ["Objetivos", "Metas", "Conceptos"], default=["Objetivos", "Metas", "Conceptos"])
            min_score = st.slider("Similitud mínima", 0.0, 1.0, 0.1, 0.05)
            sectores_filtro = st.multiselect("Sector (solo metas)", df_metas['sector'].dropna().unique())
            horizontes_filtro = st.multiselect("Horizonte (solo metas)", df_metas['horizonte'].dropna().unique())
        
        if st.button("Buscar", type="primary"):
            if not consulta.strip():
                st.warning("Ingresa una descripción.")
            else:
                with st.spinner("Buscando..."):
                    res_objs, res_metas, res_conceptos = buscar_alineacion(
                        consulta, model, emb_objs, emb_metas, emb_conceptos,
                        df_objs, df_metas, df_conceptos, df_rel, top_n)
                # Aplicar filtros
                if "Objetivos" not in mostrar:
                    res_objs = []
                if "Metas" not in mostrar:
                    res_metas = []
                if "Conceptos" not in mostrar:
                    res_conceptos = []
                res_objs = [r for r in res_objs if r['similitud'] >= min_score]
                res_metas = [r for r in res_metas if r['similitud'] >= min_score]
                res_conceptos = [r for r in res_conceptos if r['similitud'] >= min_score]
                if sectores_filtro:
                    res_metas = [r for r in res_metas if r['sector'] in sectores_filtro]
                if horizontes_filtro:
                    res_metas = [r for r in res_metas if r['horizonte'] in horizontes_filtro]
                
                if not (res_objs or res_metas or res_conceptos):
                    st.info("No se encontraron resultados con los filtros actuales.")
                else:
                    st.success(f"Encontrados: {len(res_objs)} objetivos, {len(res_metas)} metas, {len(res_conceptos)} conceptos.")
                    csv_data = exportar_resultados_csv(res_objs, res_metas, res_conceptos)
                    st.download_button("📥 Exportar resultados a CSV", data=csv_data,
                                       file_name="resultados_alineacion.csv", mime="text/csv")
                    tabs = st.tabs(["🎯 Objetivos", "📋 Metas", "🧠 Conceptos"])
                    with tabs[0]:
                        for r in res_objs:
                            with st.expander(f"{color_similitud(r['similitud'])} {r['nombre']} (score: {r['similitud']})"):
                                st.write(f"**Instrumento:** {r['instrumento']}")
                                st.write(f"**Descripción:** {r['descripcion']}")
                    with tabs[1]:
                        for r in res_metas:
                            with st.expander(f"{color_similitud(r['similitud'])} Meta (score: {r['similitud']})"):
                                st.write(f"**Instrumento:** {r['instrumento']}")
                                st.write(f"**Objetivo asociado:** {r['objetivo_nombre']}")
                                st.write(f"**Descripción:** {r['descripcion']}")
                                st.write(f"**Horizonte:** {r['horizonte']} | **Sector:** {r['sector']}")
                                if r['conceptos']:
                                    st.write("**Conceptos clave:**", ", ".join(r['conceptos'][:5]))
                    with tabs[2]:
                        for r in res_conceptos:
                            with st.expander(f"{color_similitud(r['similitud'])} {r['concepto']} (score: {r['similitud']})"):
                                st.write(f"**Definición:** {r['definicion']}")
                                if r['fuente']:
                                    st.caption(f"Fuente: {r['fuente']}")
    
    else:  # Explorar catálogos
        st.header("📚 Explorar catálogos")
        cat = st.selectbox("Selecciona un catálogo", ["Instrumentos", "Objetivos", "Metas", "Conceptos Estratégicos", "Amenazas"])
        texto_buscar = st.text_input("Filtrar por texto")
        if cat == "Instrumentos":
            df = df_inst
            cols = ['nombre', 'escala', 'año_inicio', 'año_fin', 'entidad_lider', 'tematica_principal']
        elif cat == "Objetivos":
            df = df_objs
            cols = ['id_objetivo', 'instrumento', 'nombre', 'descripción', 'nivel']
        elif cat == "Metas":
            df = df_metas
            cols = ['id_meta', 'id_objetivo', 'descripcion', 'horizonte', 'sector']
        elif cat == "Conceptos Estratégicos":
            df = df_conceptos
            cols = ['id_concepto', 'Concepto', 'Definición', 'Fuente / Marco']
        else:
            df = df_amenazas
            cols = ['id_amenazas', 'categoria', 'subcategoria', 'amenaza', 'descripcion_amenazas']
        df_filt = filtrar_tabla(df, texto_buscar)
        st.write(f"Mostrando {len(df_filt)} de {len(df)} filas")
        st.dataframe(df_filt[cols], use_container_width=True)

if __name__ == "__main__":
    main()
