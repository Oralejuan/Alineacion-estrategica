import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import sqlite3

# Configuración de página
st.set_page_config(page_title="Alineación Estratégica", layout="wide")
st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("""
Ingresa el objetivo o descripción de tu proyecto.  
La herramienta buscará los **instrumentos**, **objetivos**, **metas** y **conceptos estratégicos** más relevantes.
""")

# -------------------------------------------------------------
# Funciones de utilidad
# -------------------------------------------------------------
def color_similitud(score):
    """Devuelve un emoji de color según el score."""
    if score >= 0.7:
        return "🟢"
    elif score >= 0.4:
        return "🟡"
    else:
        return "🔴"

def preprocess(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    return text.split()

def build_vocab(corpus):
    vocab = set()
    for tokens in corpus:
        vocab.update(tokens)
    return {word: idx for idx, word in enumerate(sorted(vocab))}

def compute_tf(tokens, vocab):
    tf_vec = np.zeros(len(vocab))
    counter = Counter(tokens)
    for word, count in counter.items():
        if word in vocab:
            tf_vec[vocab[word]] = count / len(tokens)
    return tf_vec

def compute_idf(corpus_tokens, vocab):
    N = len(corpus_tokens)
    idf = np.zeros(len(vocab))
    for word, idx in vocab.items():
        df = sum(1 for tokens in corpus_tokens if word in set(tokens))
        idf[idx] = math.log((1 + N) / (1 + df)) + 1
    return idf

def vectorize_tfidf(tf_matrix, idf_vector):
    return tf_matrix * idf_vector

def cosine_similarity_vec(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

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
# Preparación índices TF-IDF
# -------------------------------------------------------------
@st.cache_resource
def preparar_indices(df_objs, df_metas):
    textos_objs = (df_objs['nombre'].fillna('') + " " + df_objs['descripción'].fillna('')).tolist()
    tokens_objs = [preprocess(t) for t in textos_objs]
    textos_metas = df_metas['descripcion'].fillna('').tolist()
    tokens_metas = [preprocess(t) for t in textos_metas]
    all_tokens = tokens_objs + tokens_metas
    vocab = build_vocab(all_tokens)
    tf_objs = np.array([compute_tf(tok, vocab) for tok in tokens_objs])
    tf_metas = np.array([compute_tf(tok, vocab) for tok in tokens_metas])
    idf = compute_idf(all_tokens, vocab)
    tfidf_objs = vectorize_tfidf(tf_objs, idf)
    tfidf_metas = vectorize_tfidf(tf_metas, idf)
    return tfidf_objs, tfidf_metas, vocab, idf

def buscar_alineacion(query, tfidf_objs, tfidf_metas, vocab, idf,
                      df_objs, df_metas, df_rel, top_n=10):
    tokens_q = preprocess(query)
    if not tokens_q:
        return []
    tf_q = compute_tf(tokens_q, vocab)
    tfidf_q = tf_q * idf
    sim_objs = [cosine_similarity_vec(tfidf_q, vec) for vec in tfidf_objs]
    sim_metas = [cosine_similarity_vec(tfidf_q, vec) for vec in tfidf_metas]
    idx_objs = np.argsort(sim_objs)[::-1][:top_n]
    idx_metas = np.argsort(sim_metas)[::-1][:top_n]
    resultados = []
    for idx in idx_objs:
        score = sim_objs[idx]
        if score < 0.01: continue
        row = df_objs.iloc[idx]
        resultados.append({
            'tipo': 'Objetivo',
            'instrumento': row['instrumento'],
            'nombre': row['nombre'],
            'descripcion': row['descripción'],
            'nivel': row['nivel'],
            'similitud': round(score, 3)
        })
    col_meta = obtener_columna_meta(df_rel)
    for idx in idx_metas:
        score = sim_metas[idx]
        if score < 0.01: continue
        row = df_metas.iloc[idx]
        obj_row = df_objs[df_objs['id_objetivo'] == row['id_objetivo']]
        instrumento = obj_row.iloc[0]['instrumento'] if not obj_row.empty else 'No encontrado'
        obj_nombre = obj_row.iloc[0]['nombre'] if not obj_row.empty else ''
        conceptos = []
        if col_meta is not None and 'Concepto' in df_rel.columns:
            conceptos_df = df_rel[df_rel[col_meta] == row['id_meta']]
            conceptos = conceptos_df['Concepto'].dropna().unique().tolist()
        resultados.append({
            'tipo': 'Meta',
            'id_meta': row['id_meta'],
            'instrumento': instrumento,
            'objetivo_nombre': obj_nombre,
            'descripcion': row['descripcion'],
            'horizonte': row['horizonte'],
            'sector': row['sector'],
            'similitud': round(score, 3),
            'conceptos': conceptos
        })
    resultados.sort(key=lambda x: x['similitud'], reverse=True)
    return resultados

# -------------------------------------------------------------
# Exploración jerárquica y descriptiva
# -------------------------------------------------------------
def mostrar_instrumentos_jerarquico(df_inst, df_objs, df_metas, df_rel):
    """Muestra estructura: Instrumento → Objetivo → Meta (con conceptos)."""
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
                            # Conceptos asociados
                            col_meta = obtener_columna_meta(df_rel)
                            conceptos = []
                            if col_meta is not None and 'Concepto' in df_rel.columns:
                                conceptos_df = df_rel[df_rel[col_meta] == meta['id_meta']]
                                conceptos = conceptos_df['Concepto'].dropna().unique().tolist()
                            conceptos_str = ", ".join(conceptos[:3]) + ("..." if len(conceptos) > 3 else "")
                            st.markdown(f"**Meta:** {meta['descripcion']}")
                            st.caption(f"Horizonte: {meta['horizonte']} | Sector: {meta['sector']} | Conceptos: {conceptos_str}")

def mostrar_tabla_descriptiva(df, col_nombre, col_definicion):
    """Selectbox para elegir un elemento y ver su definición."""
    if df.empty:
        st.warning("No hay datos disponibles.")
        return
    opciones = df[col_nombre].tolist()
    seleccion = st.selectbox(f"Selecciona un {col_nombre}", opciones)
    if seleccion:
        fila = df[df[col_nombre] == seleccion].iloc[0]
        st.info(f"**Definición:** {fila[col_definicion]}")
        if 'Fuente / Marco' in df.columns:
            st.caption(f"Fuente: {fila['Fuente / Marco']}")

# -------------------------------------------------------------
# Exportar resultados a CSV
# -------------------------------------------------------------
def exportar_resultados_csv(resultados):
    """Convierte lista de resultados a CSV."""
    df_export = pd.DataFrame(resultados)
    columnas_deseadas = ['tipo', 'instrumento', 'similitud']
    if 'nombre' in df_export.columns:
        columnas_deseadas.append('nombre')
    if 'descripcion' in df_export.columns:
        columnas_deseadas.append('descripcion')
    if 'objetivo_nombre' in df_export.columns:
        columnas_deseadas.append('objetivo_nombre')
    if 'horizonte' in df_export.columns:
        columnas_deseadas.append('horizonte')
    if 'sector' in df_export.columns:
        columnas_deseadas.append('sector')
    if 'conceptos' in df_export.columns:
        df_export['conceptos_str'] = df_export['conceptos'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
        columnas_deseadas.append('conceptos_str')
    # Filtrar solo columnas existentes
    columnas_final = [c for c in columnas_deseadas if c in df_export.columns]
    df_export = df_export[columnas_final]
    return df_export.to_csv(index=False, sep=';', encoding='utf-8-sig')

# -------------------------------------------------------------
# Interfaz principal
# -------------------------------------------------------------
def main():
    # Sidebar
    modo = st.sidebar.radio(
        "Modo de uso",
        ["🔍 Alineación de proyectos", "📚 Explorar catálogos"]
    )
    
    datos = cargar_datos()
    if datos[0] is None:
        return
    df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas = datos
    
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Objetivos: {len(df_objs)} | Metas: {len(df_metas)}")
    
    # Modo alineación
    if modo == "🔍 Alineación de proyectos":
        with st.spinner("Preparando índices de búsqueda..."):
            tfidf_objs, tfidf_metas, vocab, idf = preparar_indices(df_objs, df_metas)
        
        consulta = st.text_area("Describe tu proyecto:", height=100)
        top_n = st.slider("Número de resultados a mostrar", 5, 20, 10)
        
        if st.button("Buscar", type="primary"):
            if not consulta.strip():
                st.warning("Ingresa una descripción.")
            else:
                with st.spinner("Buscando..."):
                    resultados = buscar_alineacion(consulta, tfidf_objs, tfidf_metas,
                                                   vocab, idf, df_objs, df_metas,
                                                   df_rel, top_n)
                if not resultados:
                    st.info("No se encontraron resultados relevantes.")
                else:
                    st.success(f"Se encontraron {len(resultados)} resultados.")
                    # Botón de exportación
                    csv_data = exportar_resultados_csv(resultados)
                    st.download_button("📥 Exportar resultados a CSV", data=csv_data,
                                       file_name="resultados_alineacion.csv", mime="text/csv")
                    # Mostrar resultados con color
                    for i, res in enumerate(resultados):
                        color = color_similitud(res['similitud'])
                        with st.expander(f"{color} {i+1}. {res['tipo']} - Similitud: {res['similitud']}"):
                            col1, col2 = st.columns([1,2])
                            with col1:
                                st.metric("Score", f"{res['similitud']:.3f}")
                                if res['tipo'] == 'Meta':
                                    st.write(f"**Horizonte:** {res['horizonte']}")
                                    st.write(f"**Sector:** {res['sector']}")
                            with col2:
                                st.write(f"**Instrumento:** {res['instrumento']}")
                                if res['tipo'] == 'Objetivo':
                                    st.write(f"**Nombre:** {res['nombre']}")
                                    st.write(f"**Descripción:** {res['descripcion'][:300]}")
                                else:
                                    st.write(f"**Objetivo asociado:** {res['objetivo_nombre']}")
                                    st.write(f"**Meta:** {res['descripcion'][:300]}")
                                    if res.get('conceptos'):
                                        st.write("**Conceptos clave:**", ", ".join(res['conceptos'][:5]))
    
    # Modo explorar catálogos
    else:
        st.header("📚 Explorar catálogos")
        submodo = st.selectbox("Selecciona el tipo de exploración",
                                ["Instrumentos (Jerárquico)", "Conceptos Estratégicos", "Amenazas"])
        if submodo == "Instrumentos (Jerárquico)":
            mostrar_instrumentos_jerarquico(df_inst, df_objs, df_metas, df_rel)
        elif submodo == "Conceptos Estratégicos":
            mostrar_tabla_descriptiva(df_conceptos, "Concepto", "Definición")
        elif submodo == "Amenazas":
            mostrar_tabla_descriptiva(df_amenazas, "amenaza", "descripcion_amenazas")

if __name__ == "__main__":
    main()
