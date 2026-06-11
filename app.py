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
# Funciones TF-IDF manual
# -------------------------------------------------------------
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
    """Carga todas las tablas desde la base de datos SQLite."""
    db_path = "mi_base_de_datos.db"
    try:
        conn = sqlite3.connect(db_path)
        # Cargar cada tabla
        df_inst = pd.read_sql_query("SELECT * FROM INSTRUMENTOS", conn)
        df_objs = pd.read_sql_query("SELECT * FROM OBJETIVOS", conn)
        df_metas = pd.read_sql_query("SELECT * FROM METAS", conn)
        df_rel = pd.read_sql_query("SELECT * FROM REL_META_CONCEPTO", conn)
        df_conceptos = pd.read_sql_query("SELECT * FROM CONCEPTOS_ESTRATEGICOS", conn)
        # Intentar cargar AMENAZAS si existe
        try:
            df_amenazas = pd.read_sql_query("SELECT * FROM AMENAZAS", conn)
        except:
            df_amenazas = pd.DataFrame()  # vacío si no existe
        conn.close()
        return df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        return None, None, None, None, None, None

def obtener_columna_meta(df_rel):
    """Devuelve el nombre de la columna que contiene el id de meta."""
    posibles = ['id_meta', 'meta_id', 'ID_META', 'ID_META', 'idMeta']
    for col in posibles:
        if col in df_rel.columns:
            return col
    # Si no, buscar cualquier columna que contenga 'meta' en el nombre
    for col in df_rel.columns:
        if 'meta' in col.lower():
            return col
    return None

# -------------------------------------------------------------
# Preparación de índices TF-IDF
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
    """Búsqueda semántica de alineación (como antes)."""
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
    # Objetivos
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
    # Metas (incluyendo conceptos)
    # Determinar columna de id_meta en df_rel
    col_meta = obtener_columna_meta(df_rel)
    for idx in idx_metas:
        score = sim_metas[idx]
        if score < 0.01: continue
        row = df_metas.iloc[idx]
        obj_row = df_objs[df_objs['id_objetivo'] == row['id_objetivo']]
        instrumento = obj_row.iloc[0]['instrumento'] if not obj_row.empty else 'No encontrado'
        obj_nombre = obj_row.iloc[0]['nombre'] if not obj_row.empty else ''
        # Obtener conceptos asociados
        conceptos = []
        if col_meta is not None:
            conceptos_df = df_rel[df_rel[col_meta] == row['id_meta']]
            if 'Concepto' in conceptos_df.columns:
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
# Funciones para buscadores de catálogos
# -------------------------------------------------------------
def buscar_en_tabla(df, texto_busqueda, columnas=None):
    """Filtra filas donde el texto aparezca en cualquier columna (o en columnas específicas)."""
    if df.empty or not texto_busqueda:
        return df
    texto_busqueda = texto_busqueda.lower()
    if columnas is None:
        columnas = df.columns
    mask = False
    for col in columnas:
        if df[col].dtype == 'object':
            mask |= df[col].fillna('').astype(str).str.lower().str.contains(texto_busqueda, na=False)
    return df[mask]

# -------------------------------------------------------------
# Interfaz Principal
# -------------------------------------------------------------
def main():
    # Sidebar: opciones de navegación
    modo = st.sidebar.radio(
        "Modo de uso",
        ["🔍 Alineación de proyectos", "📚 Explorar catálogos"]
    )
    
    # Cargar datos (una sola vez)
    with st.spinner("Cargando base de datos..."):
        datos = cargar_datos()
    if datos[0] is None:
        st.error("No se pudieron cargar los datos. Asegúrate de que 'mi_base_de_datos.db' existe.")
        return
    df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas = datos
    
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Objetivos: {len(df_objs)} | Metas: {len(df_metas)}")
    if not df_amenazas.empty:
        st.sidebar.write(f"Amenazas: {len(df_amenazas)}")
    
    if modo == "🔍 Alineación de proyectos":
        # Preparar índices TF-IDF (solo necesario en este modo)
        with st.spinner("Preparando índices TF-IDF (puede tardar unos segundos)..."):
            tfidf_objs, tfidf_metas, vocab, idf = preparar_indices(df_objs, df_metas)
        
        consulta = st.text_area("Describe tu proyecto:", height=100)
        top_n = st.slider("Resultados a mostrar", 5, 20, 10)
        
        if st.button("Buscar", type="primary"):
            if not consulta.strip():
                st.warning("Por favor ingresa una descripción.")
            else:
                with st.spinner("Buscando..."):
                    resultados = buscar_alineacion(consulta, tfidf_objs, tfidf_metas,
                                                   vocab, idf, df_objs, df_metas,
                                                   df_rel, top_n)
                if not resultados:
                    st.info("No se encontraron resultados relevantes.")
                else:
                    st.success(f"Se encontraron {len(resultados)} resultados.")
                    for i, res in enumerate(resultados):
                        with st.expander(f"{i+1}. {res['tipo']} - Score: {res['similitud']}"):
                            col1, col2 = st.columns([1,2])
                            with col1:
                                st.metric("Similitud", f"{res['similitud']:.3f}")
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
                                    if res['conceptos']:
                                        st.write("**Conceptos clave:**", ", ".join(res['conceptos'][:5]))
    
    else:  # Modo explorar catálogos
        st.header("📚 Explorar catálogos")
        pestaña = st.selectbox(
            "Selecciona el catálogo a explorar",
            ["Instrumentos", "Objetivos", "Metas", "Conceptos Estratégicos", "Amenazas"]
        )
        # Buscador de texto dentro de la tabla seleccionada
        texto_buscar = st.text_input("Filtrar por texto (opcional)", key="buscador_catalogo")
        
        if pestaña == "Instrumentos":
            df = df_inst
            columnas_mostrar = ['nombre', 'escala', 'año_inicio', 'año_fin', 'entidad_lider', 'tematica_principal']
        elif pestaña == "Objetivos":
            df = df_objs
            columnas_mostrar = ['id_objetivo', 'instrumento', 'nombre', 'descripción', 'nivel']
        elif pestaña == "Metas":
            df = df_metas
            columnas_mostrar = ['id_meta', 'id_objetivo', 'descripcion', 'horizonte', 'sector']
        elif pestaña == "Conceptos Estratégicos":
            df = df_conceptos
            columnas_mostrar = ['id_concepto', 'Concepto', 'Definición', 'Fuente / Marco']
        else:  # Amenazas
            if df_amenazas.empty:
                st.warning("No hay datos de amenazas en la base de datos.")
                return
            df = df_amenazas
            columnas_mostrar = ['id_amenazas', 'categoria', 'subcategoria', 'amenaza', 'descripcion_amenazas']
        
        # Aplicar filtro
        if texto_buscar:
            df_filtrado = buscar_en_tabla(df, texto_buscar, columnas_mostrar)
            st.write(f"Mostrando {len(df_filtrado)} de {len(df)} filas que coinciden con '{texto_buscar}'")
        else:
            df_filtrado = df
            st.write(f"Mostrando {len(df_filtrado)} filas")
        
        st.dataframe(df_filtrado[columnas_mostrar], use_container_width=True)

if __name__ == "__main__":
    main()
