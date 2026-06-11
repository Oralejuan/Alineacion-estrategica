import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import io

# Configuración de página
st.set_page_config(page_title="Alineación Estratégica", layout="wide")
st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("""
Ingresa el objetivo o descripción de tu proyecto.  
La herramienta buscará los **instrumentos**, **objetivos**, **metas** y **conceptos estratégicos** más relevantes.
""")

# -------------------------------------------------------------
# 1. Funciones TF-IDF manual (sin sklearn)
# -------------------------------------------------------------
def preprocess(text):
    """Limpia y tokeniza texto."""
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    tokens = text.split()
    return tokens

def build_vocab(corpus):
    """Construye vocabulario global a partir de listas de tokens."""
    vocab = set()
    for tokens in corpus:
        vocab.update(tokens)
    return {word: idx for idx, word in enumerate(sorted(vocab))}

def compute_tf(tokens, vocab):
    """Term Frequency vector para un documento."""
    tf_vec = np.zeros(len(vocab))
    counter = Counter(tokens)
    for word, count in counter.items():
        if word in vocab:
            tf_vec[vocab[word]] = count / len(tokens)
    return tf_vec

def compute_idf(corpus_tokens, vocab):
    """IDF global para cada palabra."""
    N = len(corpus_tokens)
    idf = np.zeros(len(vocab))
    for word, idx in vocab.items():
        # número de documentos que contienen la palabra
        df = sum(1 for tokens in corpus_tokens if word in set(tokens))
        idf[idx] = math.log((1 + N) / (1 + df)) + 1  # suavizado
    return idf

def vectorize_tfidf(tf_matrix, idf_vector):
    """Multiplica cada fila de TF por IDF (broadcast)."""
    return tf_matrix * idf_vector

def cosine_similarity_vec(v1, v2):
    """Similitud coseno entre dos vectores."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# -------------------------------------------------------------
# 2. Carga de datos
# -------------------------------------------------------------
@st.cache_resource
def cargar_datos(archivo):
    df_inst = pd.read_excel(archivo, sheet_name="INSTRUMENTOS")
    df_objs = pd.read_excel(archivo, sheet_name="OBJETIVOS")
    df_metas = pd.read_excel(archivo, sheet_name="METAS")
    df_rel = pd.read_excel(archivo, sheet_name="REL_META_CONCEPTO")
    df_conceptos = pd.read_excel(archivo, sheet_name="CONCEPTOS_ESTRATEGICOS")
    df_objs = df_objs.drop_duplicates(subset=['id_objetivo'])
    if not df_rel.empty and not df_conceptos.empty:
        df_rel = df_rel.merge(df_conceptos[['id_concepto', 'Concepto']],
                              left_on='concepto', right_on='Concepto', how='left')
    return df_inst, df_objs, df_metas, df_rel, df_conceptos

@st.cache_resource
def preparar_indices(df_objs, df_metas):
    # Construir corpus de objetivos (nombre + descripción)
    textos_objs = (df_objs['nombre'].fillna('') + " " + df_objs['descripción'].fillna('')).tolist()
    tokens_objs = [preprocess(t) for t in textos_objs]
    
    # Corpus de metas (descripción)
    textos_metas = df_metas['descripcion'].fillna('').tolist()
    tokens_metas = [preprocess(t) for t in textos_metas]
    
    # Vocabulario global (objetivos + metas)
    all_tokens = tokens_objs + tokens_metas
    vocab = build_vocab(all_tokens)
    
    # TF para objetivos
    tf_objs = np.array([compute_tf(tok, vocab) for tok in tokens_objs])
    # TF para metas
    tf_metas = np.array([compute_tf(tok, vocab) for tok in tokens_metas])
    
    # IDF global (usando todos los documentos)
    idf = compute_idf(all_tokens, vocab)
    
    # Vectores TF-IDF
    tfidf_objs = vectorize_tfidf(tf_objs, idf)
    tfidf_metas = vectorize_tfidf(tf_metas, idf)
    
    return tfidf_objs, tfidf_metas, vocab, idf, textos_objs, textos_metas

def buscar(query, tfidf_objs, tfidf_metas, vocab, idf, textos_objs, textos_metas,
           df_objs, df_metas, df_rel, top_n=10):
    tokens_q = preprocess(query)
    if not tokens_q:
        return []
    # Vector TF de la consulta
    tf_q = compute_tf(tokens_q, vocab)
    # TF-IDF de la consulta
    tfidf_q = tf_q * idf  # broadcast
    # Calcular similitud coseno con cada objetivo y cada meta
    sim_objs = [cosine_similarity_vec(tfidf_q, vec) for vec in tfidf_objs]
    sim_metas = [cosine_similarity_vec(tfidf_q, vec) for vec in tfidf_metas]
    
    # Mejores índices
    idx_objs = np.argsort(sim_objs)[::-1][:top_n]
    idx_metas = np.argsort(sim_metas)[::-1][:top_n]
    
    resultados = []
    # Agregar objetivos
    for idx in idx_objs:
        score = sim_objs[idx]
        if score < 0.01:
            continue
        row = df_objs.iloc[idx]
        resultados.append({
            'tipo': 'Objetivo',
            'id': row['id_objetivo'],
            'instrumento': row['instrumento'],
            'nombre': row['nombre'],
            'descripcion': row['descripción'],
            'nivel': row['nivel'],
            'similitud': round(score, 3)
        })
    # Agregar metas
    for idx in idx_metas:
        score = sim_metas[idx]
        if score < 0.01:
            continue
        row = df_metas.iloc[idx]
        obj_row = df_objs[df_objs['id_objetivo'] == row['id_objetivo']]
        instrumento = obj_row.iloc[0]['instrumento'] if not obj_row.empty else 'No encontrado'
        obj_nombre = obj_row.iloc[0]['nombre'] if not obj_row.empty else ''
        resultados.append({
            'tipo': 'Meta',
            'id_meta': row['id_meta'],
            'instrumento': instrumento,
            'objetivo_nombre': obj_nombre,
            'descripcion': row['descripcion'],
            'horizonte': row['horizonte'],
            'sector': row['sector'],
            'similitud': round(score, 3)
        })
    resultados.sort(key=lambda x: x['similitud'], reverse=True)
    return resultados

# -------------------------------------------------------------
# 3. Main
# -------------------------------------------------------------
def main():
    st.sidebar.header("📂 Datos")
    archivo_opcion = st.sidebar.radio("Selecciona fuente", ["Usar archivo predefinido", "Subir archivo Excel"])
    if archivo_opcion == "Subir archivo Excel":
        uploaded_file = st.sidebar.file_uploader("Sube .xlsx", type=["xlsx"])
        if uploaded_file is None:
            st.info("Por favor sube un archivo Excel.")
            return
        archivo = uploaded_file
    else:
        archivo = "BASE_ALINEACION_ESTRATEGICA.xlsx"
        st.sidebar.info(f"Usando: {archivo}")
    
    with st.spinner("Cargando datos..."):
        df_inst, df_objs, df_metas, df_rel, df_conceptos = cargar_datos(archivo)
    if df_objs.empty:
        st.error("Error al cargar datos.")
        return
    
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Objetivos: {len(df_objs)} | Metas: {len(df_metas)}")
    
    with st.spinner("Preparando índices TF-IDF (primera vez puede tomar unos segundos)..."):
        tfidf_objs, tfidf_metas, vocab, idf, textos_objs, textos_metas = preparar_indices(df_objs, df_metas)
    
    consulta = st.text_area("Describe tu proyecto:", height=100)
    top_n = st.slider("Resultados a mostrar", 5, 20, 10)
    
    if st.button("Buscar", type="primary"):
        if not consulta.strip():
            st.warning("Ingresa una descripción.")
        else:
            with st.spinner("Buscando..."):
                resultados = buscar(consulta, tfidf_objs, tfidf_metas, vocab, idf,
                                    textos_objs, textos_metas, df_objs, df_metas, df_rel, top_n)
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
                                conceptos = df_rel[df_rel['id_meta'] == res['id_meta']]['Concepto'].dropna().unique()
                                if len(conceptos):
                                    st.write("**Conceptos clave:**", ", ".join(conceptos[:5]))

if __name__ == "__main__":
    main()
