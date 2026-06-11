import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import io

st.set_page_config(page_title="Alineación Estratégica", layout="wide")
st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("Ingresa el objetivo o descripción de tu proyecto. La herramienta buscará los **instrumentos**, **objetivos**, **metas** y **conceptos estratégicos** más relevantes.")

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
# Carga de datos con manejo de errores
# -------------------------------------------------------------
@st.cache_resource
def cargar_datos(archivo):
    try:
        # Intentar leer cada hoja con engine explícito
        df_inst = pd.read_excel(archivo, sheet_name="INSTRUMENTOS", engine='openpyxl')
        df_objs = pd.read_excel(archivo, sheet_name="OBJETIVOS", engine='openpyxl')
        df_metas = pd.read_excel(archivo, sheet_name="METAS", engine='openpyxl')
        df_rel = pd.read_excel(archivo, sheet_name="REL_META_CONCEPTO", engine='openpyxl')
        df_conceptos = pd.read_excel(archivo, sheet_name="CONCEPTOS_ESTRATEGICOS", engine='openpyxl')
    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")
        st.info("Asegúrate de que el archivo sea válido y tenga las hojas correctas.")
        return None, None, None, None, None
    
    df_objs = df_objs.drop_duplicates(subset=['id_objetivo'])
    if not df_rel.empty and not df_conceptos.empty:
        df_rel = df_rel.merge(df_conceptos[['id_concepto', 'Concepto']],
                              left_on='concepto', right_on='Concepto', how='left')
    return df_inst, df_objs, df_metas, df_rel, df_conceptos

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

def buscar(query, tfidf_objs, tfidf_metas, vocab, idf, df_objs, df_metas, df_rel, top_n=10):
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
    for idx in idx_metas:
        score = sim_metas[idx]
        if score < 0.01: continue
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
# Interfaz de usuario
# -------------------------------------------------------------
def main():
    st.sidebar.header("📂 Datos")
    archivo_opcion = st.sidebar.radio("Selecciona la fuente", ["Subir archivo Excel", "Usar archivo predefinido"])
    
    archivo = None
    if archivo_opcion == "Subir archivo Excel":
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo .xlsx", type=["xlsx"])
        if uploaded_file is not None:
            archivo = uploaded_file
        else:
            st.info("Por favor sube un archivo Excel para continuar.")
            return
    else:
        # Ruta predefinida (solo si el archivo existe en el sistema de Streamlit Cloud)
        archivo = "BASE_ALINEACION_ESTRATEGICA.xlsx"
        st.sidebar.info(f"Usando archivo: {archivo}")
        # Verificar si existe (opcional)
        import os
        if not os.path.exists(archivo):
            st.sidebar.error(f"El archivo {archivo} no se encuentra. Usa la opción de subir.")
            return
    
    with st.spinner("Cargando datos..."):
        df_inst, df_objs, df_metas, df_rel, df_conceptos = cargar_datos(archivo)
    if df_objs is None or df_objs.empty:
        st.error("No se pudieron cargar los datos. Revisa que el archivo Excel tenga las hojas correctas.")
        return
    
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Objetivos: {len(df_objs)} | Metas: {len(df_metas)}")
    
    with st.spinner("Preparando índices TF-IDF..."):
        tfidf_objs, tfidf_metas, vocab, idf = preparar_indices(df_objs, df_metas)
    
    consulta = st.text_area("Describe tu proyecto:", height=100)
    top_n = st.slider("Resultados a mostrar", 5, 20, 10)
    
    if st.button("Buscar", type="primary"):
        if not consulta.strip():
            st.warning("Por favor ingresa una descripción.")
        else:
            with st.spinner("Buscando..."):
                resultados = buscar(consulta, tfidf_objs, tfidf_metas, vocab, idf,
                                    df_objs, df_metas, df_rel, top_n)
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
