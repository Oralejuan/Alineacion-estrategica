import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import sqlite3

st.set_page_config(page_title="Alineación Estratégica", layout="wide")
st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("""
Ingresa el objetivo o descripción de tu proyecto.  
La herramienta buscará **objetivos**, **metas** y **conceptos estratégicos** relevantes.
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
def preparar_indices(df_objs, df_metas, df_conceptos):
    # Objetivos
    textos_objs = (df_objs['nombre'].fillna('') + " " + df_objs['descripción'].fillna('')).tolist()
    tokens_objs = [preprocess(t) for t in textos_objs]
    # Metas
    textos_metas = df_metas['descripcion'].fillna('').tolist()
    tokens_metas = [preprocess(t) for t in textos_metas]
    # Conceptos
    textos_conceptos = (df_conceptos['Concepto'].fillna('') + " " + df_conceptos['Definición'].fillna('')).tolist()
    tokens_conceptos = [preprocess(t) for t in textos_conceptos]

    all_tokens = tokens_objs + tokens_metas + tokens_conceptos
    vocab = build_vocab(all_tokens)

    # Matrices TF
    tf_objs = np.array([compute_tf(tok, vocab) for tok in tokens_objs])
    tf_metas = np.array([compute_tf(tok, vocab) for tok in tokens_metas])
    tf_conceptos = np.array([compute_tf(tok, vocab) for tok in tokens_conceptos])

    # IDF global
    idf = compute_idf(all_tokens, vocab)

    tfidf_objs = vectorize_tfidf(tf_objs, idf)
    tfidf_metas = vectorize_tfidf(tf_metas, idf)
    tfidf_conceptos = vectorize_tfidf(tf_conceptos, idf)

    return (tfidf_objs, tfidf_metas, tfidf_conceptos, vocab, idf,
            textos_objs, textos_metas, textos_conceptos)

def buscar_alineacion(query, tfidf_objs, tfidf_metas, tfidf_conceptos,
                      vocab, idf, df_objs, df_metas, df_conceptos, df_rel, top_n=10):
    tokens_q = preprocess(query)
    if not tokens_q:
        return [], [], []
    tf_q = compute_tf(tokens_q, vocab)
    tfidf_q = tf_q * idf

    # Similitudes
    sim_objs = [cosine_similarity_vec(tfidf_q, vec) for vec in tfidf_objs]
    sim_metas = [cosine_similarity_vec(tfidf_q, vec) for vec in tfidf_metas]
    sim_conceptos = [cosine_similarity_vec(tfidf_q, vec) for vec in tfidf_conceptos]

    # Índices top
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

    # Ordenar cada lista (ya lo están por el orden de los índices)
    return resultados_objs, resultados_metas, resultados_conceptos

def exportar_resultados_csv(resultados_objs, resultados_metas, resultados_conceptos):
    rows = []
    for r in resultados_objs:
        rows.append({'Tipo': 'Objetivo', 'Nombre': r['nombre'], 'Descripción': r['descripcion'],
                     'Instrumento': r['instrumento'], 'Similitud': r['similitud']})
    for r in resultados_metas:
        rows.append({'Tipo': 'Meta', 'Descripción': r['descripcion'], 'Instrumento': r['instrumento'],
                     'Objetivo asociado': r['objetivo_nombre'], 'Horizonte': r['horizonte'],
                     'Sector': r['sector'], 'Conceptos': ', '.join(r['conceptos']), 'Similitud': r['similitud']})
    for r in resultados_conceptos:
        rows.append({'Tipo': 'Concepto', 'Concepto': r['concepto'], 'Definición': r['definicion'],
                     'Fuente': r['fuente'], 'Similitud': r['similitud']})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, sep=';', encoding='utf-8-sig')

# -------------------------------------------------------------
# Exploración jerárquica y descriptiva
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
        ["🔍 Alineación de proyectos", "📚 Explorar catálogos"]
    )
    datos = cargar_datos()
    if datos[0] is None:
        return
    df_inst, df_objs, df_metas, df_rel, df_conceptos, df_amenazas = datos
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Objetivos: {len(df_objs)} | Metas: {len(df_metas)} | Conceptos: {len(df_conceptos)}")

    if modo == "🔍 Alineación de proyectos":
        with st.spinner("Preparando índices de búsqueda..."):
            (tfidf_objs, tfidf_metas, tfidf_conceptos, vocab, idf,
             _, _, _) = preparar_indices(df_objs, df_metas, df_conceptos)

        consulta = st.text_area("Describe tu proyecto:", height=100)
        top_n = st.slider("Número de resultados por categoría", 5, 20, 10)

        if st.button("Buscar", type="primary"):
            if not consulta.strip():
                st.warning("Ingresa una descripción.")
            else:
                with st.spinner("Buscando..."):
                    res_objs, res_metas, res_conceptos = buscar_alineacion(
                        consulta, tfidf_objs, tfidf_metas, tfidf_conceptos,
                        vocab, idf, df_objs, df_metas, df_conceptos, df_rel, top_n)

                if not (res_objs or res_metas or res_conceptos):
                    st.info("No se encontraron resultados relevantes.")
                else:
                    st.success(f"Se encontraron {len(res_objs)} objetivos, {len(res_metas)} metas y {len(res_conceptos)} conceptos relacionados.")
                    csv_data = exportar_resultados_csv(res_objs, res_metas, res_conceptos)
                    st.download_button("📥 Exportar resultados a CSV", data=csv_data,
                                       file_name="resultados_alineacion.csv", mime="text/csv")

                    tabs = st.tabs(["🎯 Objetivos", "📋 Metas", "🧠 Conceptos Estratégicos"])
                    with tabs[0]:
                        if not res_objs:
                            st.info("No hay objetivos coincidentes.")
                        else:
                            for r in res_objs:
                                with st.expander(f"{color_similitud(r['similitud'])} {r['nombre']} (score: {r['similitud']})"):
                                    st.write(f"**Instrumento:** {r['instrumento']}")
                                    st.write(f"**Descripción:** {r['descripcion']}")
                    with tabs[1]:
                        if not res_metas:
                            st.info("No hay metas coincidentes.")
                        else:
                            for r in res_metas:
                                with st.expander(f"{color_similitud(r['similitud'])} Meta (score: {r['similitud']})"):
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
                                with st.expander(f"{color_similitud(r['similitud'])} {r['concepto']} (score: {r['similitud']})"):
                                    st.write(f"**Definición:** {r['definicion']}")
                                    if r['fuente']:
                                        st.caption(f"Fuente: {r['fuente']}")

    else:  # Explorar catálogos
        st.header("📚 Explorar catálogos")
        cat_seleccion = st.selectbox(
            "Selecciona un catálogo",
            ["Instrumentos", "Objetivos", "Metas", "Conceptos Estratégicos", "Amenazas"]
        )
        texto_buscar = st.text_input("Filtrar por texto (búsqueda libre)", key="filtro_catalogo")
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

if __name__ == "__main__":
    main()
