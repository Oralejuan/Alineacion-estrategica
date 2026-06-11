import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Configuración de página
st.set_page_config(page_title="Alineación Estratégica", layout="wide")
st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("""
Ingresa el objetivo o descripción de tu proyecto.  
La herramienta buscará los **instrumentos**, **objetivos**, **metas** y **conceptos estratégicos** más relevantes.
""")

# -------------------------------------------------------------
# 1. Carga de datos
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

def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

@st.cache_resource
def preparar_vectorizador_y_corpus(_df_objs, _df_metas):
    textos_objs = (_df_objs['nombre'].fillna('') + " " + _df_objs['descripción'].fillna('')).apply(limpiar_texto).tolist()
    textos_metas = _df_metas['descripcion'].fillna('').apply(limpiar_texto).tolist()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='spanish')
    corpus_total = textos_objs + textos_metas
    vectorizer.fit(corpus_total)
    return vectorizer, textos_objs, textos_metas

def buscar(query, vectorizer, textos_objs, textos_metas, df_objs, df_metas, df_rel, top_n=10):
    query_limpia = limpiar_texto(query)
    q_vec = vectorizer.transform([query_limpia])
    sim_objs = cosine_similarity(q_vec, vectorizer.transform(textos_objs)).flatten()
    sim_metas = cosine_similarity(q_vec, vectorizer.transform(textos_metas)).flatten()
    
    idx_objs = np.argsort(sim_objs)[::-1][:top_n]
    idx_metas = np.argsort(sim_metas)[::-1][:top_n]
    
    resultados = []
    for idx in idx_objs:
        score = sim_objs[idx]
        if score == 0: continue
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
    for idx in idx_metas:
        score = sim_metas[idx]
        if score == 0: continue
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
# 2. Main
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
    
    with st.spinner("Preparando índices..."):
        vectorizer, textos_objs, textos_metas = preparar_vectorizador_y_corpus(df_objs, df_metas)
    
    consulta = st.text_area("Describe tu proyecto:", height=100)
    top_n = st.slider("Resultados a mostrar", 5, 20, 10)
    
    if st.button("Buscar", type="primary"):
        if not consulta.strip():
            st.warning("Ingresa una descripción.")
        else:
            with st.spinner("Buscando..."):
                resultados = buscar(consulta, vectorizer, textos_objs, textos_metas,
                                    df_objs, df_metas, df_rel, top_n)
            if not resultados:
                st.info("No se encontraron resultados.")
            else:
                st.success(f"{len(resultados)} resultados encontrados.")
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
