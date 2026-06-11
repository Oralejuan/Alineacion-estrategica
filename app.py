# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Configuración de la página
st.set_page_config(
    page_title="Alineación Estratégica - Buscador",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Buscador de Alineación Estratégica")
st.markdown("""
Ingresa el objetivo o descripción de tu proyecto.  
La herramienta buscará los **instrumentos**, **objetivos**, **metas** y **conceptos estratégicos** más relevantes en tu base de datos.
""")

# -------------------------------------------------------------
# 1. Carga de datos (con caché)
# -------------------------------------------------------------
@st.cache_resource
def cargar_datos(archivo):
    """Carga las hojas del Excel y limpia duplicados en objetivos."""
    df_inst = pd.read_excel(archivo, sheet_name="INSTRUMENTOS")
    df_objs = pd.read_excel(archivo, sheet_name="OBJETIVOS")
    df_metas = pd.read_excel(archivo, sheet_name="METAS")
    df_rel = pd.read_excel(archivo, sheet_name="REL_META_CONCEPTO")
    df_conceptos = pd.read_excel(archivo, sheet_name="CONCEPTOS_ESTRATEGICOS")
    
    # Limpiar posibles duplicados en objetivos (por id_objetivo)
    df_objs = df_objs.drop_duplicates(subset=['id_objetivo'])
    
    # Unir conceptos a la relación
    if not df_rel.empty and not df_conceptos.empty:
        df_rel = df_rel.merge(
            df_conceptos[['id_concepto', 'Concepto']],
            left_on='concepto', right_on='Concepto',
            how='left'
        )
    return df_inst, df_objs, df_metas, df_rel, df_conceptos

@st.cache_resource
def cargar_modelo_embeddings():
    """Carga el modelo multilingüe de Sentence Transformers."""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def preparar_indices(_model, df_objs, df_metas):
    """Genera embeddings para objetivos y metas."""
    # Textos de objetivos: nombre + descripción
    textos_objs = (df_objs['nombre'].fillna('') + " " + df_objs['descripción'].fillna('')).tolist()
    # Textos de metas: solo descripción
    textos_metas = df_metas['descripcion'].fillna('').tolist()
    
    with st.spinner("Generando índices de búsqueda (puede tardar unos segundos)..."):
        emb_objs = _model.encode(textos_objs, show_progress_bar=False)
        emb_metas = _model.encode(textos_metas, show_progress_bar=False)
    return emb_objs, emb_metas, textos_objs, textos_metas

# -------------------------------------------------------------
# 2. Preprocesamiento de texto (opcional, no indispensable)
# -------------------------------------------------------------
def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# -------------------------------------------------------------
# 3. Búsqueda semántica
# -------------------------------------------------------------
def buscar(query, model, emb_objs, emb_metas, textos_objs, textos_metas,
           df_objs, df_metas, df_rel, top_n=10):
    """Retorna listado de resultados (objetivos + metas) ordenados por similitud."""
    # Codificar la consulta
    emb_query = model.encode([query])
    
    # Similitud coseno con objetivos y metas
    sim_objs = cosine_similarity(emb_query, emb_objs).flatten()
    sim_metas = cosine_similarity(emb_query, emb_metas).flatten()
    
    # Índices top
    idx_objs = np.argsort(sim_objs)[::-1][:top_n]
    idx_metas = np.argsort(sim_metas)[::-1][:top_n]
    
    resultados = []
    
    # Agregar objetivos
    for idx in idx_objs:
        score = sim_objs[idx]
        if score == 0:
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
        if score == 0:
            continue
        row = df_metas.iloc[idx]
        # Obtener objetivo padre e instrumento
        obj_row = df_objs[df_objs['id_objetivo'] == row['id_objetivo']]
        if not obj_row.empty:
            instrumento = obj_row.iloc[0]['instrumento']
            obj_nombre = obj_row.iloc[0]['nombre']
        else:
            instrumento = 'No encontrado'
            obj_nombre = ''
        resultados.append({
            'tipo': 'Meta',
            'id_meta': row['id_meta'],
            'id_objetivo': row['id_objetivo'],
            'instrumento': instrumento,
            'objetivo_nombre': obj_nombre,
            'descripcion': row['descripcion'],
            'horizonte': row['horizonte'],
            'sector': row['sector'],
            'similitud': round(score, 3)
        })
    
    # Ordenar por similitud descendente
    resultados.sort(key=lambda x: x['similitud'], reverse=True)
    return resultados

# -------------------------------------------------------------
# 4. Interfaz principal
# -------------------------------------------------------------
def main():
    # Sidebar para carga de archivo o ruta fija
    st.sidebar.header("📂 Datos")
    archivo_opcion = st.sidebar.radio(
        "Selecciona la fuente de datos",
        ["Usar archivo predefinido", "Subir un archivo Excel"]
    )
    
    if archivo_opcion == "Subir un archivo Excel":
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo .xlsx", type=["xlsx"])
        if uploaded_file is None:
            st.info("Por favor sube un archivo Excel para comenzar.")
            return
        archivo = uploaded_file
    else:
        archivo = "BASE_ALINEACION_ESTRATEGICA.xlsx"
        st.sidebar.info(f"Usando archivo: `{archivo}`")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df_inst, df_objs, df_metas, df_rel, df_conceptos = cargar_datos(archivo)
    
    if df_objs is None or df_objs.empty:
        st.error("No se pudieron cargar los datos. Verifica el archivo Excel.")
        return
    
    # Mostrar estadísticas en sidebar
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"- Instrumentos: {len(df_inst)}")
    st.sidebar.write(f"- Objetivos: {len(df_objs)}")
    st.sidebar.write(f"- Metas: {len(df_metas)}")
    st.sidebar.write(f"- Relaciones meta-concepto: {len(df_rel)}")
    
    # Cargar modelo de embeddings
    with st.spinner("Cargando modelo de lenguaje (primera vez tarda unos segundos)..."):
        model = cargar_modelo_embeddings()
    
    # Generar índices (embeddings) solo si los datos cambiaron
    emb_objs, emb_metas, textos_objs, textos_metas = preparar_indices(model, df_objs, df_metas)
    
    # Entrada de búsqueda
    st.header("🔎 Búsqueda")
    consulta = st.text_area("Describe el objetivo o alcance de tu proyecto:", height=100)
    top_n = st.slider("Número de resultados a mostrar", min_value=5, max_value=20, value=10)
    
    if st.button("Buscar", type="primary"):
        if not consulta.strip():
            st.warning("Por favor ingresa una descripción.")
        else:
            with st.spinner("Buscando..."):
                resultados = buscar(
                    consulta, model, emb_objs, emb_metas,
                    textos_objs, textos_metas,
                    df_objs, df_metas, df_rel, top_n=top_n
                )
            
            if not resultados:
                st.info("No se encontraron resultados relevantes. Prueba con otros términos.")
            else:
                st.success(f"Se encontraron {len(resultados)} resultados.")
                # Mostrar resultados en pestañas
                tab1, tab2 = st.tabs(["📋 Resultados", "📄 Exportar"])
                
                with tab1:
                    for i, res in enumerate(resultados):
                        with st.expander(f"{i+1}. {res['tipo']} - Similitud: {res['similitud']}"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Score", f"{res['similitud']:.3f}")
                                if res['tipo'] == 'Meta':
                                    st.write(f"**Horizonte:** {res['horizonte']}")
                                    st.write(f"**Sector:** {res['sector']}")
                            with col2:
                                st.write(f"**Instrumento:** {res['instrumento']}")
                                if res['tipo'] == 'Objetivo':
                                    st.write(f"**Nombre:** {res['nombre']}")
                                    st.write(f"**Descripción:** {res['descripcion']}")
                                else:
                                    st.write(f"**Objetivo asociado:** {res['objetivo_nombre']}")
                                    st.write(f"**Meta:** {res['descripcion']}")
                                    # Mostrar conceptos estratégicos vinculados
                                    conceptos_meta = df_rel[df_rel['id_meta'] == res['id_meta']]
                                    if not conceptos_meta.empty:
                                        conceptos_lista = conceptos_meta['Concepto'].dropna().unique()
                                        st.write("**Conceptos estratégicos asociados:**")
                                        for conc in conceptos_lista[:5]:
                                            st.write(f"- {conc}")
                with tab2:
                    # Generar DataFrame para exportar
                    export_rows = []
                    for res in resultados:
                        if res['tipo'] == 'Objetivo':
                            export_rows.append({
                                'Tipo': 'Objetivo',
                                'Instrumento': res['instrumento'],
                                'Nombre_Objetivo': res['nombre'],
                                'Descripción': res['descripcion'],
                                'Similitud': res['similitud'],
                                'Nivel': res['nivel']
                            })
                        else:
                            conceptos_str = ', '.join(df_rel[df_rel['id_meta'] == res['id_meta']]['Concepto'].dropna().unique())
                            export_rows.append({
                                'Tipo': 'Meta',
                                'Instrumento': res['instrumento'],
                                'ID_Meta': res['id_meta'],
                                'Descripción_Meta': res['descripcion'],
                                'Similitud': res['similitud'],
                                'Horizonte': res['horizonte'],
                                'Sector': res['sector'],
                                'Conceptos_Clave': conceptos_str
                            })
                    df_export = pd.DataFrame(export_rows)
                    st.dataframe(df_export)
                    csv = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig')
                    st.download_button("📥 Descargar como CSV", data=csv,
                                       file_name="resultados_alineacion.csv",
                                       mime="text/csv")

if __name__ == "__main__":
    main()