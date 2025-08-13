import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import os
import sqlite3
import json
import shutil
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from io import BytesIO

# Configuración inicial
st.set_page_config(page_title="MiniOCR - Extracción de Datos para el Proceso de Ministración", 
                   layout="wide",
                   page_icon="🪬")
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ajustar según sistema

# Constantes
MAX_CANVAS_WIDTH = 1200
MAX_CANVAS_HEIGHT = 1600
TEMPLATE_DIR = "templates"
DB_FILE = "tezcatlipoca.db"
BACKUP_DIR = "db_backups"

# Crear directorios necesarios
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Función para preprocesar imágenes
def preprocess_image(image):
    """Aplica preprocesamiento a la imagen para mejorar el OCR"""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)

# Función para convertir PDF a imágenes
def pdf_to_images(pdf_bytes):
    """Convierte un PDF a lista de imágenes PIL"""
    return convert_from_bytes(pdf_bytes)

# Función para hacer backup de la base de datos
def backup_database():
    """Crea un backup de la base de datos con timestamp"""
    if os.path.exists(DB_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"{DB_FILE.split('.')[0]}_{timestamp}.db")
        shutil.copy(DB_FILE, backup_file)
        return backup_file
    return None

# Función para inicializar la base de datos
def init_db():
    """Inicializa la base de datos SQLite si no existe"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS templates (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT UNIQUE,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

# Función para aplicar OCR a una región
def apply_ocr(image, roi):
    """Aplica OCR a una región específica de la imagen"""
    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    cropped = image.crop((x, y, x + w, y + h))
    return pytesseract.image_to_string(cropped).strip()

# Función para guardar una plantilla
# Cambios en la función save_template
def save_template(template_name, rois, reference_image):
    """Guarda una plantilla en disco como JSON con la imagen de referencia"""
    # Verificar etiquetas duplicadas
    labels = [roi['label'] for roi in rois]
    if len(labels) != len(set(labels)):
        return False, "Error: Las etiquetas deben ser únicas. Hay etiquetas duplicadas."

    template_path = os.path.join(TEMPLATE_DIR, f"{template_name}.json")
    
    # Guardar imagen de referencia
    ref_image_path = os.path.join(TEMPLATE_DIR, f"{template_name}_ref.png")
    reference_image.save(ref_image_path)
    
    # Guardar metadatos de la plantilla
    template_data = {
        "name": template_name,
        "rois": rois,
        "reference_image": ref_image_path,
        "created_at": datetime.now().isoformat()
    }
    
    with open(template_path, 'w') as f:
        json.dump(template_data, f)
    
    # Crear tabla en la base de datos
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        # Primero registrar la plantilla
        c.execute("INSERT INTO templates (name) VALUES (?)", (template_name,))
        
        # Crear tabla con las columnas definidas
        columns = [f'"{roi["label"]}" TEXT' for roi in rois]
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{template_name}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(columns)},
            document_name TEXT,
            page_number INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        c.execute(create_table_sql)
        conn.commit()
        return True, f"Plantilla '{template_name}' guardada exitosamente!"
    except sqlite3.IntegrityError:
        return False, "Ya existe una plantilla con ese nombre."
    except sqlite3.Error as e:
        return False, f"Error de base de datos: {str(e)}"
    finally:
        conn.close()

# Función para cargar plantillas disponibles
def load_template_names():
    """Retorna la lista de nombres de plantillas disponibles"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name FROM templates ORDER BY created_at DESC")
    templates = [row[0] for row in c.fetchall()]
    conn.close()
    return templates

# Función para cargar una plantilla específica
def load_template(template_name):
    """Carga una plantilla desde disco"""
    template_path = os.path.join(TEMPLATE_DIR, f"{template_name}.json")
    if not os.path.exists(template_path):
        return None
    
    with open(template_path, 'r') as f:
        return json.load(f)

def batch_process_with_template(uploaded_files, template_data, progress_bar=None, status_text=None):
    """Procesa todos los documentos con una plantilla"""
    total_docs = len(uploaded_files)
    results = []
    inserted_records = 0

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        # Procesar cada documento
        for doc_index, pdf_file in enumerate(uploaded_files):
            doc_name = pdf_file.name
            
            # Actualizar progreso
            if progress_bar and status_text:
                progress = (doc_index + 1) / total_docs
                progress_bar.progress(progress)
                status_text.text(f"Procesando {doc_name} ({doc_index+1}/{total_docs})")
            
            # Convertir PDF a imágenes (reiniciar el puntero del archivo)
            pdf_file.seek(0)
            images = convert_from_bytes(pdf_file.read())
            
            # Procesar cada página
            for page_index, image in enumerate(images):
                try:
                    # Preprocesamiento de imagen (igual que en otros modos)
                    processed_img = preprocess_image(image)
                    
                    # Extraer datos usando la plantilla
                    page_data = {}
                    for roi in template_data['rois']:
                        # Asegurar que las coordenadas son enteras
                        roi_fixed = {
                            'x': int(roi['x']),
                            'y': int(roi['y']),
                            'width': int(roi['width']),
                            'height': int(roi['height'])
                        }
                        text = apply_ocr(processed_img, roi_fixed)
                        page_data[roi['label']] = text
                    
                    # Insertar en base de datos
                    columns = list(page_data.keys())
                    values = list(page_data.values())
                    placeholders = ', '.join(['?'] * (len(columns) + 2))
                    insert_sql = f"""
                    INSERT INTO "{template_data['name']}" ({', '.join(columns)}, document_name, page_number)
                    VALUES ({placeholders})
                    """
                    
                    c.execute(insert_sql, (*values, doc_name, page_index + 1))
                    conn.commit()
                    inserted_records += 1
                    
                    results.append({
                        "document": doc_name,
                        "page": page_index + 1,
                        "data": page_data
                    })
                
                except Exception as page_error:
                    st.error(f"Error procesando página {page_index + 1} de {doc_name}: {str(page_error)}")
                    conn.rollback()
                    continue
        
        return results, inserted_records
    
    except Exception as e:
        st.error(f"Error en procesamiento por lotes: {str(e)}")
        conn.rollback()
        return [], 0
    finally:
        conn.close()

# Interfaz principal mejorada
def main():
    st.title("MiniOCR | Extracción de Datos para el Proceso de Ministración")
    
    # Inicializar base de datos
    init_db()
    
    if 'available_labels' not in st.session_state:
        predefined_labels = ["RFC", "CLABE", "MONTO", "NUMERO_SOLICITUD", "FECHA", 
                             "NOMBRE", "CUENTA", "BANCO", "REFERENCIA", "IMPORTE"]
        st.session_state.available_labels = predefined_labels

    # Usar pestañas para separar funcionalidades
    tab1, tab2 = st.tabs(["📥 Extracción de Datos", "📋 Crear Plantillas"])
    
    # Pestaña 1: Extracción de Datos
    with tab1:
        st.header("Extracción de Datos")
        
        # Estado de sesión para extracción
        if 'extraction_processed_images' not in st.session_state:
            st.session_state.extraction_processed_images = []
        if 'extraction_current_page' not in st.session_state:
            st.session_state.extraction_current_page = 0
        if 'extraction_rois' not in st.session_state:
            st.session_state.extraction_rois = []
        if 'extraction_labels' not in st.session_state:
            st.session_state.extraction_labels = {}
        if 'extraction_extracted_data' not in st.session_state:
            st.session_state.extraction_extracted_data = []
        
        # Carga de documentos
        uploaded_files = st.file_uploader(
            "Cargar documentos PDF", 
            type="pdf", 
            accept_multiple_files=True,
            key="extraction_uploader"
        )
        
        if not uploaded_files:
            st.info("Por favor, cargue uno o más documentos PDF para comenzar.")
        else:
            # Selección de documento
            doc_names = [f.name for f in uploaded_files]
            selected_doc = st.selectbox("Documento seleccionado", doc_names, index=0)
            pdf_file = uploaded_files[doc_names.index(selected_doc)]
            
            # Procesar PDF
            if not st.session_state.extraction_processed_images or st.session_state.get('extraction_current_doc') != selected_doc:
                with st.spinner(f"Procesando {selected_doc}..."):
                    images = pdf_to_images(pdf_file.read())
                    st.session_state.extraction_processed_images = [preprocess_image(img) for img in images]
                    st.session_state.extraction_current_doc = selected_doc
                    st.session_state.extraction_current_page = 0
                    st.session_state.extraction_rois = []
                    st.session_state.extraction_labels = {}
                    st.session_state.extraction_extracted_data = []
            
            # Navegación de páginas
            total_pages = len(st.session_state.extraction_processed_images)
            st.markdown(f"**Documento:** {selected_doc} - **Página:** {st.session_state.extraction_current_page + 1}/{total_pages}")
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("◀ Página", key="extraction_prev") and st.session_state.extraction_current_page > 0:
                    st.session_state.extraction_current_page -= 1
            with col2:
                st.markdown(f"**Página {st.session_state.extraction_current_page + 1}/{total_pages}**")
            with col3:
                if st.button("Página ▶", key="extraction_next") and st.session_state.extraction_current_page < total_pages - 1:
                    st.session_state.extraction_current_page += 1
            
            # Mostrar imagen actual
            current_image = st.session_state.extraction_processed_images[st.session_state.extraction_current_page]
            
            # Selección de modo de extracción
            mode = st.radio("Modo de extracción", 
                           ["Manual", "Usar plantilla existente"],
                           horizontal=True)
            
            # Canvas para dibujar ROIs
            scale_factor = 1.0
            if current_image.width > MAX_CANVAS_WIDTH or current_image.height > MAX_CANVAS_HEIGHT:
                scale_factor = min(MAX_CANVAS_WIDTH / current_image.width, 
                                  MAX_CANVAS_HEIGHT / current_image.height)
                canvas_width = int(current_image.width * scale_factor)
                canvas_height = int(current_image.height * scale_factor)
                display_image = current_image.resize((canvas_width, canvas_height))
            else:
                canvas_width = current_image.width
                canvas_height = current_image.height
                display_image = current_image
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=display_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key="extraction_canvas",
            )
            
            # Procesar dibujos del canvas
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                st.session_state.extraction_rois = []
                for obj in objects:
                    if obj["type"] == "rect":
                        x = int(obj["left"] / scale_factor)
                        y = int(obj["top"] / scale_factor)
                        width = int(obj["width"] / scale_factor)
                        height = int(obj["height"] / scale_factor)
                        roi_id = f"roi_{x}_{y}_{width}_{height}"
                        st.session_state.extraction_rois.append({
                            "id": roi_id,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height
                        })
            
            # ... (código anterior sin cambios)

# Dentro de la pestaña 1 (Extracción de Datos), modo Manual:

            # ... (código anterior)

            if mode == "Manual":
                st.subheader("Extracción Manual")
                
                # Resetear ROIs al cambiar de página o documento
                current_context = f"{selected_doc}_{st.session_state.extraction_current_page}"
                if 'last_context' not in st.session_state or st.session_state.last_context != current_context:
                    st.session_state.extraction_rois = []
                    st.session_state.extraction_labels = {}
                    st.session_state.last_context = current_context
                
                # Etiquetado de ROIs con sistema mejorado
                if st.session_state.extraction_rois:
                    st.write("**Etiquetado de Regiones:**")
                    for i, roi in enumerate(st.session_state.extraction_rois):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Usar selectbox con etiquetas disponibles
                            label = st.selectbox(
                                f"Etiqueta para ROI {i+1}", 
                                options=st.session_state.available_labels + ["[Nueva etiqueta]"],
                                index=0,
                                key=f"label_select_{roi['id']}_{current_context}"  # CORRECCIÓN: usar current_context
                            )
                            
                            # Si selecciona nueva etiqueta, mostrar campo de texto
                            if label == "[Nueva etiqueta]":
                                new_label = st.text_input(
                                    f"Nombre de nueva etiqueta para ROI {i+1}",
                                    key=f"new_label_{roi['id']}_{current_context}"  # CORRECCIÓN: usar current_context
                                )
                                if new_label:
                                    label = new_label
                                    if new_label not in st.session_state.available_labels:
                                        st.session_state.available_labels.append(new_label)
                            
                            st.session_state.extraction_labels[roi['id']] = label
                        
                        with col2:
                            # Vista previa del ROI
                            x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
                            roi_img = current_image.crop((x, y, x + w, y + h))
                            st.image(roi_img, caption=f"Vista previa", width=80)
                
                # ... (resto del código)
                
                # Botones de acción mejorados
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Extraer datos", key="manual_extract", help="Extrae datos de las regiones seleccionadas"):
                        data = {}
                        for roi in st.session_state.extraction_rois:
                            label = st.session_state.extraction_labels.get(roi['id'], f"ROI {roi['id']}")
                            text = apply_ocr(current_image, roi)
                            data[label] = text
                        
                        # Crear identificador único para esta página/documento
                        page_id = f"{selected_doc}_page_{st.session_state.extraction_current_page + 1}"
                        
                        # Guardar o actualizar los datos
                        st.session_state[page_id] = data
                        st.success("Datos extraídos correctamente!")
                
                with col2:
                    if st.button("Verificar coincidencias", key="verify_data", 
                                help="Compara valores entre documentos y páginas"):
                        # Recopilar todos los datos de todas las páginas
                        all_data = []
                        for key in st.session_state:
                            if key.startswith("_") or key in ["extraction_processed_images", "extraction_current_page", 
                                                            "extraction_rois", "extraction_labels", "available_labels",
                                                            "last_page_key"]:
                                continue
                            
                            if "_page_" in key:
                                all_data.append(st.session_state[key])
                        
                        if not all_data:
                            st.warning("No hay datos para verificar")
                            return
                            
                        # Agrupar valores por etiqueta, ignorando "NA"
                        value_map = {}
                        for data in all_data:
                            for label, value in data.items():
                                if value == "NA":
                                    continue
                                if label not in value_map:
                                    value_map[label] = set()
                                value_map[label].add(value)
                        
                        # Detectar discrepancias
                        discrepancies = []
                        for label, values in value_map.items():
                            if len(values) > 1:
                                discrepancies.append({
                                    "campo": label,
                                    "valores": ", ".join(values),
                                    "estado": "⚠️ Discrepancia"
                                })
                            else:
                                discrepancies.append({
                                    "campo": label,
                                    "valores": next(iter(values)),
                                    "estado": "✅ Coincide"
                                })
                        
                        # Mostrar resultados
                        if discrepancies:
                            df_discrepancies = pd.DataFrame(discrepancies)
                            st.subheader("Resultado de Verificación Global")
                            st.dataframe(df_discrepancies)
                            
                            # Alertar si hay discrepancias
                            if any(d['estado'] == "⚠️ Discrepancia" for d in discrepancies):
                                st.error("Se detectaron discrepancias en los valores entre documentos y páginas!")
                            else:
                                st.success("Todos los valores coinciden entre documentos y páginas!")
                        else:
                            st.info("No hay datos para verificar")
            
                # Mostrar y exportar datos con persistencia completa
                st.subheader("Datos Extraídos (Todos los Documentos)")
                
                # Recopilar todas las etiquetas únicas
                all_labels = set()
                all_rows = []
                
                # Recorrer todas las páginas y documentos
                for key in st.session_state:
                    if key.startswith("_") or key in ["extraction_processed_images", "extraction_current_page", 
                                                    "extraction_rois", "extraction_labels", "available_labels",
                                                    "last_context"]:  # CORRECCIÓN: cambiar last_page_key por last_context
                        continue
                        
                    if "_page_" in key:
                        doc_name, page_info = key.split("_page_")
                        page_num = int(page_info)
                        
                        # Crear fila con todos los campos
                        row = {
                            "Documento": doc_name,
                            "Página": page_num,
                        }
                        
                        # Obtener datos para esta página
                        page_data = st.session_state[key]
                        
                        # Añadir etiquetas encontradas
                        for label in page_data:
                            row[label] = page_data[label]
                            all_labels.add(label)
                        
                        all_rows.append(row)
                
                # Crear DataFrame con todas las columnas necesarias
                if all_rows:
                    # Ordenar etiquetas
                    sorted_labels = sorted(all_labels)
                    
                    # Crear DataFrame completo con "NA" para valores faltantes
                    full_data = []
                    for row in all_rows:
                        complete_row = {
                            "Documento": row["Documento"],
                            "Página": row["Página"]
                        }
                        # Añadir todas las etiquetas conocidas
                        for label in sorted_labels:
                            complete_row[label] = row.get(label, "NA")
                        full_data.append(complete_row)
                    
                    result_df = pd.DataFrame(full_data)
                    
                    # Función para resaltar coincidencias con fila anterior
                    def highlight_coincidences(row):
                        styles = [''] * len(row)
                        current_idx = row.name
                        
                        # No aplicar estilo a la primera fila
                        if current_idx == 0:
                            return styles
                            
                        prev_row = result_df.iloc[current_idx - 1]
                        
                        # Comparar cada campo con la fila anterior
                        for i, col in enumerate(result_df.columns):
                            # Saltar columnas de Documento y Página
                            if col in ["Documento", "Página"]:
                                continue
                                
                            current_val = row[col]
                            prev_val = prev_row[col]
                            
                            # Resaltar solo si no es NA y coincide con el valor anterior
                            if current_val != "NA" and current_val == prev_val:
                                styles[i] = 'background-color: #ccffcc'
                                
                        return styles
                    
                    # Aplicar resaltado
                    if not result_df.empty:
                        styled_df = result_df.style.apply(highlight_coincidences, axis=1)
                        st.dataframe(styled_df)
                        
                        # Crear columnas para los botones de exportación
                        col1, col2 = st.columns(2)

                        with col1:
                            # Exportar a CSV (existente)
                            csv_data = result_df.to_csv(index=False)
                            st.download_button(
                                label="Descargar como CSV",
                                data=csv_data,
                                file_name=f"MiniOCR_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # Nueva opción para exportar a Excel
                            excel_buffer = BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                result_df.to_excel(writer, index=False, sheet_name='DatosExtraidos')

                            st.download_button(
                                label="Descargar como Excel",
                                data=excel_buffer.getvalue(),
                                file_name=f"MiniOCR_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.info("No hay datos extraídos todavía")
                else:
                    st.info("Aún no se han extraído datos. Seleccione regiones y haga clic en 'Extraer datos'")


            # Modo Plantilla
            elif mode == "Usar plantilla existente":
                st.subheader("Extracción con Plantilla")
                templates = load_template_names()
                
                if not templates:
                    st.warning("No hay plantillas disponibles. Cree una en la pestaña 'Crear Plantillas'.")
                else:
                    selected_template = st.selectbox("Seleccionar plantilla", templates)
                    template_data = load_template(selected_template)
                    
                    if template_data:
                        st.info(f"**Plantilla:** {selected_template} - **Creada:** {template_data['created_at']}")
                        
                        # Opciones de procesamiento
                        processing_option = st.radio("Alcance del procesamiento",
                                                    ["Página actual", "Todo el documento", "Todos los documentos"],
                                                    horizontal=True)
                        
                        if st.button("Ejecutar extracción", key="template_extract"):
                            # Hacer backup antes de cualquier operación
                            backup_path = backup_database()
                            st.success(f"Backup creado: {os.path.basename(backup_path)}")
                            
                            if processing_option == "Página actual":
                                # Procesar solo página actual
                                data = {}
                                for roi in template_data['rois']:
                                    text = apply_ocr(current_image, roi)
                                    data[roi['label']] = text
                                
                                # Guardar en base de datos
                                conn = sqlite3.connect(DB_FILE)
                                c = conn.cursor()
                                
                                columns = list(data.keys())
                                values = list(data.values())
                                placeholders = ', '.join(['?'] * (len(columns) + 2))
                                insert_sql = f"""
                                INSERT INTO "{selected_template}" ({', '.join(columns)}, document_name, page_number)
                                VALUES ({placeholders})
                                """
                                c.execute(insert_sql, (*values, selected_doc, st.session_state.extraction_current_page + 1))
                                conn.commit()
                                conn.close()
                                
                                st.success("Datos extraídos y guardados en la base de datos!")
                                
                                # Mostrar datos
                                st.subheader("Datos Extraídos")
                                df = pd.DataFrame([data])
                                df.insert(0, "Documento", selected_doc)
                                df.insert(1, "Página", st.session_state.extraction_current_page + 1)
                                st.dataframe(df)
                            
                            elif processing_option == "Todo el documento":
                                # Procesar todas las páginas del documento actual
                                with st.spinner(f"Procesando {selected_doc}..."):
                                    results = []
                                    conn = sqlite3.connect(DB_FILE)
                                    c = conn.cursor()
                                    
                                    for page_index, image in enumerate(st.session_state.extraction_processed_images):
                                        data = {}
                                        for roi in template_data['rois']:
                                            text = apply_ocr(image, roi)
                                            data[roi['label']] = text
                                        
                                        # Guardar en base de datos
                                        columns = list(data.keys())
                                        values = list(data.values())
                                        placeholders = ', '.join(['?'] * (len(columns) + 2))
                                        insert_sql = f"""
                                        INSERT INTO "{selected_template}" ({', '.join(columns)}, document_name, page_number)
                                        VALUES ({placeholders})
                                        """
                                        c.execute(insert_sql, (*values, selected_doc, page_index + 1))
                                        conn.commit()
                                        
                                        results.append({
                                            "document": selected_doc,
                                            "page": page_index + 1,
                                            "data": data
                                        })
                                    
                                    conn.close()
                                    st.success(f"Documento completo procesado! {len(results)} páginas guardadas.")
                                    
                                    # Mostrar resumen
                                    st.subheader("Resumen de Extracción")
                                    summary = []
                                    for entry in results:
                                        summary.append({
                                            "Documento": entry['document'],
                                            "Página": entry['page'],
                                            "Campos Extraídos": len(entry['data'])
                                        })
                                    st.dataframe(pd.DataFrame(summary))
                            
                            elif processing_option == "Todos los documentos":
                                # Procesar todos los documentos cargados
                                with st.spinner("Procesando todos los documentos..."):
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    # Hacer backup antes de procesar
                                    backup_path = backup_database()
                                    
                                    results, inserted_count = batch_process_with_template(
                                        uploaded_files,
                                        template_data,
                                        progress_bar,
                                        status_text
                                    )
                                    
                                    progress_bar.empty()
                                    status_text.empty()

                                    if inserted_count > 0:
                                        st.success(f"Datos extraídos y guardados en la base de datos! {inserted_count} registros insertados.")
                                    
                                        # Mostrar resumen
                                        st.subheader("Resumen de Procesamiento")
                                        summary = []
                                        for entry in results:
                                            summary.append({
                                                "Documento": entry['document'],
                                                "Página": entry['page'],
                                                "Campos Extraídos": len(entry['data'])
                                            })
                                        st.dataframe(pd.DataFrame(summary))

                                        # Mostrar primeros registros insertados
                                        st.subheader("Primeros registros insertados")
                                        conn = sqlite3.connect(DB_FILE)
                                        df_inserted = pd.read_sql_query(
                                            f'SELECT * FROM "{selected_template}" ORDER BY id DESC LIMIT 5',
                                            conn
                                        )
                                        conn.close()
                                        st.dataframe(df_inserted)
                                    else:
                                        st.error("No se insertaron registros. Verifique los errores arriba.")
    
    # Pestaña 2: Crear Plantillas
    with tab2:
        st.header("Crear Nueva Plantilla")
        
        # Estado de sesión para creación de plantillas
        if 'template_processed_images' not in st.session_state:
            st.session_state.template_processed_images = []
        if 'template_current_page' not in st.session_state:
            st.session_state.template_current_page = 0
        if 'template_rois' not in st.session_state:
            st.session_state.template_rois = []
        if 'template_labels' not in st.session_state:
            st.session_state.template_labels = {}
        
        # Carga de documento para plantilla
        template_file = st.file_uploader(
            "Cargar documento PDF para plantilla", 
            type="pdf",
            key="template_uploader"
        )
        
        if not template_file:
            st.info("Por favor, cargue un documento PDF para crear una plantilla.")
        else:
            # Procesar PDF para plantilla
            if not st.session_state.template_processed_images or st.session_state.get('template_current_doc') != template_file.name:
                with st.spinner(f"Procesando {template_file.name}..."):
                    images = pdf_to_images(template_file.read())
                    st.session_state.template_processed_images = [preprocess_image(img) for img in images]
                    st.session_state.template_current_doc = template_file.name
                    st.session_state.template_current_page = 0
                    st.session_state.template_rois = []
                    st.session_state.template_labels = {}
            
            # Navegación de páginas para plantilla
            total_pages = len(st.session_state.template_processed_images)
            st.markdown(f"**Documento:** {template_file.name} - **Página:** {st.session_state.template_current_page + 1}/{total_pages}")
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("◀ Página", key="template_prev") and st.session_state.template_current_page > 0:
                    st.session_state.template_current_page -= 1
            with col2:
                st.markdown(f"**Página {st.session_state.template_current_page + 1}/{total_pages}**")
            with col3:
                if st.button("Página ▶", key="template_next") and st.session_state.template_current_page < total_pages - 1:
                    st.session_state.template_current_page += 1
            
            # Mostrar imagen actual para plantilla
            current_image = st.session_state.template_processed_images[st.session_state.template_current_page]
            
            # Canvas para plantilla
            scale_factor = 1.0
            if current_image.width > MAX_CANVAS_WIDTH or current_image.height > MAX_CANVAS_HEIGHT:
                scale_factor = min(MAX_CANVAS_WIDTH / current_image.width, 
                                  MAX_CANVAS_HEIGHT / current_image.height)
                canvas_width = int(current_image.width * scale_factor)
                canvas_height = int(current_image.height * scale_factor)
                display_image = current_image.resize((canvas_width, canvas_height))
            else:
                canvas_width = current_image.width
                canvas_height = current_image.height
                display_image = current_image
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=display_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key="template_canvas",
            )
            
            # Procesar dibujos del canvas para plantilla
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                st.session_state.template_rois = []
                for obj in objects:
                    if obj["type"] == "rect":
                        x = int(obj["left"] / scale_factor)
                        y = int(obj["top"] / scale_factor)
                        width = int(obj["width"] / scale_factor)
                        height = int(obj["height"] / scale_factor)
                        roi_id = f"tpl_roi_{x}_{y}_{width}_{height}"
                        st.session_state.template_rois.append({
                            "id": roi_id,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height
                        })
            
            # Etiquetado para plantilla
            if st.session_state.template_rois:
                st.subheader("Definición de Campos")
                for i, roi in enumerate(st.session_state.template_rois):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        # Vista previa ROI
                        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
                        roi_img = current_image.crop((x, y, x + w, y + h))
                        st.image(roi_img, caption=f"ROI {i+1}", width=100)
                    with col2:
                        label = st.text_input(f"Nombre del campo para ROI {i+1}", 
                                             value=st.session_state.template_labels.get(roi['id'], ""),
                                             key=f"tpl_label_{roi['id']}")
                        st.session_state.template_labels[roi['id']] = label
            
            template_name = st.text_input("Nombre de la plantilla", help="Este será el nombre de la tabla en la base de datos")

            if st.button("Guardar Plantilla", key="save_template"):
                if not template_name:
                    st.error("Por favor ingrese un nombre para la plantilla")
                elif not st.session_state.template_rois:
                    st.error("Debe definir al menos una región de interés")
                else:
                    # Validar etiquetas
                    valid = True
                    rois_to_save = []
                    labels = set()

                    for roi in st.session_state.template_rois:
                        label = st.session_state.template_labels.get(roi['id'], "")
                        if not label:
                            st.error(f"ROI en posición ({roi['x']}, {roi['y']}) no tiene etiqueta")
                            valid = False
                            break
                        
                        # Verificar duplicados
                        if label in labels:
                            st.error(f"Etiqueta duplicada: '{label}'. Las etiquetas deben ser únicas.")
                            valid = False
                            break

                        labels.add(label)

                        rois_to_save.append({
                            "x": roi['x'],
                            "y": roi['y'],
                            "width": roi['width'],
                            "height": roi['height'],
                            "label": label
                        })

                    if valid:
                        success, message = save_template(template_name, rois_to_save, current_image)
                        if success:
                            st.success(message)
                            # Resetear estado
                            st.session_state.template_rois = []
                            st.session_state.template_labels = {}
                        else:
                            st.error(message)

if __name__ == "__main__":
    main()