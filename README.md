# 📄 CV Assistant — RAG + Groq + Pinecone + Streamlit

Un **chatbot inteligente** que responde preguntas sobre mi perfil profesional usando **RAG (Retrieval-Augmented Generation)**, un LLM acelerado por **Groq**, y un índice vectorial en **Pinecone**.

Funciona como una versión conversacional de mi CV, ideal para reclutadores, entrevistas técnicas y networking profesional.

👉 **Demo en vivo**: *Próximamente*

---

## 🎯 ¿Qué hace este proyecto?

Este asistente de CV permite a cualquier persona hacer preguntas sobre mi experiencia profesional, habilidades, formación y proyectos de forma natural y conversacional. El sistema recupera información relevante de mi CV y la presenta de manera clara y profesional, evitando alucinaciones gracias al uso de RAG.

### Ejemplo de uso:
- **Pregunta:** "¿Qué experiencia tenés en Machine Learning?"
- **Respuesta:** Información precisa extraída del CV sobre proyectos, herramientas y años de experiencia.

---

## ✨ Características principales

- 🔍 **RAG real**: Las respuestas provienen de información auténtica del CV (sin inventar datos)
- ⚡ **LLM ultrarrápido**: Usa Groq con modelos Llama optimizados para latencias muy bajas (<1s)
- 📚 **Vector Database**: Almacenamiento eficiente de embeddings en Pinecone (serverless)
- 📝 **Metadata enriquecida**: Información estructurada como nombre, email, skills, experiencia, etc.
- 🎨 **Interfaz moderna**: UI profesional construida con Streamlit
- 🧠 **Memoria de sesión**: Mantiene el contexto de la conversación
- 🚫 **Sin alucinaciones**: Reglas estrictas para evitar información inventada
- 📸 **Header personalizado**: Incluye foto personal y diseño profesional

---

## 🏗️ Arquitectura del proyecto

```
┌─────────────────┐
│   Usuario       │
│   (Pregunta)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   Streamlit UI              │
│   (rag_app.py)              │
└──────┬──────────────────────┘
       │
       ├──► 1. Embed pregunta (Sentence Transformers)
       │
       ├──► 2. Buscar contexto en Pinecone (top-k chunks)
       │
       ├──► 3. Cargar metadata (metadata.json)
       │
       └──► 4. Generar respuesta con Groq (Llama 3.1)
                ▼
       ┌──────────────────┐
       │   Respuesta      │
       │   (sin inventar) │
       └──────────────────┘
```

### Flujo RAG:
1. **Ingesta** (`rag_ingest.py`): El CV en PDF se divide en chunks, se generan embeddings y se almacenan en Pinecone
2. **Consulta** (`rag_app.py`): 
   - La pregunta del usuario se convierte en embedding
   - Se buscan los chunks más relevantes en Pinecone
   - Se construye un prompt con metadata + chunks recuperados
   - Groq genera una respuesta natural basada únicamente en esa información

---

## 🛠️ Stack tecnológico

| Componente | Tecnología | Propósito |
|------------|-----------|-----------|
| **Frontend** | Streamlit | Interfaz web interactiva |
| **LLM** | Groq (Llama 3.1 8B Instant) | Generación de respuestas en lenguaje natural |
| **Vector DB** | Pinecone (Serverless) | Almacenamiento y búsqueda de embeddings |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Conversión de texto a vectores |
| **PDF Processing** | pdfplumber | Extracción de texto del CV |
| **Lenguaje** | Python 3.9+ | Backend y procesamiento |

---

## 📦 Instalación y configuración

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/TP2-NLP2.git
cd TP2-NLP2
```

### 2️⃣ Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3️⃣ Instalar dependencias

```bash
pip install -r rag/requirements.txt
```

### 4️⃣ Configurar variables de entorno

Copia el archivo `.env.example` a `.env` y completa con tus credenciales:

```bash
cp .env.example .env
```

Edita `.env` con tus claves:

```env
PINECONE_API_KEY=tu_clave_de_pinecone
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_INDEX=cv-alumno
GROQ_API_KEY=tu_clave_de_groq
```

**Dónde obtener las claves:**
- **Pinecone**: [Registrate aquí](https://www.pinecone.io/) (plan gratuito disponible)
- **Groq**: [Consigue tu API key](https://console.groq.com/) (gratuito, muy generoso)

### 5️⃣ Preparar tu CV

1. Coloca tu CV en PDF en `docs/Tu_Nombre_CV.pdf`
2. Coloca tu foto en `docs/foto.jpg` (o actualiza la ruta en `rag_app.py`)
3. Edita `docs/metadata.json` con tu información personal

Ejemplo de `metadata.json`:

```json
{
  "nombre": "Tu Nombre",
  "titulo": "Data Scientist",
  "profesion": "Científico de Datos",
  "ubicacion": "Buenos Aires, Argentina",
  "fecha_nacimiento": "2000-01-01",
  "email": "tu@email.com",
  "linkedin": "linkedin.com/in/tu-perfil",
  "nivel_ingles": "Avanzado (C1)",
  "seniority": "Semi Senior",
  "experiencia_anios": 3,
  "skills_clave": ["Python", "Machine Learning", "SQL"]
}
```

---

## 🚀 Uso

### Paso 1: Ingestar el CV en Pinecone

Antes de usar el chatbot, debes procesar tu CV y subirlo a Pinecone:

```bash
python rag/rag_ingest.py
```

Esto:
- Lee tu CV en PDF
- Lo divide en chunks inteligentes
- Genera embeddings con Sentence Transformers
- Sube todo a Pinecone

**Salida esperada:**
```
✅ Cliente Pinecone inicializado correctamente
✅ Modelo de embeddings cargado (384 dimensiones)
📄 Texto extraído del PDF (5432 caracteres)
✂️ Generados 12 chunks
🚀 Iniciando ingesta de 12 chunks...
   ➜ 12/12 chunks subidos
🎉 Ingesta completada correctamente
```

### Paso 2: Lanzar la aplicación Streamlit

```bash
streamlit run rag/rag_app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Paso 3: ¡Empezar a conversar! 💬

Haz preguntas como:
- "¿Qué experiencia tenés en NLP?"
- "Contame sobre tus proyectos de Machine Learning"
- "¿Qué herramientas de MLOps manejás?"
- "¿Cuál es tu nivel de inglés?"

---

## 📁 Estructura del proyecto

```
TP2-NLP2/
│
├── docs/
│   ├── Abril Noguera - CV.pdf    # CV en formato PDF
│   ├── foto.jpg                   # Foto personal para el header
│   └── metadata.json              # Información estructurada del CV
│
├── rag/
│   ├── __init__.py
│   ├── rag_ingest.py             # Script de ingesta a Pinecone
│   ├── rag_app.py                # Aplicación Streamlit principal
│   ├── validate_env.py           # Validador de variables de entorno
│   └── requirements.txt          # Dependencias del proyecto
│
├── .env.example                   # Plantilla de variables de entorno
├── .gitignore                     # Archivos ignorados por Git
└── README.md                      # Este archivo
```

---

## 🔧 Configuración avanzada

### Ajustar parámetros de RAG

En `rag_app.py` puedes modificar:

```python
# Número de chunks recuperados
def retrieve(question: str, top_k: int = 5):  # Aumenta para más contexto

# Parámetros del LLM
resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    temperature=0.2,        # Creatividad (0-1)
    max_tokens=600,         # Longitud máxima de respuesta
)
```

### Cambiar modelo de embeddings

En `rag_ingest.py` y `rag_app.py`:

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Rápido y ligero
# Alternativas:
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Mejor multiidioma
# "sentence-transformers/all-mpnet-base-v2"  # Mejor calidad, más lento
```

### Chunking personalizado

En `rag_ingest.py`:

```python
def chunkear_texto(texto: str, max_chars=700, overlap=100):
    # max_chars: Tamaño de cada chunk
    # overlap: Solapamiento entre chunks (evita perder contexto)
```

---

## 🎨 Personalización de la UI

### Cambiar colores y estilos

Edita el CSS en `rag_app.py`:

```python
st.markdown("""
<style>
body { background-color: #0f172a; }  /* Fondo oscuro */
.chat-user { background:#334155; }    /* Mensajes del usuario */
.chat-bot { background:#1e293b; }     /* Mensajes del bot */
</style>
""", unsafe_allow_html=True)
```

### Modificar header

```python
st.markdown("<h1 class='header-name'>Tu Nombre</h1>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>Tu título profesional</div>", unsafe_allow_html=True)
```

---

## 🧪 Validación de entorno

Para verificar que todas las variables de entorno están configuradas correctamente:

```bash
python rag/validate_env.py
```

---

## 🚨 Solución de problemas comunes

### Error: "PINECONE_API_KEY no está configurada"

**Solución:** Asegúrate de tener un archivo `.env` en la raíz del proyecto con todas las claves necesarias.

### Error: "No se pudo extraer texto del PDF"

**Solución:** Tu PDF puede ser una imagen escaneada. Necesitarás un PDF con texto seleccionable o usar OCR.

### La aplicación no muestra respuestas

**Solución:** 
1. Verifica que ejecutaste `rag_ingest.py` primero
2. Confirma que el índice de Pinecone tiene datos:
   ```python
   from pinecone import Pinecone
   pc = Pinecone(api_key="tu_clave")
   index = pc.Index("cv-alumno")
   print(index.describe_index_stats())
   ```

### Groq responde muy lento

**Solución:** Groq es extremadamente rápido. Si hay lentitud, probablemente sea tu conexión a internet o límites de tasa (espera unos segundos y reintenta).

---

## 📊 Consideraciones técnicas

### Embeddings

- **Modelo**: `all-MiniLM-L6-v2` (384 dimensiones)
- **Ventajas**: Rápido, ligero, bueno para español e inglés
- **Desventajas**: Para CVs muy técnicos, considera modelos más grandes

### LLM (Groq)

- **Modelo**: Llama 3.1 8B Instant
- **Latencia**: ~200-500ms por respuesta
- **Límites**: ~30 req/min en plan gratuito (muy generoso para este uso)

### Pinecone

- **Plan gratuito**: 1 índice, hasta 100k vectores (suficiente para decenas de CVs)
- **Latencia**: ~50-100ms por query
- **Escalabilidad**: Serverless → se ajusta automáticamente

### Costos estimados

- **Pinecone Free**: $0/mes (hasta 100k vectores)
- **Groq**: $0/mes (límites generosos)
- **Total**: **GRATIS** para uso personal

---

## 🔐 Seguridad y privacidad

- ✅ Las claves API se manejan mediante variables de entorno
- ✅ El archivo `.env` está en `.gitignore` (no se sube a Git)
- ✅ Los datos del CV se almacenan en tu instancia de Pinecone (privada)
- ⚠️ **Importante**: No compartas tu archivo `.env` ni lo subas a repositorios públicos

---

## 🤝 Contribuciones

Este es un proyecto académico/personal, pero si encontrás bugs o mejoras:

1. **Reportá issues** en GitHub
2. **Propone mejoras** via Pull Requests
3. **Comparte tu feedback** en LinkedIn

---

## 📚 Recursos adicionales

- [Documentación de Pinecone](https://docs.pinecone.io/)
- [Groq API Reference](https://console.groq.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

## 📝 Licencia

Este proyecto fue desarrollado como parte del **TP2 de Procesamiento de Lenguaje Natural (NLP2)**.

**Autor**: Abril Noguera  
**Contacto**: abrilnoguera@gmail.com  
**LinkedIn**: [linkedin.com/in/abrilnoguera](https://linkedin.com/in/abrilnoguera)

---

## 🎓 Créditos académicos

**Materia**: Procesamiento de Lenguaje Natural 2  
**Trabajo Práctico**: TP2 - RAG Application  
**Año**: 2025

