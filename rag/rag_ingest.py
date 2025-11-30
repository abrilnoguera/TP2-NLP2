"""
Ingesta RAG del CV en Pinecone
===============================================

Lee un PDF, chunkéa el texto, genera embeddings y los sube a un índice Pinecone
usando la API.
"""

import os
import time
import pdfplumber
from typing import List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()


# =============================================================
# 1. CONFIGURACIÓN: CLIENTE DE PINECONE
# =============================================================

def conectar_pinecone() -> Pinecone:
    """Crea la instancia Pinecone client (serverless)."""

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("❌ PINECONE_API_KEY no está configurada en variables de entorno")

    pc = Pinecone(api_key=api_key)
    print("✅ Cliente Pinecone inicializado correctamente")

    return pc


# =============================================================
# 2. CREAR ÍNDICE (COMPATIBLE SERVERLESS)
# =============================================================

def crear_indice(pc: Pinecone, nombre_indice: str, dimension: int):
    """Crea índice serverless si no existe (sin usar list_indexes, incompatible con serverless)."""

    try:
        # Intentar conectarse: si existe, no falla
        pc.Index(nombre_indice)
        print(f"⚠️ El índice '{nombre_indice}' ya existe")
        return
    except Exception:
        print(f"🔄 El índice '{nombre_indice}' no existe. Creando...")

    pc.create_index(
        name=nombre_indice,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )

    # Esperar hasta que el endpoint responda
    while True:
        try:
            pc.Index(nombre_indice)
            break
        except Exception:
            time.sleep(1)

    print(f"✅ Índice '{nombre_indice}' creado exitosamente.")


# =============================================================
# 3. EMBEDDINGS
# =============================================================

class GeneradorEmbeddings:
    """Modelo generador de embeddings."""

    def __init__(self, modelo="sentence-transformers/all-MiniLM-L6-v2"):
        self.modelo = SentenceTransformer(modelo)
        self.dimension = self.modelo.get_sentence_embedding_dimension()
        print(f"✅ Modelo de embeddings cargado ({self.dimension} dimensiones)")

    def generar(self, texto: str) -> List[float]:
        return self.modelo.encode(texto).tolist()

    def generar_lote(self, textos: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.modelo.encode(textos)]


# =============================================================
# 4. EXTRAER TEXTO DEL PDF
# =============================================================

def cargar_cv_pdf(ruta_pdf: str) -> str:
    """Leer todas las páginas del PDF del CV y extraer su texto."""

    texto = ""

    with pdfplumber.open(ruta_pdf) as pdf:
        for page in pdf.pages:
            contenido = page.extract_text()
            if contenido:
                texto += contenido + "\n"

    texto = texto.strip()

    if len(texto) < 40:
        raise ValueError("❌ No se pudo extraer texto del PDF. ¿Es un PDF escaneado?")

    print(f"📄 Texto extraído del PDF ({len(texto)} caracteres)")
    return texto


# =============================================================
# 5. CHUNKEAR TEXTO
# =============================================================

def chunkear_texto(texto: str, max_chars=700, overlap=100) -> List[Dict[str, Any]]:
    """Chunking simple con overlap."""
    texto = texto.replace("\r", "")
    chunks = []
    start = 0
    idx = 0

    while start < len(texto):
        end = start + max_chars
        chunk = texto[start:end].strip()

        if chunk:
            chunks.append({
                "id": f"cv_chunk_{idx:03d}",
                "texto": chunk,
                "seccion": "cv"
            })
            idx += 1

        start = end - overlap

    print(f"✂️ Generados {len(chunks)} chunks")
    return chunks


# =============================================================
# 6. INGESTAR EN PINECONE
# =============================================================

def ingestar_cv_en_pinecone(ruta_pdf: str):
    """Pipeline principal."""
    pc = conectar_pinecone()

    embedder = GeneradorEmbeddings()

    nombre_indice =os.getenv("PINECONE_INDEX")

    crear_indice(pc, nombre_indice, embedder.dimension)

    # Conectar al índice ya creado
    index = pc.Index(nombre_indice)

    texto_cv = cargar_cv_pdf(ruta_pdf)
    documentos = chunkear_texto(texto_cv)

    batch_size = 64
    total = len(documentos)
    procesados = 0

    print(f"🚀 Iniciando ingesta de {total} chunks...")

    for i in range(0, total, batch_size):
        lote = documentos[i:i + batch_size]
        textos = [d["texto"] for d in lote]
        embeddings = embedder.generar_lote(textos)

        registros = []
        for j, doc in enumerate(lote):
            registros.append({
                "id": doc["id"],
                "values": embeddings[j],
                "metadata": {
                    "texto": doc["texto"],
                    "seccion": doc["seccion"]
                }
            })

        index.upsert(vectors=registros)
        procesados += len(lote)

        print(f"   ➜ {procesados}/{total} chunks subidos")

    print("\n🎉 Ingesta completada correctamente")
    print(index.describe_index_stats())


# =============================================================
# 7. MAIN
# =============================================================

if __name__ == "__main__":

    RUTA_CV = "docs/Abril Noguera - CV.pdf"

    if not os.path.exists(RUTA_CV):
        print(f"❌ No se encontró el archivo: {RUTA_CV}")
        exit(1)

    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Falta la variable de entorno PINECONE_API_KEY")
        exit(1)

    ingestar_cv_en_pinecone(RUTA_CV, nombre_indice=os.getenv("PINECONE_INDEX"))