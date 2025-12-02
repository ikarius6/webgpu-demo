# Clasificador de Servicios con WebGPU - By Mr.Jack

Una aplicación web moderna que utiliza inteligencia artificial para clasificar servicios basándose en descripciones en lenguaje natural. Construida con React, TypeScript, y Transformers.js con soporte para aceleración por WebGPU.

## 🎬 Demo

[Demo](https://hackers.army/webgpu/)

## 🚀 Características

- **10 Modelos de IA**: Soporte para 5 familias de embeddings (E5, MiniLM, Paraphrase, BGE, MPNet/GTE) desde 23MB a 438MB
- **Aceleración por WebGPU**: Aprovecha la GPU del navegador cuando está disponible
- **Multi-Matcher Strategy**: Combina 3 algoritmos independientes (keyword 35%, fuzzy 30%, embedding 35%) con pesos adaptativos
- **Interfaz Moderna**: UI responsive con TailwindCSS, selector de modelos y score breakdown detallado
- **Sin Backend**: Funciona completamente en el navegador
- **Caché Inteligente**: Los modelos se descargan una vez y se guardan en caché del navegador

## 📋 Requisitos Previos

- Node.js 18 o superior
- npm o pnpm
- Navegador moderno (Chrome, Edge, Opera recomendados para soporte WebGPU)

## 🛠️ Instalación

1. **Clonar el repositorio**
```bash
git clone git@github.com:ikarius6/webgpu-demo.git
cd webgpu
```

2. **Instalar dependencias**
```bash
npm install
```

3. **Iniciar servidor de desarrollo**
```bash
npm run dev
```

4. **Abrir en el navegador**
```
http://localhost:3000
```

## 📦 Scripts Disponibles

```bash
# Desarrollo local con hot-reload
npm run dev

# Compilar TypeScript y construir para producción
npm run build

# Vista previa de la build de producción
npm run preview
```

## 🏗️ Estructura del Proyecto

```
webgpu/
├── src/
│   ├── ServiceClassifier.tsx  # Componente principal con Multi-Matcher Strategy
│   ├── ModelComparison.tsx    # Comparador de modelos
│   ├── ModelTester.tsx        # Testing de modelos
│   ├── categories.json        # 300+ categorías con sinónimos
│   ├── modelsConfig.json      # Configuración de modelos disponibles
│   ├── testCases.json         # Casos de prueba para validación
│   ├── main.tsx               # Punto de entrada de React
│   └── index.css              # Estilos globales con Tailwind
├── index.html                 # HTML base con carga de Transformers.js
├── package.json               # Dependencias y configuración
├── tsconfig.json              # Configuración de TypeScript
├── tailwind.config.js         # Configuración de TailwindCSS
└── vite.config.ts             # Configuración de Vite
```

## 🧠 Cómo Funciona

### 1. Modelos de IA

El proyecto utiliza **Transformers.js** para ejecutar modelos de machine learning directamente en el navegador. Incluye **10 modelos diferentes** organizados en 5 familias.

**Modelos Disponibles:**

#### E5 Family (Multilingües con prefijos)
- **Multilingual E5 Small** - 118MB, 384 dims - Rendimiento equilibrado
- **Multilingual E5 Base** - 278MB, 768 dims - Mayor precisión, más pesado

#### MiniLM Family (Ligeros y rápidos)
- **All-MiniLM L6 v2** ⭐ (Recomendado) - 23MB, 384 dims - Más ligero, enfocado en inglés
- **All-MiniLM L12 v2** - 66MB, 384 dims - Mayor precisión que L6

#### Paraphrase Family (Optimizados para paráfrasis)
- **Paraphrase MiniLM L6 v2** - 23MB, 384 dims - Ligero para paráfrasis
- **Paraphrase Multilingual MiniLM** - 118MB, 384 dims - Multilingüe alternativo

#### BGE Family (Alto rendimiento para inglés)
- **BGE Small EN v1.5** - 133MB, 384 dims - Compacto
- **BGE Base EN v1.5** - 438MB, 768 dims - Alto rendimiento

#### MPNet & GTE
- **All-MPNet Base v2** - 438MB, 768 dims - Alta calidad para inglés
- **GTE Small** - 133MB, 384 dims - Eficiente y rápido

**Notas Técnicas:**
- Los modelos E5 requieren prefijos "query:" y "passage:" para mejor rendimiento
- Feature extraction con similitud coseno
- Embeddings enriquecidos con sinónimos completos
- Configuración editable en `modelsConfig.json`

### 2. Algoritmo de Clasificación

El sistema implementa una **Multi-Matcher Strategy** que combina 3 matchers independientes con weighted voting:

1. **Keyword Matcher (35%)**
   - Coincidencias exactas y substring matching
   - Detecta términos específicos dentro del query
   - Ideal para búsquedas directas de servicios conocidos

2. **Fuzzy Matcher (30%)**
   - Utiliza Levenshtein distance para detectar variaciones ortográficas
   - Tolera errores de tipeo y variaciones en escritura
   - Umbral de similitud del 80% para activarse

3. **Embedding Matcher (35%)**
   - Similitud semántica usando embeddings multilingües
   - Captura el significado contextual profundo
   - Compara vectores mediante similitud coseno

**Características Avanzadas:**
- **Pesos Adaptativos**: Los pesos se ajustan dinámicamente según la calidad de los matches
  - Si keyword match > 80%: aumenta a 50% keyword, 20% fuzzy, 30% embedding
  - Si fuzzy match > 85%: aumenta a 25% keyword, 45% fuzzy, 30% embedding
- **Position-based Weighting**: Bonus del 30% para resultados mejor posicionados
- **Confidence Threshold**: Filtro mínimo del 15% para eliminar resultados poco relevantes
- **Score Breakdown**: Muestra la contribución de cada matcher en los resultados

### 3. Categorías

El archivo `categories.json` contiene:
- Más de 300 categorías de servicios
- Múltiples sinónimos por categoría para mejor cobertura
- Nombres en español optimizados para el contexto local

## 💡 Ejemplos de Uso

Prueba con estas consultas:

```
"necesito arreglar una fuga de agua"
→ Resultado: Accesorios de drenaje, Plomería

"mi jardín necesita poda"
→ Resultado: Jardinería, Poda de árboles

"me duele un diente"
→ Resultado: Dentistas, Odontología

"quiero pintar mi sala"
→ Resultado: Pintores, Decoración de interiores

"mi refrigerador no enfría"
→ Resultado: Refrigeración, Reparación de electrodomésticos
```

## 🎨 Tecnologías Utilizadas

### Frontend
- **React 18** - Librería de UI
- **TypeScript** - Tipado estático
- **Vite** - Build tool y dev server
- **TailwindCSS** - Framework de estilos
- **Lucide React** - Iconos SVG

### IA/ML
- **Transformers.js 2.17.2** - Modelos de ML en el navegador
- **WebGPU** - Aceleración por GPU
- **Multi-Matcher Strategy** - Sistema de 3 algoritmos combinados
- **Embeddings Multilingües** - Soporte para múltiples modelos (E5, MiniLM)
- **Levenshtein Distance** - Algoritmo de fuzzy matching

## ⚙️ Configuración Avanzada

### Ajustar Pesos del Algoritmo

En `ServiceClassifier.tsx`, líneas 347-380:

```typescript
// Weighted voting con pesos configurables
const weights = {
  keyword: 0.35,   // Coincidencias exactas son muy importantes
  fuzzy: 0.30,     // Fuzzy matching para variaciones
  embedding: 0.35  // Semántica para entender contexto
};

// Pesos adaptativos según calidad del match
similarities.forEach((item: any) => {
  let finalWeights = { ...weights };
  
  // Si hay keyword match fuerte (>0.8), aumentar su peso
  if (item.keywordScore >= 0.8) {
    finalWeights = {
      keyword: 0.50,
      fuzzy: 0.20,
      embedding: 0.30
    };
  }
  // Si hay fuzzy match fuerte (>0.85), ajustar pesos
  else if (item.fuzzyScore >= 0.85) {
    finalWeights = {
      keyword: 0.25,
      fuzzy: 0.45,
      embedding: 0.30
    };
  }
  
  // Calcular score final combinado
  item.score = (
    item.keywordScore * finalWeights.keyword +
    item.fuzzyScore * finalWeights.fuzzy +
    item.embeddingScore * finalWeights.embedding
  );
});
```

### Agregar Nuevas Categorías

Editar `src/categories.json`:

```json
{
  "items": [
    {
      "id": 999,
      "name": "Nueva Categoría",
      "synonyms": ["sinónimo1", "sinónimo2", "término relacionado", "variación"]
    }
  ]
}
```

**Tips:**
- Incluir mínimo 3-5 sinónimos por categoría
- Agregar variaciones comunes y errores de escritura
- Los sinónimos mejoran keyword y fuzzy matching

### Agregar Nuevos Modelos

Editar `src/modelsConfig.json`:

```json
{
  "models": [
    {
      "id": "nuevo-modelo",
      "name": "Nombre del Modelo",
      "huggingFaceId": "Xenova/nombre-modelo",
      "size": "200MB",
      "dimensions": 384,
      "requiresPrefixes": false,
      "description": "Descripción breve del modelo",
      "recommended": false,
      "category": "Familia del Modelo"
    }
  ]
}
```

**Campos:**
- `id`: Identificador único (kebab-case)
- `name`: Nombre para mostrar en la UI
- `huggingFaceId`: ID en HuggingFace (formato: `Xenova/modelo`)
- `size`: Tamaño aproximado del modelo
- `dimensions`: Dimensiones del vector embedding (384 o 768 típicamente)
- `requiresPrefixes`: `true` solo para modelos E5 (requieren "query:"/"passage:")
- `description`: Breve descripción para el usuario
- `recommended`: `true` para marcar con ⭐ en el selector
- `category`: Familia del modelo (E5, MiniLM, BGE, etc.)

## 🌐 Compatibilidad de Navegadores

| Navegador | WebGPU | CPU Fallback |
|-----------|--------|--------------|
| Chrome 113+ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ |
| Opera 99+ | ✅ | ✅ |
| Firefox | ❌ (experimental) | ✅ |
| Safari | ⚠️ (experimental) | ✅ |

**Nota**: Si WebGPU no está disponible, la aplicación funciona automáticamente con CPU (más lento pero funcional).

## 📊 Rendimiento

**Carga del Modelo** (varía según tamaño y familia):

| Modelo | Tamaño | Primera Carga | Con Caché |
|--------|--------|---------------|-----------|
| All-MiniLM L6 v2 ⭐ | 23MB | ~2-4s | ~1s |
| Paraphrase MiniLM L6 | 23MB | ~2-4s | ~1s |
| All-MiniLM L12 v2 | 66MB | ~4-7s | ~1-2s |
| Multilingual E5 Small | 118MB | ~6-10s | ~2-3s |
| Paraphrase Multilingual | 118MB | ~6-10s | ~2-3s |
| BGE/GTE Small | 133MB | ~7-11s | ~2-3s |
| Multilingual E5 Base | 278MB | ~12-18s | ~3-5s |
| All-MPNet/BGE Base | 438MB | ~20-30s | ~5-8s |

**Clasificación:**
- **Con WebGPU**: <200ms
- **Con CPU**: ~500ms-1s
- **Score breakdown**: incluido en UI sin impacto perceptible

**Recomendación:** Para balance entre velocidad y calidad, usa **All-MiniLM L6 v2** (23MB, recomendado).

## 🔧 Solución de Problemas

### El modelo no carga

1. Verificar consola del navegador para errores
2. Asegurar conexión a internet (primera vez)
3. Limpiar caché del navegador y reintentar
4. Verificar que el CDN de jsDelivr esté accesible

### WebGPU no se detecta

1. Usar Chrome/Edge actualizado
2. Habilitar flags experimentales:
   - Chrome: `chrome://flags/#enable-unsafe-webgpu`
3. La app funcionará con CPU de todos modos

### Resultados imprecisos

1. Agregar más sinónimos relevantes en `categories.json`
2. Ajustar los pesos del algoritmo en `ServiceClassifier.tsx` (líneas 347-380)
3. Probar con diferentes modelos según necesidad:
   - **All-MiniLM L6 v2** ⭐: Recomendado para balance velocidad/calidad
   - **Multilingual E5**: Mejor para español y multilingüe
   - **BGE/MPNet Base**: Mayor precisión (más pesados)
4. Verificar que el query esté bien escrito
5. Revisar el score breakdown para entender qué matcher está fallando

## 🚀 Deployment

### Build de Producción

```bash
npm run build
```

Los archivos se generarán en `dist/`. Puede desplegarse en:
- Netlify
- Vercel
- GitHub Pages
- Cualquier hosting estático

### Variables de Entorno

No requiere variables de entorno - todo funciona client-side.

## 📝 Licencia

ISC

## 👨‍💻 Desarrollo

### Estructura de Componentes

```tsx
ServiceClassifier
├── Estado (hooks)
│   ├── input / result / loading
│   ├── modelLoading / error
│   ├── selectedModelId / selectedModel
│   └── extractor / categoryEmbeddings
├── Efectos
│   ├── checkWebGPU()
│   └── loadModel() (se ejecuta al seleccionar modelo)
└── Funciones (Multi-Matcher)
    ├── cosineSimilarity() - para embeddings
    ├── keywordMatch() - coincidencias exactas/substring
    ├── levenshteinDistance() - distancia de edición
    ├── fuzzyMatch() - matching tolerante a errores
    └── classify() - orquesta los 3 matchers
```

### Flujo de Carga del Modelo

1. Verificar WebGPU disponible
2. Selección de modelo desde dropdown (configurado en `modelsConfig.json`)
3. Cargar Transformers.js desde CDN
4. Crear pipeline de feature-extraction con el modelo seleccionado
5. Generar embeddings para todas las categorías:
   - Si `requiresPrefixes: true` (E5): agregar prefijo "passage:"
   - Si `false`: usar texto directo
6. Guardar pipeline y embeddings en estado para reutilizar

### Flujo de Clasificación

1. Usuario ingresa query
2. Generar embedding del query:
   - Si modelo tiene `requiresPrefixes: true`: agregar prefijo "query:"
   - Si `false`: usar query directo
3. Para cada categoría calcular:
   - **Keyword Score**: coincidencias exactas y substring matching
   - **Fuzzy Score**: Levenshtein distance con sinónimos
   - **Embedding Score**: similitud coseno entre vectores
4. Aplicar weighted voting con pesos adaptativos:
   - Pesos base: 35% keyword, 30% fuzzy, 35% embedding
   - Si keyword match > 80%: ajustar a 50/20/30
   - Si fuzzy match > 85%: ajustar a 25/45/30
5. Aplicar position-based weighting (bonus del 30%)
6. Filtrar por confianza mínima (15%)
7. Ordenar y mostrar top 10 resultados con breakdown de scores

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add: amazing feature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, por favor abrir un issue en el repositorio.

---

**Hecho con ❤️ usando React, TypeScript, y Transformers.js**
