# Clasificador de Servicios con WebGPU - By Mr.Jack

Una aplicación web moderna que utiliza inteligencia artificial para clasificar servicios basándose en descripciones en lenguaje natural. Construida con React, TypeScript, y Transformers.js con soporte para aceleración por WebGPU.

## 🚀 Características

- **IA Multilingüe**: Utiliza el modelo `multilingual-e5-small` para embeddings semánticos
- **Aceleración por WebGPU**: Aprovecha la GPU del navegador cuando está disponible
- **Búsqueda Híbrida**: Combina búsqueda semántica (80%) con coincidencias fuzzy (20%) para mayor precisión
- **Interfaz Moderna**: UI responsive con TailwindCSS y componentes de React
- **Sin Backend**: Funciona completamente en el navegador
- **Caché Inteligente**: El modelo se descarga una vez (~118MB) y se guarda en caché del navegador

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
│   ├── ServiceClassifier.tsx  # Componente principal con la lógica de IA
│   ├── categories.json        # Categorías de servicios y sinónimos
│   ├── main.tsx              # Punto de entrada de React
│   └── index.css             # Estilos globales con Tailwind
├── index.html                # HTML base con carga de Transformers.js
├── package.json              # Dependencias y configuración
├── tsconfig.json            # Configuración de TypeScript
├── tailwind.config.js       # Configuración de TailwindCSS
└── vite.config.ts           # Configuración de Vite
```

## 🧠 Cómo Funciona

### 1. Modelo de IA

El proyecto utiliza **Transformers.js** para ejecutar modelos de machine learning directamente en el navegador:

- **Modelo**: `Xenova/multilingual-e5-small`
- **Tipo**: Embeddings semánticos multilingües
- **Técnica**: Feature extraction con similitud coseno
- **Optimización**: Prefijos "query:" y "passage:" para mejor rendimiento (best practice E5)

### 2. Algoritmo de Clasificación

La clasificación combina dos enfoques:

1. **Embeddings Semánticos (80%)**
   - Convierte el texto en vectores numéricos
   - Compara la similitud coseno entre el query y las categorías
   - Captura el significado semántico profundo

2. **Búsqueda Fuzzy (20-60%)**
   - Busca coincidencias directas en sinónimos
   - Si encuentra coincidencias altas (>0.7), aumenta su peso al 60%
   - Ideal para términos técnicos específicos

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
- **multilingual-e5-small** - Modelo de embeddings

## ⚙️ Configuración Avanzada

### Ajustar Pesos del Algoritmo

En `ServiceClassifier.tsx`, líneas 228-235:

```typescript
// Combinar scores
let finalScore;
if (fuzzyScore >= 0.7) {
  finalScore = 0.4 * embeddingSimilarity + 0.6 * fuzzyScore;
} else {
  finalScore = 0.8 * embeddingSimilarity + 0.2 * fuzzyScore;
}
```

### Agregar Nuevas Categorías

Editar `src/categories.json`:

```json
{
  "id": 999,
  "name": "Nueva Categoría",
  "synonyms": ["sinónimo1", "sinónimo2", "término relacionado"]
}
```

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

- **Primera carga**: ~5-10 segundos (descarga del modelo 118MB)
- **Cargas posteriores**: ~2-3 segundos (modelo en caché)
- **Clasificación con WebGPU**: <200ms
- **Clasificación con CPU**: ~500ms-1s

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
2. Ajustar los pesos del algoritmo híbrido
3. Verificar que el query esté bien escrito en español

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
│   └── extractor / categoryEmbeddings
├── Efectos
│   ├── checkWebGPU()
│   └── loadModel()
└── Funciones
    ├── cosineSimilarity()
    ├── fuzzyMatch()
    └── classify()
```

### Flujo de Carga del Modelo

1. Verificar WebGPU disponible
2. Cargar Transformers.js desde CDN
3. Crear pipeline de feature-extraction
4. Generar embeddings para todas las categorías (con prefijo "passage:")
5. Guardar en estado para reutilizar

### Flujo de Clasificación

1. Usuario ingresa query
2. Generar embedding del query (con prefijo "query:")
3. Calcular similitud coseno con cada categoría
4. Calcular fuzzy match con sinónimos
5. Combinar scores con pesos adaptativos
6. Ordenar y mostrar top 10 resultados

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
