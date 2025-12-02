import { useState, useEffect, useRef } from 'react';
import { AlertCircle, Zap, Search, Loader2 } from 'lucide-react';
import categoriesData from './categories.json';

/**
 * Clasificador de Servicios con IA
 * 
 * Este componente usa embeddings semánticos multilingües + búsqueda fuzzy para clasificar servicios.
 * 
 * Mejoras implementadas:
 * - Modelo: multilingual-e5-small con prefijos "query:" y "passage:" (best practice)
 * - Usa TODOS los sinónimos para mejor contexto semántico
 * - Búsqueda fuzzy en sinónimos para coincidencias directas (prioriza matches exactos)
 * - Combina scores: 80% embeddings + 20% fuzzy (60% fuzzy si match > 0.7)
 */
const ServiceClassifier = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<Array<{ category: string; score: number }> | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [webGPUSupported, setWebGPUSupported] = useState<boolean | null>(null);
  const [extractor, setExtractor] = useState<any>(null);
  const [categoryEmbeddings, setCategoryEmbeddings] = useState<any>(null);
  const isLoadingRef = useRef(false);

  // Crear etiquetas enriquecidas con sinónimos para mejor contexto
  const categoriesWithSynonyms = categoriesData.items.map(item => ({
    name: item.name,
    label: `${item.name}: ${item.synonyms.join(', ')}`, // TODOS los sinónimos
    synonyms: item.synonyms
  }));
  
  const categories = categoriesData.items.map(item => item.name);
  const categoryLabels = categoriesWithSynonyms.map(item => item.label);

  const examples = [
    'necesito arreglar una fuga de agua',
    'mi jardín necesita poda',
    'me duele un diente',
    'quiero pintar mi sala',
    'necesito reparar una puerta',
    'mi refrigerador no enfría',
    'limpieza dental'
  ];

  useEffect(() => {
    checkWebGPU();
    loadModel();
  }, []);

  const checkWebGPU = async () => {
    if ('gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        setWebGPUSupported(adapter !== null);
      } catch (e) {
        setWebGPUSupported(false);
      }
    } else {
      setWebGPUSupported(false);
    }
  };

  const loadModel = async () => {
    // Prevenir doble inicialización (React StrictMode)
    if (isLoadingRef.current) {
      console.log('[SKIP] Ya se está cargando el modelo');
      return;
    }

    try {
      isLoadingRef.current = true;
      console.log('[1] Iniciando carga del modelo...');
      setModelLoading(true);
      setError("");

      // Esperar a que transformers.js esté disponible en window
      console.log('[2] Esperando transformers.js desde CDN...');
      let attempts = 0;
      while (!(window as any).transformers && attempts < 50) {
        await new Promise(resolve => setTimeout(resolve, 100));
        attempts++;
      }

      if (!(window as any).transformers) {
        throw new Error('Transformers.js no se cargó desde el CDN');
      }

      const { pipeline, env } = (window as any).transformers;
      console.log('[3] Transformers.js disponible');

      // Configurar el entorno de transformers.js (usar defaults)
      console.log('[4] Configurando entorno...');
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      // No sobrescribir remoteHost y remotePathTemplate - usar los defaults del paquete
      console.log('[5] Configuración aplicada (usando defaults):', {
        allowRemoteModels: env.allowRemoteModels,
        remoteHost: env.remoteHost,
        remotePathTemplate: env.remotePathTemplate
      });
      
      // Cargar el modelo de embeddings multilingüe
      console.log('[6] Cargando pipeline de feature-extraction...');
      const pipe = await pipeline(
        'feature-extraction',
        'Xenova/multilingual-e5-small',
        {
          progress_callback: (progress: any) => {
            console.log('[PROGRESS]', progress);
          }
        }
      );
      console.log('[7] Pipeline cargado exitosamente:', pipe);
      console.log('[7.1] Tipo de pipe:', typeof pipe);
      console.log('[7.2] Es Promise?:', pipe instanceof Promise);

      // Validar que el pipeline es callable
      if (typeof pipe !== 'function') {
        throw new Error(`El pipeline no es una función callable. Tipo: ${typeof pipe}`);
      }

      // Hacer una prueba simple para verificar que funciona
      console.log('[7.5] Probando pipeline con texto de prueba...');
      try {
        await pipe('test');
        console.log('[7.6] Pipeline funcionando correctamente');
      } catch (testErr) {
        console.error('[7.6] Error en prueba de pipeline:', testErr);
        throw new Error('Pipeline no pasó la prueba de validación');
      }

      // Generar embeddings para todas las categorías
      // Usar prefijo "passage:" para documentos (best practice E5)
      console.log('[8] Generando embeddings para categorías...');
      const embeddingsPromises = categoryLabels.map(label => pipe(`passage: ${label}`));
      const embeddings = await Promise.all(embeddingsPromises);
      console.log('[9] Embeddings generados:', embeddings.length);

      // Usar callback setter para evitar conflictos de estado
      setExtractor(() => pipe);
      setCategoryEmbeddings(embeddings);
      setModelLoading(false);
      isLoadingRef.current = false;
      console.log('[10] Modelo listo para usar');
    } catch (err) {
      console.error('[ERROR] Error durante la carga del modelo:');
      console.error('Error object:', err);
      console.error('Error stack:', err instanceof Error ? err.stack : 'No stack trace');
      const errorMessage = err instanceof Error ? err.message : 'Error desconocido';
      setError(`Error al cargar el modelo: ${errorMessage}`);
      setModelLoading(false);
      isLoadingRef.current = false;
    }
  };

  // Función para calcular similitud coseno
  const cosineSimilarity = (a: number[], b: number[]) => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  };

  // Función para buscar coincidencias fuzzy en sinónimos
  const fuzzyMatch = (query: string, synonyms: string[]): number => {
    const queryLower = query.toLowerCase().trim();
    const queryWords = queryLower.split(/\s+/);
    
    let bestScore = 0;
    
    for (const synonym of synonyms) {
      const synLower = synonym.toLowerCase();
      
      // Coincidencia exacta completa
      if (synLower === queryLower) {
        return 1.0;
      }
      
      // Coincidencia exacta de palabra
      if (synLower.includes(queryLower) || queryLower.includes(synLower)) {
        bestScore = Math.max(bestScore, 0.9);
      }
      
      // Coincidencia de palabras individuales
      for (const word of queryWords) {
        if (word.length >= 3 && synLower.includes(word)) {
          bestScore = Math.max(bestScore, 0.7);
        }
      }
    }
    
    return bestScore;
  };

  const classify = async () => {
    if (!input.trim() || !extractor || !categoryEmbeddings) return;

    setLoading(true);
    setResult(null);
    setError("");

    try {
      console.log('[CLASSIFY] Generando embedding del input...');
      
      // Prefijo "query:" es best practice para modelos E5
      const prefixedInput = `query: ${input}`;
      
      // Generar embedding del input
      const inputEmbedding = await extractor(prefixedInput);
      const inputVector = Array.from(inputEmbedding.data) as number[];
      
      console.log('[CLASSIFY] Calculando similitudes y fuzzy matches...');
      
      // Calcular similitud con cada categoría
      const similarities = categoryEmbeddings.map((catEmb: any, idx: number) => {
        const catVector = Array.from(catEmb.data) as number[];
        const embeddingSimilarity = cosineSimilarity(inputVector, catVector);
        
        // Extraer nombre de categoría y sinónimos
        const categoryName = categoryLabels[idx].split(':')[0].trim();
        const synonyms = categoriesWithSynonyms[idx].synonyms;
        
        // Calcular fuzzy match score
        const fuzzyScore = fuzzyMatch(input, synonyms);
        
        // Combinar scores: 70% embedding + 30% fuzzy
        // Si hay fuzzy match alto (>0.7), darle más peso
        let finalScore;
        if (fuzzyScore >= 0.7) {
          finalScore = 0.4 * embeddingSimilarity + 0.6 * fuzzyScore;
        } else {
          finalScore = 0.8 * embeddingSimilarity + 0.2 * fuzzyScore;
        }
        
        return {
          category: categoryName,
          score: finalScore,
          embeddingScore: embeddingSimilarity,
          fuzzyScore: fuzzyScore
        };
      });
      
      // Ordenar por similitud descendente
      similarities.sort((a: any, b: any) => b.score - a.score);
      
      // Tomar top 10
      const topResults = similarities.slice(0, 10);
      
      console.log('[CLASSIFY] Top 3 resultados:', topResults.slice(0, 3));
      setResult(topResults);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error desconocido';
      setError(`Error en la clasificación: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example: string) => {
    setInput(example);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Clasificador de Servicios
            </h1>
            <p className="text-gray-600">
              Escribe lo que necesitas y el modelo te sugerirá el servicio más apropiado
            </p>
          </div>

          {/* WebGPU Status */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2">
              <Zap className={`w-5 h-5 ${webGPUSupported ? 'text-green-500' : 'text-orange-500'}`} />
              <span className="text-sm font-medium">
                WebGPU: {webGPUSupported === null ? 'Verificando...' : webGPUSupported ? 'Soportado ✓' : 'No disponible (usando CPU)'}
              </span>
            </div>
          </div>

          {/* Model Loading */}
          {modelLoading && (
            <div className="mb-6 p-6 bg-blue-50 rounded-lg text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-3 text-blue-600" />
              <p className="text-blue-800 font-medium">Cargando modelo y generando embeddings...</p>
              <p className="text-sm text-blue-600 mt-1">Primera vez: ~118MB. Luego se guarda en caché.</p>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          {/* Categories */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Categorías disponibles:</h3>
            <div className="flex flex-wrap gap-2">
              {categories.map((cat) => (
                <span
                  key={cat}
                  className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-medium"
                >
                  {cat}
                </span>
              ))}
            </div>
          </div>

          {/* Search Input */}
          <div className="mb-6">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && classify()}
                placeholder="Ej: necesito reparar una tubería rota"
                disabled={modelLoading}
                className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:outline-none text-lg disabled:bg-gray-100"
              />
            </div>
            <button
              onClick={classify}
              disabled={!input.trim() || loading || modelLoading}
              className="w-full mt-3 bg-indigo-600 text-white py-3 rounded-xl font-semibold hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Clasificando...
                </span>
              ) : (
                'Clasificar'
              )}
            </button>
          </div>

          {/* Examples */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Prueba estos ejemplos:</h3>
            <div className="flex flex-wrap gap-2">
              {examples.map((example) => (
                <button
                  key={example}
                  onClick={() => handleExampleClick(example)}
                  disabled={modelLoading}
                  className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-sm transition-colors disabled:opacity-50"
                >
                  "{example}"
                </button>
              ))}
            </div>
          </div>

          {/* Results */}
          {result && (
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Resultados:</h3>
              <div className="space-y-3">
                {result.map((item, idx) => (
                  <div key={item.category} className="bg-white rounded-lg p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <span className={`font-semibold ${idx === 0 ? 'text-green-700 text-lg' : 'text-gray-700'}`}>
                        {idx === 0 && '🎯 '}{item.category}
                      </span>
                      <span className={`font-bold ${idx === 0 ? 'text-green-700' : 'text-gray-600'}`}>
                        {(item.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${
                          idx === 0 ? 'bg-green-500' : 'bg-gray-400'
                        }`}
                        style={{ width: `${item.score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ServiceClassifier;