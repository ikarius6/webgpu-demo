import { useState, useEffect, useRef } from 'react';
import { AlertCircle, Zap, Search, Loader2, Github } from 'lucide-react';
import { pipeline, env } from '@huggingface/transformers';
import categoriesData from './categories.json';
import modelsConfig from './modelsConfig.json';
import WeightControls from './WeightControls';

/**
 * Clasificador de Servicios con IA - Multi-Matcher Strategy
 * 
 * Este componente implementa un sistema avanzado de clasificación inspirado en los samples:
 * 
 * ARQUITECTURA:
 * - Combina 3 matchers independientes con weighted voting:
 *   1. Keyword Matcher: Coincidencias exactas y substring matching
 *   2. Fuzzy Matcher: Levenshtein distance para variaciones ortográficas
 *   3. Embedding Matcher: Similitud semántica con multilingual-e5-small
 * 
 * MEJORAS IMPLEMENTADAS:
 * - Modelo: multilingual-e5-small con prefijos "query:" y "passage:" (best practice)
 * - Multi-matcher con pesos adaptativos (35% keyword, 30% fuzzy, 35% embedding)
 * - Position-based weighting para mejorar ranking (inspirado en search_engine.py)
 * - Levenshtein distance para fuzzy matching robusto
 * - Confidence threshold filtering (mínimo 15%)
 * - Score breakdown detallado en la UI
 * - Pesos dinámicos: aumenta peso de keyword si match >80%, fuzzy si >85%
 */
const ServiceClassifier = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<Array<{ category: string; score: number; embeddingScore: number; fuzzyScore: number; keywordScore: number }> | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [webGPUSupported, setWebGPUSupported] = useState<boolean | null>(null);
  const [extractor, setExtractor] = useState<any>(null);
  const [categoryEmbeddings, setCategoryEmbeddings] = useState<any>(null);
  const [showAllCategories, setShowAllCategories] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [weights, setWeights] = useState({ keyword: 0.35, fuzzy: 0.30, embedding: 0.35 });
  const isLoadingRef = useRef(false);
  const pipelineRef = useRef<any>(null);
  const availableModels = modelsConfig.models;
  const selectedModel = availableModels.find(m => m.id === selectedModelId);

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
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      loadModel();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pipelineRef.current) {
        console.log('[CLEANUP] Liberando recursos del pipeline al desmontar...');
        try {
          if (typeof pipelineRef.current.dispose === 'function') {
            pipelineRef.current.dispose();
          }
        } catch (err) {
          console.warn('[CLEANUP] Error al liberar recursos:', err);
        }
      }
    };
  }, []);

  const checkWebGPU = async () => {
    // Detect Firefox (WebGPU is 21x slower than Chrome)
    const isFirefox = navigator.userAgent.toLowerCase().includes('firefox');
    
    if ('gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        const hasWebGPU = adapter !== null;
        // Use WASM on Firefox even if WebGPU is available (performance reasons)
        setWebGPUSupported(hasWebGPU && !isFirefox);
        
        if (hasWebGPU && isFirefox) {
          console.log('[WebGPU] Firefox detectado - usando WASM por rendimiento (WebGPU es 21x más lento)');
        }
      } catch (e) {
        setWebGPUSupported(false);
      }
    } else {
      setWebGPUSupported(false);
    }
  };

  const loadModel = async () => {
    if (!selectedModel) {
      console.log('[SKIP] No model selected');
      setModelLoading(false);
      return;
    }

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
      
      // Limpiar GPU resources del modelo anterior
      if (pipelineRef.current) {
        console.log('[1.1] Liberando recursos del modelo anterior...');
        try {
          if (typeof pipelineRef.current.dispose === 'function') {
            await pipelineRef.current.dispose();
          }
        } catch (disposeErr) {
          console.warn('[1.1] Error al liberar recursos:', disposeErr);
        }
        pipelineRef.current = null;
      }
      
      // Limpiar estado previo
      setExtractor(null);
      setCategoryEmbeddings(null);
      setResult(null);

      // Transformers.js ya disponible via npm import
      console.log('[2] Transformers.js disponible via npm');

      // Configurar el entorno de transformers.js
      console.log('[3] Configurando entorno...');
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      console.log('[4] Configuración aplicada:', {
        allowRemoteModels: env.allowRemoteModels,
        remoteHost: env.remoteHost,
        remotePathTemplate: env.remotePathTemplate
      });
      
      // Cargar el modelo de embeddings seleccionado con WebGPU si está disponible
      console.log(`[5] Cargando modelo: ${selectedModel.name} (${selectedModel.huggingFaceId})...`);
      const pipelineStartTime = performance.now();
      const pipe = await pipeline(
        'feature-extraction',
        selectedModel.huggingFaceId,
        {
          device: webGPUSupported ? 'webgpu' : 'wasm',
          progress_callback: (progress: any) => {
            console.log('[PROGRESS]', progress);
          }
        }
      );
      const pipelineLoadTime = performance.now() - pipelineStartTime;
      console.log(`[5.1] ⏱️ Pipeline cargado en ${pipelineLoadTime.toFixed(2)}ms`);
      console.log('[6] Pipeline cargado exitosamente:', pipe);
      console.log('[6.1] Tipo de pipe:', typeof pipe);
      console.log('[6.2] Dispositivo:', webGPUSupported ? 'WebGPU' : 'WASM');

      // Validar que el pipeline es callable
      if (typeof pipe !== 'function') {
        throw new Error(`El pipeline no es una función callable. Tipo: ${typeof pipe}`);
      }

      // Hacer una prueba simple para verificar que funciona
      console.log('[7] Probando pipeline con texto de prueba...');
      try {
        const testStartTime = performance.now();
        await pipe('test');
        const testInferenceTime = performance.now() - testStartTime;
        console.log(`[7.1] ⏱️ Pipeline funcionando correctamente. Test inference: ${testInferenceTime.toFixed(2)}ms`);
      } catch (testErr) {
        console.error('[7.1] Error en prueba de pipeline:', testErr);
        throw new Error('Pipeline no pasó la prueba de validación');
      }

      // Generar embeddings para todas las categorías
      console.log('[8] Generando embeddings para categorías...');
      const embeddingsStartTime = performance.now();
      const prefix = selectedModel.requiresPrefixes ? 'passage: ' : '';
      const embeddingsPromises = categoryLabels.map(label => pipe(`${prefix}${label}`));
      const embeddings = await Promise.all(embeddingsPromises);
      const embeddingsTime = performance.now() - embeddingsStartTime;
      console.log(`[9] ⏱️ Embeddings generados: ${embeddings.length} en ${embeddingsTime.toFixed(2)}ms (${(embeddingsTime/embeddings.length).toFixed(2)}ms/embedding)`);

      // Guardar referencia del pipeline y actualizar estado
      pipelineRef.current = pipe;
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

  // Función para keyword matching (coincidencias exactas/substring)
  const keywordMatch = (query: string, synonyms: string[]): number => {
    const queryLower = query.toLowerCase().trim();
    const queryWords = queryLower.split(/\s+/);
    
    let matchCount = 0;
    let maxWordLength = 0;
    
    for (const synonym of synonyms) {
      const synLower = synonym.toLowerCase();
      
      // Coincidencia exacta completa - score muy alto
      if (synLower === queryLower) {
        return 1.0;
      }
      
      // Coincidencia de substring
      if (synLower.includes(queryLower)) {
        return 0.95;
      }
      
      if (queryLower.includes(synLower)) {
        return 0.9;
      }
      
      // Contar palabras individuales que coinciden
      for (const word of queryWords) {
        if (word.length >= 3 && synLower.includes(word)) {
          matchCount++;
          maxWordLength = Math.max(maxWordLength, word.length);
        }
      }
    }
    
    // Score basado en número de matches y longitud de palabras
    if (matchCount > 0) {
      return Math.min(0.7 * (matchCount / queryWords.length) + 0.1 * (maxWordLength / 10), 0.85);
    }
    
    return 0;
  };

  // Función mejorada para fuzzy matching usando Levenshtein distance
  const levenshteinDistance = (a: string, b: string): number => {
    const matrix: number[][] = [];
    
    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return matrix[b.length][a.length];
  };

  // Fuzzy matching mejorado con Levenshtein distance
  const fuzzyMatch = (query: string, synonyms: string[]): number => {
    const queryLower = query.toLowerCase().trim();
    const queryWords = queryLower.split(/\s+/);
    
    let bestScore = 0;
    const threshold = 80; // Umbral mínimo de similitud (0-100)
    
    for (const synonym of synonyms) {
      const synLower = synonym.toLowerCase();
      
      // Calcular ratio de similitud (0-100)
      const maxLen = Math.max(queryLower.length, synLower.length);
      const distance = levenshteinDistance(queryLower, synLower);
      const ratio = ((maxLen - distance) / maxLen) * 100;
      
      if (ratio > threshold) {
        bestScore = Math.max(bestScore, ratio / 100);
      }
      
      // También comparar palabras individuales
      for (const word of queryWords) {
        if (word.length >= 3) {
          const wordMaxLen = Math.max(word.length, synLower.length);
          const wordDistance = levenshteinDistance(word, synLower);
          const wordRatio = ((wordMaxLen - wordDistance) / wordMaxLen) * 100;
          
          if (wordRatio > threshold) {
            bestScore = Math.max(bestScore, wordRatio / 100 * 0.8);
          }
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
      const classifyStartTime = performance.now();
      console.log('[CLASSIFY] Generando embedding del input...');
      
      // Usar prefijo según el modelo seleccionado
      const queryPrefix = selectedModel?.requiresPrefixes ? 'query: ' : '';
      const prefixedInput = `${queryPrefix}${input}`;
      
      // Generar embedding del input
      const embeddingStartTime = performance.now();
      const inputEmbedding = await extractor(prefixedInput);
      const embeddingTime = performance.now() - embeddingStartTime;
      console.log(`[CLASSIFY] ⏱️ Embedding generado en ${embeddingTime.toFixed(2)}ms`);
      const inputVector = Array.from(inputEmbedding.data) as number[];
      
      const matchingStartTime = performance.now();
      console.log('[CLASSIFY] Calculando similitudes y fuzzy matches...');
      
      // Multi-matcher strategy: combinar keyword, fuzzy y embedding
      const similarities = categoryEmbeddings.map((catEmb: any, idx: number) => {
        const catVector = Array.from(catEmb.data) as number[];
        const embeddingSimilarity = cosineSimilarity(inputVector, catVector);
        
        // Extraer nombre de categoría y sinónimos
        const categoryName = categoryLabels[idx].split(':')[0].trim();
        const synonyms = categoriesWithSynonyms[idx].synonyms;
        
        // Calcular scores de los 3 matchers
        const keywordScore = keywordMatch(input, synonyms);
        const fuzzyScore = fuzzyMatch(input, synonyms);
        
        return {
          category: categoryName,
          embeddingScore: embeddingSimilarity,
          fuzzyScore: fuzzyScore,
          keywordScore: keywordScore,
          score: 0 // Se calculará después con weighted voting
        };
      });
      
      // Weighted voting con pesos configurables
      const baseWeights = weights;
      
      // Si hay keyword match fuerte, ajustar pesos
      similarities.forEach((item: any) => {
        let finalWeights = { ...baseWeights };
        
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
      
      // Ordenar por score combinado descendente
      similarities.sort((a: any, b: any) => b.score - a.score);
      
      // Aplicar position-based weighting (inspirado en los samples)
      const positionWeightedResults = similarities.map((item: any, idx: number) => {
        const positionWeight = 1.0 / (idx + 1);
        return {
          ...item,
          finalScore: item.score * (0.7 + 0.3 * positionWeight) // 70% score original + 30% bonus por posición
        };
      });
      
      // Re-ordenar con scores ajustados por posición
      positionWeightedResults.sort((a: any, b: any) => b.finalScore - a.finalScore);
      
      // Filtrar por confianza mínima (threshold)
      const minConfidence = 0.15; // Umbral mínimo
      const filteredResults = positionWeightedResults.filter((item: any) => item.finalScore >= minConfidence);
      
      // Tomar top 10
      const topResults = filteredResults.slice(0, 10).map((item: any) => ({
        category: item.category,
        score: item.finalScore,
        embeddingScore: item.embeddingScore,
        fuzzyScore: item.fuzzyScore,
        keywordScore: item.keywordScore
      }));
      
      const matchingTime = performance.now() - matchingStartTime;
      const totalClassifyTime = performance.now() - classifyStartTime;
      
      console.log(`[CLASSIFY] ⏱️ Matching completado en ${matchingTime.toFixed(2)}ms`);
      console.log(`[CLASSIFY] ⏱️ Clasificación total: ${totalClassifyTime.toFixed(2)}ms`);
      console.log('[CLASSIFY] Top 3 resultados:');
      topResults.slice(0, 3).forEach((r: any, i: number) => {
        console.log(`  ${i+1}. ${r.category}:`, {
          final: (r.score * 100).toFixed(1) + '%',
          keyword: (r.keywordScore * 100).toFixed(1) + '%',
          fuzzy: (r.fuzzyScore * 100).toFixed(1) + '%',
          embedding: (r.embeddingScore * 100).toFixed(1) + '%'
        });
      });
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
            <div className="flex items-center justify-between mb-2">
              <h1 className="text-3xl font-bold text-gray-800">
                Clasificador de Servicios
              </h1>
              <a
                href="https://github.com/ikarius6/webgpu-demo"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-gray-900 hover:bg-gray-800 text-white rounded-lg transition-colors shadow-md hover:shadow-lg"
                title="Ver repositorio en GitHub"
              >
                <Github className="w-5 h-5" />
                <span className="font-medium">GitHub</span>
              </a>
            </div>
            <p className="text-gray-600">
              Escribe lo que necesitas y el modelo te sugerirá el servicio más apropiado
            </p>
          </div>

          {/* WebGPU Status */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2">
              <Zap className={`w-5 h-5 ${webGPUSupported ? 'text-green-500' : 'text-orange-500'}`} />
              <span className="text-sm font-medium">
                {webGPUSupported === null ? 'Verificando...' : 
                 webGPUSupported ? 'WebGPU activado' : 
                 navigator.userAgent.toLowerCase().includes('firefox') ? 'Usando WASM (optimizado para Firefox)' :
                 'WebGPU no disponible (usando WASM)'}
              </span>
            </div>
          </div>

          {/* Model Loading */}
          {modelLoading && selectedModel && (
            <div className="mb-6 p-6 bg-blue-50 rounded-lg text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-3 text-blue-600" />
              <p className="text-blue-800 font-medium">Cargando {selectedModel.name}...</p>
              <p className="text-sm text-blue-600 mt-1">Primera vez: ~{selectedModel.size}. Luego se guarda en caché.</p>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          {/* Weight Controls */}
          <WeightControls
            weights={weights}
            setWeights={setWeights}
            disabled={modelLoading || loading}
            defaultCollapsed={true}
          />

          {/* Model Selector */}
          <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border border-purple-200">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Modelo de embeddings:
            </label>
            <select
              value={selectedModelId}
              onChange={(e) => {
                setSelectedModelId(e.target.value);
                isLoadingRef.current = false; // Reset para permitir nueva carga
              }}
              disabled={modelLoading || loading}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
            >
              <option value="">📥 Selecciona un modelo para comenzar...</option>
              {availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.size}) {model.recommended ? '⭐ Recomendado' : ''}
                </option>
              ))}
            </select>
            {selectedModel && (
              <p className="text-xs text-gray-600 mt-2">
                {selectedModel?.description} - Dimensiones: {selectedModel?.dimensions}
              </p>
            )}
          </div>

          {/* Categories */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              Categorías disponibles ({categories.length}):
            </h3>
            <div className="flex flex-wrap gap-2">
              {(showAllCategories ? categories : categories.slice(0, 20)).map((cat) => (
                <span
                  key={cat}
                  className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-medium"
                >
                  {cat}
                </span>
              ))}
              {!showAllCategories && categories.length > 20 && (
                <button
                  onClick={() => setShowAllCategories(true)}
                  className="px-3 py-1 bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700 rounded-full text-sm font-bold hover:from-purple-200 hover:to-pink-200 transition-all duration-200 shadow-sm hover:shadow-md"
                >
                  +{categories.length - 20} más
                </button>
              )}
              {showAllCategories && (
                <button
                  onClick={() => setShowAllCategories(false)}
                  className="px-3 py-1 bg-gray-200 text-gray-700 rounded-full text-sm font-medium hover:bg-gray-300 transition-colors"
                >
                  Mostrar menos
                </button>
              )}
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
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div
                        className={`h-2 rounded-full transition-all ${
                          idx === 0 ? 'bg-green-500' : 'bg-gray-400'
                        }`}
                        style={{ width: `${item.score * 100}%` }}
                      />
                    </div>
                    {/* Score breakdown */}
                    <div className="flex gap-2 text-xs mt-2">
                      <div className="flex items-center gap-1">
                        <span className="font-medium text-blue-600">Keyword:</span>
                        <span className="text-gray-600">{(item.keywordScore * 100).toFixed(0)}%</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="font-medium text-purple-600">Fuzzy:</span>
                        <span className="text-gray-600">{(item.fuzzyScore * 100).toFixed(0)}%</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="font-medium text-orange-600">Semantic:</span>
                        <span className="text-gray-600">{(item.embeddingScore * 100).toFixed(0)}%</span>
                      </div>
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