import { useState, useEffect, useRef } from 'react';
import { Play, Download, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import categoriesData from './categories.json';
import testCasesData from './testCases.json';
import modelsConfig from './modelsConfig.json';
import WeightControls from './WeightControls';

interface TestCase {
  id: number;
  query: string;
  expectedCategory: string;
  alternativeCategories: string[];
  description: string;
}

interface TestResult {
  testId: number;
  query: string;
  expectedCategory: string;
  alternativeCategories: string[];
  predictions: Array<{
    category: string;
    score: number;
    embeddingScore: number;
    fuzzyScore: number;
    keywordScore: number;
  }>;
  correctTopK: {
    top1: boolean;
    top3: boolean;
    top5: boolean;
  };
  rank: number | null; // Position of correct answer, null if not in top 10
}

const ModelTester = () => {
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testing, setTesting] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>("");
  const [extractor, setExtractor] = useState<any>(null);
  const [categoryEmbeddings, setCategoryEmbeddings] = useState<any>(null);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [weights, setWeights] = useState({ keyword: 0.35, fuzzy: 0.30, embedding: 0.35 });
  const isLoadingRef = useRef(false);
  const availableModels = modelsConfig.models;
  const selectedModel = availableModels.find(m => m.id === selectedModelId);

  // Same category setup as ServiceClassifier
  const categoriesWithSynonyms = categoriesData.items.map(item => ({
    name: item.name,
    label: `${item.name}: ${item.synonyms.join(', ')}`,
    synonyms: item.synonyms
  }));
  
  const categoryLabels = categoriesWithSynonyms.map(item => item.label);
  const testCases: TestCase[] = testCasesData.testCases;

  useEffect(() => {
    if (selectedModelId) {
      loadModel();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId]);

  const loadModel = async () => {
    if (!selectedModel) {
      console.log('[SKIP] No model selected');
      setModelLoading(false);
      return;
    }

    if (isLoadingRef.current) {
      console.log('[SKIP] Model already loading');
      return;
    }

    try {
      isLoadingRef.current = true;
      console.log('[TEST] Loading model...');
      setModelLoading(true);
      setError("");
      
      // Clear previous state
      setExtractor(null);
      setCategoryEmbeddings(null);
      setTestResults([]);

      // Wait for transformers.js
      let attempts = 0;
      while (!(window as any).transformers && attempts < 50) {
        await new Promise(resolve => setTimeout(resolve, 100));
        attempts++;
      }

      if (!(window as any).transformers) {
        throw new Error('Transformers.js not loaded from CDN');
      }

      const { pipeline, env } = (window as any).transformers;
      console.log('[TEST] Transformers.js available');

      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      
      // Load embedding model
      console.log(`[TEST] Loading model: ${selectedModel.name}...`);
      const pipe = await pipeline(
        'feature-extraction',
        selectedModel.huggingFaceId,
        {
          progress_callback: (progress: any) => {
            console.log('[TEST PROGRESS]', progress);
          }
        }
      );

      if (typeof pipe !== 'function') {
        throw new Error(`Pipeline is not callable. Type: ${typeof pipe}`);
      }

      // Test pipeline
      console.log('[TEST] Testing pipeline...');
      await pipe('test');
      console.log('[TEST] Pipeline working correctly');

      // Generate embeddings for all categories
      console.log('[TEST] Generating category embeddings...');
      const prefix = selectedModel.requiresPrefixes ? 'passage: ' : '';
      const embeddingsPromises = categoryLabels.map(label => pipe(`${prefix}${label}`));
      const embeddings = await Promise.all(embeddingsPromises);
      console.log('[TEST] Embeddings generated:', embeddings.length);

      setExtractor(() => pipe);
      setCategoryEmbeddings(embeddings);
      setModelLoading(false);
      isLoadingRef.current = false;
      console.log('[TEST] Model ready');
    } catch (err) {
      console.error('[TEST ERROR]', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error loading model: ${errorMessage}`);
      setModelLoading(false);
      isLoadingRef.current = false;
    }
  };

  // Cosine similarity
  const cosineSimilarity = (a: number[], b: number[]) => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  };

  // Keyword matching
  const keywordMatch = (query: string, synonyms: string[]): number => {
    const queryLower = query.toLowerCase().trim();
    const queryWords = queryLower.split(/\s+/);
    
    let matchCount = 0;
    let maxWordLength = 0;
    
    for (const synonym of synonyms) {
      const synLower = synonym.toLowerCase();
      
      if (synLower === queryLower) return 1.0;
      if (synLower.includes(queryLower)) return 0.95;
      if (queryLower.includes(synLower)) return 0.9;
      
      for (const word of queryWords) {
        if (word.length >= 3 && synLower.includes(word)) {
          matchCount++;
          maxWordLength = Math.max(maxWordLength, word.length);
        }
      }
    }
    
    if (matchCount > 0) {
      return Math.min(0.7 * (matchCount / queryWords.length) + 0.1 * (maxWordLength / 10), 0.85);
    }
    
    return 0;
  };

  // Levenshtein distance
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

  // Fuzzy matching
  const fuzzyMatch = (query: string, synonyms: string[]): number => {
    const queryLower = query.toLowerCase().trim();
    const queryWords = queryLower.split(/\s+/);
    
    let bestScore = 0;
    const threshold = 80;
    
    for (const synonym of synonyms) {
      const synLower = synonym.toLowerCase();
      
      const maxLen = Math.max(queryLower.length, synLower.length);
      const distance = levenshteinDistance(queryLower, synLower);
      const ratio = ((maxLen - distance) / maxLen) * 100;
      
      if (ratio > threshold) {
        bestScore = Math.max(bestScore, ratio / 100);
      }
      
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

  // Classify a single query (same logic as ServiceClassifier)
  const classifyQuery = async (query: string, customWeights = weights) => {
    if (!extractor || !categoryEmbeddings) {
      throw new Error('Model not ready');
    }

    const queryPrefix = selectedModel?.requiresPrefixes ? 'query: ' : '';
    const prefixedInput = `${queryPrefix}${query}`;
    const inputEmbedding = await extractor(prefixedInput);
    const inputVector = Array.from(inputEmbedding.data) as number[];
    
    const similarities = categoryEmbeddings.map((catEmb: any, idx: number) => {
      const catVector = Array.from(catEmb.data) as number[];
      const embeddingSimilarity = cosineSimilarity(inputVector, catVector);
      
      const categoryName = categoryLabels[idx].split(':')[0].trim();
      const synonyms = categoriesWithSynonyms[idx].synonyms;
      
      const keywordScore = keywordMatch(query, synonyms);
      const fuzzyScore = fuzzyMatch(query, synonyms);
      
      return {
        category: categoryName,
        embeddingScore: embeddingSimilarity,
        fuzzyScore: fuzzyScore,
        keywordScore: keywordScore,
        score: 0
      };
    });
    
    // Weighted voting
    const baseWeights = customWeights;
    
    similarities.forEach((item: any) => {
      let finalWeights = { ...baseWeights };
      
      if (item.keywordScore >= 0.8) {
        finalWeights = { keyword: 0.50, fuzzy: 0.20, embedding: 0.30 };
      } else if (item.fuzzyScore >= 0.85) {
        finalWeights = { keyword: 0.25, fuzzy: 0.45, embedding: 0.30 };
      }
      
      item.score = (
        item.keywordScore * finalWeights.keyword +
        item.fuzzyScore * finalWeights.fuzzy +
        item.embeddingScore * finalWeights.embedding
      );
    });
    
    similarities.sort((a: any, b: any) => b.score - a.score);
    
    // Position-based weighting
    const positionWeightedResults = similarities.map((item: any, idx: number) => {
      const positionWeight = 1.0 / (idx + 1);
      return {
        ...item,
        finalScore: item.score * (0.7 + 0.3 * positionWeight)
      };
    });
    
    positionWeightedResults.sort((a: any, b: any) => b.finalScore - a.finalScore);
    
    // Filter by confidence
    const minConfidence = 0.15;
    const filteredResults = positionWeightedResults.filter((item: any) => item.finalScore >= minConfidence);
    
    // Return top 10
    return filteredResults.slice(0, 10).map((item: any) => ({
      category: item.category,
      score: item.finalScore,
      embeddingScore: item.embeddingScore,
      fuzzyScore: item.fuzzyScore,
      keywordScore: item.keywordScore
    }));
  };

  // Run all tests
  const runTests = async () => {
    if (!extractor || !categoryEmbeddings) {
      setError('Model not ready');
      return;
    }

    setTesting(true);
    setTestResults([]);
    setProgress(0);
    setError("");

    const results: TestResult[] = [];

    try {
      for (let i = 0; i < testCases.length; i++) {
        const testCase = testCases[i];
        console.log(`[TEST ${i + 1}/${testCases.length}] Testing: "${testCase.query}"`);

        const predictions = await classifyQuery(testCase.query, weights);
        
        // Check if correct answer is in top-k
        const allAcceptableAnswers = [testCase.expectedCategory, ...testCase.alternativeCategories];
        
        const top1Match = allAcceptableAnswers.includes(predictions[0]?.category);
        const top3Match = predictions.slice(0, 3).some((p: any) => allAcceptableAnswers.includes(p.category));
        const top5Match = predictions.slice(0, 5).some((p: any) => allAcceptableAnswers.includes(p.category));
        
        // Find rank of correct answer
        let rank: number | null = null;
        for (let j = 0; j < predictions.length; j++) {
          if (allAcceptableAnswers.includes(predictions[j].category)) {
            rank = j + 1;
            break;
          }
        }

        const result: TestResult = {
          testId: testCase.id,
          query: testCase.query,
          expectedCategory: testCase.expectedCategory,
          alternativeCategories: testCase.alternativeCategories,
          predictions: predictions.slice(0, 5), // Store top 5
          correctTopK: {
            top1: top1Match,
            top3: top3Match,
            top5: top5Match
          },
          rank: rank
        };

        results.push(result);
        setTestResults([...results]);
        setProgress(((i + 1) / testCases.length) * 100);

        // Small delay to avoid overwhelming the system
        await new Promise(resolve => setTimeout(resolve, 50));
      }

      console.log('[TEST] All tests completed');
    } catch (err) {
      console.error('[TEST ERROR]', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error during testing: ${errorMessage}`);
    } finally {
      setTesting(false);
    }
  };

  // Download results as JSON
  const downloadResults = () => {
    const dataStr = JSON.stringify(testResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `model-test-results-${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Calculate statistics
  const calculateStats = () => {
    if (testResults.length === 0) return null;

    const top1Correct = testResults.filter(r => r.correctTopK.top1).length;
    const top3Correct = testResults.filter(r => r.correctTopK.top3).length;
    const top5Correct = testResults.filter(r => r.correctTopK.top5).length;
    const total = testResults.length;

    return {
      top1Accuracy: (top1Correct / total) * 100,
      top3Accuracy: (top3Correct / total) * 100,
      top5Accuracy: (top5Correct / total) * 100,
      total: total,
      top1Correct: top1Correct,
      top3Correct: top3Correct,
      top5Correct: top5Correct
    };
  };

  const stats = calculateStats();

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Probador de modelos
          </h1>
          <p className="text-gray-600 mb-6">
            {selectedModel ? `Evaluando ${selectedModel.name} con ${testCases.length} casos de prueba` : `Evalúa cualquier modelo con ${testCases.length} casos de prueba`}
          </p>

          {/* Model Status */}
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
            disabled={testing || modelLoading}
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
              disabled={modelLoading || testing}
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
                {selectedModel.description} - Dimensiones: {selectedModel.dimensions}
              </p>
            )}
          </div>

          {/* Control Buttons */}
          {!modelLoading && (
            <div className="mb-6 flex gap-4">
              <button
                onClick={runTests}
                disabled={testing || modelLoading}
                className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {testing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Testing... {progress.toFixed(0)}%
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Run Tests
                  </>
                )}
              </button>

              {testResults.length > 0 && (
                <button
                  onClick={downloadResults}
                  className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors"
                >
                  <Download className="w-5 h-5" />
                  Download Results
                </button>
              )}
            </div>
          )}

          {/* Progress Bar */}
          {testing && (
            <div className="mb-6">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-purple-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Statistics */}
          {stats && (
            <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-green-700 mb-1">Top-1 Accuracy</h3>
                <p className="text-3xl font-bold text-green-800">{stats.top1Accuracy.toFixed(1)}%</p>
                <p className="text-sm text-green-600">{stats.top1Correct} / {stats.total} correct</p>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-blue-700 mb-1">Top-3 Accuracy</h3>
                <p className="text-3xl font-bold text-blue-800">{stats.top3Accuracy.toFixed(1)}%</p>
                <p className="text-sm text-blue-600">{stats.top3Correct} / {stats.total} correct</p>
              </div>
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-purple-700 mb-1">Top-5 Accuracy</h3>
                <p className="text-3xl font-bold text-purple-800">{stats.top5Accuracy.toFixed(1)}%</p>
                <p className="text-sm text-purple-600">{stats.top5Correct} / {stats.total} correct</p>
              </div>
            </div>
          )}

          {/* Results Table */}
          {testResults.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border border-gray-300 px-3 py-2 text-left text-sm font-semibold">ID</th>
                    <th className="border border-gray-300 px-3 py-2 text-left text-sm font-semibold">Query</th>
                    <th className="border border-gray-300 px-3 py-2 text-left text-sm font-semibold">Expected</th>
                    <th className="border border-gray-300 px-3 py-2 text-left text-sm font-semibold">Top-1</th>
                    <th className="border border-gray-300 px-3 py-2 text-left text-sm font-semibold">Top-3</th>
                    <th className="border border-gray-300 px-3 py-2 text-left text-sm font-semibold">Top-5</th>
                    <th className="border border-gray-300 px-3 py-2 text-left text-sm font-semibold">Rank</th>
                  </tr>
                </thead>
                <tbody>
                  {testResults.map((result) => (
                    <tr key={result.testId} className={result.correctTopK.top1 ? 'bg-green-50' : 'bg-white'}>
                      <td className="border border-gray-300 px-3 py-2 text-sm">{result.testId}</td>
                      <td className="border border-gray-300 px-3 py-2 text-sm">{result.query}</td>
                      <td className="border border-gray-300 px-3 py-2 text-sm font-medium">{result.expectedCategory}</td>
                      <td className="border border-gray-300 px-3 py-2 text-center">
                        {result.correctTopK.top1 ? (
                          <CheckCircle className="w-5 h-5 text-green-600 mx-auto" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-600 mx-auto" />
                        )}
                      </td>
                      <td className="border border-gray-300 px-3 py-2 text-center">
                        {result.correctTopK.top3 ? (
                          <CheckCircle className="w-5 h-5 text-green-600 mx-auto" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-600 mx-auto" />
                        )}
                      </td>
                      <td className="border border-gray-300 px-3 py-2 text-center">
                        {result.correctTopK.top5 ? (
                          <CheckCircle className="w-5 h-5 text-green-600 mx-auto" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-600 mx-auto" />
                        )}
                      </td>
                      <td className="border border-gray-300 px-3 py-2 text-sm text-center">
                        {result.rank ? `#${result.rank}` : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Detailed Results (expandable) */}
          {testResults.length > 0 && (
            <div className="mt-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Detailed Results</h2>
              <div className="space-y-4">
                {testResults.map((result) => (
                  <details key={result.testId} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <summary className="cursor-pointer font-semibold text-gray-800 hover:text-purple-600">
                      Test #{result.testId}: {result.query} 
                      {result.correctTopK.top1 ? ' ✓' : ' ✗'}
                    </summary>
                    <div className="mt-4 space-y-2">
                      <p className="text-sm">
                        <span className="font-semibold">Expected:</span> {result.expectedCategory}
                        {result.alternativeCategories.length > 0 && (
                          <span className="text-gray-600"> (or: {result.alternativeCategories.join(', ')})</span>
                        )}
                      </p>
                      <div className="mt-2">
                        <p className="font-semibold text-sm mb-2">Top 5 Predictions:</p>
                        {result.predictions.map((pred, idx) => (
                          <div key={idx} className="text-sm mb-2 pl-4 border-l-2 border-gray-300">
                            <p className="font-medium">
                              #{idx + 1}: {pred.category} - {(pred.score * 100).toFixed(1)}%
                            </p>
                            <p className="text-xs text-gray-600">
                              Keyword: {(pred.keywordScore * 100).toFixed(0)}% | 
                              Fuzzy: {(pred.fuzzyScore * 100).toFixed(0)}% | 
                              Semantic: {(pred.embeddingScore * 100).toFixed(0)}%
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </details>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelTester;
