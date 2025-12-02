import { useState } from 'react';
import { Play, Download, Loader2, BarChart3, AlertCircle } from 'lucide-react';
import categoriesData from './categories.json';
import testCasesData from './testCases.json';
import modelsConfig from './modelsConfig.json';

interface TestCase {
  id: number;
  query: string;
  expectedCategory: string;
  alternativeCategories: string[];
  description: string;
}

interface ModelResult {
  modelId: string;
  modelName: string;
  top1Accuracy: number;
  top3Accuracy: number;
  top5Accuracy: number;
  averageInferenceTime: number;
  totalTestTime: number;
  testsPassed: number;
  testsFailed: number;
}

interface ModelConfig {
  id: string;
  name: string;
  huggingFaceId: string;
  size: string;
  dimensions: number;
  requiresPrefixes: boolean;
  description: string;
  recommended: boolean;
  category: string;
}

const ModelComparison = () => {
  const [results, setResults] = useState<ModelResult[]>([]);
  const [testing, setTesting] = useState(false);
  const [currentModel, setCurrentModel] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>("");
  const [selectedModels, setSelectedModels] = useState<string[]>(['multilingual-e5-small']);

  const categoriesWithSynonyms = categoriesData.items.map(item => ({
    name: item.name,
    label: `${item.name}: ${item.synonyms.join(', ')}`,
    synonyms: item.synonyms
  }));
  
  const categoryLabels = categoriesWithSynonyms.map(item => item.label);
  const testCases: TestCase[] = testCasesData.testCases;
  const availableModels: ModelConfig[] = modelsConfig.models;

  const loadModel = async (modelConfig: ModelConfig) => {
    console.log(`[COMPARISON] Loading model: ${modelConfig.name}`);

    let attempts = 0;
    while (!(window as any).transformers && attempts < 50) {
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }

    if (!(window as any).transformers) {
      throw new Error('Transformers.js not loaded from CDN');
    }

    const { pipeline, env } = (window as any).transformers;
    
    env.allowRemoteModels = true;
    env.allowLocalModels = false;
    
    const pipe = await pipeline(
      'feature-extraction',
      modelConfig.huggingFaceId,
      {
        progress_callback: (progress: any) => {
          console.log(`[${modelConfig.id}]`, progress);
        }
      }
    );

    if (typeof pipe !== 'function') {
      throw new Error(`Pipeline is not callable for ${modelConfig.name}`);
    }

    await pipe('test');
    
    // Generate embeddings for all categories
    const prefix = modelConfig.requiresPrefixes ? 'passage: ' : '';
    const embeddingsPromises = categoryLabels.map(label => pipe(`${prefix}${label}`));
    const embeddings = await Promise.all(embeddingsPromises);
    
    return { pipe, embeddings };
  };

  const cosineSimilarity = (a: number[], b: number[]) => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  };

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

  const classifyQuery = async (query: string, extractor: any, categoryEmbeddings: any, requiresPrefixes: boolean) => {
    const prefix = requiresPrefixes ? 'query: ' : '';
    const prefixedInput = `${prefix}${query}`;
    
    const startTime = performance.now();
    const inputEmbedding = await extractor(prefixedInput);
    const inferenceTime = performance.now() - startTime;
    
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
    
    const weights = { keyword: 0.35, fuzzy: 0.30, embedding: 0.35 };
    
    similarities.forEach((item: any) => {
      let finalWeights = { ...weights };
      
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
    
    const positionWeightedResults = similarities.map((item: any, idx: number) => {
      const positionWeight = 1.0 / (idx + 1);
      return {
        ...item,
        finalScore: item.score * (0.7 + 0.3 * positionWeight)
      };
    });
    
    positionWeightedResults.sort((a: any, b: any) => b.finalScore - a.finalScore);
    
    const minConfidence = 0.15;
    const filteredResults = positionWeightedResults.filter((item: any) => item.finalScore >= minConfidence);
    
    return {
      predictions: filteredResults.slice(0, 10).map((item: any) => ({
        category: item.category,
        score: item.finalScore
      })),
      inferenceTime
    };
  };

  const testModel = async (modelConfig: ModelConfig) => {
    console.log(`[COMPARISON] Testing model: ${modelConfig.name}`);
    setCurrentModel(modelConfig.name);
    
    const startTime = performance.now();
    
    // Load model
    const { pipe, embeddings } = await loadModel(modelConfig);
    
    let top1Correct = 0;
    let top3Correct = 0;
    let top5Correct = 0;
    let totalInferenceTime = 0;
    
    // Run all test cases
    for (let i = 0; i < testCases.length; i++) {
      const testCase = testCases[i];
      
      const { predictions, inferenceTime } = await classifyQuery(
        testCase.query,
        pipe,
        embeddings,
        modelConfig.requiresPrefixes
      );
      
      totalInferenceTime += inferenceTime;
      
      const allAcceptableAnswers = [testCase.expectedCategory, ...testCase.alternativeCategories];
      
      if (predictions[0] && allAcceptableAnswers.includes(predictions[0].category)) {
        top1Correct++;
      }
      
      if (predictions.slice(0, 3).some((p: any) => allAcceptableAnswers.includes(p.category))) {
        top3Correct++;
      }
      
      if (predictions.slice(0, 5).some((p: any) => allAcceptableAnswers.includes(p.category))) {
        top5Correct++;
      }
      
      setProgress(((i + 1) / testCases.length) * 100);
      
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    
    const totalTestTime = performance.now() - startTime;
    const total = testCases.length;
    
    return {
      modelId: modelConfig.id,
      modelName: modelConfig.name,
      top1Accuracy: (top1Correct / total) * 100,
      top3Accuracy: (top3Correct / total) * 100,
      top5Accuracy: (top5Correct / total) * 100,
      averageInferenceTime: totalInferenceTime / total,
      totalTestTime: totalTestTime / 1000, // Convert to seconds
      testsPassed: top1Correct,
      testsFailed: total - top1Correct
    };
  };

  const runComparison = async () => {
    if (selectedModels.length === 0) {
      setError('Please select at least one model to test');
      return;
    }

    setTesting(true);
    setResults([]);
    setProgress(0);
    setError("");

    const comparisonResults: ModelResult[] = [];

    try {
      for (const modelId of selectedModels) {
        const modelConfig = availableModels.find(m => m.id === modelId);
        if (!modelConfig) continue;

        const result = await testModel(modelConfig);
        comparisonResults.push(result);
        setResults([...comparisonResults]);
      }

      console.log('[COMPARISON] All models tested');
    } catch (err) {
      console.error('[COMPARISON ERROR]', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error during comparison: ${errorMessage}`);
    } finally {
      setTesting(false);
      setCurrentModel('');
    }
  };

  const downloadResults = () => {
    const dataStr = JSON.stringify({
      timestamp: new Date().toISOString(),
      totalTestCases: testCases.length,
      results: results
    }, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `model-comparison-${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Model Comparison
          </h1>
          <p className="text-gray-600 mb-6">
            Test multiple embedding models and compare their performance
          </p>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          {/* Model Selection */}
          <div className="mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Select Models to Test</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {availableModels.map(model => (
                <div
                  key={model.id}
                  onClick={() => !testing && toggleModelSelection(model.id)}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                    selectedModels.includes(model.id)
                      ? 'border-indigo-600 bg-indigo-50'
                      : 'border-gray-200 hover:border-gray-300'
                  } ${testing ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={() => {}}
                        className="w-4 h-4"
                        disabled={testing}
                      />
                      <h3 className="font-bold text-gray-800">{model.name}</h3>
                    </div>
                    {model.recommended && (
                      <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-bold rounded">
                        Recommended
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{model.description}</p>
                  <div className="flex gap-3 text-xs text-gray-500">
                    <span>📦 {model.size}</span>
                    <span>📐 {model.dimensions}D</span>
                    <span>🏷️ {model.category}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Control Buttons */}
          <div className="mb-6 flex gap-4">
            <button
              onClick={runComparison}
              disabled={testing || selectedModels.length === 0}
              className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {testing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Testing {currentModel}... {progress.toFixed(0)}%
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Run Comparison
                </>
              )}
            </button>

            {results.length > 0 && (
              <button
                onClick={downloadResults}
                className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors"
              >
                <Download className="w-5 h-5" />
                Download Results
              </button>
            )}
          </div>

          {/* Progress Bar */}
          {testing && (
            <div className="mb-6">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-indigo-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Results Comparison */}
          {results.length > 0 && (
            <div className="space-y-6">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-6 h-6 text-indigo-600" />
                <h2 className="text-2xl font-bold text-gray-800">Results Comparison</h2>
              </div>

              {/* Summary Table */}
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border border-gray-300 px-4 py-3 text-left text-sm font-semibold">Model</th>
                      <th className="border border-gray-300 px-4 py-3 text-center text-sm font-semibold">Top-1</th>
                      <th className="border border-gray-300 px-4 py-3 text-center text-sm font-semibold">Top-3</th>
                      <th className="border border-gray-300 px-4 py-3 text-center text-sm font-semibold">Top-5</th>
                      <th className="border border-gray-300 px-4 py-3 text-center text-sm font-semibold">Avg Time</th>
                      <th className="border border-gray-300 px-4 py-3 text-center text-sm font-semibold">Total Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, idx) => (
                      <tr key={result.modelId} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="border border-gray-300 px-4 py-3 font-medium">{result.modelName}</td>
                        <td className="border border-gray-300 px-4 py-3 text-center">
                          <span className="font-bold text-lg text-green-700">{result.top1Accuracy.toFixed(1)}%</span>
                        </td>
                        <td className="border border-gray-300 px-4 py-3 text-center">
                          <span className="font-bold text-blue-700">{result.top3Accuracy.toFixed(1)}%</span>
                        </td>
                        <td className="border border-gray-300 px-4 py-3 text-center">
                          <span className="font-bold text-purple-700">{result.top5Accuracy.toFixed(1)}%</span>
                        </td>
                        <td className="border border-gray-300 px-4 py-3 text-center text-sm">
                          {result.averageInferenceTime.toFixed(1)}ms
                        </td>
                        <td className="border border-gray-300 px-4 py-3 text-center text-sm">
                          {result.totalTestTime.toFixed(1)}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Visual Comparison */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {results.map(result => (
                  <div key={result.modelId} className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-lg p-4 border border-indigo-200">
                    <h3 className="font-bold text-gray-800 mb-3">{result.modelName}</h3>
                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-600">Top-1 Accuracy</span>
                          <span className="font-bold text-green-700">{result.top1Accuracy.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${result.top1Accuracy}%` }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-600">Top-3 Accuracy</span>
                          <span className="font-bold text-blue-700">{result.top3Accuracy.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${result.top3Accuracy}%` }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-600">Top-5 Accuracy</span>
                          <span className="font-bold text-purple-700">{result.top5Accuracy.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-purple-500 h-2 rounded-full"
                            style={{ width: `${result.top5Accuracy}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Winner Declaration */}
              {results.length > 1 && (
                <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border-2 border-yellow-400 rounded-lg p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">🏆 Best Model</h3>
                  {(() => {
                    const bestByTop1 = [...results].sort((a, b) => b.top1Accuracy - a.top1Accuracy)[0];
                    const fastestModel = [...results].sort((a, b) => a.averageInferenceTime - b.averageInferenceTime)[0];
                    
                    return (
                      <div className="space-y-3">
                        <p className="text-gray-700">
                          <span className="font-bold text-green-700">{bestByTop1.modelName}</span> has the highest Top-1 accuracy at{' '}
                          <span className="font-bold">{bestByTop1.top1Accuracy.toFixed(1)}%</span>
                        </p>
                        <p className="text-gray-700">
                          <span className="font-bold text-blue-700">{fastestModel.modelName}</span> is the fastest with{' '}
                          <span className="font-bold">{fastestModel.averageInferenceTime.toFixed(1)}ms</span> average inference time
                        </p>
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
