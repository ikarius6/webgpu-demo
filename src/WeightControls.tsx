import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface WeightControlsProps {
  weights: {
    keyword: number;
    fuzzy: number;
    embedding: number;
  };
  setWeights: (weights: { keyword: number; fuzzy: number; embedding: number }) => void;
  disabled?: boolean;
  defaultCollapsed?: boolean;
}

const WeightControls = ({ weights, setWeights, disabled = false, defaultCollapsed = false }: WeightControlsProps) => {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  return (
    <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border border-purple-200">
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between mb-2 hover:opacity-80 transition-opacity"
      >
        <h2 className="block text-sm font-semibold text-gray-700 mb-2">
          ⚖️ Configuración de Pesos
        </h2>
        {isCollapsed ? (
          <ChevronDown className="w-5 h-5 text-gray-600" />
        ) : (
          <ChevronUp className="w-5 h-5 text-gray-600" />
        )}
      </button>

      {!isCollapsed && (
        <>
          <p className="text-sm text-gray-600 mb-4">
            Ajusta los pesos para cada tipo de matching. Los valores se normalizan automáticamente para sumar 1.0.
          </p>
          
          <div className="space-y-4">
            {/* Keyword Weight */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-semibold text-gray-700">
                  Keyword Matching
                </label>
                <span className="px-3 py-1 bg-blue-100 text-blue-700 font-bold text-sm rounded-full">
                  {(weights.keyword * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={weights.keyword * 100}
                onChange={(e) => {
                  const newKeyword = parseFloat(e.target.value) / 100;
                  const remaining = 1 - newKeyword;
                  const sum = weights.fuzzy + weights.embedding;
                  // Avoid NaN when both weights are 0 - distribute equally
                  const ratio = sum > 0.001 ? weights.fuzzy / sum : 0.5;
                  setWeights({
                    keyword: newKeyword,
                    fuzzy: remaining * ratio,
                    embedding: remaining * (1 - ratio)
                  });
                }}
                disabled={disabled}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${weights.keyword * 100}%, #e5e7eb ${weights.keyword * 100}%, #e5e7eb 100%)`
                }}
              />
              <p className="text-xs text-gray-500 mt-1">Coincidencias exactas y substring matching</p>
            </div>

            {/* Fuzzy Weight */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-semibold text-gray-700">
                  Fuzzy Matching
                </label>
                <span className="px-3 py-1 bg-purple-100 text-purple-700 font-bold text-sm rounded-full">
                  {(weights.fuzzy * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={weights.fuzzy * 100}
                onChange={(e) => {
                  const newFuzzy = parseFloat(e.target.value) / 100;
                  const remaining = 1 - newFuzzy;
                  const sum = weights.keyword + weights.embedding;
                  // Avoid NaN when both weights are 0 - distribute equally
                  const ratio = sum > 0.001 ? weights.keyword / sum : 0.5;
                  setWeights({
                    keyword: remaining * ratio,
                    fuzzy: newFuzzy,
                    embedding: remaining * (1 - ratio)
                  });
                }}
                disabled={disabled}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                style={{
                  background: `linear-gradient(to right, #9333ea 0%, #9333ea ${weights.fuzzy * 100}%, #e5e7eb ${weights.fuzzy * 100}%, #e5e7eb 100%)`
                }}
              />
              <p className="text-xs text-gray-500 mt-1">Levenshtein distance para variaciones ortográficas</p>
            </div>

            {/* Embedding Weight */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-semibold text-gray-700">
                  Semantic Embedding
                </label>
                <span className="px-3 py-1 bg-orange-100 text-orange-700 font-bold text-sm rounded-full">
                  {(weights.embedding * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={weights.embedding * 100}
                onChange={(e) => {
                  const newEmbedding = parseFloat(e.target.value) / 100;
                  const remaining = 1 - newEmbedding;
                  const sum = weights.keyword + weights.fuzzy;
                  // Avoid NaN when both weights are 0 - distribute equally
                  const ratio = sum > 0.001 ? weights.keyword / sum : 0.5;
                  setWeights({
                    keyword: remaining * ratio,
                    fuzzy: remaining * (1 - ratio),
                    embedding: newEmbedding
                  });
                }}
                disabled={disabled}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                style={{
                  background: `linear-gradient(to right, #f97316 0%, #f97316 ${weights.embedding * 100}%, #e5e7eb ${weights.embedding * 100}%, #e5e7eb 100%)`
                }}
              />
              <p className="text-xs text-gray-500 mt-1">Similitud semántica con embeddings del modelo</p>
            </div>

            {/* Reset Button */}
            <div className="flex justify-between items-center pt-2 border-t border-purple-200">
              <div className="text-sm text-gray-600">
                Total: <span className="font-bold">{((weights.keyword + weights.fuzzy + weights.embedding) * 100).toFixed(1)}%</span>
              </div>
              <button
                onClick={() => setWeights({ keyword: 0.35, fuzzy: 0.30, embedding: 0.35 })}
                disabled={disabled}
                className="px-4 py-2 bg-white border border-purple-300 text-purple-700 rounded-lg text-sm font-semibold hover:bg-purple-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Resetear Valores
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default WeightControls;
