import { useState } from 'react';
import { FlaskConical, TestTube2, BarChart3 } from 'lucide-react';
import ServiceClassifier from './ServiceClassifier';
import ModelTester from './ModelTester';
import ModelComparison from './ModelComparison';

type View = 'classifier' | 'tester' | 'comparison';

const App = () => {
  const [currentView, setCurrentView] = useState<View>('classifier');

  return (
    <div>
      {/* Navigation Bar */}
      <div className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-4 py-3">
            <button
              onClick={() => setCurrentView('classifier')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-colors ${
                currentView === 'classifier'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <FlaskConical className="w-5 h-5" />
              Clasificador
            </button>
            <button
              onClick={() => setCurrentView('tester')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-colors ${
                currentView === 'tester'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <TestTube2 className="w-5 h-5" />
              Probador de Modelos
            </button>
            <button
              onClick={() => setCurrentView('comparison')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-colors ${
                currentView === 'comparison'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <BarChart3 className="w-5 h-5" />
              Comparador de Modelos
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div>
        {currentView === 'classifier' && <ServiceClassifier />}
        {currentView === 'tester' && <ModelTester />}
        {currentView === 'comparison' && <ModelComparison />}
      </div>
    </div>
  );
};

export default App;
