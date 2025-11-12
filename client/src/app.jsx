import { useState } from 'react';
const API_URL = 'http://localhost:8000/api/analyze';
function App() {
    const [sequence, setSequence] = useState('MKTFFVLVLLLAAAGVAGTQATQGNVKAAW');
    const [isLoading, setIsLoading] = useState(false);
    const [showResults, setShowResults] = useState(false);

    const [model1Status, setModel1Status] = useState('Waiting');
    const [model1Data, setModel1Data] = useState(null);

    const [model2Status, setModel2Status] = useState('Waiting');
    const [model2Data, setModel2Data] = useState(null);

    const [model3Status, setModel3Status] = useState('Waiting');
    const [model3Data, setModel3Data] = useState(null);
    const validateSequence = (seq) => {
        const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
        return validAminoAcids.test(seq);
    };
    const handleAnalyze = async () => {
        if (!sequence.trim()) {
            alert('Please provide a protein sequence');
            return;
        }
        if (!validateSequence(sequence)) {
            alert('Invalid sequence. Please use only valid amino acid letters.');
            return;
        }
        setIsLoading(true);
        setShowResults(false);
        setModel1Status('Waiting');
        setModel2Status('Waiting');
        setModel3Status('Waiting');
        setModel1Data(null);
        setModel2Data(null);
        setModel3Data(null);
        try {
            setModel1Status('Processing...');

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sequence: sequence.trim().toUpperCase()
                })
            })
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'API call failed');
            }
            const data = await response.json();
            setModel1Data(data.model1_output);
            setModel1Status(data.model1_output.cached ? 'Complete (Cached)' : 'Complete');
            await new Promise(resolve => setTimeout(resolve, 500));
            setModel2Status('Processing...');
            await new Promise(resolve => setTimeout(resolve, 800));
            setModel2Data(data.model2_output);
            setModel2Status('Complete');
            await new Promise(resolve => setTimeout(resolve, 500));
            setModel3Status('Processing...');
            await new Promise(resolve => setTimeout(resolve, 800));
            setModel3Data(data.model3_output);
            setModel3Status(data.model3_output.model_loaded ? 'Complete' : 'Complete (Placeholder)');
            setShowResults(true);
            setIsLoading(false);
        } catch (err) {
            console.error('Pipeline Error:', err);
            setIsLoading(false);
            alert(`Error: ${err.message}`);
        }
    };
    const handleReset = () => {
        setShowResults(false);
        setModel1Status('Waiting');
        setModel2Status('Waiting');
        setModel3Status('Waiting');
        setModel1Data(null);
        setModel2Data(null);
        setModel3Data(null);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    const getStatusColor = (status) => {
        const colors = {
            'Waiting': 'bg-gray-500',
            'Processing...': 'bg-yellow-500',
            'Complete': 'bg-green-500',
            'Complete (Cached)': 'bg-green-500',
            'Complete (Placeholder)': 'bg-green-500',
            'Error': 'bg-red-500'
        };
        return colors[status] || 'bg-gray-500';
    };
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
            <div className="container mx-auto px-4 py-8 max-w-6xl">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                        Protein Analysis Pipeline
                    </h1>
                    <p className="text-gray-300 text-lg">ESM2 Embedding → Neural Network → Multilabel Classifier</p>
                </div>
                {/* Input Section */}
                <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 mb-8 border border-white/20">
                    <h2 className="text-2xl font-bold mb-6 flex items-center">
                        <span className="bg-blue-500 rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm">1</span>
                        Input Protein Sequence
                    </h2>

                    <div className="space-y-4">
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2">Protein Sequence</label>
                            <textarea
                                value={sequence}
                                onChange={(e) => setSequence(e.target.value)}
                                rows="4"
                                placeholder="Enter amino acid sequence (e.g., MKTFFVLVLLLAAAGVAGTQATQGNVKAAW)"
                                className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono transition text-white"
                            />
                            <p className="text-xs text-gray-400 mt-1">Valid amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y</p>
                        </div>
                        <button
                            onClick={handleAnalyze}
                            disabled={isLoading}
                            className={`w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-bold py-4 px-6 rounded-lg transition-all transform hover:scale-105 shadow-lg ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            {isLoading ? '🔄 Processing...' : '🧬 Run Analysis Pipeline'}
                        </button>
                    </div>
                </div>
                {/* Pipeline Progress */}
                {(isLoading || showResults) && (
                    <div className="space-y-6">
                        {/* Model 1 */}
                        <div className={`bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20 transition-all ${model1Status === 'Processing...' ? 'processing' : ''}`}>
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-bold flex items-center">
                                    <span className="bg-purple-500 rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm">M1</span>
                                    ESM2-150M Embedding Model
                                </h3>
                                <span className={`px-3 py-1 ${getStatusColor(model1Status)} rounded-full text-xs`}>
                                    {model1Status}
                                </span>
                            </div>
                            <p className="text-gray-400 text-sm mb-4">Generating 640-dimensional protein embeddings...</p>
                            {model1Data && (
                                <div className="slide-down">
                                    <div className="bg-black/30 rounded-lg p-4 border border-green-500/30">
                                        <p className="text-xs text-green-400 mb-2">✓ Embedding Generated</p>
                                        <p className="text-sm mb-2"><strong>Shape:</strong> <span className="text-cyan-400">({model1Data.shape.join(', ')})</span></p>
                                        <p className="text-sm mb-2"><strong>Type:</strong> <span className="text-cyan-400">torch.Tensor (CPU)</span></p>
                                        <p className="text-xs text-gray-400 mb-2">First 10 values:</p>
                                        <div className="code-block text-xs">
                                            [{model1Data.first_10_values.map(v => v.toFixed(4)).join(', ')}, ...]
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                        {/* Model 2 */}
                        <div className={`bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20 transition-all ${model1Status === 'Waiting' ? 'opacity-50' : ''} ${model2Status === 'Processing...' ? 'processing' : ''}`}>
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-bold flex items-center">
                                    <span className="bg-green-500 rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm">M2</span>
                                    Neural Network Prediction
                                </h3>
                                <span className={`px-3 py-1 ${getStatusColor(model2Status)} rounded-full text-xs`}>
                                    {model2Status}
                                </span>
                            </div>
                            <p className="text-gray-400 text-sm mb-4">Processing embeddings through neural network...</p>
                            {model2Data && (
                                <div className="slide-down">
                                    <div className="bg-black/30 rounded-lg p-4 border border-green-500/30">
                                        <p className="text-xs text-green-400 mb-3">✓ Predictions Complete</p>
                                        <p className="text-sm mb-2"><strong>Output Shape:</strong> <span className="text-cyan-400">({model2Data.shape.join(', ')})</span></p>
                                        <p className="text-sm mb-2"><strong>Positive Predictions:</strong> <span className="text-cyan-400">{model2Data.output_summary.positive_count} / {model2Data.shape[0]}</span></p>
                                        <div className="mt-3 space-y-1 text-xs">
                                            <p><strong>Min:</strong> {model2Data.output_summary.min.toFixed(4)}</p>
                                            <p><strong>Max:</strong> {model2Data.output_summary.max.toFixed(4)}</p>
                                            <p><strong>Mean:</strong> {model2Data.output_summary.mean.toFixed(4)}</p>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                        {/* Model 3 */}
                        <div className={`bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20 transition-all ${model2Status === 'Waiting' ? 'opacity-50' : ''} ${model3Status === 'Processing...' ? 'processing' : ''}`}>
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-bold flex items-center">
                                    <span className="bg-yellow-500 rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm">M3</span>
                                    Multilabel Function Classification
                                </h3>
                                <span className={`px-3 py-1 ${getStatusColor(model3Status)} rounded-full text-xs`}>
                                    {model3Status}
                                </span>
                            </div>
                            <p className="text-gray-400 text-sm mb-4">Classifying protein functions from predictions...</p>
                            {model3Data && (
                                <div className="slide-down">
                                    <div className="bg-black/30 rounded-lg p-4 border border-green-500/30">
                                        {!model3Data.model_loaded ? (
                                            <p className="text-yellow-400 text-sm">⚠️ Model 3 not loaded. Please add model3.pkl file.</p>
                                        ) : (
                                            <>
                                                <p className="text-xs text-green-400 mb-3">✓ Classification Complete</p>

                                                {model3Data.labels.length > 0 ? (
                                                    <p className="text-green-400 text-sm mb-3">
                                                        ✓ Predicted Functions: {model3Data.labels.join(', ')}
                                                    </p>
                                                ) : (
                                                    <p className="text-gray-400 text-sm mb-3">
                                                        No functions above threshold ({model3Data.threshold || 0.5})
                                                    </p>
                                                )}
                                                <div className="space-y-3">
                                                    {Object.entries(model3Data.probabilities).map(([label, prob]) => {
                                                        const percentage = (prob * 100).toFixed(1);
                                                        const isAboveThreshold = model3Data.labels.includes(label);
                                                        const barColor = isAboveThreshold
                                                            ? 'from-green-500 to-emerald-500'
                                                            : 'from-blue-500 to-cyan-500';

                                                        return (
                                                            <div key={label}>
                                                                <div className="flex justify-between mb-1">
                                                                    <span className={`text-sm ${isAboveThreshold ? 'font-bold text-green-300' : ''}`}>
                                                                        {label}
                                                                    </span>
                                                                    <span className="text-sm text-cyan-400">{percentage}%</span>
                                                                </div>
                                                                <div className="w-full bg-gray-700 rounded-full h-2">
                                                                    <div
                                                                        className={`bg-gradient-to-r ${barColor} h-2 rounded-full transition-all`}
                                                                        style={{ width: `${percentage}%` }}
                                                                    />
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
                {/* Final Results */}
                {showResults && (
                    <div className="mt-8 bg-gradient-to-r from-green-500/20 to-blue-500/20 backdrop-blur-md rounded-2xl p-8 border border-green-500/30 slide-down">
                        <h2 className="text-2xl font-bold mb-4 flex items-center">
                            <span className="text-3xl mr-3">✓</span>
                            Analysis Complete
                        </h2>
                        <p className="text-gray-300">All models have successfully processed the sequence. Results are displayed above.</p>
                        <button
                            onClick={handleReset}
                            className="mt-4 bg-white/10 hover:bg-white/20 px-6 py-2 rounded-lg transition"
                        >
                            Run New Analysis
                        </button>
                    </div>
                )}
                {/* API Documentation */}
                <div className="mt-12 bg-white/5 backdrop-blur-md rounded-2xl p-8 border border-white/10">
                    <h3 className="text-xl font-bold mb-4">📡 API Integration</h3>
                    <p className="text-gray-400 mb-4">FastAPI endpoint structure:</p>
                    <div className="code-block">
                        {`POST /api/analyze
{
  "seq_id": "seq_001",
  "sequence": "MKTFFVLVLLLAAAGVAGTQATQGNVKAAW"
}
// Response:
{
  "model1_output": { "embedding": [...], "shape": [640] },
  "model2_output": { "predictions": [...], "shape": [26121] },
  "model3_output": { "labels": [...], "probabilities": {...} }
}`}
                    </div>
                </div>
            </div>
        
    );
}
export default App;
