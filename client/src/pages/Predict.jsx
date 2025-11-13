import React, { useState } from 'react'
import axios from 'axios'
const Predict = () => {
    const [inputType, setInputType] = useState('text')
    const [sequence, setSequence] = useState('')
    const [file, setFile] = useState(null)
    const [predictions, setPredictions] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const exampleSequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    const handlePredict = async () => {
        setError('')
        setPredictions(null)
        let sequenceToPredict = sequence
        if (inputType === 'file' && file) {
            const reader = new FileReader()
            reader.onload = async (e) => {
                const content = e.target.result
                const lines = content.split('\n')
                const seq = lines.filter(line => !line.startsWith('>')).join('').trim()
                sequenceToPredict = seq
                await executePrediction(sequenceToPredict)
            }
            reader.readAsText(file)
        } else if (inputType === 'text' && sequence) {
            await executePrediction(sequenceToPredict)
        } else {
            setError('Please provide a protein sequence or upload a FASTA file')
        }
    }
    const executePrediction = async (seq) => {
        if (!seq || seq.length < 10) {
            setError('Sequence is too short. Please provide a valid protein sequence (minimum 10 amino acids)')
            return
        }
        setLoading(true)
        try {
            const response = await axios.post('/api/predict', { sequence: seq })
            setPredictions(response.data.predictions)
        } catch (err) {
            setError('Prediction failed. Please try again.')
            console.error('Prediction error:', err)
        } finally {
            setLoading(false)
        }
    }
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0]
        if (selectedFile) {
            if (selectedFile.name.endsWith('.fasta') || selectedFile.name.endsWith('.fa') || selectedFile.name.endsWith('.txt')) {
                setFile(selectedFile)
                setError('')
            } else {
                setError('Please upload a valid FASTA file (.fasta, .fa, or .txt)')
                setFile(null)
            }
        }
    }
    const loadExample = () => {
        setInputType('text')
        setSequence(exampleSequence)
        setFile(null)
    }
    return (
        <div className="animate-fadeIn max-w-5xl mx-auto px-4 py-12">
            <h1 className="text-4xl font-bold text-gray-800 mb-4">Protein Function Prediction Demo</h1>
            <p className="text-gray-600 mb-8">
                Upload a FASTA file or paste a protein sequence to predict associated Gene Ontology (GO) terms.
            </p>
            <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
                {/* Input Type Selection */}
                <div className="mb-6">
                    <label className="block text-sm font-semibold text-gray-700 mb-3">Input Method:</label>
                    <div className="flex gap-4">
                        <button
                            onClick={() => setInputType('text')}
                            className={`px-6 py-2 rounded-lg font-medium transition-all ${inputType === 'text'
                                    ? 'bg-purple-600 text-white shadow-md'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                        >
                            ✏️ Paste Sequence
                        </button>
                        <button
                            onClick={() => setInputType('file')}
                            className={`px-6 py-2 rounded-lg font-medium transition-all ${inputType === 'file'
                                    ? 'bg-purple-600 text-white shadow-md'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                        >
                            📁 Upload FASTA
                        </button>
                    </div>
                </div>
                {/* Text Input */}
                {inputType === 'text' && (
                    <div className="mb-6">
                        <label className="block text-sm font-semibold text-gray-700 mb-2">
                            Protein Sequence (amino acids):
                        </label>
                        <textarea
                            value={sequence}
                            onChange={(e) => setSequence(e.target.value.toUpperCase())}
                            placeholder="Enter amino acid sequence (e.g., MKTAYIAKQRQISFVK...)"
                            className="w-full h-40 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent font-mono text-sm"
                        />
                        <button
                            onClick={loadExample}
                            className="mt-2 text-sm text-purple-600 hover:text-purple-800 font-medium"
                        >
                            📋 Load Example Sequence
                        </button>
                    </div>
                )}
                {/* File Input */}
                {inputType === 'file' && (
                    <div className="mb-6">
                        <label className="block text-sm font-semibold text-gray-700 mb-2">
                            Upload FASTA File:
                        </label>
                        <input
                            type="file"
                            accept=".fasta,.fa,.txt"
                            onChange={handleFileChange}
                            className="w-full px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-purple-500 transition-colors"
                        />
                        {file && (
                            <p className="mt-2 text-sm text-green-600 flex items-center">
                                <span className="mr-2">✓</span> File selected: {file.name}
                            </p>
                        )}
                    </div>
                )}
                {/* Error Message */}
                {error && (
                    <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                        <p className="font-medium">⚠️ {error}</p>
                    </div>
                )}
                {/* Submit Button */}
                <button
                    onClick={handlePredict}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                    {loading ? '🔄 Analyzing Protein...' : '🚀 Predict GO Terms'}
                </button>
            </div>
            {/* Loading State */}
            {loading && (
                <div className="bg-white rounded-lg shadow-lg p-12 text-center">
                    <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-600 mb-4"></div>
                    <p className="text-gray-600 font-medium">Running ensemble prediction models...</p>
                    <p className="text-sm text-gray-500 mt-2">ESM-2 + AlphaFold 2 Integration</p>
                </div>
            )}
            {/* Prediction Results */}
            {predictions && !loading && (
                <div className="bg-white rounded-lg shadow-lg p-8">
                    <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                        <span className="text-3xl mr-3">📊</span>
                        Prediction Results
                    </h2>

                    {/* Threshold Info Banner */}
                    <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-200">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-semibold text-purple-900">
                                    🎯 Confidence Threshold: <span className="text-purple-600">0.7 (70%)</span>
                                </p>
                                <p className="text-xs text-gray-600 mt-1">
                                    Only showing predictions with probability greater than 0.7
                                </p>
                            </div>
                            <div className="text-right">
                                <p className="text-2xl font-bold text-purple-600">{predictions.filter(p => p.probability > 0.7).length}</p>
                                <p className="text-xs text-gray-600">High-confidence<br />predictions</p>
                            </div>
                        </div>
                    </div>

                    {predictions.filter(p => p.probability > 0.7).length === 0 && (
                        <div className="p-6 bg-yellow-50 border-l-4 border-yellow-400 rounded-lg">
                            <p className="text-yellow-800 font-medium">
                                ⚠️ No predictions above threshold (0.7)
                            </p>
                            <p className="text-sm text-yellow-700 mt-2">
                                The model did not find any GO terms with confidence greater than 70% for this sequence.
                                Try a different protein sequence.
                            </p>
                        </div>
                    )}

                    {predictions.filter(p => p.probability > 0.7).length > 0 && (
                        <p className="text-gray-600 mb-6">
                            Found <strong>{predictions.filter(p => p.probability > 0.7).length}</strong> GO term predictions with probability <strong>&gt; 0.7</strong>:
                        </p>
                    )}
                    <div className="space-y-4">
                        {predictions.filter(p => p.probability > 0.7).map((pred, idx) => (
                            <div
                                key={idx}
                                className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow"
                            >
                                <div className="flex justify-between items-start mb-3">
                                    <div>
                                        <h3 className="font-mono text-lg font-bold text-purple-700">
                                            {pred.GO_term}
                                        </h3>
                                        <p className="text-gray-700 font-medium">{pred.name}</p>
                                    </div>
                                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${pred.ontology === 'MF' ? 'bg-blue-100 text-blue-700' :
                                            pred.ontology === 'BP' ? 'bg-green-100 text-green-700' :
                                                'bg-orange-100 text-orange-700'
                                        }`}>
                                        {pred.ontology === 'MF' ? 'Molecular Function' :
                                            pred.ontology === 'BP' ? 'Biological Process' :
                                                'Cellular Component'}
                                    </span>
                                </div>
                                <div>
                                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                                        <span>Confidence Score</span>
                                        <span className="font-bold text-purple-600">
                                            {(pred.probability * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                                        <div
                                            className="bg-gradient-to-r from-purple-600 to-blue-600 h-2.5 rounded-full transition-all duration-500"
                                            style={{ width: `${pred.probability * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <p className="text-sm text-blue-900">
                            <strong>💡 Note:</strong> These predictions are generated using the backend API.
                            In production, the ensemble system combines ESM-2 sequence embeddings and
                            AlphaFold 2 structural features for enhanced accuracy.
                        </p>
                    </div>
                </div>
            )}
        </div>
    )
}
export default Predict