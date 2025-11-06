import React from 'react'
const DatasetPage = () => {
    return (
        <div className="animate-fadeIn max-w-6xl mx-auto px-4 py-12">
            <h1 className="text-4xl font-bold text-gray-800 mb-8">Dataset & Methodology</h1>
            {/* Dataset Details */}
            <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <span className="text-3xl mr-3">📁</span>
                    Dataset Overview
                </h2>
                <p className="text-gray-700 mb-6">
                    The CAFA 6 challenge provides a comprehensive dataset designed for training and evaluating protein function prediction models.
                </p>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">File Name</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            <tr>
                                <td className="px-6 py-4 whitespace-nowrap font-mono text-sm text-purple-600">train_sequences.fasta</td>
                                <td className="px-6 py-4 text-sm text-gray-700">Contains amino acid sequences of training proteins in FASTA format</td>
                            </tr>
                            <tr>
                                <td className="px-6 py-4 whitespace-nowrap font-mono text-sm text-purple-600">train_terms.tsv</td>
                                <td className="px-6 py-4 text-sm text-gray-700">Lists the known GO term annotations (labels) for each protein</td>
                            </tr>
                            <tr>
                                <td className="px-6 py-4 whitespace-nowrap font-mono text-sm text-purple-600">train_taxonomy.tsv</td>
                                <td className="px-6 py-4 text-sm text-gray-700">Specifies the taxonomic classification of each protein (organism-level information)</td>
                            </tr>
                            <tr>
                                <td className="px-6 py-4 whitespace-nowrap font-mono text-sm text-purple-600">go-basic.obo</td>
                                <td className="px-6 py-4 text-sm text-gray-700">The Gene Ontology hierarchy file defining relationships between GO terms (parent–child structure)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            {/* Output Format */}
            <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <span className="text-3xl mr-3">📤</span>
                    Output Format
                </h2>
                <p className="text-gray-700 mb-4">
                    Each prediction includes three columns: <strong>Protein ID</strong>, <strong>GO Term</strong>, and <strong>Predicted Probability</strong> (0-1 range).
                </p>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                    <div className="grid grid-cols-3 gap-4 mb-2 font-bold text-green-400">
                        <div>Protein ID</div>
                        <div>GO Term</div>
                        <div>Probability</div>
                    </div>
                    <div className="grid grid-cols-3 gap-4 mb-1">
                        <div>P9WHI7</div>
                        <div>GO:0009274</div>
                        <div>0.931</div>
                    </div>
                    <div className="grid grid-cols-3 gap-4 mb-1">
                        <div>P9WHI7</div>
                        <div>GO:0071944</div>
                        <div>0.540</div>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                        <div>P04637</div>
                        <div>GO:0031625</div>
                        <div>0.989</div>
                    </div>
                </div>
                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-900">
                        <strong>Evaluation:</strong> Conducted separately for three sub-ontologies:
                        <span className="font-semibold"> Molecular Function (MF)</span>,
                        <span className="font-semibold"> Biological Process (BP)</span>, and
                        <span className="font-semibold"> Cellular Component (CC)</span>.
                        Final score is the average of all three.
                    </p>
                </div>
            </div>
            {/* Model Architecture */}
            <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <span className="text-3xl mr-3">🏗️</span>
                    Proposed Solution: Ensemble Architecture
                </h2>

                {/* Model 1: ESM-2 */}
                <div className="mb-8 border-l-4 border-blue-500 pl-6">
                    <h3 className="text-xl font-bold text-gray-800 mb-3">Model 1: Sequence-Based (ESM-2 Pipeline)</h3>
                    <ul className="space-y-2 text-gray-700">
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">▸</span>
                            <span>Amino acid sequence encoded using <strong>ESM-2</strong>, a large pretrained transformer model trained on millions of protein sequences</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">▸</span>
                            <span>Converts sequence into high-dimensional embeddings capturing biochemical and evolutionary patterns</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">▸</span>
                            <span>Pretrained GO-prediction head maps embeddings to probabilities for ~1500 GO terms</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">▸</span>
                            <span>Outputs sigmoid-activated probability vector for each GO term</span>
                        </li>
                    </ul>
                </div>
                {/* Model 2: AlphaFold */}
                <div className="mb-8 border-l-4 border-green-500 pl-6">
                    <h3 className="text-xl font-bold text-gray-800 mb-3">Model 2: Structure-Based (AlphaFold 2 Pipeline)</h3>
                    <ul className="space-y-2 text-gray-700">
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">▸</span>
                            <span>Sequence processed through <strong>AlphaFold 2</strong> to predict 3D protein structure</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">▸</span>
                            <span>Extracts geometric and spatial features including:
                                <ul className="ml-6 mt-1 space-y-1">
                                    <li>• Residue–residue distance matrices (Cα–Cα distances)</li>
                                    <li>• Secondary structure patterns (α-helices, β-sheets)</li>
                                    <li>• Surface accessibility and contact maps</li>
                                </ul>
                            </span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">▸</span>
                            <span>Features flattened/pooled and passed through Multi-Layer Perceptron (MLP) classifier</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">▸</span>
                            <span>MLP outputs 1500-dimensional probability vector for predicted GO terms</span>
                        </li>
                    </ul>
                </div>
                {/* Model 3: Ensemble */}
                <div className="border-l-4 border-purple-500 pl-6">
                    <h3 className="text-xl font-bold text-gray-800 mb-3">Model 3: Ensemble Integration</h3>
                    <ul className="space-y-2 text-gray-700">
                        <li className="flex items-start">
                            <span className="text-purple-500 mr-2">▸</span>
                            <span>Combines outputs from both ESM-2 and AlphaFold 2 models using ensemble learning</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-purple-500 mr-2">▸</span>
                            <span>Final GO term predictions obtained by applying confidence threshold (e.g., 0.5)</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-purple-500 mr-2">▸</span>
                            <span>Retains only terms with sufficiently high confidence scores</span>
                        </li>
                    </ul>
                </div>
            </div>
            {/* Advantages */}
            <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <span className="text-3xl mr-3">✨</span>
                    Advantages of the Proposed Approach
                </h2>
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-white rounded-lg p-5 shadow">
                        <h4 className="font-bold text-blue-700 mb-2">ESM-2 Component</h4>
                        <p className="text-sm text-gray-700 mb-2"><strong>Captures:</strong> Sequence-level biochemical and evolutionary features</p>
                        <p className="text-sm text-gray-600"><strong>Advantage:</strong> Fast, pretrained, robust baseline</p>
                    </div>
                    <div className="bg-white rounded-lg p-5 shadow">
                        <h4 className="font-bold text-green-700 mb-2">AlphaFold 2 Component</h4>
                        <p className="text-sm text-gray-700 mb-2"><strong>Captures:</strong> 3D structural and spatial relationships</p>
                        <p className="text-sm text-gray-600"><strong>Advantage:</strong> Adds physical insight into protein function</p>
                    </div>
                    <div className="bg-white rounded-lg p-5 shadow md:col-span-2">
                        <h4 className="font-bold text-purple-700 mb-2">Ensemble Integration</h4>
                        <p className="text-sm text-gray-700 mb-2"><strong>Captures:</strong> Combined perspective of sequence and structure</p>
                        <p className="text-sm text-gray-600"><strong>Advantage:</strong> Improves generalization and predictive power</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
export default DatasetPage