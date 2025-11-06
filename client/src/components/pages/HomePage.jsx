import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
const HomePage = () => {
    const navigate = useNavigate()
    const [projectInfo, setProjectInfo] = useState(null)
    const [loading, setLoading] = useState(true)
    useEffect(() => {
        // Fetch project info from backend
        axios.get('/api/project-info')
            .then(response => {
                setProjectInfo(response.data)
                setLoading(false)
            })
            .catch(error => {
                console.error('Error fetching project info:', error)
                setLoading(false)
            })
    }, [])
    return (
        <div className="animate-fadeIn">
            {/* Hero Section */}
            <div className="protein-helix text-white py-20 px-4">
                <div className="max-w-4xl mx-auto text-center">
                    <h1 className="text-5xl md:text-6xl font-bold mb-6">
                        Protein Function Prediction
                    </h1>
                    <p className="text-xl md:text-2xl mb-8 opacity-90">
                        Leveraging AI to decode the functions of proteins from their sequences
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <button
                            onClick={() => navigate('/demo')}
                            className="bg-white text-purple-700 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-all transform hover:scale-105"
                        >
                            🚀 Explore Model
                        </button>
                        <button
                            onClick={() => navigate('/dataset')}
                            className="bg-purple-800 text-white px-8 py-3 rounded-lg font-semibold hover:bg-purple-900 transition-all transform hover:scale-105"
                        >
                            📖 Learn More
                        </button>
                    </div>
                </div>
            </div>
            {/* Problem Statement */}
            <div className="max-w-6xl mx-auto px-4 py-16">
                <div className="bg-white rounded-lg shadow-lg p-8 mb-12">
                    <h2 className="text-3xl font-bold text-gray-800 mb-4 flex items-center">
                        <span className="text-4xl mr-3">🎯</span>
                        The Challenge
                    </h2>
                    {loading ? (
                        <div className="text-gray-600">Loading...</div>
                    ) : (
                        <>
                            <p className="text-lg text-gray-700 leading-relaxed">
                                {projectInfo?.problemStatement}
                            </p>
                            <div className="mt-6 p-4 bg-purple-50 rounded-lg border-l-4 border-purple-500">
                                <p className="text-purple-900 font-semibold">CAFA 6 Challenge</p>
                                <p className="text-gray-700 mt-2">
                                    Critical Assessment of Functional Annotation - advancing computational biology through predictive modeling
                                </p>
                            </div>
                        </>
                    )}
                </div>
                {/* Project Objectives */}
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
                        Project Objectives
                    </h2>
                    {loading ? (
                        <div className="text-center text-gray-600">Loading objectives...</div>
                    ) : (
                        <div className="grid md:grid-cols-2 gap-6">
                            {projectInfo?.objectives.map((objective, idx) => (
                                <div key={idx} className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                                    <div className="flex items-start">
                                        <div className="flex-shrink-0">
                                            <div className="flex items-center justify-center h-12 w-12 rounded-md bg-purple-500 text-white text-xl font-bold">
                                                {idx + 1}
                                            </div>
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-gray-700 leading-relaxed">{objective}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                {/* Key Features */}
                <div className="grid md:grid-cols-3 gap-8 mb-12">
                    <div className="text-center bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                        <div className="text-5xl mb-4">🧬</div>
                        <h3 className="text-xl font-bold text-gray-800 mb-2">ESM-2 Model</h3>
                        <p className="text-gray-600">Sequence-based transformer model capturing evolutionary patterns</p>
                    </div>
                    <div className="text-center bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                        <div className="text-5xl mb-4">🔬</div>
                        <h3 className="text-xl font-bold text-gray-800 mb-2">AlphaFold 2</h3>
                        <p className="text-gray-600">Structure-based predictions using 3D protein folding</p>
                    </div>
                    <div className="text-center bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                        <div className="text-5xl mb-4">🤝</div>
                        <h3 className="text-xl font-bold text-gray-800 mb-2">Ensemble Learning</h3>
                        <p className="text-gray-600">Combined predictions for optimal accuracy</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
export default HomePage