import React from 'react'
const Footer = () => {
    return (
        <footer className="bg-gray-800 text-white py-8 mt-16">
            <div className="max-w-6xl mx-auto px-4 text-center">
                <p className="mb-2">
                    <strong>Protein Function Prediction Project</strong> | CAFA 6 Challenge
                </p>
                <p className="text-gray-400 text-sm mb-4">
                    Advancing computational biology through AI-powered protein function annotation
                </p>
                <p className="text-gray-500 text-xs">
                    © 2024 ProteinPredict Research Team. Demo application for educational purposes.
                </p>
                <div className="mt-4 text-sm text-gray-400">
                    <p>Built with React • Node.js • Tailwind CSS</p>
                </div>
            </div>
        </footer>
    )
}
export default Footer