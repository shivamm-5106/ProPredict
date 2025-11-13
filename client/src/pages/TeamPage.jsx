import React from 'react'
const TeamPage = () => {
    const teamMembers = [
        {
            name: "Dr. Sarah Chen",
            role: "Project Lead & ML Engineer",
            bio: "PhD in Computational Biology with expertise in protein structure prediction and deep learning.",
            avatar: "👩‍🔬"
        },
        {
            name: "Dr. Michael Rodriguez",
            role: "Bioinformatics Specialist",
            bio: "Expert in Gene Ontology annotation and functional genomics with 10+ years of research experience.",
            avatar: "👨‍💻"
        },
        {
            name: "Dr. Aisha Patel",
            role: "Data Scientist",
            bio: "Specializes in ensemble learning methods and neural network optimization for biological applications.",
            avatar: "👩‍💼"
        },
        {
            name: "Dr. James Kim",
            role: "Structural Biology Advisor",
            bio: "AlphaFold expert focused on leveraging protein structure for functional prediction.",
            avatar: "👨‍🔬"
        }
    ]
    return (
        <div className="animate-fadeIn max-w-6xl mx-auto px-4 py-12">
            <h1 className="text-4xl font-bold text-gray-800 mb-4">About the Team</h1>
            <p className="text-gray-600 mb-12">
                Meet the researchers and scientists behind this protein function prediction project.
            </p>
            {/* Team Members */}
            <div className="grid md:grid-cols-2 gap-8 mb-12">
                {teamMembers.map((member, idx) => (
                    <div key={idx} className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow">
                        <div className="flex items-start">
                            <div className="text-6xl mr-6">{member.avatar}</div>
                            <div>
                                <h3 className="text-xl font-bold text-gray-800 mb-1">{member.name}</h3>
                                <p className="text-purple-600 font-semibold mb-3">{member.role}</p>
                                <p className="text-gray-700 text-sm leading-relaxed">{member.bio}</p>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            {/* CAFA 6 Acknowledgment */}
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg shadow-xl p-8 mb-8">
                <h2 className="text-3xl font-bold mb-4 flex items-center">
                    <span className="text-4xl mr-3">🏆</span>
                    CAFA 6 Challenge
                </h2>
                <p className="text-lg leading-relaxed mb-4">
                    This project is part of the <strong>Critical Assessment of Functional Annotation (CAFA) 6</strong> challenge,
                    a community-wide experiment designed to evaluate computational methods for protein function prediction.
                </p>
                <p className="leading-relaxed mb-4">
                    CAFA brings together researchers from around the world to advance the state-of-the-art in automated
                    protein function annotation, helping to bridge the gap between sequence data and biological understanding.
                </p>
                <div className="bg-white bg-opacity-20 rounded-lg p-4 mt-6">
                    <p className="text-sm">
                        <strong>Organizers:</strong> Iddo Friedberg (Iowa State University), Predrag Radivojac (Northeastern University)
                    </p>
                </div>
            </div>
            {/* Technologies */}
            <div className="bg-white rounded-lg shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <span className="text-3xl mr-3">🔧</span>
                    Technologies & Tools
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { name: 'ESM-2', desc: 'Sequence Embeddings' },
                        { name: 'AlphaFold 2', desc: 'Structure Prediction' },
                        { name: 'PyTorch', desc: 'Deep Learning' },
                        { name: 'Scikit-learn', desc: 'Ensemble Methods' },
                        { name: 'BioPython', desc: 'Sequence Analysis' },
                        { name: 'React', desc: 'Frontend Framework' },
                        { name: 'Node.js', desc: 'Backend Runtime' },
                        { name: 'Tailwind CSS', desc: 'UI Styling' }
                    ].map((tech, idx) => (
                        <div key={idx} className="text-center p-4 bg-gray-50 rounded-lg hover:bg-purple-50 transition-colors">
                            <p className="font-bold text-gray-800">{tech.name}</p>
                            <p className="text-xs text-gray-600 mt-1">{tech.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
            {/* Contact */}
            <div className="mt-8 bg-gray-50 rounded-lg p-6 text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-3">Get in Touch</h3>
                <p className="text-gray-600 mb-4">
                    Interested in collaborating or learning more about our research?
                </p>
                <a
                    href="mailto:contact@proteinpredict.research"
                    className="inline-block bg-purple-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-purple-700 transition-colors"
                >
                    📧 Contact Us
                </a>
            </div>
        </div>
    )
}
export default TeamPage
