import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navigation from './components/Navigation'
import Footer from './components/Footer'
import HomePage from './components/pages/HomePage'
import DatasetPage from './components/pages/DatasetPage'
import DemoPage from './components/pages/DemoPage'
import TeamPage from './components/pages/TeamPage'
function App() {
    return (
        <div className="min-h-screen flex flex-col">
            <Navigation />
            <main className="flex-grow">
                <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/dataset" element={<DatasetPage />} />
                    <Route path="/demo" element={<DemoPage />} />
                    <Route path="/team" element={<TeamPage />} />
                </Routes>
            </main>
            <Footer />
        </div>
    )
}
export default App