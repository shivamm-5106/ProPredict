import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import Analysis from "./pages/Analysis";
import Predict from "./pages/Predict";
import DatasetPage from "./pages/DatasetPage";
import TeamPage from "./pages/TeamPage";

export default function app() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <div className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analysis" element={<Analysis />} />

          <Route path="/predict" element={<Predict />} />
          <Route path="/datasetpage" element={<DatasetPage />} />
          <Route path="/teampage" element={<TeamPage />} />
        </Routes>
      </div>

      <Footer />
    </div>
  );
}
