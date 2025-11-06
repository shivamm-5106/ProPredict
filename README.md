# Protein Function Prediction - CAFA 6 Challenge
A full-stack web application for protein function prediction using ESM-2 and AlphaFold 2 ensemble models.
## 🚀 Quick Start
### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
### Installation
1. **Clone the repository**
```bash
git clone <your-repo-url>
cd protein-prediction-app
```
2. **Install Backend Dependencies**
```bash
cd server
npm install
```
3. **Install Frontend Dependencies**
```bash
cd ../client
npm install
```
### Running the Application
1. **Start the Backend Server** (Terminal 1)
```bash
cd server
npm run dev
```
Backend will run on `http://localhost:5000`
2. **Start the Frontend** (Terminal 2)
```bash
cd client
npm run dev
```
Frontend will run on `http://localhost:5173`
3. **Open your browser** and navigate to `http://localhost:5173`
## 📁 Project Structure
```
protein-prediction-app/
├── server/              # Backend (Node.js + Express)
│   ├── server.js       # Main server file
│   ├── routes/         # API routes
│   └── package.json    # Backend dependencies
├── client/              # Frontend (React + Vite)
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── pages/      # Page components
│   │   ├── App.jsx     # Main app component
│   │   └── main.jsx    # Entry point
│   └── package.json    # Frontend dependencies
└── README.md
```
## 🔌 API Endpoints
### POST /api/predict
Accepts protein sequence and returns GO term predictions.
**Request:**
```json
{
  "sequence": "MKTAYIAKQRQISFVK..."
}
```
**Response:**
```json
[
  {
    "GO_term": "GO:0009274",
    "name": "Peptidoglycan-based cell wall",
    "probability": 0.931,
    "ontology": "CC"
  }
]
```
### GET /api/project-info
Returns project metadata and objectives.
## 🧬 Features
- **Interactive Demo**: Upload FASTA files or paste sequences
- **Mock Predictions**: Simulated GO term predictions
- **Responsive Design**: Works on all devices
- **Modern UI**: Built with Tailwind CSS and React
## 🛠️ Technologies
- **Frontend**: React, Vite, Tailwind CSS
- **Backend**: Node.js, Express, CORS
- **Styling**: Tailwind CSS
## 📝 Notes
- Current version uses mock predictions
- To integrate real models (ESM-2, AlphaFold 2), update `/api/predict` endpoint in `server/routes/api.js`
## 👥 Team
Research project for CAFA 6 Challenge - Critical Assessment of Functional Annotation
## 📄 License
MIT License