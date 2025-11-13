import React, { useState } from "react";
import axios from "axios";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";

const Predict = () => {
  const [inputType, setInputType] = useState("text");
  const [sequence, setSequence] = useState("");
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const exampleSequence =
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL";

  const handlePredict = async () => {
    setError("");
    setPredictions(null);
    let sequenceToPredict = sequence;

    if (inputType === "file" && file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const content = e.target.result;
        const lines = content.split("\n");
        const seq = lines.filter((line) => !line.startsWith(">")).join("").trim();
        await executePrediction(seq);
      };
      reader.readAsText(file);
    } else if (sequence) {
      await executePrediction(sequenceToPredict);
    } else {
      setError("Please provide a protein sequence or upload a FASTA file");
    }
  };

  const executePrediction = async (seq) => {
    if (!seq || seq.length < 10) {
      setError("Sequence is too short. Minimum 10 amino acids required.");
      return;
    }
    setLoading(true);

    try {
      const response = await axios.post("/api/predict", { sequence: seq });
      setPredictions(response.data.predictions);
    } catch (err) {
      console.error(err);
      setError("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    if (
      selected.name.endsWith(".fasta") ||
      selected.name.endsWith(".fa") ||
      selected.name.endsWith(".txt")
    ) {
      setFile(selected);
      setError("");
    } else {
      setError("Invalid file. Only .fasta, .fa, .txt allowed.");
      setFile(null);
    }
  };

  return (
    <main className="flex-1 pt-24 pb-12">
      <div className="container mx-auto px-4 max-w-6xl space-y-10">

        {/* Header Like Analysis Page */}
        <header className="text-center animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-orbitron font-bold">
            <span className="text-white">Protein Function</span>{" "}
            <span className="gradient-text">Prediction</span>
          </h1>
          <p className="text-muted-foreground mt-2">
            Upload a FASTA file or paste a protein sequence to predict GO terms.
          </p>
        </header>

        {/* INPUT CARD — Styled Like PipelineInput */}
        <Card className="glass-card p-8 hover-glow space-y-6">

          {/* Input Type Switch */}
          <div className="flex gap-4">
            <Button
              variant={inputType === "text" ? "default" : "outline"}
              onClick={() => setInputType("text")}
            >
              ✏️ Paste Sequence
            </Button>

            <Button
              variant={inputType === "file" ? "default" : "outline"}
              onClick={() => setInputType("file")}
            >
              📁 Upload FASTA
            </Button>
          </div>

          {/* TEXTAREA INPUT */}
          {inputType === "text" && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Protein Sequence</label>
              <Textarea
                value={sequence}
                onChange={(e) => setSequence(e.target.value.toUpperCase())}
                className="min-h-[140px] font-mono text-sm bg-input/50"
                placeholder="Enter amino acid sequence..."
              />
              <Button
                variant="ghost"
                className="text-primary px-0"
                onClick={() => setSequence(exampleSequence)}
              >
                📋 Load Example Sequence
              </Button>
            </div>
          )}

          {/* FILE INPUT */}
          {inputType === "file" && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Upload FASTA File</label>
              <input
                type="file"
                accept=".fasta,.fa,.txt"
                onChange={handleFileChange}
                className="w-full p-4 border border-border rounded-lg bg-input/40 hover:border-primary transition"
              />
              {file && (
                <p className="text-green-500 text-sm mt-1">
                  ✓ Selected: {file.name}
                </p>
              )}
            </div>
          )}

          {/* ERROR */}
          {error && (
            <div className="p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-300">
              ⚠️ {error}
            </div>
          )}

          {/* SUBMIT BUTTON */}
          <Button
            disabled={loading}
            onClick={handlePredict}
            className="w-full bg-primary hover:bg-primary-glow text-background font-semibold"
          >
            {loading ? "Analyzing..." : "🚀 Predict GO Terms"}
          </Button>
        </Card>

        {/* LOADING */}
        {loading && (
          <Card className="glass-card p-12 text-center">
            <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4" />
            <p className="text-muted-foreground">Running prediction models...</p>
          </Card>
        )}

        {/* RESULTS */}
        {predictions && !loading && (
          <Card className="glass-card p-8 space-y-6">
            <h2 className="text-2xl font-orbitron font-bold text-primary">
              Prediction Results
            </h2>

            {predictions
              .filter((p) => p.probability > 0.7)
              .map((pred, idx) => (
                <div
                  key={idx}
                  className="p-5 border border-border rounded-lg hover-glow"
                >
                  <p className="font-mono text-lg text-primary">
                    {pred.GO_term}
                  </p>
                  <p className="text-muted-foreground">{pred.name}</p>
                  <div className="mt-3">
                    <div className="flex justify-between text-sm">
                      <span>Confidence</span>
                      <span className="text-primary font-bold">
                        {(pred.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-muted h-2 rounded-full mt-1">
                      <div
                        className="bg-primary h-2 rounded-full"
                        style={{ width: `${pred.probability * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
          </Card>
        )}
      </div>
    </main>
  );
};

export default Predict;
