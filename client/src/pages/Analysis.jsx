// PipelinePage.tsx
import { useState } from "react";
import PipelineInput from "../components/PipelineInput";
import PipelineStep from "../components/PipelineStep";
import PipelineResults from "../components/PipelineResults";

const API_URL = "http://localhost:8000/api/analyze";

const Analysis = () => {
  const [sequence, setSequence] = useState("MKTFFVLVLLLAAAGVAGTQATQGNVKAAW");
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const [model1Status, setModel1Status] = useState("Waiting");
  const [model1Data, setModel1Data] = useState(null);

  const [model2Status, setModel2Status] = useState("Waiting");
  const [model2Data, setModel2Data] = useState(null);

  const [model3Status, setModel3Status] = useState("Waiting");
  const [model3Data, setModel3Data] = useState(null);

  const validateSequence = (seq) => {
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
    return validAminoAcids.test(seq);
  };

  const handleAnalyze = async () => {
    if (!sequence.trim()) {
      alert("Please provide a protein sequence");
      return;
    }
    if (!validateSequence(sequence)) {
      alert("Invalid sequence. Please use only valid amino acid letters.");
      return;
    }

    setIsLoading(true);
    setShowResults(false);

    setModel1Status("Waiting");
    setModel2Status("Waiting");
    setModel3Status("Waiting");

    setModel1Data(null);
    setModel2Data(null);
    setModel3Data(null);

    try {
      setModel1Status("Processing...");

      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sequence: sequence.trim().toUpperCase(),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "API call failed");
      }

      const data = await response.json();

      setModel1Data(data.model1_output);
      setModel1Status(data.model1_output.cached ? "Complete (Cached)" : "Complete");

      await new Promise((resolve) => setTimeout(resolve, 500));
      setModel2Status("Processing...");

      await new Promise((resolve) => setTimeout(resolve, 800));
      setModel2Data(data.model2_output);
      setModel2Status("Complete");

      await new Promise((resolve) => setTimeout(resolve, 500));
      setModel3Status("Processing...");

      await new Promise((resolve) => setTimeout(resolve, 800));
      setModel3Data(data.model3_output);
      setModel3Status(
        data.model3_output.model_loaded ? "Complete" : "Complete (Placeholder)"
      );

      setShowResults(true);
      setIsLoading(false);
    } catch (err) {
      console.error("Pipeline Error:", err);
      setIsLoading(false);
      alert(`Error: ${err?.message || err}`);
    }
  };

  const handleReset = () => {
    setShowResults(false);

    setModel1Status("Waiting");
    setModel2Status("Waiting");
    setModel3Status("Waiting");

    setModel1Data(null);
    setModel2Data(null);
    setModel3Data(null);

    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {/* <Navbar /> */}

      <main className="flex-1 pt-24 pb-12">
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto space-y-8">
            
            {/* Header */}
            <header className="text-center mb-2 animate-fade-in">
              <h1 className="text-4xl md:text-5xl font-orbitron font-bold">
                <span className="text-white">Protein Analysis</span>{" "}
                <span className="gradient-text">Pipeline</span>
              </h1>
              <p className="text-muted-foreground mt-2">
                ESM2 Embedding → Neural Network → Multilabel Classifier
              </p>
            </header>

            {/* Input */}
            <PipelineInput
              sequence={sequence}
              setSequence={setSequence}
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
            />

            {/* Pipeline Steps */}
            {(isLoading || showResults) && (
              <div className="space-y-6">
                {/* Model 1 */}
                <PipelineStep
                  title="ESM2-150M Embedding Model"
                  badge="M1"
                  status={model1Status}
                  description="Generating 640-dimensional protein embeddings..."
                >
                  {model1Data && (
                    <div className="bg-card/30 p-4 rounded-lg border border-green-500/30">
                      <p className="text-xs text-green-400 mb-2">✓ Embedding Generated</p>
                      <p className="text-sm mb-2">
                        <strong>Shape:</strong>{" "}
                        <span className="text-cyan-400">
                          ({model1Data.shape.join(", ")})
                        </span>
                      </p>
                      <p className="text-sm mb-2">
                        <strong>Type:</strong>{" "}
                        <span className="text-cyan-400">torch.Tensor (CPU)</span>
                      </p>
                      <div className="code-block text-xs font-mono">
                        [{model1Data.first_10_values.map((v) => v.toFixed(4)).join(", ")}, ...]
                      </div>
                    </div>
                  )}
                </PipelineStep>

                {/* Model 2 */}
                <PipelineStep
                  title="Neural Network Prediction"
                  badge="M2"
                  status={model2Status}
                  description="Processing embeddings through neural network..."
                  dimmed={model1Status === "Waiting"}
                >
                  {model2Data && (
                    <div className="bg-card/30 p-4 rounded-lg border border-green-500/30">
                      <p className="text-xs text-green-400 mb-3">✓ Predictions Complete</p>
                      <p className="text-sm mb-2">
                        <strong>Output Shape:</strong>{" "}
                        <span className="text-cyan-400">
                          ({model2Data.shape.join(", ")})
                        </span>
                      </p>
                      <p className="text-sm mb-2">
                        <strong>Positive Predictions:</strong>{" "}
                        <span className="text-cyan-400">
                          {model2Data.output_summary.positive_count} / {model2Data.shape[0]}
                        </span>
                      </p>
                    </div>
                  )}
                </PipelineStep>

                {/* Model 3 */}
                <PipelineStep
                  title="Multilabel Function Classification"
                  badge="M3"
                  status={model3Status}
                  description="Classifying protein functions from predictions..."
                  dimmed={model2Status === "Waiting"}
                >
                  {model3Data && (
                    <div className="bg-card/30 p-4 rounded-lg border border-green-500/30">
                      {!model3Data.model_loaded ? (
                        <p className="text-yellow-400 text-sm">
                          ⚠️ Model 3 not loaded. Please add model3.pkl file.
                        </p>
                      ) : (
                        <>
                          <p className="text-xs text-green-400 mb-3">✓ Classification Complete</p>

                          {model3Data.labels.length > 0 ? (
                            <p className="text-green-400 text-sm mb-3">
                              ✓ Predicted Functions: {model3Data.labels.join(", ")}
                            </p>
                          ) : (
                            <p className="text-gray-400 text-sm mb-3">
                              No functions above threshold ({model3Data.threshold || 0.5})
                            </p>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </PipelineStep>
              </div>
            )}

            {/* Final Results */}
            {showResults && <PipelineResults onReset={handleReset} />}
          </div>
        </div>
      </main>

    </div>
  );
};

export default Analysis;
