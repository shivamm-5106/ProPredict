import React from "react";

const DatasetPage = () => {
  return (
    <main className="flex-1 pt-24 pb-12">
      <div className="animate-fadeIn max-w-6xl mx-auto px-4">

        {/* PAGE HEADER */}
        <h1 className="text-4xl md:text-5xl font-orbitron font-bold mb-8">
                <span className="text-white">Dataset</span>{" "}
                <span className="gradient-text">&</span>{" "}
                <span className="text-white">Methodology</span>
            </h1>

        {/* DATASET OVERVIEW */}
        <div className="glass-card p-8 rounded-lg hover-glow mb-10 border border-border/40 bg-card/50">
          <h2 className="text-2xl font-orbitron font-bold mb-6 flex items-center gradient-text">
            <span className="text-3xl mr-3">📁</span>
            Dataset Overview
          </h2>

          <p className="text-muted-foreground mb-6">
            The CAFA 6 challenge provides a comprehensive dataset designed for
            training and evaluating protein function prediction models.
          </p>

          <div className="overflow-x-auto rounded-lg border border-border/40">
            <table className="min-w-full divide-y divide-border">
              <thead className="bg-muted/30">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    File Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-card/50">
                <tr>
                  <td className="px-6 py-4 font-mono text-sm text-primary">
                    train_sequences.fasta
                  </td>
                  <td className="px-6 py-4 text-sm text-foreground">
                    Contains amino acid sequences of training proteins in FASTA format
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 font-mono text-sm text-primary">
                    train_terms.tsv
                  </td>
                  <td className="px-6 py-4 text-sm text-foreground">
                    Lists the known GO term annotations (labels) for each protein
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 font-mono text-sm text-primary">
                    train_taxonomy.tsv
                  </td>
                  <td className="px-6 py-4 text-sm text-foreground">
                    Specifies the taxonomic classification of each protein
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 font-mono text-sm text-primary">
                    go-basic.obo
                  </td>
                  <td className="px-6 py-4 text-sm text-foreground">
                    Gene Ontology hierarchy defining parent–child term relationships
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* OUTPUT FORMAT */}
        <div className="glass-card p-8 rounded-lg hover-glow mb-10 border border-border/40 bg-card/50">
          <h2 className="text-2xl font-orbitron font-bold mb-6 flex items-center gradient-text">
            <span className="text-3xl mr-3">📤</span>
            Output Format
          </h2>

          <p className="text-muted-foreground mb-4">
            Each prediction includes: <strong>Protein ID</strong>,{" "}
            <strong>GO Term</strong>, <strong>Predicted Probability</strong>.
          </p>

          <div className="bg-black/40 border border-border/40 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
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

          <div className="mt-4 p-4 bg-blue-500/10 rounded-lg border border-blue-400/30">
            <p className="text-sm text-blue-300">
              <strong>Evaluation:</strong> Performed for three GO sub-ontologies:{" "}
              <strong>MF</strong>, <strong>BP</strong>, <strong>CC</strong>.
              Final performance = average of all three.
            </p>
          </div>
        </div>

        {/* MODEL ARCHITECTURE */}
        <div className="glass-card p-8 rounded-lg hover-glow mb-10 border border-border/40 bg-card/50">
          <h2 className="text-2xl font-orbitron font-bold mb-6 flex items-center gradient-text">
            <span className="text-3xl mr-3">🏗️</span>
            Proposed Solution: Ensemble Architecture
          </h2>

          {/* MODEL 1 */}
          <div className="mb-8 border-l-4 border-blue-500 pl-6">
            <h3 className="text-xl font-bold text-foreground mb-3">
              Model 1: ESM-2 (Sequence-based)
            </h3>
            <ul className="space-y-2 text-muted-foreground">
              <li>▸ Produces high-dimensional embeddings of amino-acid sequences</li>
              <li>▸ Captures biochemical + evolutionary protein patterns</li>
              <li>▸ Pretrained head predicts ~1500 GO terms</li>
              <li>▸ Outputs probability vector using sigmoid activation</li>
            </ul>
          </div>

          {/* MODEL 2 */}
          <div className="mb-8 border-l-4 border-green-500 pl-6">
            <h3 className="text-xl font-bold text-foreground mb-3">
              Model 2: AlphaFold 2 (Structure-based)
            </h3>
            <ul className="space-y-2 text-muted-foreground">
              <li>▸ Predicts 3D structure from sequence</li>
              <li>
                ▸ Extracts structural features like:
                <ul className="ml-6 mt-1 space-y-1">
                  <li>• Distance matrices</li>
                  <li>• Secondary structure maps</li>
                  <li>• Surface accessibility</li>
                </ul>
              </li>
              <li>▸ MLP classifier predicts GO-term probabilities</li>
            </ul>
          </div>

          {/* MODEL 3 */}
          <div className="border-l-4 border-purple-500 pl-6">
            <h3 className="text-xl font-bold text-foreground mb-3">
              Model 3: Ensemble Integration
            </h3>
            <ul className="space-y-2 text-muted-foreground">
              <li>▸ Combines ESM-2 + AlphaFold outputs</li>
              <li>▸ Applies confidence threshold</li>
              <li>▸ Produces final GO-term predictions</li>
            </ul>
          </div>
        </div>

        {/* ADVANTAGES */}
        <div className="glass-card p-8 rounded-lg hover-glow bg-gradient-to-r from-purple-950/40 to-blue-950/30 border border-border/40">
          <h2 className="text-2xl font-orbitron font-bold mb-6 flex items-center gradient-text">
            <span className="text-3xl mr-3">✨</span>
            Advantages of the Proposed Approach
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="glass-card bg-card p-5 rounded-lg border border-border/40">
              <h4 className="font-bold text-blue-400 mb-2">ESM-2 Component</h4>
              <p className="text-sm text-muted-foreground mb-2">
                <strong>Captures:</strong> Sequence-level biochemical features
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Advantage:</strong> Fast & robust pretrained baseline
              </p>
            </div>

            <div className="glass-card bg-card p-5 rounded-lg border border-border/40">
              <h4 className="font-bold text-green-400 mb-2">AlphaFold Component</h4>
              <p className="text-sm text-muted-foreground mb-2">
                <strong>Captures:</strong> 3D structural relationships
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Advantage:</strong> Adds structural insight
              </p>
            </div>

            <div className="glass-card bg-card p-5 rounded-lg border border-border/40 md:col-span-2">
              <h4 className="font-bold text-purple-400 mb-2">Ensemble Integration</h4>
              <p className="text-sm text-muted-foreground mb-2">
                <strong>Captures:</strong> Combined perspective of sequence + structure
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Advantage:</strong> Better generalization & accuracy
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};

export default DatasetPage;
