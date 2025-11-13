// components/PipelineInput.tsx
import React from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Loader2 } from "lucide-react";

interface PipelineInputProps {
  sequence: string;
  setSequence: (s: string) => void;
  onAnalyze: () => void;
  isLoading: boolean;
}

const PipelineInput: React.FC<PipelineInputProps> = ({ sequence, setSequence, onAnalyze, isLoading }) => {
  return (
    <Card className="glass-card p-6 hover-glow">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-2xl font-orbitron font-bold mb-1 gradient-text">Input Protein Sequence</h2>
          <p className="text-sm text-muted-foreground">Paste the amino-acid sequence below (single-letter codes)</p>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Protein Sequence</label>
          <Textarea
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            rows={4}
            placeholder="Enter amino acid sequence (e.g., MKTFFVLVLLLAAAGVAGTQATQGNVKAAW)"
            className="min-h-[140px] font-mono text-sm bg-input/50 border-border"
          />
          <p className="text-xs text-muted-foreground mt-2">
            Valid amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
          </p>
        </div>

        <div className="flex gap-3">
          <Button
            onClick={onAnalyze}
            disabled={isLoading}
            className="flex-1 bg-primary hover:bg-primary-glow text-background font-semibold"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                🧬 Run Analysis Pipeline
              </>
            )}
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default PipelineInput;
