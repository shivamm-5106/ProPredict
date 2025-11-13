// components/PipelineResults.tsx
import React from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";

interface PipelineResultsProps {
  onReset: () => void;
}

const PipelineResults: React.FC<PipelineResultsProps> = ({ onReset }) => {
  return (
    <Card className="glass-card p-6 animate-fade-in bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <span className="text-3xl mr-3">✓</span>
          <div>
            <h2 className="text-2xl font-orbitron font-bold">Analysis Complete</h2>
            <p className="text-muted-foreground">All models have successfully processed the sequence. Results are displayed above.</p>
          </div>
        </div>

        <div>
          <Button onClick={onReset} className="bg-white/10 hover:bg-white/20">
            Run New Analysis
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default PipelineResults;
