// components/PipelineStep.tsx
import React from "react";
import { Card } from "./ui/card";

interface PipelineStepProps {
  title: string;
  badge?: string;
  status: string;
  description?: string;
  dimmed?: boolean;
  children?: React.ReactNode;
}

const statusColor = (status: string) => {
  const map: Record<string, string> = {
    Waiting: "bg-gray-500",
    "Processing...": "bg-yellow-500",
    Complete: "bg-green-500",
    "Complete (Cached)": "bg-green-500",
    "Complete (Placeholder)": "bg-green-500",
    Error: "bg-red-500",
  };
  return map[status] || "bg-gray-500";
};

const PipelineStep: React.FC<PipelineStepProps> = ({ title, badge, status, description, dimmed, children }) => {
  return (
    <Card className={`glass-card p-6 hover-glow transition-all ${dimmed ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center mr-3 font-bold" style={{ background: "linear-gradient(90deg,#7c3aed,#06b6d4)" }}>
            <span className="text-sm">{badge}</span>
          </div>
          <div>
            <h3 className="text-xl font-semibold">{title}</h3>
            {description && <p className="text-sm text-muted-foreground">{description}</p>}
          </div>
        </div>

        <div>
          <span className={`px-3 py-1 ${statusColor(status)} rounded-full text-xs`}>{status}</span>
        </div>
      </div>

      {children}
    </Card>
  );
};

export default PipelineStep;
