import { Suspense } from "react";
import TaskSelector from "@/components/demo/TaskSelector";
import LoadingSkeleton from "@/components/LoadingSkeleton";

export const metadata = {
  title: "Live Model Demos — Lume AI Sandbox",
  description: "Try Lume AI's classification, clustering, sentiment, semantic search, and forecasting models live in the sandbox.",
};

export default function DemoPage() {
  return (
    <div className="space-y-6 py-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-100 sm:text-4xl">Live Sandbox Demos</h1>
        <p className="text-slate-400 text-sm max-w-3xl leading-relaxed">
          Interact directly with our pre-trained model bundles using real-time API inference. Toggle between the tabs below to submit payloads.
        </p>
      </div>

      <Suspense fallback={<div className="p-8 border border-[var(--border-subtle)] rounded-xl"><LoadingSkeleton /></div>}>
        <TaskSelector />
      </Suspense>
    </div>
  );
}
