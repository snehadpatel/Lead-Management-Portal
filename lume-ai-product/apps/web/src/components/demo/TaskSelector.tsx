"use client";

import { useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Target, Users, MessageSquare, Search, LineChart } from "lucide-react";
import LeadScoringDemo from "./LeadScoringDemo";
import PersonaClusterDemo from "./PersonaClusterDemo";
import SentimentDemo from "./SentimentDemo";
import SemanticSearchDemo from "./SemanticSearchDemo";
import NavForecastDemo from "./NavForecastDemo";

const TASKS = [
  { id: "lead_scoring", label: "Lead Scoring", icon: Target, component: LeadScoringDemo },
  { id: "investor_cluster", label: "Investor Persona", icon: Users, component: PersonaClusterDemo },
  { id: "sentiment", label: "Sentiment Analysis", icon: MessageSquare, component: SentimentDemo },
  { id: "semantic_search", label: "Semantic Search", icon: Search, component: SemanticSearchDemo },
  { id: "nav_forecast", label: "NAV Forecast", icon: LineChart, component: NavForecastDemo },
];

export default function TaskSelector() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState("lead_scoring");

  useEffect(() => {
    const task = searchParams.get("task");
    if (task && TASKS.some((t) => t.id === task)) {
      setActiveTab(task);
    }
  }, [searchParams]);

  const handleTabChange = (id: string) => {
    setActiveTab(id);
    const params = new URLSearchParams(searchParams);
    params.set("task", id);
    router.replace(`/demo?${params.toString()}`);
  };

  const SelectedComponent = TASKS.find((t) => t.id === activeTab)?.component || LeadScoringDemo;

  return (
    <div className="space-y-8">
      {/* Segmented controls tabs */}
      <div className="flex border-b border-[var(--border-subtle)] overflow-x-auto pb-px gap-2">
        {TASKS.map((task) => {
          const Icon = task.icon;
          const isActive = activeTab === task.id;
          return (
            <button
              key={task.id}
              onClick={() => handleTabChange(task.id)}
              className={`tab-button flex items-center gap-2 pb-3 px-4 ${isActive ? "active font-bold border-b-2 border-[var(--accent-mint)]" : ""}`}
            >
              <Icon className="h-4 w-4" />
              <span>{task.label}</span>
            </button>
          );
        })}
      </div>

      {/* Render the active demo */}
      <div className="transition-all duration-300">
        <SelectedComponent />
      </div>
    </div>
  );
}
