"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Target, AlertCircle, CheckCircle2, RefreshCw } from "lucide-react";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import Disclaimer from "@/components/Disclaimer";

const DEFAULT_LEAD = {
  TotalVisits: 5,
  "Total Time Spent on Website": 800,
  "Page Views Per Visit": 2.5,
  "Asymmetrique Activity Score": 15,
  "Asymmetrique Profile Score": 15,
  "Lead Origin": "API",
  "Lead Source": "Organic Search",
  Specialization: "Select",
  "What is your current occupation": "Unemployed",
  "Last Activity": "Page Visited on Website",
  Country: "India",
  "Lead Quality": "Low in Relevance",
  "Do Not Email": "No",
  "Do Not Call": "No"
};

export default function LeadScoringDemo() {
  const [formData, setFormData] = useState(DEFAULT_LEAD);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? (value === "" ? 0 : Number(value)) : value
    }));
  };

  const handleReset = () => {
    setFormData(DEFAULT_LEAD);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // Map frontend fields to API expectations if needed
      const res = await api.predict("lead_scoring", { lead: formData });
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to get lead score. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--accent-mint-dim)] text-[var(--accent-mint)]">
          <Target className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-100">Lead Conversion Scorer</h2>
          <p className="text-slate-400 text-xs mt-0.5">Determine the probability of conversion based on activity markers and CRM metrics.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <form onSubmit={handleSubmit} className="lg:col-span-2 glass-card p-6 border-[var(--border-subtle)] space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {/* Numeric fields */}
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Total Visits</label>
              <input
                type="number"
                name="TotalVisits"
                value={formData.TotalVisits}
                onChange={handleInputChange}
                className="input-field"
                min="0"
                max="1000"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Time on Site (Sec)</label>
              <input
                type="number"
                name="Total Time Spent on Website"
                value={formData["Total Time Spent on Website"]}
                onChange={handleInputChange}
                className="input-field"
                min="0"
                max="86400"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Pages Per Visit</label>
              <input
                type="number"
                step="0.1"
                name="Page Views Per Visit"
                value={formData["Page Views Per Visit"]}
                onChange={handleInputChange}
                className="input-field"
                min="0"
                max="100"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Asymmetrique Activity</label>
              <input
                type="number"
                name="Asymmetrique Activity Score"
                value={formData["Asymmetrique Activity Score"]}
                onChange={handleInputChange}
                className="input-field"
                min="0"
                max="20"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Asymmetrique Profile</label>
              <input
                type="number"
                name="Asymmetrique Profile Score"
                value={formData["Asymmetrique Profile Score"]}
                onChange={handleInputChange}
                className="input-field"
                min="0"
                max="20"
                required
              />
            </div>

            {/* Select fields */}
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Lead Origin</label>
              <select name="Lead Origin" value={formData["Lead Origin"]} onChange={handleInputChange} className="input-field">
                <option value="API">API</option>
                <option value="Landing Page Submission">Landing Page</option>
                <option value="Lead Add Form">Lead Add Form</option>
                <option value="Quick Add Form">Quick Add Form</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Lead Source</label>
              <select name="Lead Source" value={formData["Lead Source"]} onChange={handleInputChange} className="input-field">
                <option value="Organic Search">Organic Search</option>
                <option value="Google">Google</option>
                <option value="Direct Traffic">Direct Traffic</option>
                <option value="Olark Chat">Olark Chat</option>
                <option value="Reference">Reference</option>
                <option value="Referral Sites">Referral Sites</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Specialization</label>
              <select name="Specialization" value={formData.Specialization} onChange={handleInputChange} className="input-field">
                <option value="Select">Select</option>
                <option value="Finance Management">Finance Management</option>
                <option value="Human Resource Management">HR Management</option>
                <option value="Marketing Management">Marketing Management</option>
                <option value="Operations Management">Operations Management</option>
                <option value="IT Projects Management">IT Projects Management</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Current Occupation</label>
              <select name="What is your current occupation" value={formData["What is your current occupation"]} onChange={handleInputChange} className="input-field">
                <option value="Unemployed">Unemployed</option>
                <option value="Student">Student</option>
                <option value="Working Professional">Working Professional</option>
                <option value="Businessman">Businessman</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Last Activity</label>
              <select name="Last Activity" value={formData["Last Activity"]} onChange={handleInputChange} className="input-field">
                <option value="Page Visited on Website">Page Visited</option>
                <option value="Email Opened">Email Opened</option>
                <option value="SMS Sent">SMS Sent</option>
                <option value="Olark Chat Conversation">Olark Chat</option>
                <option value="Modified">Modified</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Lead Quality</label>
              <select name="Lead Quality" value={formData["Lead Quality"]} onChange={handleInputChange} className="input-field">
                <option value="Low in Relevance">Low Relevance</option>
                <option value="Might be">Might be</option>
                <option value="Not Sure">Not Sure</option>
                <option value="Worst">Worst</option>
                <option value="High in Relevance">High Relevance</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Country</label>
              <select name="Country" value={formData.Country} onChange={handleInputChange} className="input-field">
                <option value="India">India</option>
                <option value="United States">United States</option>
                <option value="United Arab Emirates">UAE</option>
                <option value="Singapore">Singapore</option>
                <option value="United Kingdom">United Kingdom</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Do Not Email</label>
              <select name="Do Not Email" value={formData["Do Not Email"]} onChange={handleInputChange} className="input-field">
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">Do Not Call</label>
              <select name="Do Not Call" value={formData["Do Not Call"]} onChange={handleInputChange} className="input-field">
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-3 pt-2 border-t border-[var(--border-subtle)]">
            <button type="submit" disabled={loading} className="btn-primary flex-1 sm:flex-initial">
              {loading ? "Calculating Score..." : "Calculate Conversion Probability"}
            </button>
            <button type="button" onClick={handleReset} className="btn-secondary flex items-center justify-center p-2.5">
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </form>

        <div className="flex flex-col gap-6">
          <div className="glass-card p-6 border-[var(--border-subtle)] bg-slate-900/10 flex-1 flex flex-col justify-center">
            {loading && <LoadingSkeleton />}
            
            {error && (
              <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/5 text-slate-200 space-y-2">
                <div className="flex items-center gap-2 text-red-500 font-bold text-sm">
                  <AlertCircle className="h-4 w-4" /> Inference Failure
                </div>
                <p className="text-xs leading-relaxed text-slate-400">{error}</p>
              </div>
            )}

            {!loading && !result && !error && (
              <div className="text-center space-y-3 py-12">
                <Target className="h-10 w-10 text-slate-650 mx-auto" />
                <div className="space-y-1">
                  <h3 className="text-sm font-bold text-slate-350">Awaiting Input</h3>
                  <p className="text-slate-500 text-xs max-w-xs mx-auto leading-relaxed">
                    Submit the feature parameters on the left to invoke the model registry and view the score card.
                  </p>
                </div>
              </div>
            )}

            {!loading && result && (
              <div className="space-y-6">
                <div className="space-y-1">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Prediction Outcome</h3>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-4xl font-extrabold tracking-tight ${result.prediction.converted ? "text-[var(--accent-mint)]" : "text-amber-500"}`}>
                      {(result.prediction.conversion_probability * 100).toFixed(0)}%
                    </span>
                    <span className="text-xs font-semibold text-slate-400">conversion probability</span>
                  </div>
                </div>

                {/* Score badge indicator */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs font-medium text-slate-400">
                    <span>Probability Intensity</span>
                    <span>{result.prediction.conversion_probability >= 0.85 ? "Hot" : result.prediction.conversion_probability >= 0.65 ? "Warm" : "Cold"}</span>
                  </div>
                  <div className="confidence-bar">
                    <div
                      className={`confidence-bar-fill ${
                        result.prediction.conversion_probability >= 0.85
                          ? "bg-[var(--accent-mint)]"
                          : result.prediction.conversion_probability >= 0.65
                          ? "bg-amber-500"
                          : "bg-blue-500"
                      }`}
                      style={{ width: `${result.prediction.conversion_probability * 100}%` }}
                    />
                  </div>
                </div>

                <div className="space-y-2 border-t border-[var(--border-subtle)] pt-4">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                    <CheckCircle2 className="h-4 w-4 text-[var(--accent-mint)]" /> Explanation
                  </div>
                  <p className="text-xs leading-relaxed text-slate-400">{result.explanation}</p>
                </div>

                <Disclaimer />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
